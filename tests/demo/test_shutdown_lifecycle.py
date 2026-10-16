"""Offline regressions for interrupted shutdown and owned launch children."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("demo_shutdown_runtime", ROOT / "tools/demo/demo_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


@pytest.fixture
def isolated_runtime(tmp_path, monkeypatch):
    folder = tmp_path / ".runtime"
    folder.mkdir()
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "RUNTIME", folder)
    monkeypatch.setattr(runtime, "STATE", folder / "state.json")
    monkeypatch.setattr(runtime, "ABORT", folder / "abort_requested")
    # Any accidental ROS or hardware probe in these regressions must fail.
    monkeypatch.setattr(runtime, "probe", lambda *args, **kwargs: pytest.fail("unexpected ROS probe"))
    return tmp_path


@pytest.fixture
def orphaned_robot(isolated_runtime):
    child_path = isolated_runtime / "child.pid"
    source = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_path)!r}).write_text(str(child.pid)); "
        "time.sleep(0.05)"
    )
    record = runtime.spawn("robot", [sys.executable, "-c", source], isolated_runtime / "logs")
    deadline = time.monotonic() + 3
    try:
        while time.monotonic() < deadline:
            if child_path.exists() and not runtime.alive(record) and runtime.group_members(record):
                break
            time.sleep(.02)
        child_pid = int(child_path.read_text())
        assert not runtime.alive(record), "inert launch parent did not exit"
        assert child_pid in runtime.group_members(record)
        yield record, child_pid
    finally:
        if runtime.group_members(record):
            os.killpg(record["pid"], signal.SIGKILL)
            deadline = time.monotonic() + 2
            while runtime.group_members(record) and time.monotonic() < deadline:
                time.sleep(.02)


def confirmed_state(record):
    return {
        "services": {"robot": record},
        "motor_enable_requested": False,
        "shutdown": {"phase": "disable_confirmed", "robot": runtime.robot_identity(record)},
    }


def test_orphan_is_owned_duplicate_rejected_and_group_stops(orphaned_robot, isolated_runtime):
    record, child_pid = orphaned_robot
    assert runtime.owned_processes(runtime.read_state(), "robot")
    with pytest.raises(RuntimeError, match="已运行"):
        runtime.spawn("robot", [sys.executable, "-c", "raise SystemExit(99)"], isolated_runtime / "other_logs")
    assert runtime.read_state()["services"]["robot"]["pid"] == record["pid"]
    runtime.stop_service("robot")
    assert not runtime.group_members(record)
    assert runtime.pid_identity(child_pid) is None
    assert "robot" not in runtime.read_state()["services"]


@pytest.mark.parametrize("mutation", [
    "receipt_pid", "receipt_start_ticks", "receipt_boot", "old_boot", "motor_enabled", "motor_unknown", "wrong_phase",
])
def test_disable_receipt_must_match_process_boot_and_motor_state(isolated_runtime, mutation):
    record = {"pid": 12345, "start_ticks": "900", "boot_id": runtime.BOOT_ID}
    state = confirmed_state(record)
    assert runtime.disable_confirmed(state)
    invalid = copy.deepcopy(state)
    if mutation == "receipt_pid":
        invalid["shutdown"]["robot"]["pid"] += 1
    elif mutation == "receipt_start_ticks":
        invalid["shutdown"]["robot"]["start_ticks"] = "901"
    elif mutation == "receipt_boot":
        invalid["shutdown"]["robot"]["boot_id"] = "another-boot"
    elif mutation == "old_boot":
        invalid["services"]["robot"]["boot_id"] = "another-boot"
        invalid["shutdown"]["robot"]["boot_id"] = "another-boot"
    elif mutation == "motor_enabled":
        invalid["motor_enable_requested"] = True
    elif mutation == "motor_unknown":
        invalid.pop("motor_enable_requested")
    else:
        invalid["shutdown"]["phase"] = "requested"
    assert not runtime.disable_confirmed(invalid)
    runtime.write_json(runtime.STATE, invalid)
    with pytest.raises(RuntimeError, match="失能确认"):
        runtime.finish_robot_shutdown()


def test_reconcile_resumes_verified_disable_without_ros(orphaned_robot, isolated_runtime, monkeypatch):
    record, _ = orphaned_robot
    state = confirmed_state(record)
    state["startup_dir"] = str(isolated_runtime)
    runtime.write_json(runtime.STATE, state)
    original_stop = runtime.stop_service

    def interrupted_stop(name, **kwargs):
        if name == "robot":
            raise KeyboardInterrupt("simulated exit between controller and driver cleanup")
        original_stop(name, **kwargs)

    monkeypatch.setattr(runtime, "stop_service", interrupted_stop)
    with pytest.raises(KeyboardInterrupt):
        runtime.finish_robot_shutdown()
    assert runtime.disable_confirmed(runtime.read_state())
    assert runtime.group_members(record)
    monkeypatch.setattr(runtime, "stop_service", original_stop)
    runtime.reconcile_shutdown()
    result = runtime.read_state()
    assert result["shutdown"]["phase"] == "complete"
    assert result["motor_enable_requested"] is False
    assert not result["services"]
    assert not runtime.group_members(record)


def test_prepare_reports_own_orphan_before_ros_discovery(orphaned_robot, monkeypatch):
    record, child_pid = orphaned_robot
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: pytest.fail("orphan detection must precede ROS discovery"))
    monkeypatch.setattr(runtime, "stop_service", lambda *args, **kwargs: pytest.fail("unverified orphan must not be killed"))
    with pytest.raises(RuntimeError, match="本脚本上轮启动器已退出") as error:
        runtime.prepare(argparse.Namespace())
    assert "不是其他终端" in str(error.value)
    assert child_pid in runtime.group_members(record)


def test_prepare_never_cleans_foreign_node_process(isolated_runtime, monkeypatch):
    # A real inert process exists outside the state registry, while a mocked
    # discovery response names a conflicting driver node. No ROS is contacted.
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "/agx_arm_ctrl_single_node\n"))
        monkeypatch.setattr(runtime, "stop_service", lambda *args, **kwargs: pytest.fail("foreign node must not be cleaned"))
        monkeypatch.setattr(runtime, "spawn", lambda *args, **kwargs: pytest.fail("must not launch duplicate driver"))
        with pytest.raises(RuntimeError, match="其他终端"):
            runtime.prepare(argparse.Namespace())
        assert foreign.poll() is None
        assert not runtime.STATE.exists()
    finally:
        foreign.terminate()
        foreign.wait(timeout=3)
