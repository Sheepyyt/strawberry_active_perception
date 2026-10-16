"""Offline checks for run isolation and the demo process lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("demo_runtime", ROOT / "tools/demo/demo_runtime.py")
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
    return tmp_path


def test_missing_process_identity_is_never_alive(monkeypatch):
    monkeypatch.setattr(runtime, "pid_identity", lambda pid: None)
    assert not runtime.alive({"pid": 999999, "start_ticks": None})
    assert not runtime.alive({"pid": 999999, "start_ticks": "123"})


def test_new_runs_never_reuse_directory_or_scene(isolated_runtime):
    first, second = runtime.new_directory(), runtime.new_directory()
    assert first != second
    assert first.name != second.name
    assert first.is_dir() and second.is_dir()


def test_motion_plan_hash_and_scene_bind_to_this_run(isolated_runtime):
    directory = runtime.new_directory()
    payload = b'{"status":"passed_preview_only","scene_id":"fresh"}\n'
    (directory / "preview.json").write_bytes(payload)
    args = argparse.Namespace(max_step=.025, coverage=.8, steps=8)
    command = runtime.supervisor_command(directory, args, True)
    assert f"scene_id:={directory.name}" in command
    assert f"execution_plan_sha256:={hashlib.sha256(payload).hexdigest()}" in command
    assert f"authorization_receipt_path:={directory / 'authorization_receipt.json'}" in command
    assert "max_step_m:=0.025" in command
    assert "execution_authorization_token:=EXECUTE_REAL_NBV_LARGE_SESSION_8" in command


def test_default_step_comes_from_current_planner_profile(isolated_runtime, monkeypatch):
    profile = isolated_runtime / "supervisor.yaml"
    profile.write_text("real_nbv_supervisor:\n  ros__parameters:\n    max_step_m: 0.075\n")
    monkeypatch.setattr(runtime, "SUPERVISOR_CONFIG", profile)
    calls = []
    monkeypatch.setattr(runtime, "run_session", lambda args, **kwargs: calls.append(args))
    assert runtime.main(["run", "--workspace-clear"]) == 0
    assert calls[0].max_step == .075


def test_optional_step_can_narrow_but_not_expand_native_profile(isolated_runtime, monkeypatch):
    monkeypatch.setattr(runtime, "profile_max_step", lambda: .1)
    calls = []
    monkeypatch.setattr(runtime, "run_session", lambda args, **kwargs: calls.append(args))
    assert runtime.main(["preview", "--max-step", ".05"]) == 0
    assert calls[0].max_step == .05
    with pytest.raises(SystemExit) as error:
        runtime.main(["run", "--workspace-clear", "--max-step", ".101"])
    assert error.value.code == 2
    assert len(calls) == 1


def test_prepare_robot_command_never_enables_or_moves(isolated_runtime):
    command = runtime.service_command("robot", isolated_runtime)
    assert "startup_enable:=false" in command
    assert "speed_percent:=10" in command
    assert "precision_test_mode:=true" in command
    assert "allow_limit_recovery_execution:=true" in command
    assert not any("move_home" in word for word in command)


def test_optional_standalone_controller_must_be_alive(isolated_runtime, monkeypatch):
    runtime.write_json(runtime.STATE, {"services": {
        **{name: {"running": True} for name in runtime.REQUIRED_SERVICES},
        "controller": {"running": False},
    }})
    monkeypatch.setattr(runtime, "alive", lambda record: bool(record and record["running"]))
    with pytest.raises(RuntimeError, match="controller"):
        runtime.check_services()


def test_shutdown_disables_before_stopping_standalone_controller(isolated_runtime, monkeypatch):
    runtime.write_json(runtime.STATE, {"startup_dir": str(isolated_runtime), "services": {
        "robot": {"running": True, "pid": 100, "start_ticks": "123"},
        "controller": {"running": True},
    }})
    monkeypatch.setattr(runtime, "alive", lambda record: bool(record and record["running"]))
    monkeypatch.setattr(runtime, "owned_processes", lambda state, name: name in state["services"])
    monkeypatch.setattr(runtime, "controller_present", lambda state: True)
    monkeypatch.setattr(runtime, "interrupt_supervisor", lambda: None)
    calls = []
    monkeypatch.setattr(runtime, "probe", lambda operation, **kwargs: calls.append(operation))

    def stop_service(name, **kwargs):
        if name in ("controller", "robot"):
            assert runtime.disable_confirmed(runtime.read_state())
        calls.append(f"stop_{name}")
        state = runtime.read_state()
        state["services"].pop(name, None)
        runtime.write_json(runtime.STATE, state)

    monkeypatch.setattr(runtime, "stop_service", stop_service)
    runtime.stop_locked(argparse.Namespace(arm_supported=True), shutdown=True)
    assert calls.index("disable-driver") < calls.index("stop_controller") < calls.index("stop_robot")
    assert calls.index("stop_tf") < calls.index("disable-driver")
    assert runtime.read_state()["shutdown"]["phase"] == "complete"
    assert runtime.read_state()["motor_enable_requested"] is False


def test_prepare_existing_session_adds_missing_read_only_tf(isolated_runtime, monkeypatch):
    runtime.write_json(runtime.STATE, {"startup_dir": str(isolated_runtime), "services": {
        name: {"running": True} for name in ("robot", "controller", "camera", "mask", "nbv")
    }})
    monkeypatch.setattr(runtime, "alive", lambda record: bool(record and record["running"]))
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, ""))
    spawned = []
    monkeypatch.setattr(runtime, "spawn", lambda name, cmd, directory: spawned.append((name, cmd)))
    probes = []
    monkeypatch.setattr(runtime, "probe", lambda operation, **kwargs: probes.append(operation))
    runtime.prepare(argparse.Namespace())
    assert spawned == [("tf", [runtime.PYTHON, str(isolated_runtime / "tools/calibration/read_only_nero_tf.launch.py")])]
    assert probes == ["ready"]


def test_run_requires_read_only_tf_process(isolated_runtime, monkeypatch):
    runtime.write_json(runtime.STATE, {"services": {
        name: {"running": True} for name in ("robot", "camera", "mask", "nbv")
    }})
    monkeypatch.setattr(runtime, "alive", lambda record: bool(record and record["running"]))
    with pytest.raises(RuntimeError, match="tf"):
        runtime.check_services()


def test_abort_prevents_launch_before_any_process(isolated_runtime):
    runtime.ABORT.touch()
    with pytest.raises(RuntimeError, match="请求停止"):
        runtime.spawn("fake", [sys.executable, "-c", "raise SystemExit(99)"], isolated_runtime / "logs")
    assert not runtime.STATE.exists()


def test_failed_initialization_has_shareable_result(isolated_runtime):
    directory = runtime.new_directory()
    runtime.write_json(directory / "initialization/report.json", {"passed": False, "error": "no strawberry"})
    runtime.write_json(directory / "failure.json", {"error": "no strawberry"})
    runtime.render(directory)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["initialization_passed"] is False
    assert "no strawberry" in (directory / "index.html").read_text()
    assert (directory / "manifest.json").exists()


def test_zero_motion_report_and_latest_link_remain_truthful(isolated_runtime, monkeypatch):
    directory = runtime.new_directory()
    runtime.write_json(runtime.STATE, {"services": {}, "current_run": str(directory)})
    runtime.write_json(directory / "execution.json", {
        "status": "failed", "reason": "before dispatch", "motion_goal_count": 0, "motion_steps": [],
    })
    # No motion or map means no trajectory chart can truthfully be generated.
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: pytest.fail("unexpected renderer"))
    runtime.render(directory)
    overview = json.loads((directory / "overview.json").read_text())
    assert overview["completed_nbv_actions"] == 0
    assert overview["final_roi_ray_coverage"] is None
    assert not json.loads((directory / "summary.json").read_text())["render_errors"]
    latest = isolated_runtime / "results/index.html"
    original = latest.read_text()
    assert directory.name in original
    assert 'http-equiv="refresh"' not in original
    assert "主动感知" in original
    explanation = (isolated_runtime / "results/结果说明.txt").read_text()
    assert "before dispatch" in explanation
    assert "<!doctype" not in explanation
    historical = runtime.new_directory()
    runtime.render(historical)
    assert latest.read_text() == original
    assert (isolated_runtime / "results/结果说明.txt").read_text() == explanation


def test_result_command_reads_without_robot_services_or_lock(isolated_runtime, monkeypatch, capsys):
    directory = runtime.new_directory()
    runtime.write_json(runtime.STATE, {"services": {}, "current_run": str(directory)})
    (directory / "结果说明.txt").write_text("本轮完成 4 次主动感知运动。\n")
    (directory / "index.html").write_text("<html>report</html>")
    monkeypatch.setattr(runtime, "check_services", lambda: pytest.fail("hardware service check"))
    monkeypatch.setattr(runtime, "acquire_operation_lock", lambda: pytest.fail("robot operation lock"))
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: pytest.fail("subprocess requested"))
    assert runtime.main(["result"]) == 0
    assert "完成 4 次" in capsys.readouterr().out


def test_weekly_analysis_uses_saved_run_without_robot_lock_or_state_changes(isolated_runtime, monkeypatch):
    from tools.analysis import weekly_report

    directory = runtime.new_directory()
    historical = runtime.new_directory()
    runtime.write_json(runtime.STATE, {"services": {}, "current_run": str(directory)})
    state_bytes = runtime.STATE.read_bytes()
    calls = []
    monkeypatch.setattr(weekly_report, "generate_weekly", lambda *args: calls.append(args))
    monkeypatch.setattr(runtime, "check_services", lambda: pytest.fail("hardware service check"))
    monkeypatch.setattr(runtime, "acquire_operation_lock", lambda: pytest.fail("robot operation lock"))
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: pytest.fail("hardware subprocess"))
    assert runtime.main(["analyze"]) == 0
    assert runtime.main(["analyze", str(historical)]) == 0
    assert calls == [(directory, isolated_runtime / "results/weekly" / directory.name),
                     (historical, isolated_runtime / "results/weekly" / historical.name)]
    assert runtime.STATE.read_bytes() == state_bytes


def test_stop_reaps_entire_owned_group(isolated_runtime):
    # These are inert local Python sleeps, not ROS or hardware processes.
    source = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"
    record = runtime.spawn("camera", [sys.executable, "-c", source], isolated_runtime / "logs")
    deadline = time.monotonic() + 3
    try:
        while len(runtime.group_members(record)) < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert len(runtime.group_members(record)) == 2
        runtime.stop_service("camera")
        assert not runtime.group_members(record)
        assert "camera" not in runtime.read_state()["services"]
    finally:
        if runtime.group_members(record):
            import os
            import signal
            os.killpg(record["pid"], signal.SIGKILL)


def test_reused_leader_pid_is_not_owned(isolated_runtime):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        ticks = runtime.pid_identity(process.pid)
        assert not runtime.group_members({"pid": process.pid, "start_ticks": str(int(ticks) - 1)})
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_execution_map_selection_never_uses_preview(isolated_runtime):
    import numpy as np

    paths = []
    for identity, scene in (("preview", "this_run"), ("exec_start", "this_run"), ("post_1", "this_run"), ("post_1", "other_run")):
        path = isolated_runtime / f"{scene}_{identity}.npz"
        np.savez(path, observation_id=identity, scene_id=scene)
        paths.append(path)
    document = {"scene_id": "this_run", "corrected_observation": {"observation_id": "exec_start"},
                "motion_steps": [{"post_map_update": {"observation_id": "post_1"}}]}
    selected = runtime.execution_map_snapshots(document, paths)
    assert selected == [paths[1], paths[2]]
    assert runtime.execution_map_snapshots({"scene_id": "this_run"}, paths) == []


def test_stop_lock_blocks_new_actions_but_can_interrupt_existing(isolated_runtime):
    import fcntl

    operation_lock = runtime.acquire_operation_lock()
    try:
        with (runtime.RUNTIME / "stop.lock").open("a") as stop_lock:
            # An operation holds only operation.lock after entering; stop can
            # immediately obtain its independent lock and cancel that operation.
            fcntl.flock(stop_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            operation_lock.close()
            # Even after the old operation exits, no new action can slip in
            # while stop is still closing services or disabling the arm.
            with pytest.raises(RuntimeError, match="停止流程正在进行"):
                runtime.acquire_operation_lock()
        next_operation = runtime.acquire_operation_lock()
        next_operation.close()
    finally:
        operation_lock.close()
