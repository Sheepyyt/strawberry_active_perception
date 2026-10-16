"""An interrupted real run must not erase its already recorded motion evidence."""

import json
from types import SimpleNamespace

import pytest

from strawberry_active_perception_bridge import real_nbv_supervisor as supervisor


def test_interrupt_keeps_latest_motion_evidence_without_ros_or_hardware(tmp_path, monkeypatch):
    destination = tmp_path / "execution.json"

    class InterruptedSupervisor:
        output_path = destination
        _audit = {"steps": [{"step_index": 1, "evidence": "already checkpointed"}]}
        _session_ledger = None
        _motion_command_count = 20
        _motion_goal_count = 1
        execute_requested = True
        scene_id = "this_run"
        report = SimpleNamespace(path="calibration.json", sha256="a" * 64, session_id="calibration")

        def run_once(self):
            raise KeyboardInterrupt()

        def destroy_node(self):
            pass

    monkeypatch.setattr(supervisor, "RealNBVSupervisor", InterruptedSupervisor)
    monkeypatch.setattr(supervisor.rclpy, "init", lambda **kwargs: None)
    monkeypatch.setattr(supervisor.rclpy, "ok", lambda: False)
    with pytest.raises(SystemExit, match="operator interrupted"):
        supervisor.main()
    document = json.loads(destination.read_text())
    assert document["status"] == "failed"
    assert document["safe_to_execute"] is False
    assert document["motion_goal_count"] == 1
    assert document["authorization_consumed"] is True
    assert document["operator_interrupted"] is True
    assert document["onsite_stop_required"] is True
    assert document["steps"][0]["evidence"] == "already checkpointed"
