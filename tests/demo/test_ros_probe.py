"""Stopping must cover recovery actions even if another cancel service fails."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("demo_ros_probe", ROOT / "tools/demo/ros_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def cancellation_fixture(first_failure=None):
    events = []
    report = {}

    class Client:
        def __init__(self, name):
            self.action = name.split("/")[-3]

        def wait_for_service(self, **kwargs):
            events.append(("wait", self.action))
            return not (self.action == "move_to_pose" and first_failure == "missing")

        def call_async(self, request):
            events.append(("cancel", self.action))
            failing = self.action == "move_to_pose"
            response = SimpleNamespace(return_code=1 if failing and first_failure == "rejected" else 0,
                                       goals_canceling=[object()] * (2 if failing else 1))
            return SimpleNamespace(done=lambda: not (failing and first_failure == "timeout"),
                                   result=lambda: response)

    node = SimpleNamespace(create_client=lambda kind, name: Client(name),
                           destroy_client=lambda client: events.append(("destroy", client.action)))
    service = SimpleNamespace(Request=object)
    spin = lambda *args, **kwargs: None
    close = lambda: events.append(("close", "gates"))
    return node, service, spin, close, report, events


def test_halt_counts_normal_and_recovery_goals_and_closes_gates():
    *args, events = cancellation_fixture()
    probe.cancel_motion_goals(*args)
    report = args[-1]
    assert report["goals_canceling"] == 3
    assert report["cancel_actions"]["recover_to_safe"]["goals_canceling"] == 1
    assert events[-1] == ("close", "gates")


@pytest.mark.parametrize("failure", ["missing", "timeout", "rejected"])
def test_first_cancel_failure_still_cancels_recovery_and_closes_gates(failure):
    *args, events = cancellation_fixture(failure)
    with pytest.raises(RuntimeError, match="move_to_pose"):
        probe.cancel_motion_goals(*args)
    assert ("cancel", "recover_to_safe") in events
    assert events[-1] == ("close", "gates")
    assert "error" in args[-1]["cancel_actions"]["move_to_pose"]


def driver_cleanup_fixture(failing_call=None):
    calls = []
    report = {"success": False}
    service = SimpleNamespace(Request=lambda **kwargs: SimpleNamespace(**kwargs))

    def call(kind, name, request, timeout):
        # No controller services exist in this fixture, as after its exit.
        assert name in ("/control_enable", "/enable_agx_arm")
        assert request.data is False
        calls.append((name, request.data))
        if len(calls) == failing_call:
            raise RuntimeError("driver did not acknowledge request")
        return SimpleNamespace(success=True, message="acknowledged")

    return call, service, report, calls


def test_close_driver_gate_requires_no_controller_or_enable_service():
    call, service, report, calls = driver_cleanup_fixture()
    probe.close_driver_gate(call, service, report)
    assert calls == [("/control_enable", False)]
    assert report["driver_gate_false_ack_count"] == 1
    assert "driver_disable_acknowledged" not in report


def test_disable_driver_only_sends_false_and_requires_both_close_acknowledgements():
    call, service, report, calls = driver_cleanup_fixture()
    probe.disable_driver(call, service, report)
    assert calls == [("/control_enable", False), ("/enable_agx_arm", False),
                     ("/control_enable", False)]
    assert report["driver_gate_false_ack_count"] == 2
    assert report["driver_disable_acknowledged"] is True


@pytest.mark.parametrize("failing_call", [1, 2, 3])
def test_disable_driver_does_not_report_success_when_any_ack_is_missing(failing_call):
    call, service, report, calls = driver_cleanup_fixture(failing_call)
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        probe.disable_driver(call, service, report)
    assert report["success"] is False
    if failing_call == 1:
        assert calls == [("/control_enable", False)]
    else:
        # Even a failed disable is followed by another gate-close attempt.
        assert calls[-1] == ("/control_enable", False)
        assert len(calls) == 3
