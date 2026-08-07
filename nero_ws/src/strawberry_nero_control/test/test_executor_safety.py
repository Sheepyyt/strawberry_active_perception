"""Safety regression tests for the single NERO command backend."""

import numpy as np
import pytest

from builtin_interfaces.msg import Time

from strawberry_nero_control.control_node import NeroExecutor


JOINT_NAMES = [f"joint{index}" for index in range(1, 8)]


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def error(self, _message):
        pass


class _FakeNode:
    def __init__(self):
        self.logger = _FakeLogger()
        self.states = []

    def get_logger(self):
        return self.logger

    def set_simulated_state(self, positions, velocities):
        self.states.append((positions.copy(), velocities.copy()))

    def get_clock(self):
        return _FakeClock()


class _FakeNow:
    @staticmethod
    def to_msg():
        return Time()


class _FakeClock:
    @staticmethod
    def now():
        return _FakeNow()


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _FakeEmergencyClient:
    def __init__(self):
        self.calls = 0

    @staticmethod
    def service_is_ready():
        return True

    def call_async(self, _request):
        self.calls += 1


def test_soft_hold_is_skipped_until_this_action_sends_motion():
    """A rejected or pre-planning canceled goal must emit no joint command."""
    node = _FakeNode()
    executor = NeroExecutor(node, None, None, JOINT_NAMES, True)
    command_count_before = executor.command_count

    held = executor.hold_if_commanded_since(
        np.zeros(7),
        "canceled before execution",
        command_count_before,
    )

    assert not held
    assert executor.command_count == 0
    assert node.states == []
    assert "sent no motion command" in node.logger.warnings[-1]


def test_soft_hold_is_allowed_after_this_action_sends_motion():
    """A tracking failure during motion may still command a measured hold."""
    node = _FakeNode()
    publisher = _FakePublisher()
    emergency_client = _FakeEmergencyClient()
    executor = NeroExecutor(
        node,
        publisher,
        emergency_client,
        JOINT_NAMES,
        False,
    )
    command_count_before = executor.command_count
    commanded = np.full(7, 0.1)
    measured = np.full(7, 0.08)

    executor.send(commanded)
    held = executor.hold_if_commanded_since(
        measured,
        "tracking error",
        command_count_before,
    )

    assert held
    assert executor.command_count == 2
    assert len(publisher.messages) == 2
    np.testing.assert_allclose(publisher.messages[-1].position, measured)
    assert emergency_client.calls == 1


def test_previous_action_command_does_not_trigger_new_action_hold():
    """A new action uses a fresh baseline instead of all-time history."""
    node = _FakeNode()
    executor = NeroExecutor(node, None, None, JOINT_NAMES, True)
    executor.send(np.full(7, 0.1))
    command_count_before = executor.command_count

    held = executor.hold_if_commanded_since(
        np.zeros(7),
        "new action canceled before execution",
        command_count_before,
    )

    assert not held
    assert executor.command_count == 1
    assert len(node.states) == 1


def test_invalid_send_does_not_unlock_soft_hold():
    """A rejected partial command is never counted as actual motion."""
    node = _FakeNode()
    executor = NeroExecutor(node, None, None, JOINT_NAMES, True)
    command_count_before = executor.command_count

    with pytest.raises(ValueError):
        executor.send(np.zeros(6))

    held = executor.hold_if_commanded_since(
        np.zeros(7),
        "invalid command",
        command_count_before,
    )

    assert not held
    assert executor.command_count == 0
    assert node.states == []
