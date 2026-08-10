"""Safety-state tests for the first supervised NERO real-arm tool."""

from contextlib import contextmanager
from types import SimpleNamespace
import time

import numpy as np
import pytest

from action_msgs.msg import GoalStatus
from agx_arm_msgs.msg import AgxArmStatus
from sensor_msgs.msg import JointState
from strawberry_nero_control.models import NERO_JOINT_NAMES
from strawberry_nero_control.real_smoke_test import (
    MOTION_START_POSITION_DELTA_RAD,
    MeasuredState,
    NeroRealSmokeTest,
    ROUNDTRIP_DISTANCE_M,
    SMOKE_DISTANCE_M,
    SmokeTestError,
    _MotionWatch,
    _argument_parser,
    local_x_target,
    recovery_result_allows_placo,
    require_confirmation,
    roundtrip_local_x_target,
    run_command,
    validate_smoke_ik,
)
from strawberry_nero_interfaces.action import RecoverToSafe
from strawberry_nero_interfaces.msg import IKResult


class _FakeSolver:
    @staticmethod
    def forward_kinematics(_joints, _frame):
        transform = np.eye(4)
        transform[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform[:3, 3] = [0.1, 0.2, 0.3]
        return transform


def _recovery_result(code, *, success, executed, safe):
    result = RecoverToSafe.Result()
    result.code = code
    result.success = success
    result.executed = executed
    result.robot_is_safe = safe
    return result


def _valid_ik_result():
    result = IKResult()
    result.success = True
    result.code = IKResult.SUCCESS
    result.reason = "ok"
    result.solution_joint_state.name = list(NERO_JOINT_NAMES)
    result.solution_joint_state.position = [0.0] * 7
    result.position_error_m = 0.0015
    result.orientation_error_rad = 0.001
    result.solve_time_ms = 3.0
    result.sigma_min = 0.18
    result.condition_number = 9.0
    result.max_joint_delta_rad = 0.048
    return result


class _FakeBackend:
    def __init__(self, recovery_preview):
        self.recovery_preview = recovery_preview
        self.calls = []
        self.target = object()
        self.state = object()
        self.roundtrip_plan = object()
        self.center_plan = SimpleNamespace(target_poses=(object(),))
        self.axis_plan = object()
        self.axis_writer = object()
        self.axis_rows = []

    def preview_recovery(self):
        self.calls.append("recover_preview")
        return self.recovery_preview

    def execute_recovery(self):
        self.calls.append("recover_execute")
        return _recovery_result(
            RecoverToSafe.Result.SUCCESS,
            success=True,
            executed=True,
            safe=True,
        )

    def preview_placo(self, distance_m):
        self.calls.append(("placo_preview", distance_m))
        return self.target, _valid_ik_result(), self.state

    def execute_placo(self, target, state):
        assert target is self.target
        assert state is self.state
        self.calls.append("placo_execute")

    def preview_roundtrip(self):
        self.calls.append("roundtrip_preview")
        return self.roundtrip_plan

    def execute_roundtrip(self, plan):
        assert plan is self.roundtrip_plan
        self.calls.append("roundtrip_execute")

    def preview_center_ready(self):
        self.calls.append("center_preview")
        return self.center_plan

    def execute_center_ready(self, plan):
        assert plan is self.center_plan
        self.calls.append("center_execute")

    def preview_axis_suite(self):
        self.calls.append("axis_preview")
        return self.axis_plan

    def save_axis_preview(self, plan, output_directory):
        assert plan is self.axis_plan
        self.calls.append(("axis_report", output_directory))
        return self.axis_writer, self.axis_rows

    def execute_axis_suite(self, plan, writer, rows):
        assert plan is self.axis_plan
        assert writer is self.axis_writer
        assert rows is self.axis_rows
        self.calls.append("axis_execute")


def test_local_x_target_uses_link_rotation_and_locks_authorized_offset():
    """The target is exactly link7-local +X 15 mm, not base-frame +X."""
    target = local_x_target(_FakeSolver(), np.zeros(7), SMOKE_DISTANCE_M)

    np.testing.assert_allclose(target[:3, 3], [0.1, 0.215, 0.3])
    np.testing.assert_allclose(target[:3, :3], _FakeSolver.forward_kinematics(
        np.zeros(7), "link7"
    )[:3, :3])
    with pytest.raises(ValueError):
        local_x_target(_FakeSolver(), np.zeros(7), -SMOKE_DISTANCE_M)
    with pytest.raises(ValueError):
        local_x_target(_FakeSolver(), np.zeros(7), 0.010)


def test_roundtrip_target_is_exactly_local_x_ten_millimeters():
    """The second-stage test cannot expose an arbitrary displacement."""
    target = roundtrip_local_x_target(
        _FakeSolver(), np.zeros(7), ROUNDTRIP_DISTANCE_M
    )

    np.testing.assert_allclose(target[:3, 3], [0.1, 0.21, 0.3])
    with pytest.raises(ValueError):
        roundtrip_local_x_target(_FakeSolver(), np.zeros(7), 0.015)


def test_only_physical_or_already_safe_recovery_allows_placo():
    """PREVIEW_READY is useful but never proof that the arm is safe."""
    preview = _recovery_result(
        RecoverToSafe.Result.PREVIEW_READY,
        success=True,
        executed=False,
        safe=False,
    )
    executed = _recovery_result(
        RecoverToSafe.Result.SUCCESS,
        success=True,
        executed=True,
        safe=True,
    )
    already = _recovery_result(
        RecoverToSafe.Result.ALREADY_SAFE,
        success=True,
        executed=False,
        safe=True,
    )

    assert not recovery_result_allows_placo(preview)
    assert recovery_result_allows_placo(executed)
    assert recovery_result_allows_placo(already)


def test_smoke_ik_requires_complete_finite_tightly_bounded_success():
    """Malformed, discontinuous or singular service output cannot move."""
    validate_smoke_ik(_valid_ik_result())

    bad_code = _valid_ik_result()
    bad_code.code = IKResult.ALREADY_AT_TARGET
    with pytest.raises(SmokeTestError):
        validate_smoke_ik(bad_code)

    bad_joint = _valid_ik_result()
    bad_joint.solution_joint_state.position = [0.0] * 6
    with pytest.raises(SmokeTestError):
        validate_smoke_ik(bad_joint)

    singular = _valid_ik_result()
    singular.sigma_min = 0.01
    with pytest.raises(SmokeTestError):
        validate_smoke_ik(singular)

    nan_result = _valid_ik_result()
    nan_result.position_error_m = float("nan")
    with pytest.raises(SmokeTestError):
        validate_smoke_ik(nan_result)


def test_parser_defaults_are_preview_only_and_expose_no_offset():
    """Neither command can execute or select an arbitrary target by default."""
    parser = _argument_parser()
    recover = parser.parse_args(["recover"])
    placo = parser.parse_args(["placo"])
    roundtrip = parser.parse_args(["roundtrip"])
    center_ready = parser.parse_args(["center-ready"])
    axis_suite = parser.parse_args(["axis-suite"])

    assert not recover.execute
    assert not placo.execute
    assert not roundtrip.execute
    assert not center_ready.execute
    assert not axis_suite.execute
    assert not hasattr(placo, "direction")
    assert not hasattr(roundtrip, "distance")
    assert not hasattr(center_ready, "distance")
    assert not hasattr(axis_suite, "direction")
    assert not hasattr(axis_suite, "distance")


def test_recovery_preview_never_calls_execution_or_confirmation():
    """The default recovery command is guaranteed preview-only."""
    backend = _FakeBackend(_recovery_result(
        RecoverToSafe.Result.PREVIEW_READY,
        success=True,
        executed=False,
        safe=False,
    ))
    arguments = _argument_parser().parse_args(["recover"])

    assert run_command(
        backend,
        arguments,
        input_function=lambda _prompt: pytest.fail("must not prompt"),
    ) == 0
    assert backend.calls == ["recover_preview"]


def test_recovery_execute_requires_exact_confirmation():
    """A wrong confirmation leaves only the no-command preview call."""
    backend = _FakeBackend(_recovery_result(
        RecoverToSafe.Result.PREVIEW_READY,
        success=True,
        executed=False,
        safe=False,
    ))
    arguments = _argument_parser().parse_args(["recover", "--execute"])

    with pytest.raises(SmokeTestError):
        run_command(backend, arguments, input_function=lambda _prompt: "wrong")
    assert backend.calls == ["recover_preview"]

    assert run_command(
        backend, arguments, input_function=lambda _prompt: "RECOVER"
    ) == 0
    assert backend.calls[-2:] == ["recover_preview", "recover_execute"]


def test_placo_is_blocked_until_recovery_preview_reports_already_safe():
    """An unsafe preview never reaches FK, SolveIK or MoveToPose."""
    backend = _FakeBackend(_recovery_result(
        RecoverToSafe.Result.PREVIEW_READY,
        success=True,
        executed=False,
        safe=False,
    ))
    arguments = _argument_parser().parse_args(["placo"])

    with pytest.raises(SmokeTestError):
        run_command(backend, arguments)
    assert backend.calls == ["recover_preview"]


def test_placo_preview_and_execute_have_separate_explicit_gates(capsys):
    """Preview calls no Move action; execute reuses its frozen target."""
    safe = _recovery_result(
        RecoverToSafe.Result.ALREADY_SAFE,
        success=True,
        executed=False,
        safe=True,
    )
    backend = _FakeBackend(safe)
    preview_arguments = _argument_parser().parse_args(["placo"])

    assert run_command(backend, preview_arguments) == 0
    assert backend.calls == [
        "recover_preview",
        ("placo_preview", SMOKE_DISTANCE_M),
    ]

    execute_arguments = _argument_parser().parse_args(["placo", "--execute"])
    assert run_command(
        backend, execute_arguments, input_function=lambda _prompt: "PLACO"
    ) == 0
    assert backend.calls[-3:] == [
        "recover_preview",
        ("placo_preview", SMOKE_DISTANCE_M),
        "placo_execute",
    ]
    output = capsys.readouterr().out
    assert "不要在机械臂悬空时直接失能" in output
    assert "人工托住" in output


def test_roundtrip_preview_and_execution_are_fixed_and_confirmed(capsys):
    """The guarded CLI previews both legs and requires its exact token."""
    safe = _recovery_result(
        RecoverToSafe.Result.ALREADY_SAFE,
        success=True,
        executed=False,
        safe=True,
    )
    backend = _FakeBackend(safe)
    preview_arguments = _argument_parser().parse_args(["roundtrip"])

    assert run_command(backend, preview_arguments) == 0
    assert backend.calls == ["recover_preview", "roundtrip_preview"]

    execute_arguments = _argument_parser().parse_args([
        "roundtrip", "--execute"
    ])
    with pytest.raises(SmokeTestError):
        run_command(
            backend,
            execute_arguments,
            input_function=lambda _prompt: "PLACO",
        )
    assert backend.calls[-2:] == ["recover_preview", "roundtrip_preview"]

    assert run_command(
        backend,
        execute_arguments,
        input_function=lambda _prompt: "ROUNDTRIP",
    ) == 0
    assert backend.calls[-3:] == [
        "recover_preview",
        "roundtrip_preview",
        "roundtrip_execute",
    ]
    assert "人工托住" in capsys.readouterr().out


def test_axis_suite_preview_and_execution_require_exact_token(tmp_path, capsys):
    """Six fixed pairs are previewed before one explicit supervised run."""
    safe = _recovery_result(
        RecoverToSafe.Result.ALREADY_SAFE,
        success=True,
        executed=False,
        safe=True,
    )
    backend = _FakeBackend(safe)
    preview_arguments = _argument_parser().parse_args([
        "axis-suite",
        "--output-dir",
        str(tmp_path),
    ])

    assert run_command(backend, preview_arguments) == 0
    assert backend.calls == [
        "recover_preview",
        "axis_preview",
        ("axis_report", tmp_path),
    ]

    execute_arguments = _argument_parser().parse_args([
        "axis-suite",
        "--execute",
        "--output-dir",
        str(tmp_path),
    ])
    with pytest.raises(SmokeTestError):
        run_command(
            backend,
            execute_arguments,
            input_function=lambda _prompt: "ROUNDTRIP",
        )
    assert backend.calls[-2:] == [
        "axis_preview",
        ("axis_report", tmp_path),
    ]

    assert run_command(
        backend,
        execute_arguments,
        input_function=lambda _prompt: "AXIS_SUITE",
    ) == 0
    assert backend.calls[-3:] == [
        "axis_preview",
        ("axis_report", tmp_path),
        "axis_execute",
    ]
    assert "人工托住" in capsys.readouterr().out


def test_center_ready_preview_and_execution_are_explicit(capsys):
    """The ready path remains preview-only until its exact token is typed."""
    safe = _recovery_result(
        RecoverToSafe.Result.ALREADY_SAFE,
        success=True,
        executed=False,
        safe=True,
    )
    backend = _FakeBackend(safe)
    preview_arguments = _argument_parser().parse_args(["center-ready"])

    assert run_command(backend, preview_arguments) == 0
    assert backend.calls == ["recover_preview", "center_preview"]

    execute_arguments = _argument_parser().parse_args([
        "center-ready",
        "--execute",
    ])
    with pytest.raises(SmokeTestError):
        run_command(
            backend,
            execute_arguments,
            input_function=lambda _prompt: "READY",
        )
    assert backend.calls[-1] == "center_preview"

    assert run_command(
        backend,
        execute_arguments,
        input_function=lambda _prompt: "CENTER_READY",
    ) == 0
    assert backend.calls[-2:] == ["center_preview", "center_execute"]
    assert "人工托住" in capsys.readouterr().out


def test_confirmation_token_is_exact_and_case_sensitive():
    """A similar-looking token cannot authorize physical motion."""
    require_confirmation("PLACO", lambda _prompt: "PLACO")
    with pytest.raises(SmokeTestError):
        require_confirmation("PLACO", lambda _prompt: "placo")


def test_joint_feedback_detects_end_to_end_motion_start_threshold():
    """Latency starts only after encoder movement exceeds the fixed threshold."""
    class Harness:
        _joint_state = None
        _motion_watch = _MotionWatch(
            baseline_positions=np.zeros(7),
            target_submitted_monotonic=time.monotonic() - 0.01,
        )

    harness = Harness()
    message = JointState()
    message.name = [f"joint{index}" for index in range(1, 8)]
    message.position = [0.0] * 7
    message.velocity = [0.0] * 7
    NeroRealSmokeTest._joint_callback(harness, message)
    assert harness._motion_watch.motion_started_monotonic is None

    message.position[3] = MOTION_START_POSITION_DELTA_RAD * 1.01
    NeroRealSmokeTest._joint_callback(harness, message)
    assert harness._motion_watch.motion_started_monotonic is not None


def test_axis_suite_requires_ready_center_even_when_inside_safe_limits():
    """A limit-near recovery pose cannot enter the 15 mm suite directly."""
    state = MeasuredState(
        np.array([
            -0.017174,
            -1.695081,
            0.072047,
            2.099055,
            -0.179891,
            -0.029566,
            -0.199282,
        ]),
        np.zeros(7),
        time.monotonic(),
    )

    class Solver:
        safe_joint_limits = np.column_stack((
            np.full(7, -3.0),
            np.full(7, 3.0),
        ))

    class Harness:
        _solver = Solver()

        @staticmethod
        def measured_state():
            return state

    with pytest.raises(SmokeTestError, match="center-ready"):
        NeroRealSmokeTest.preview_axis_suite(Harness())


def test_real_recovery_preview_builds_an_execute_false_goal():
    """The concrete ROS helper, not only its wrapper, requests no motion."""
    result = _recovery_result(
        RecoverToSafe.Result.PREVIEW_READY,
        success=True,
        executed=False,
        safe=False,
    )

    class Harness:
        _recovery_client = object()

        def _send_action_goal(self, client, goal, label, timeout_s):
            assert client is self._recovery_client
            assert label == "恢复预览"
            assert timeout_s == 12.0
            self.goal = goal
            return SimpleNamespace(
                status=GoalStatus.STATUS_SUCCEEDED,
                result=result,
            )

        @staticmethod
        def _print_recovery_result(_result):
            pass

    harness = Harness()
    returned = NeroRealSmokeTest.preview_recovery(harness)

    assert returned is result
    assert harness.goal.execute is False


def test_execution_preflight_returns_fresh_continuously_stationary_state():
    """The healthy path uses the measured state and complete status arrays."""
    now = time.monotonic()
    state = MeasuredState(np.zeros(7), np.zeros(7), now)
    status = AgxArmStatus()
    status.ctrl_mode = 1
    status.arm_status = 0
    status.joint_angle_limit = [False] * 7
    status.communication_status_joint = [False] * 7

    class Harness:
        _arm_status = status
        _arm_status_monotonic = now

        @staticmethod
        def _verify_controller_mode(
            _require_recovery_unlock, _precision_test
        ):
            pass

        @staticmethod
        def _validate_command_publishers():
            pass

        @staticmethod
        def measured_state():
            return state

        def _spin_until(self, predicate, _timeout_s, _description):
            self._arm_status_monotonic = time.monotonic()
            assert predicate()

        @staticmethod
        def _wait_stationary_state():
            return state

    returned = NeroRealSmokeTest._execution_preflight(
        Harness(), require_recovery_unlock=True
    )
    assert returned is state


def test_ctrl_c_after_goal_acceptance_requests_and_confirms_cancel(monkeypatch):
    """An interrupt cannot clear the active goal before cancellation."""
    send_future = object()
    result_future = object()
    cancel_future = object()
    wrapped_canceled = SimpleNamespace(status=GoalStatus.STATUS_CANCELED)

    class GoalHandle:
        accepted = True

        def __init__(self):
            self.cancel_calls = 0

        @staticmethod
        def get_result_async():
            return result_future

        def cancel_goal_async(self):
            self.cancel_calls += 1
            return cancel_future

    goal_handle = GoalHandle()

    class Client:
        @staticmethod
        def wait_for_server(timeout_sec):
            assert timeout_sec == 5.0
            return True

        @staticmethod
        def send_goal_async(_goal, feedback_callback):
            assert callable(feedback_callback)
            return send_future

    class Harness:
        _active_goal_handle = None
        _active_result_future = None
        _last_feedback_stage = None
        result_waits = 0

        @staticmethod
        def _feedback_callback(_label):
            return lambda _feedback: None

        def _wait_future(self, future, _timeout_s, _description):
            if future is send_future:
                return goal_handle
            if future is cancel_future:
                return SimpleNamespace(goals_canceling=[object()])
            if future is result_future:
                self.result_waits += 1
                if self.result_waits == 1:
                    raise KeyboardInterrupt
                return wrapped_canceled
            raise AssertionError("unexpected future")

        def _cancel_goal_and_wait(self, handle, future, label):
            return NeroRealSmokeTest._cancel_goal_and_wait(
                self, handle, future, label
            )

    monkeypatch.setattr("rclpy.ok", lambda: True)
    harness = Harness()

    with pytest.raises(KeyboardInterrupt):
        NeroRealSmokeTest._send_action_goal(
            harness, Client(), object(), "测试动作", 20.0
        )

    assert goal_handle.cancel_calls == 1
    assert harness.result_waits == 2
    assert harness._active_goal_handle is None
    assert harness._active_result_future is None


@pytest.mark.parametrize("action_raises", [False, True])
def test_recovery_action_is_wrapped_by_temporary_gates(action_raises):
    """Both success and failure close Placo then driver gates."""
    events = []
    result = _recovery_result(
        RecoverToSafe.Result.SUCCESS,
        success=True,
        executed=True,
        safe=True,
    )

    class Harness:
        _recovery_client = object()

        @staticmethod
        def _execution_preflight(
            require_recovery_unlock, precision_test
        ):
            assert require_recovery_unlock
            assert precision_test is None
            events.append("preflight")

        def _set_gate(self, _client, service_name, enabled):
            events.append((service_name, enabled))

        _driver_gate_client = object()
        _controller_gate_client = object()

        @contextmanager
        def _temporary_command_gates(self):
            with NeroRealSmokeTest._temporary_command_gates(self):
                yield

        @staticmethod
        def _send_action_goal(_client, goal, _label, _timeout_s):
            assert goal.execute
            events.append("action")
            if action_raises:
                raise SmokeTestError("simulated failure")
            return SimpleNamespace(
                status=GoalStatus.STATUS_SUCCEEDED,
                result=result,
            )

        @staticmethod
        def _print_recovery_result(_result):
            pass

    harness = Harness()
    if action_raises:
        with pytest.raises(SmokeTestError):
            NeroRealSmokeTest.execute_recovery(harness)
    else:
        assert NeroRealSmokeTest.execute_recovery(harness) is result

    assert events == [
        "preflight",
        ("/control_enable", True),
        ("/strawberry_nero/enable_execution", True),
        "action",
        ("/strawberry_nero/enable_execution", False),
        ("/control_enable", False),
    ]
