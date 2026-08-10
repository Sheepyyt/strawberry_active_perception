"""Integration tests for the guarded RecoverToSafe action callback."""

import time

import numpy as np
import pytest

from builtin_interfaces.msg import Duration
import rclpy
from sensor_msgs.msg import JointState

from strawberry_nero_control.control_node import NeroControlNode
from strawberry_nero_control.models import READY_JOINT_POSITIONS
from strawberry_nero_control.ros_utils import matrix_to_pose_stamped
from strawberry_nero_interfaces.action import MoveToPose, RecoverToSafe


MEASURED_OUTSIDE_LIMITS = np.array([
    -0.11927580108129249,
    -1.7599376578335222,
    0.07869689597242432,
    2.192726952450556,
    0.023160519173964753,
    -0.015114551322270894,
    -0.18298031877908552,
])


class _Request:
    def __init__(self, execute):
        self.execute = execute
        self.timeout = Duration(sec=10)


class _GoalHandle:
    def __init__(self, execute, canceled=False):
        self.request = _Request(execute)
        self.is_cancel_requested = canceled
        self.feedback = []
        self.status = None

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)

    def succeed(self):
        self.status = 'succeeded'

    def abort(self):
        self.status = 'aborted'

    def canceled(self):
        self.status = 'canceled'


class _MoveRequest:
    def __init__(self, target_pose):
        self.target_pose = target_pose
        self.controlled_frame = 'link7'
        self.timeout = Duration(sec=10)


class _MoveGoalHandle(_GoalHandle):
    def __init__(self, target_pose):
        super().__init__(execute=False)
        self.request = _MoveRequest(target_pose)


def _node(
    *,
    allow_execution=False,
    first_motion_test=False,
    precision_test=False,
):
    arguments = [
        '--ros-args',
        '-p',
        'metrics_log_enabled:=false',
        '-p',
        'feedback_timeout_sec:=2.0',
        '-p',
        f'allow_limit_recovery_execution:={str(allow_execution).lower()}',
        '-p',
        'verified_driver_speed_percent:='
        f'{10 if allow_execution or first_motion_test or precision_test else 0}',
        '-p',
        f'first_motion_test_mode:={str(first_motion_test).lower()}',
        '-p',
        f'precision_test_mode:={str(precision_test).lower()}',
    ]
    rclpy.init(args=arguments)
    return NeroControlNode()


def _set_measured_start(node):
    node.set_simulated_state(MEASURED_OUTSIDE_LIMITS, np.zeros(7))


def _close(node):
    node.destroy_node()
    rclpy.shutdown()


def test_recovery_preview_is_explicit_and_sends_no_command():
    """A successful preview is never reported as completed robot motion."""
    node = _node()
    try:
        _set_measured_start(node)
        goal = _GoalHandle(execute=False)
        before = node._executor_backend.command_count

        result = node._execute_recovery_action(goal)

        assert goal.status == 'succeeded'
        assert result.success
        assert result.code == RecoverToSafe.Result.PREVIEW_READY
        assert not result.executed
        assert not result.robot_is_safe
        assert node._executor_backend.command_count == before
        assert result.ingress_joint_state.position[1] == pytest.approx(-1.735)
        assert result.target_joint_state.position[3] == pytest.approx(
            2.0950934149601137
        )
        assert result.phase_a_duration.sec > 0
        assert result.phase_b_duration.sec > 0
    finally:
        _close(node)


def test_locked_or_pre_command_canceled_recovery_sends_nothing():
    """Both the second execution lock and early cancel preserve zero commands."""
    node = _node()
    try:
        _set_measured_start(node)
        before = node._executor_backend.command_count

        locked = node._execute_recovery_action(_GoalHandle(execute=True))
        assert locked.code == RecoverToSafe.Result.DRIVER_FAULT
        assert node._executor_backend.command_count == before

        _set_measured_start(node)
        canceled_goal = _GoalHandle(execute=False, canceled=True)
        canceled = node._execute_recovery_action(canceled_goal)
        assert canceled.code == RecoverToSafe.Result.CANCELED
        assert canceled_goal.status == 'canceled'
        assert node._executor_backend.command_count == before
    finally:
        _close(node)


def test_simulated_recovery_executes_two_smooth_phases_to_safe_state():
    """The complete Action state machine reaches a measured safe result."""
    node = _node(allow_execution=True)
    try:
        _set_measured_start(node)
        node._stationary_since = time.monotonic() - 1.0
        goal = _GoalHandle(execute=True)
        before = node._executor_backend.command_count
        commands = []
        original_send = node._executor_backend.send

        def record_send(positions, velocities=None):
            commands.append(np.asarray(positions, dtype=float).copy())
            original_send(positions, velocities)

        node._executor_backend.send = record_send

        result = node._execute_recovery_action(goal)

        assert goal.status == 'succeeded'
        assert result.success
        assert result.code == RecoverToSafe.Result.SUCCESS
        assert result.executed
        assert result.robot_is_safe
        assert node._executor_backend.command_count > before + 2
        assert commands
        np.testing.assert_allclose(
            commands[0],
            result.ingress_joint_state.position,
            atol=1.0e-12,
        )
        assert np.all(commands[0] >= node._raw_joint_limits[:, 0])
        assert np.all(commands[0] <= node._raw_joint_limits[:, 1])
        np.testing.assert_allclose(
            result.achieved_joint_state.position,
            result.target_joint_state.position,
            atol=1.0e-12,
        )
    finally:
        _close(node)


def test_failed_recovery_reports_that_physical_motion_started():
    """A post-command failure must not misleadingly print executed=false."""
    node = _node()
    try:
        _set_measured_start(node)
        result = node._recovery_result(
            None,
            RecoverToSafe.Result.TRACKING_ERROR,
            "simulated tracking failure",
            time.monotonic(),
            0.17,
            10.0,
            motion_started=True,
        )

        assert not result.success
        assert result.executed
    finally:
        _close(node)


def test_first_motion_mode_tightens_small_move_acceptance_gates():
    """The 15 mm hardware mode cannot inherit broad research tolerances."""
    node = _node(first_motion_test=True)
    try:
        assert node._tracking_pause_error == pytest.approx(0.015)
        assert node._tracking_abort_error == pytest.approx(0.030)
        assert node._tracking_abort_duration == pytest.approx(0.20)
        assert node._settle_joint_tolerance == pytest.approx(0.005)
        assert node._final_position_tolerance == pytest.approx(0.003)
        assert node._final_orientation_tolerance == pytest.approx(
            np.radians(2.0)
        )
    finally:
        _close(node)


def test_first_motion_server_locks_exact_target_and_one_shot_use():
    """Bypassing the CLI cannot request another pose or repeat the 15 mm move."""
    node = _node(first_motion_test=True)
    try:
        node._stationary_since = time.monotonic() - 1.0
        snapshot, _, _ = node._snapshot()
        assert snapshot is not None
        current = node._solver.forward_kinematics(snapshot.positions)
        target = current.copy()
        target[:3, 3] += current[:3, 0] * 0.015
        target_message = matrix_to_pose_stamped(
            target, "base_link", node.get_clock().now().to_msg()
        )

        result, context, solved_snapshot = node._solve(target_message, "link7")
        assert result.success, result.message
        assert node._first_motion_request_error(
            result, context, solved_snapshot
        ) is None

        wrong_target = current.copy()
        wrong_target[:3, 3] -= current[:3, 0] * 0.015
        wrong_message = matrix_to_pose_stamped(
            wrong_target, "base_link", node.get_clock().now().to_msg()
        )
        wrong_result, wrong_context, wrong_snapshot = node._solve(
            wrong_message, "link7"
        )
        assert wrong_result.success, wrong_result.message
        assert node._first_motion_request_error(
            wrong_result, wrong_context, wrong_snapshot
        ) is not None

        node._first_motion_test_consumed = True
        assert node._first_motion_request_error(
            result, context, solved_snapshot
        ) is not None
    finally:
        _close(node)


def test_precision_mode_accepts_small_placo_targets_without_one_shot_lock():
    """The second stage keeps strict IK gates but permits a return leg."""
    node = _node(precision_test=True)
    try:
        start = np.asarray(READY_JOINT_POSITIONS, dtype=float)
        current = node._solver.forward_kinematics(start, 'link7')
        target = current.copy()
        target[:3, 3] += current[:3, 0] * 0.010
        target_pose = matrix_to_pose_stamped(
            target,
            'base_link',
            node.get_clock().now().to_msg(),
        )

        result, context, snapshot = node._solve(target_pose, 'link7')
        assert result.success, result.message
        assert node._precision_motion_request_error(
            result, context, snapshot
        ) is None
        assert node._precision_motion_request_error(
            result, context, snapshot
        ) is None
        assert not node._first_motion_test_consumed
        assert node._tracking_abort_error == pytest.approx(0.030)
        assert node._final_position_tolerance == pytest.approx(0.003)
    finally:
        _close(node)


def test_ros_solve_path_honors_center_ready_posture_reference():
    """The ROS integration selects ready's redundant branch in ten steps."""
    node = _node(precision_test=True)
    try:
        measured = np.array([
            0.167761,
            -1.695081,
            -0.041015,
            2.020620,
            -0.023440,
            0.033772,
            0.003735,
        ])
        ready = np.asarray(READY_JOINT_POSITIONS)
        predicted = measured.copy()
        node.set_simulated_state(predicted, np.zeros(7))

        for index in range(1, 11):
            reference = measured + (ready - measured) * index / 10.0
            target = node._solver.forward_kinematics(reference, "link7")
            target_pose = matrix_to_pose_stamped(
                target,
                "base_link",
                node.get_clock().now().to_msg(),
            )
            posture = JointState()
            posture.name = [f"joint{joint}" for joint in range(1, 8)]
            posture.position = reference.tolist()
            result, _, _ = node._solve(target_pose, "link7", posture)
            assert result.success, result.message
            predicted = np.asarray(result.joint_positions)
            node.set_simulated_state(predicted, np.zeros(7))

        assert np.max(np.abs(predicted - ready)) < 0.005
    finally:
        _close(node)


def test_first_motion_mode_executes_full_placo_chain_in_simulation():
    """The strict 15 mm mode still completes IK, trajectory and feedback."""
    node = _node(first_motion_test=True)
    try:
        node._stationary_since = time.monotonic() - 1.0
        start = np.asarray(READY_JOINT_POSITIONS, dtype=float)
        current = node._solver.forward_kinematics(start, 'link7')
        target = current.copy()
        target[:3, 3] += current[:3, 0] * 0.015
        target_pose = matrix_to_pose_stamped(
            target,
            'base_link',
            node.get_clock().now().to_msg(),
        )
        goal = _MoveGoalHandle(target_pose)
        before = node._executor_backend.command_count

        result = node._execute_action(goal)

        assert goal.status == 'succeeded'
        assert result.ik_result.success
        assert result.ik_result.code == MoveToPose.Result().ik_result.SUCCESS
        assert result.final_position_error_m <= 0.003
        assert result.final_orientation_error_rad <= np.radians(2.0)
        assert node._executor_backend.command_count > before + 2

        commands_after_first = node._executor_backend.command_count
        second_goal = _MoveGoalHandle(target_pose)
        second_result = node._execute_action(second_goal)
        assert second_goal.status == 'aborted'
        assert not second_result.ik_result.success
        assert node._executor_backend.command_count == commands_after_first
        assert not node._execution_enabled
    finally:
        _close(node)
