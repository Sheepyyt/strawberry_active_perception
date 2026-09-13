"""Unit tests for NERO's ROS-independent quintic trajectory generator."""

from dataclasses import replace

import numpy as np
import pytest

from strawberry_nero_control.models import (
    IKErrorCode,
    NERO_SDK_JOINT_LIMITS,
    NERO_URDF_JOINT_LIMITS,
    READY_JOINT_POSITIONS,
    TrajectoryConfig,
    TrajectoryErrorCode,
)
from strawberry_nero_control.trajectory import (
    JointLimitRecoveryMonitor,
    TrajectoryGenerator,
)


def test_auto_trajectory_is_complete_smooth_and_limit_compliant():
    """All 50 Hz samples contain seven bounded position/velocity states."""
    generator = TrajectoryGenerator()
    start = np.asarray(READY_JOINT_POSITIONS)
    goal = start + np.array([0.10, -0.04, 0.03, 0.08, -0.02, 0.01, 0.05])

    result = generator.generate(start, goal)

    assert result.success, result.message
    assert result.error_code == TrajectoryErrorCode.SUCCESS
    assert result.duration_s * generator.config.frequency_hz == pytest.approx(
        len(result.points) - 1
    )
    assert result.points[0].time_from_start_s == 0.0
    assert result.points[-1].time_from_start_s == result.duration_s
    assert result.points[0].positions == pytest.approx(tuple(start))
    assert result.points[-1].positions == pytest.approx(tuple(goal))
    assert result.points[0].velocities == pytest.approx((0.0,) * 7)
    assert result.points[-1].velocities == pytest.approx((0.0,) * 7)
    assert result.points[0].accelerations == pytest.approx((0.0,) * 7)
    assert result.points[-1].accelerations == pytest.approx((0.0,) * 7)
    assert all(len(point.positions) == 7 for point in result.points)
    assert all(len(point.velocities) == 7 for point in result.points)
    assert all(len(point.accelerations) == 7 for point in result.points)
    assert result.peak_velocity_rad_s <= generator.config.max_velocity_rad_s
    assert (
        result.peak_acceleration_rad_s2
        <= generator.config.max_acceleration_rad_s2
    )


def test_samples_are_monotonic_for_each_nonzero_displacement():
    """Minimum jerk does not overshoot any joint endpoint."""
    generator = TrajectoryGenerator()
    start = np.asarray(READY_JOINT_POSITIONS)
    goal = start + np.array([0.10, -0.10, 0.05, -0.08, 0.03, 0.02, -0.04])
    result = generator.generate(start, goal)
    positions = np.asarray([point.positions for point in result.points])

    assert result.success
    for joint_index, delta in enumerate(goal - start):
        differences = np.diff(positions[:, joint_index])
        if delta > 0.0:
            assert np.all(differences >= -1.0e-12)
        else:
            assert np.all(differences <= 1.0e-12)


def test_explicit_duration_below_motion_limits_is_rejected():
    """An explicit fast jump is refused instead of silently stretched."""
    generator = TrajectoryGenerator()
    start = np.asarray(READY_JOINT_POSITIONS)
    goal = start.copy()
    goal[0] += 0.20

    result = generator.generate(start, goal, duration_s=0.10)

    assert not result.success
    assert result.error_code == TrajectoryErrorCode.DURATION_TOO_SHORT
    assert not result.points


def test_start_or_goal_outside_joint_limits_is_rejected():
    """Trajectory interpolation never receives an unsafe endpoint."""
    generator = TrajectoryGenerator()
    start = np.asarray(READY_JOINT_POSITIONS)
    goal = start.copy()
    goal[5] = generator.joint_limits[5, 1] + 0.01

    result = generator.generate(start, goal)

    assert not result.success
    assert result.error_code == TrajectoryErrorCode.JOINT_LIMIT
    assert not result.points


def test_excessive_safe_duration_is_rejected():
    """A configured maximum duration can reject an otherwise smooth move."""
    config = replace(TrajectoryConfig(), max_duration_s=0.25)
    generator = TrajectoryGenerator(config)
    start = np.asarray(READY_JOINT_POSITIONS)
    goal = start.copy()
    goal[0] += 0.20

    result = generator.generate(start, goal)

    assert not result.success
    assert result.error_code == TrajectoryErrorCode.DURATION_TOO_LONG


@pytest.mark.parametrize(
    'start, goal',
    [
        ([0.0] * 6, [0.0] * 7),
        ([0.0] * 7, [float('nan')] * 7),
    ],
)
def test_invalid_joint_vectors_are_structured_failures(start, goal):
    """Bad vector size or values return an error without partial points."""
    result = TrajectoryGenerator().generate(start, goal)

    assert not result.success
    assert result.error_code == TrajectoryErrorCode.INVALID_INPUT
    assert not result.points


MEASURED_OUTSIDE_LIMITS = np.array([
    -0.11927580108129249,
    -1.7599376578335222,
    0.07869689597242432,
    2.192726952450556,
    0.023160519173964753,
    -0.015114551322270894,
    -0.18298031877908552,
])


def _raw_joint_limits():
    """Return the URDF/SDK intersection before the two-degree margin."""
    urdf = np.asarray(NERO_URDF_JOINT_LIMITS)
    sdk = np.asarray(NERO_SDK_JOINT_LIMITS)
    return np.column_stack((
        np.maximum(urdf[:, 0], sdk[:, 0]),
        np.minimum(urdf[:, 1], sdk[:, 1]),
    ))


def test_limit_recovery_only_moves_joints_two_and_four_inward():
    """The measured startup state gets the exact bounded two-stage plan."""
    config = replace(
        TrajectoryConfig(),
        max_velocity_rad_s=0.05,
        max_acceleration_rad_s2=0.10,
    )
    generator = TrajectoryGenerator(config)

    plan = generator.generate_limit_recovery(
        MEASURED_OUTSIDE_LIMITS,
        _raw_joint_limits(),
    )

    assert plan.success, plan.message
    assert plan.error_code == IKErrorCode.SUCCESS
    assert plan.recovering_joint_indices == (2, 4)
    assert plan.ingress_positions[1] == pytest.approx(-1.735)
    assert plan.ingress_positions[3] == pytest.approx(2.135)
    assert plan.target_positions[1] == pytest.approx(
        generator.joint_limits[1, 0] + 0.010
    )
    assert plan.target_positions[3] == pytest.approx(
        generator.joint_limits[3, 1] - 0.010
    )
    assert plan.max_raw_violation_rad == pytest.approx(0.05272695245)
    assert plan.max_safe_violation_rad == pytest.approx(0.08763353749)
    assert plan.max_joint_delta_rad < 0.10

    phase_a = plan.phase_a_trajectory
    assert phase_a is not None and phase_a.success
    assert phase_a.points[0].positions == pytest.approx(
        tuple(MEASURED_OUTSIDE_LIMITS)
    )
    assert phase_a.points[-1].positions == pytest.approx(
        plan.ingress_positions
    )
    assert phase_a.peak_velocity_rad_s <= 0.05
    assert phase_a.peak_acceleration_rad_s2 <= 0.10

    fixed = [0, 2, 4, 5, 6]
    np.testing.assert_allclose(
        np.asarray(plan.ingress_positions)[fixed],
        MEASURED_OUTSIDE_LIMITS[fixed],
    )
    np.testing.assert_allclose(
        np.asarray(plan.target_positions)[fixed],
        MEASURED_OUTSIDE_LIMITS[fixed],
    )


def test_limit_recovery_phase_b_is_monotonic_and_slow():
    """All sampled commands move inward without exceeding recovery limits."""
    config = replace(
        TrajectoryConfig(),
        max_velocity_rad_s=0.05,
        max_acceleration_rad_s2=0.10,
    )
    generator = TrajectoryGenerator(config)
    plan = generator.generate_limit_recovery(
        MEASURED_OUTSIDE_LIMITS,
        _raw_joint_limits(),
    )
    trajectory = plan.phase_b_trajectory

    assert trajectory is not None and trajectory.success
    positions = np.asarray([point.positions for point in trajectory.points])
    assert np.all(np.diff(positions[:, 1]) >= -1.0e-12)
    assert np.all(np.diff(positions[:, 3]) <= 1.0e-12)
    np.testing.assert_allclose(
        positions[:, [0, 2, 4, 5, 6]],
        np.broadcast_to(
            MEASURED_OUTSIDE_LIMITS[[0, 2, 4, 5, 6]],
            (len(positions), 5),
        ),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert trajectory.peak_velocity_rad_s <= 0.05
    assert trajectory.peak_acceleration_rad_s2 <= 0.10

    phase_a_positions = np.asarray([
        point.positions for point in plan.phase_a_trajectory.points
    ])
    assert np.all(np.diff(phase_a_positions[:, 1]) >= -1.0e-12)
    assert np.all(np.diff(phase_a_positions[:, 3]) <= 1.0e-12)


def test_limit_recovery_refuses_a_large_raw_violation():
    """A state far beyond the hard envelope cannot use automatic recovery."""
    generator = TrajectoryGenerator()
    start = MEASURED_OUTSIDE_LIMITS.copy()
    start[1] = _raw_joint_limits()[1, 0] - 0.061

    plan = generator.generate_limit_recovery(start, _raw_joint_limits())

    assert not plan.success
    assert plan.error_code == IKErrorCode.JOINT_LIMIT_VIOLATION
    assert plan.phase_a_trajectory is None
    assert plan.phase_b_trajectory is None


def test_limit_recovery_is_a_no_command_result_when_already_safe():
    """A safe measured state produces no executable trajectory."""
    generator = TrajectoryGenerator()

    plan = generator.generate_limit_recovery(
        READY_JOINT_POSITIONS,
        _raw_joint_limits(),
    )

    assert plan.success
    assert plan.already_safe
    assert plan.error_code == IKErrorCode.ALREADY_AT_TARGET
    assert plan.phase_a_trajectory is None
    assert plan.phase_b_trajectory is None


def test_normal_trajectory_limits_remain_strict_after_recovery_support():
    """The exceptional recovery path never weakens ordinary motion checks."""
    generator = TrajectoryGenerator()

    result = generator.generate(MEASURED_OUTSIDE_LIMITS, READY_JOINT_POSITIONS)

    assert not result.success
    assert result.error_code == TrajectoryErrorCode.JOINT_LIMIT


def _recovery_monitor(plan, generator):
    return JointLimitRecoveryMonitor(
        plan,
        generator.joint_limits,
        fixed_joint_tolerance_rad=0.003,
        progress_tolerance_rad=0.001,
    )


def test_recovery_monitor_accepts_the_complete_planned_path():
    """Every planned sample satisfies the same guard used for real feedback."""
    config = replace(
        TrajectoryConfig(),
        max_velocity_rad_s=0.05,
        max_acceleration_rad_s2=0.10,
    )
    generator = TrajectoryGenerator(config)
    plan = generator.generate_limit_recovery(
        MEASURED_OUTSIDE_LIMITS,
        _raw_joint_limits(),
    )
    monitor = _recovery_monitor(plan, generator)

    samples = list(plan.phase_a_trajectory.points)
    samples.extend(plan.phase_b_trajectory.points[1:])

    assert all(monitor.validate(point.positions) is None for point in samples)


def test_recovery_monitor_rejects_reverse_motion_and_fixed_joint_drift():
    """Wrong-way recovery or movement of another axis is stopped immediately."""
    generator = TrajectoryGenerator()
    plan = generator.generate_limit_recovery(
        MEASURED_OUTSIDE_LIMITS,
        _raw_joint_limits(),
    )

    reverse = MEASURED_OUTSIDE_LIMITS.copy()
    reverse[1] -= 0.002
    assert '错误方向' in _recovery_monitor(
        plan, generator
    ).validate(reverse)

    drift = MEASURED_OUTSIDE_LIMITS.copy()
    drift[0] += 0.004
    assert '非预期移动' in _recovery_monitor(
        plan, generator
    ).validate(drift)


def test_recovery_monitor_allows_small_startup_settling_then_rejects_reverse():
    """A real encoder settling step must not hide meaningful reverse motion."""
    generator = TrajectoryGenerator()
    plan = generator.generate_limit_recovery(
        MEASURED_OUTSIDE_LIMITS,
        _raw_joint_limits(),
    )
    monitor = JointLimitRecoveryMonitor(
        plan,
        generator.joint_limits,
        fixed_joint_tolerance_rad=0.003,
        progress_tolerance_rad=0.003,
    )

    encoder_settling = MEASURED_OUTSIDE_LIMITS.copy()
    encoder_settling[1] -= 0.0013
    assert monitor.validate(encoder_settling) is None

    meaningful_reverse = MEASURED_OUTSIDE_LIMITS.copy()
    meaningful_reverse[1] -= 0.0031
    assert '错误方向' in monitor.validate(meaningful_reverse)
