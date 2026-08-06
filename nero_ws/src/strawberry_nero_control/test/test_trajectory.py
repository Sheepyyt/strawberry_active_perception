"""Unit tests for NERO's ROS-independent quintic trajectory generator."""

from dataclasses import replace

import numpy as np
import pytest

from strawberry_nero_control.models import (
    READY_JOINT_POSITIONS,
    TrajectoryConfig,
    TrajectoryErrorCode,
)
from strawberry_nero_control.trajectory import TrajectoryGenerator


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
