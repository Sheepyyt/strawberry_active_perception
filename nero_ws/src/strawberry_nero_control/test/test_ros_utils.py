"""Unit tests for the ROS message conversion helpers."""

import math

from builtin_interfaces.msg import Time

from geometry_msgs.msg import Pose

import numpy as np

import pytest

from scipy.spatial.transform import Rotation

from sensor_msgs.msg import JointState

from strawberry_nero_control.ros_utils import duration_to_seconds
from strawberry_nero_control.ros_utils import joint_state_from_arrays
from strawberry_nero_control.ros_utils import matrix_to_pose
from strawberry_nero_control.ros_utils import ordered_joint_arrays
from strawberry_nero_control.ros_utils import pose_error
from strawberry_nero_control.ros_utils import pose_to_matrix
from strawberry_nero_control.ros_utils import seconds_to_duration


JOINT_NAMES = [f'joint{index}' for index in range(1, 8)]


def _pose(position, quaternion) -> Pose:
    """Create a Pose without obscuring the values under test."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = quaternion
    return pose


def _joint_state(names, positions, velocities=None) -> JointState:
    """Create a JointState with deliberately caller-controlled ordering."""
    message = JointState()
    message.name = list(names)
    message.position = list(positions)
    if velocities is not None:
        message.velocity = list(velocities)
    return message


def test_pose_matrix_round_trip_preserves_transform():
    """A pose converted to a matrix and back keeps the same transform."""
    quaternion = Rotation.from_euler(
        'xyz',
        [0.37, -0.81, 1.42],
    ).as_quat()
    original = _pose([0.24, -0.13, 0.67], quaternion)

    transform = pose_to_matrix(original)
    recovered = matrix_to_pose(transform)

    np.testing.assert_allclose(
        pose_to_matrix(recovered),
        transform,
        atol=1.0e-12,
    )


def test_pose_to_matrix_normalizes_quaternion():
    """Quaternion scale does not change the represented rotation."""
    unit_quaternion = Rotation.from_euler('z', math.pi / 2.0).as_quat()
    pose = _pose([0.0, 0.0, 0.0], 7.5 * unit_quaternion)

    transform = pose_to_matrix(pose)

    np.testing.assert_allclose(
        transform[:3, :3],
        Rotation.from_quat(unit_quaternion).as_matrix(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        transform[:3, :3].T @ transform[:3, :3],
        np.eye(3),
        atol=1.0e-12,
    )


def test_pose_to_matrix_rejects_zero_quaternion():
    """A zero quaternion is invalid instead of silently becoming identity."""
    pose = _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match='quaternion has zero length'):
        pose_to_matrix(pose)


def test_pose_error_uses_shortest_rotation():
    """Rotations at +179 and -179 degrees differ by two degrees."""
    target = np.eye(4)
    actual = np.eye(4)
    target[:3, :3] = Rotation.from_euler('z', 179, degrees=True).as_matrix()
    actual[:3, :3] = Rotation.from_euler('z', -179, degrees=True).as_matrix()
    target[:3, 3] = [0.10, -0.20, 0.30]
    actual[:3, 3] = [0.10, -0.20, 0.25]

    position_error, orientation_error = pose_error(target, actual)

    assert position_error == pytest.approx(0.05)
    assert orientation_error == pytest.approx(math.radians(2.0))


def test_joint_state_creation_and_ordered_extraction_are_complete():
    """All seven joints are created and restored to the expected order."""
    scrambled_names = [
        'joint4',
        'joint1',
        'joint7',
        'joint2',
        'joint6',
        'joint3',
        'joint5',
    ]
    position_by_name = {
        name: index * 0.1 for index, name in enumerate(JOINT_NAMES, start=1)
    }
    velocity_by_name = {
        name: -index * 0.01 for index, name in enumerate(JOINT_NAMES, start=1)
    }
    stamp = Time(sec=42, nanosec=123)
    message = joint_state_from_arrays(
        scrambled_names,
        [position_by_name[name] for name in scrambled_names],
        stamp,
        velocities=[velocity_by_name[name] for name in scrambled_names],
    )

    positions, velocities = ordered_joint_arrays(message, JOINT_NAMES)

    assert message.header.stamp == stamp
    assert len(message.name) == 7
    assert len(message.position) == 7
    np.testing.assert_allclose(
        positions,
        [position_by_name[name] for name in JOINT_NAMES],
    )
    np.testing.assert_allclose(
        velocities,
        [velocity_by_name[name] for name in JOINT_NAMES],
    )


def test_ordered_joint_arrays_rejects_missing_joint():
    """A partial state cannot be used as an IK seed."""
    message = _joint_state(JOINT_NAMES[:-1], np.zeros(6))

    with pytest.raises(ValueError, match='joint7'):
        ordered_joint_arrays(message, JOINT_NAMES)


def test_ordered_joint_arrays_rejects_duplicate_name():
    """Duplicate names are rejected before dictionary conversion loses data."""
    duplicate_names = JOINT_NAMES[:-1] + ['joint1']
    message = _joint_state(duplicate_names, np.zeros(7))

    with pytest.raises(ValueError, match='duplicate names'):
        ordered_joint_arrays(message, JOINT_NAMES)


def test_ordered_joint_arrays_rejects_non_finite_position():
    """Non-finite joint positions cannot initialize Placo."""
    positions = np.zeros(7)
    positions[3] = np.nan
    message = _joint_state(JOINT_NAMES, positions)

    with pytest.raises(ValueError, match='positions contain NaN or infinity'):
        ordered_joint_arrays(message, JOINT_NAMES)


def test_ordered_joint_arrays_replaces_non_finite_velocity():
    """Unusable optional velocity feedback safely falls back to zero."""
    velocities = np.arange(7, dtype=float)
    velocities[2] = np.nan
    message = _joint_state(JOINT_NAMES, np.zeros(7), velocities)

    _, ordered_velocities = ordered_joint_arrays(message, JOINT_NAMES)

    np.testing.assert_array_equal(ordered_velocities, np.zeros(7))


@pytest.mark.parametrize(
    'seconds',
    [0.0, 1.0e-9, 0.25, 1.9999999996, 12.345678901],
)
def test_duration_round_trip(seconds):
    """Duration values round-trip to the nearest nanosecond."""
    duration = seconds_to_duration(seconds)

    assert 0 <= duration.nanosec < 1_000_000_000
    assert duration_to_seconds(duration) == pytest.approx(
        seconds,
        abs=0.5e-9,
    )


def test_negative_duration_is_clamped_to_zero():
    """Negative action timeouts become the documented zero default."""
    duration = seconds_to_duration(-3.5)

    assert duration.sec == 0
    assert duration.nanosec == 0
    assert duration_to_seconds(duration) == 0.0
