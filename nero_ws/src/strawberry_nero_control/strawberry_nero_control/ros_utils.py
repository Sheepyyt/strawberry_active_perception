"""Small, testable conversions shared by the ROS 2 control node."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert a ROS pose to a homogeneous matrix and normalize its quaternion."""
    values = np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("pose contains NaN or infinity")

    quaternion = values[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise ValueError("pose quaternion has zero length")
    quaternion /= norm

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def matrix_to_pose(transform: np.ndarray) -> Pose:
    """Convert a finite 4x4 homogeneous matrix to a ROS pose."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix")
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = matrix[:3, 3]
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = quaternion
    return pose


def matrix_to_pose_stamped(
    transform: np.ndarray,
    frame_id: str,
    stamp,
) -> PoseStamped:
    """Build a stamped ROS pose from a homogeneous matrix."""
    message = PoseStamped()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.pose = matrix_to_pose(transform)
    return message


def pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return translation distance in metres and shortest rotation in radians."""
    target_matrix = np.asarray(target, dtype=float)
    actual_matrix = np.asarray(actual, dtype=float)
    position_error = float(
        np.linalg.norm(target_matrix[:3, 3] - actual_matrix[:3, 3])
    )
    relative_rotation = target_matrix[:3, :3] @ actual_matrix[:3, :3].T
    orientation_error = float(Rotation.from_matrix(relative_rotation).magnitude())
    return position_error, orientation_error


def joint_state_from_arrays(
    names: Sequence[str],
    positions: Iterable[float],
    stamp,
    *,
    velocities: Iterable[float] | None = None,
    efforts: Iterable[float] | None = None,
) -> JointState:
    """Create a complete, ordered joint-state message."""
    message = JointState()
    message.header.stamp = stamp
    message.name = list(names)
    message.position = [float(value) for value in positions]
    if len(message.position) != len(message.name):
        raise ValueError("joint name and position lengths differ")
    if velocities is not None:
        message.velocity = [float(value) for value in velocities]
    if efforts is not None:
        message.effort = [float(value) for value in efforts]
    return message


def ordered_joint_arrays(
    message: JointState,
    expected_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract complete ordered position and velocity arrays from JointState."""
    if len(message.name) != len(set(message.name)):
        raise ValueError("joint state contains duplicate names")
    positions = dict(zip(message.name, message.position))
    missing = [name for name in expected_names if name not in positions]
    if missing:
        raise ValueError(f"joint state is missing: {', '.join(missing)}")

    position_array = np.array([positions[name] for name in expected_names], dtype=float)
    velocity_by_name = dict(zip(message.name, message.velocity))
    velocity_array = np.array(
        [velocity_by_name.get(name, 0.0) for name in expected_names], dtype=float
    )
    if not np.all(np.isfinite(position_array)):
        raise ValueError("joint positions contain NaN or infinity")
    if not np.all(np.isfinite(velocity_array)):
        velocity_array = np.zeros(len(expected_names), dtype=float)
    return position_array, velocity_array


def duration_to_seconds(duration: Duration) -> float:
    """Convert a ROS duration message to seconds."""
    return float(duration.sec) + float(duration.nanosec) * 1.0e-9


def seconds_to_duration(seconds: float) -> Duration:
    """Convert non-negative seconds to a normalized ROS duration message."""
    value = max(0.0, float(seconds))
    whole = math.floor(value)
    duration = Duration()
    duration.sec = int(whole)
    duration.nanosec = int(round((value - whole) * 1.0e9))
    if duration.nanosec >= 1_000_000_000:
        duration.sec += 1
        duration.nanosec -= 1_000_000_000
    return duration
