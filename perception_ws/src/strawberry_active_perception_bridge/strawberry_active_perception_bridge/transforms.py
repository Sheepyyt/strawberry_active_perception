"""Middleware-independent rigid-transform helpers for IK preview.

All transforms follow ``T_parent_child`` semantics.  ROS-facing quaternions are
always ordered ``xyzw``.  Keeping these operations here makes the safety-
critical camera-to-link7 conversion independently testable without ROS, Placo,
or a robot driver.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_EPS = 1.0e-12


def _normalise_quaternion(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= _EPS:
        raise ValueError("quaternion norm must be greater than zero")
    return quaternion / norm


def pose_components_to_matrix(
    position_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Create a validated homogeneous transform from ROS pose components."""
    position = np.asarray(position_xyz, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_xyz must contain three finite values")
    x, y, z, w = _normalise_quaternion(quaternion_xyzw)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )
    transform[:3, 3] = position
    return transform


def _validate_transform(transform: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-7):
        raise ValueError(f"{label} rotation must have determinant +1")
    return matrix


def validate_rigid_transform(
    transform: np.ndarray,
    label: str = "transform",
) -> np.ndarray:
    """Return a defensive copy of one finite proper rigid transform.

    This public wrapper is used at trust boundaries such as a signed-off
    hand-eye report.  Returning a copy prevents callers from mutating the
    validated array after the check.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    return _validate_transform(transform, label.strip()).copy()


def matrix_to_pose_components(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return position and normalized ``xyzw`` quaternion for a transform."""
    matrix = _validate_transform(transform, "transform")
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    quaternion = _normalise_quaternion(quaternion)
    # Canonicalize the double cover so recorded previews are deterministic.
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return matrix[:3, 3].copy(), quaternion


def camera_target_to_link7(
    target_world_camera: np.ndarray,
    transform_link7_camera: np.ndarray,
) -> np.ndarray:
    """Convert ``T_world_camera`` into the matching ``T_world_link7``."""
    target = _validate_transform(target_world_camera, "target_world_camera")
    mount = _validate_transform(transform_link7_camera, "transform_link7_camera")
    return target @ np.linalg.inv(mount)


def _slerp_xyzw(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    first = _normalise_quaternion(start)
    second = _normalise_quaternion(end)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalise_quaternion(first + alpha * (second - first))
    theta = float(np.arccos(dot))
    sine = float(np.sin(theta))
    return (
        np.sin((1.0 - alpha) * theta) / sine * first
        + np.sin(alpha * theta) / sine * second
    )


def interpolate_transform(
    current_world_camera: np.ndarray,
    target_world_camera: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate translation and rotation between camera poses."""
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    current_position, current_quaternion = matrix_to_pose_components(
        current_world_camera
    )
    target_position, target_quaternion = matrix_to_pose_components(
        target_world_camera
    )
    position = current_position + alpha * (target_position - current_position)
    quaternion = _slerp_xyzw(current_quaternion, target_quaternion, alpha)
    return pose_components_to_matrix(position, quaternion)


def preview_camera_targets(
    current_world_camera: np.ndarray,
    target_world_camera: np.ndarray,
    transform_link7_camera: np.ndarray,
    alphas: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
) -> tuple[tuple[float, np.ndarray], ...]:
    """Build the ordered full/step-halved link7 targets for SolveIK."""
    if not alphas:
        raise ValueError("at least one preview alpha is required")
    previews = []
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        camera = interpolate_transform(
            current_world_camera,
            target_world_camera,
            alpha,
        )
        previews.append(
            (alpha, camera_target_to_link7(camera, transform_link7_camera))
        )
    return tuple(previews)
