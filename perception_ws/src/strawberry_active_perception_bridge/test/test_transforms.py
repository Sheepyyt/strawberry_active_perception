"""Pure tests for the motion-free camera-to-link7 conversion."""

import numpy as np
import pytest

from strawberry_active_perception_bridge.transforms import (
    camera_target_to_link7,
    interpolate_transform,
    matrix_to_pose_components,
    pose_components_to_matrix,
    preview_camera_targets,
)


MOUNT = np.array(
    [
        [0.0, 0.0, 1.0, 0.10],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, -1.0, 0.0, 0.00],
        [0.0, 0.0, 0.0, 1.00],
    ]
)


def test_pose_matrix_round_trip() -> None:
    transform = pose_components_to_matrix(
        (0.4, -0.2, 0.7),
        (0.2, -0.3, 0.1, 0.9),
    )
    position, quaternion = matrix_to_pose_components(transform)
    recovered = pose_components_to_matrix(position, quaternion)
    np.testing.assert_allclose(recovered, transform, atol=1.0e-12)


def test_camera_link7_round_trip_is_exact() -> None:
    target_camera = pose_components_to_matrix(
        (0.45, -0.05, 0.55),
        (0.1, 0.2, -0.1, 0.95),
    )
    target_link7 = camera_target_to_link7(target_camera, MOUNT)
    np.testing.assert_allclose(target_link7 @ MOUNT, target_camera, atol=1.0e-12)


def test_step_halving_preserves_order_and_endpoint() -> None:
    current = np.eye(4)
    target = pose_components_to_matrix((0.08, 0.02, -0.04), (0.0, 0.0, 1.0, 0.0))
    previews = preview_camera_targets(current, target, MOUNT)
    assert tuple(item[0] for item in previews) == (1.0, 0.5, 0.25, 0.125)
    np.testing.assert_allclose(previews[0][1] @ MOUNT, target, atol=1.0e-12)
    expected_half = interpolate_transform(current, target, 0.5)
    np.testing.assert_allclose(previews[1][1] @ MOUNT, expected_half, atol=1.0e-12)


@pytest.mark.parametrize("alpha", [0.0, -0.2, 1.01, float("nan")])
def test_invalid_interpolation_fraction_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        interpolate_transform(np.eye(4), np.eye(4), alpha)


def test_reflection_mount_is_rejected() -> None:
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        camera_target_to_link7(np.eye(4), reflection)
