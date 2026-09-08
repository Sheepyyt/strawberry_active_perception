"""Tests for the reproducible G4 camera target fixture."""

import numpy as np

from strawberry_active_perception_bridge.simulation_fixture import (
    make_lateral_camera_target,
)


def test_fixture_moves_100_mm_and_keeps_optical_axis_on_target() -> None:
    """The fixture target has the agreed displacement and look-at direction."""
    current = np.eye(4)
    target = make_lateral_camera_target(current)
    np.testing.assert_allclose(target[:3, 3], (0.1, 0.0, 0.0), atol=1.0e-12)
    target_center = np.array((0.0, 0.0, 0.35))
    expected_forward = target_center - target[:3, 3]
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(target[:3, 2], expected_forward, atol=1.0e-12)
    np.testing.assert_allclose(
        target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1.0e-12
    )
    assert np.linalg.det(target[:3, :3]) > 0.999999999
