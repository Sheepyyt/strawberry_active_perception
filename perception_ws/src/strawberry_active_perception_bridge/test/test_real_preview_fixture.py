"""Pure geometry tests for the real-camera no-motion preview fixture."""

import numpy as np
import pytest

from strawberry_active_perception_bridge.real_preview_fixture import (
    make_small_nbv_target,
)


def test_small_target_moves_5mm_and_keeps_optical_z_on_target() -> None:
    current = np.eye(4)
    candidate = make_small_nbv_target(current, 0.005, 0.35)
    np.testing.assert_allclose(candidate[:3, 3], (0.005, 0.0, 0.0))
    target = np.array((0.0, 0.0, 0.35))
    direction = target - candidate[:3, 3]
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(candidate[:3, 2], direction, atol=1.0e-12)
    np.testing.assert_allclose(
        candidate[:3, :3].T @ candidate[:3, :3],
        np.eye(3),
        atol=1.0e-12,
    )
    assert np.linalg.det(candidate[:3, :3]) == pytest.approx(1.0)


@pytest.mark.parametrize("step", [0.0, -0.001, 0.010001])
def test_small_target_rejects_non_small_step(step) -> None:
    with pytest.raises(ValueError, match="small_nbv_step_m"):
        make_small_nbv_target(np.eye(4), step, 0.35)
