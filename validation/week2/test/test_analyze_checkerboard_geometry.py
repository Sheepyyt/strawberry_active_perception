"""Pure numerical tests for checkerboard RGB-D geometry validation."""

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "analyze_checkerboard_geometry.py"
SPEC = importlib.util.spec_from_file_location("checkerboard_geometry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_plane_fit_recovers_a_deterministic_plane_with_outliers() -> None:
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 30), np.linspace(-0.1, 0.1, 20))
    z = 0.6 + 0.1 * x - 0.05 * y
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    points[::100, 2] += 0.05

    normal, offset, inliers = MODULE._fit_plane(points)
    expected = np.asarray([-0.1, 0.05, 1.0])
    expected /= np.linalg.norm(expected)

    assert np.allclose(normal, expected, atol=1e-9)
    assert np.isclose(offset, -0.6 / np.linalg.norm([-0.1, 0.05, 1.0]))
    assert np.count_nonzero(~inliers) == 6


def test_upward_crossing_uses_subpixel_linear_interpolation() -> None:
    offsets = np.arange(-1.0, 4.0, 0.25, dtype=np.float32)
    profile = 2.0 * offsets + 1.0

    crossing = MODULE._upward_crossing(profile, offsets, threshold=4.0)

    assert np.isclose(crossing, 1.5)


def test_upward_crossing_does_not_invent_a_missing_edge() -> None:
    offsets = np.arange(-1.0, 4.0, 0.25, dtype=np.float32)
    profile = np.zeros_like(offsets)

    assert np.isnan(MODULE._upward_crossing(profile, offsets, threshold=1.0))
