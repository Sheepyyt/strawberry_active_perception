"""Tests for multi-holdout hand-eye stability acceptance."""

import numpy as np

from strawberry_handeye_calibration.fixture import make_synthetic_dataset
from strawberry_handeye_calibration.schema import (
    CalibrationDataset,
    CalibrationSample,
)
from strawberry_handeye_calibration.stability import (
    StabilityConfig,
    validate_cross_split_stability,
)


def test_synthetic_dataset_passes_repeated_holdouts() -> None:
    report = validate_cross_split_stability(
        make_synthetic_dataset(sample_count=15),
        stability_config=StabilityConfig(split_seeds=tuple(range(20))),
    )
    assert report["success"]
    assert report["status"] == "passed"
    assert report["successful_split_count"] == 20
    assert report["safe_for_robot_use"]
    assert report["T_link7_camera_optical"] is not None
    assert not report["physical_prior_used_for_acceptance"]
    assert report["cross_split_transform_stability"]["translation_mm"]["max"] < 10.0
    assert report["residual_scale_sensitivity"]["translation_mm"]["max"] < 10.0


def test_fewer_than_twenty_splits_never_yields_robot_usable_transform() -> None:
    report = validate_cross_split_stability(
        make_synthetic_dataset(sample_count=15),
        stability_config=StabilityConfig(split_seeds=(0, 1, 2, 3)),
    )
    assert not report["success"]
    assert "at least 20" in report["reason"]
    assert not report["safe_for_robot_use"]
    assert report["T_link7_camera_optical"] is None


def test_degenerate_dataset_has_no_robot_usable_transform() -> None:
    samples = tuple(
        CalibrationSample(
            f"pose_{index:03d}",
            np.eye(4),
            np.eye(4),
        )
        for index in range(8)
    )
    dataset = CalibrationDataset(
        session_id="degenerate",
        samples=samples,
        checkerboard_columns=11,
        checkerboard_rows=8,
        square_size_m=0.03,
    )
    report = validate_cross_split_stability(
        dataset,
        stability_config=StabilityConfig(split_seeds=(0, 1)),
    )
    assert not report["success"]
    assert report["status"] == "failed_cross_split_stability"
    assert not report["safe_for_robot_use"]
    assert report["T_link7_camera_optical"] is None
    assert report["successful_split_count"] == 0
