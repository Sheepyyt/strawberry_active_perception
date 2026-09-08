"""Tests for the explicit, versioned hand-eye sample contract."""

import json

import numpy as np
import pytest

from strawberry_handeye_calibration.fixture import make_synthetic_dataset
from strawberry_handeye_calibration.schema import (
    CalibrationDataset,
    CalibrationSample,
    SCHEMA_VERSION,
    load_dataset,
    save_dataset,
)


def test_dataset_json_round_trip_preserves_direction_and_values(tmp_path) -> None:
    dataset = make_synthetic_dataset(sample_count=8, translation_noise_mm=0.0)
    path = tmp_path / "session.json"
    save_dataset(dataset, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert "p_base = T_base_link7 p_link7" in raw["transform_convention"]["T_base_link7"]
    assert "p_camera = T_camera_checkerboard" in raw["transform_convention"][
        "T_camera_checkerboard"
    ]
    recovered = load_dataset(path)
    assert recovered.session_id == dataset.session_id
    assert len(recovered.samples) == len(dataset.samples)
    np.testing.assert_allclose(
        recovered.samples[3].T_base_link7,
        dataset.samples[3].T_base_link7,
    )
    np.testing.assert_allclose(
        recovered.samples[3].T_camera_checkerboard,
        dataset.samples[3].T_camera_checkerboard,
    )


def test_duplicate_sample_ids_are_rejected() -> None:
    sample = CalibrationSample("same", np.eye(4), np.eye(4))
    with pytest.raises(ValueError, match="unique"):
        CalibrationDataset(
            session_id="bad",
            samples=(sample, sample),
            checkerboard_columns=11,
            checkerboard_rows=8,
            square_size_m=0.03,
        )


def test_non_rigid_pose_is_rejected_before_solver() -> None:
    scaled = np.eye(4)
    scaled[0, 0] = 1.02
    with pytest.raises(ValueError, match="orthonormal"):
        CalibrationSample("bad_pose", scaled, np.eye(4))


def test_wrong_schema_or_transform_convention_is_rejected() -> None:
    raw = make_synthetic_dataset(sample_count=8).to_dict()
    raw["schema_version"] = "future/v99"
    with pytest.raises(ValueError, match="schema_version"):
        CalibrationDataset.from_dict(raw)
    raw = make_synthetic_dataset(sample_count=8).to_dict()
    raw["transform_convention"]["T_base_link7"] = "ambiguous"
    with pytest.raises(ValueError, match="transform_convention"):
        CalibrationDataset.from_dict(raw)
