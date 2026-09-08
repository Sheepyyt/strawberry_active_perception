"""Tests for frozen train/test hand-eye validation and its audit trail."""

from dataclasses import replace
import hashlib

import numpy as np

from strawberry_handeye_calibration.fixture import make_synthetic_dataset
from strawberry_handeye_calibration.prospective import validate_prospective_file
from strawberry_handeye_calibration.schema import (
    CalibrationDataset,
    CalibrationSample,
    save_dataset,
)


def _split_ids(dataset: CalibrationDataset) -> tuple[list[str], list[str]]:
    ids = [sample.sample_id for sample in dataset.samples]
    return ids[:25], ids[25:]


def test_exact_prospective_check_freezes_fit_and_binds_input_hash(tmp_path) -> None:
    source = tmp_path / "samples.json"
    dataset = make_synthetic_dataset(
        sample_count=30,
        translation_noise_mm=0.0,
        rotation_noise_deg=0.0,
    )
    save_dataset(dataset, source)
    training_ids, test_ids = _split_ids(dataset)

    report = validate_prospective_file(source, training_ids, test_ids)

    assert report["passed"]
    assert not report["safe_for_robot_use"]
    assert report["training_fit"]["selected_method"] == "ROBUST_PAIRWISE_AX_XB"
    assert report["training_fit"]["solver_training_sample_ids"] == training_ids
    assert report["training_fit"]["solver_holdout_sample_ids"] == []
    assert report["evaluation_protocol"]["transform_refit_with_test_samples"] is False
    assert report["prospective_metrics"]["sample_ids"] == test_ids
    assert report["prospective_metrics"]["translation_mm"]["max"] < 1.0e-5
    assert report["prospective_metrics"]["rotation_deg"]["max"] < 1.0e-5
    assert report["input_dataset"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_prospective_outlier_fails_p95_without_changing_training_fit(tmp_path) -> None:
    source = tmp_path / "samples_with_late_error.json"
    dataset = make_synthetic_dataset(
        sample_count=30,
        translation_noise_mm=0.0,
        rotation_noise_deg=0.0,
    )
    samples = list(dataset.samples)
    corrupted = samples[-1].T_camera_checkerboard.copy()
    corrupted[0, 3] += 0.03
    samples[-1] = CalibrationSample(
        samples[-1].sample_id,
        samples[-1].T_base_link7,
        corrupted,
    )
    changed = replace(dataset, samples=tuple(samples))
    save_dataset(changed, source)
    training_ids, test_ids = _split_ids(changed)

    report = validate_prospective_file(source, training_ids, test_ids)

    assert not report["passed"]
    assert not report["safe_for_robot_use"]
    assert report["acceptance_gate"]["translation_passed"] is False
    assert report["prospective_metrics"]["translation_mm"]["p95"] > 10.0
    assert report["training_fit"]["solver_training_sample_ids"] == training_ids


def test_selection_must_be_disjoint_complete_partition(tmp_path) -> None:
    source = tmp_path / "samples.json"
    dataset = make_synthetic_dataset(sample_count=10)
    save_dataset(dataset, source)
    ids = [sample.sample_id for sample in dataset.samples]

    with np.testing.assert_raises_regex(ValueError, "overlap"):
        validate_prospective_file(source, ids[:6], ids[5:])
    with np.testing.assert_raises_regex(ValueError, "unselected"):
        validate_prospective_file(source, ids[:6], ids[7:])
