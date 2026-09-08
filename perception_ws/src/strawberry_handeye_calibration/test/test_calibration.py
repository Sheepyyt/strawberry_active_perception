"""Numerical and failure-mode tests for the offline hand-eye core."""

import numpy as np

from strawberry_handeye_calibration.calibration import (
    CalibrationConfig,
    _method_consensus_initialization,
    _refine_pairwise_ax_xb,
    calibrate_eye_in_hand,
)
from strawberry_handeye_calibration.fixture import make_synthetic_dataset
from strawberry_handeye_calibration.schema import CalibrationDataset, CalibrationSample
from strawberry_handeye_calibration.transforms import transform_error


def _ground_truth(dataset: CalibrationDataset) -> np.ndarray:
    return np.asarray(
        dataset.metadata["ground_truth_T_link7_camera_optical"],
        dtype=np.float64,
    )


def test_exact_fixture_recovers_link7_camera_transform() -> None:
    dataset = make_synthetic_dataset(
        sample_count=12,
        translation_noise_mm=0.0,
        rotation_noise_deg=0.0,
    )
    report = calibrate_eye_in_hand(dataset)
    assert report["success"]
    assert report["status"] == "passed"
    assert report["selected_method"] == "ROBUST_PAIRWISE_AX_XB"
    estimate = np.asarray(report["T_link7_camera_optical"])
    translation_mm, rotation_deg = transform_error(estimate, _ground_truth(dataset))
    assert translation_mm < 1.0e-6
    assert rotation_deg < 1.0e-5
    # This identity is the defining eye-in-hand direction, not its inverse.
    board_poses = [
        sample.T_base_link7 @ estimate @ sample.T_camera_checkerboard
        for sample in dataset.samples
    ]
    for pose in board_poses[1:]:
        np.testing.assert_allclose(pose, board_poses[0], atol=1.0e-9)


def test_noisy_fixture_passes_independent_holdout() -> None:
    dataset = make_synthetic_dataset(sample_count=16)
    report = calibrate_eye_in_hand(dataset)
    assert report["success"]
    assert len(report["holdout_sample_ids"]) == 3
    assert set(report["holdout_sample_ids"]).isdisjoint(
        report["selected_solver_sample_ids"]
    )
    translation_mm, rotation_deg = transform_error(
        np.asarray(report["T_link7_camera_optical"]),
        _ground_truth(dataset),
    )
    assert translation_mm < 1.0
    assert rotation_deg < 0.2
    assert report["metrics"]["holdout"]["translation_mm"]["p95"] < 1.0
    assert report["metrics"]["holdout"]["rotation_deg"]["p95"] < 0.2


def test_training_outlier_is_excluded_and_solution_remains_accurate() -> None:
    dataset = make_synthetic_dataset(sample_count=16, outlier_index=5)
    report = calibrate_eye_in_hand(dataset)
    assert report["success"]
    assert report["outlier_sample_ids"] == ["fixture_005"]
    translation_mm, rotation_deg = transform_error(
        np.asarray(report["T_link7_camera_optical"]),
        _ground_truth(dataset),
    )
    assert translation_mm < 1.0
    assert rotation_deg < 0.2


def test_holdout_outlier_fails_validation_instead_of_being_hidden() -> None:
    # Seed zero places fixture_002 in the independent holdout set.
    dataset = make_synthetic_dataset(sample_count=16, outlier_index=2)
    report = calibrate_eye_in_hand(dataset, CalibrationConfig(random_seed=0))
    assert not report["success"]
    assert report["status"] == "failed_holdout_validation"
    assert "fixture_002" in report["holdout_sample_ids"]
    assert "holdout translation" in report["reason"]
    assert report["T_link7_camera_optical"] is not None


def test_parallel_rotation_axes_are_reported_as_degenerate() -> None:
    samples = []
    for index in range(8):
        angle = np.deg2rad(index * 5.0)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        pose = np.eye(4)
        pose[:3, :3] = (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
        pose[0, 3] = index * 0.01
        samples.append(CalibrationSample(f"axis_{index}", pose, np.eye(4)))
    dataset = CalibrationDataset(
        session_id="degenerate",
        samples=tuple(samples),
        checkerboard_columns=11,
        checkerboard_rows=8,
        square_size_m=0.03,
    )
    report = calibrate_eye_in_hand(dataset)
    assert not report["success"]
    assert report["status"] == "failed_degenerate_motion"
    assert "one axis" in report["reason"]
    assert report["T_link7_camera_optical"] is None


def test_insufficient_samples_return_a_machine_readable_failure() -> None:
    dataset = make_synthetic_dataset(sample_count=6)
    short = CalibrationDataset(
        session_id="short",
        samples=dataset.samples[:5],
        checkerboard_columns=11,
        checkerboard_rows=8,
        square_size_m=0.03,
    )
    report = calibrate_eye_in_hand(short)
    assert not report["success"]
    assert report["status"] == "failed_insufficient_samples"
    assert report["sample_count"] == 5


def test_pairwise_refinement_is_invariant_to_sample_order() -> None:
    dataset = make_synthetic_dataset(sample_count=12)
    config = CalibrationConfig(holdout_fraction=0.0)
    initial, _, _ = _method_consensus_initialization(
        dataset.samples, tuple(range(12)), config
    )
    assert initial is not None
    forward, _ = _refine_pairwise_ax_xb(
        dataset.samples, tuple(range(12)), initial, config
    )
    reverse, _ = _refine_pairwise_ax_xb(
        dataset.samples, tuple(reversed(range(12))), initial, config
    )
    translation_mm, rotation_deg = transform_error(forward, reverse)
    assert translation_mm < 1.0e-6
    assert rotation_deg < 2.0e-6


def test_group_robust_refinement_is_rotation_basis_covariant() -> None:
    dataset = make_synthetic_dataset(sample_count=12)
    config = CalibrationConfig(holdout_fraction=0.0)
    initial, _, _ = _method_consensus_initialization(
        dataset.samples, tuple(range(12)), config
    )
    assert initial is not None
    original, _ = _refine_pairwise_ax_xb(
        dataset.samples, tuple(range(12)), initial, config
    )
    angle_link = np.deg2rad(31.0)
    angle_camera = np.deg2rad(-47.0)
    basis_link = np.eye(4)
    basis_link[:3, :3] = (
        (np.cos(angle_link), -np.sin(angle_link), 0.0),
        (np.sin(angle_link), np.cos(angle_link), 0.0),
        (0.0, 0.0, 1.0),
    )
    basis_camera = np.eye(4)
    basis_camera[:3, :3] = (
        (1.0, 0.0, 0.0),
        (0.0, np.cos(angle_camera), -np.sin(angle_camera)),
        (0.0, np.sin(angle_camera), np.cos(angle_camera)),
    )
    transformed_samples = tuple(
        CalibrationSample(
            sample.sample_id,
            sample.T_base_link7 @ basis_link,
            np.linalg.inv(basis_camera) @ sample.T_camera_checkerboard,
        )
        for sample in dataset.samples
    )
    expected_initial = np.linalg.inv(basis_link) @ initial @ basis_camera
    transformed, _ = _refine_pairwise_ax_xb(
        transformed_samples,
        tuple(range(12)),
        expected_initial,
        config,
    )
    expected = np.linalg.inv(basis_link) @ original @ basis_camera
    translation_mm, rotation_deg = transform_error(transformed, expected)
    assert translation_mm < 1.0e-5
    assert rotation_deg < 1.0e-5


def test_divergent_andreff_initialization_is_diagnosed_and_excluded() -> None:
    dataset = make_synthetic_dataset(sample_count=16, outlier_index=5)
    _, method_reports, _ = _method_consensus_initialization(
        dataset.samples,
        tuple(range(16)),
        CalibrationConfig(),
    )
    by_method = {report["method"]: report for report in method_reports}
    assert not by_method["ANDREFF"]["accepted_for_consensus"]
    assert any(
        "consensus" in reason
        for reason in by_method["ANDREFF"]["rejection_reasons"]
    )
