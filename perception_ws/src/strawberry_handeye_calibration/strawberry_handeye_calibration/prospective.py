"""Frozen prospective holdout validation for offline eye-in-hand calibration.

The training samples are solved exactly once.  The resulting hand-eye transform
and stationary-checkerboard reference are then held fixed while later samples
are evaluated.  Nothing in this module imports ROS or contacts robot hardware.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .calibration import CalibrationConfig, calibrate_eye_in_hand
from .schema import (
    CalibrationDataset,
    SCHEMA_VERSION,
    TRANSFORM_CONVENTION,
    load_dataset,
)
from .transforms import transform_error, validate_transform


PROSPECTIVE_REPORT_VERSION = "strawberry_handeye_prospective_holdout/v2"
ROBUST_SOLVER_NAME = "ROBUST_PAIRWISE_AX_XB"
DIAGNOSTIC_NOTE = (
    "This is a frozen prospective diagnostic, not robot-use authorization. "
    "safe_for_robot_use remains false; only the formal 20-split stability "
    "report may authorize a transform for robot use."
)


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _validate_ids(
    dataset: CalibrationDataset,
    training_sample_ids: Sequence[str],
    prospective_sample_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    training = tuple(str(value) for value in training_sample_ids)
    prospective = tuple(str(value) for value in prospective_sample_ids)
    if not training or not prospective:
        raise ValueError("training and prospective sample ID lists must be non-empty")
    if len(set(training)) != len(training):
        raise ValueError("training sample IDs must be unique")
    if len(set(prospective)) != len(prospective):
        raise ValueError("prospective sample IDs must be unique")
    overlap = sorted(set(training) & set(prospective))
    if overlap:
        raise ValueError(
            "training and prospective sample IDs overlap: " + ", ".join(overlap)
        )
    available = {sample.sample_id for sample in dataset.samples}
    missing = sorted((set(training) | set(prospective)) - available)
    if missing:
        raise ValueError("unknown sample IDs: " + ", ".join(missing))
    unselected = sorted(available - (set(training) | set(prospective)))
    if unselected:
        raise ValueError(
            "training and prospective IDs must partition the input dataset; "
            "unselected IDs: " + ", ".join(unselected)
        )
    return training, prospective


def _subset_dataset(
    dataset: CalibrationDataset,
    sample_ids: tuple[str, ...],
) -> CalibrationDataset:
    by_id = {sample.sample_id: sample for sample in dataset.samples}
    return CalibrationDataset(
        session_id=dataset.session_id + "/prospective_training",
        samples=tuple(by_id[sample_id] for sample_id in sample_ids),
        checkerboard_columns=dataset.checkerboard_columns,
        checkerboard_rows=dataset.checkerboard_rows,
        square_size_m=dataset.square_size_m,
        base_frame=dataset.base_frame,
        link_frame=dataset.link_frame,
        camera_frame=dataset.camera_frame,
        checkerboard_frame=dataset.checkerboard_frame,
        metadata={
            "source_session_id": dataset.session_id,
            "selection": "frozen prospective training subset",
        },
    )


def _empty_metrics() -> dict[str, Any]:
    return {
        "sample_ids": [],
        "translation_mm": _summary(()),
        "rotation_deg": _summary(()),
        "per_sample": [],
    }


def validate_prospective_holdout(
    dataset: CalibrationDataset,
    training_sample_ids: Sequence[str],
    prospective_sample_ids: Sequence[str],
    *,
    input_path: str,
    input_sha256: str,
    input_byte_count: int,
    maximum_translation_p95_mm: float = 10.0,
    maximum_rotation_p95_deg: float = 2.0,
    calibration_config: CalibrationConfig | None = None,
) -> dict[str, Any]:
    """Fit once on declared training IDs and score only the later IDs.

    The acceptance thresholds apply to prospective P95 errors.  Maximum errors
    are always reported separately so a single large residual cannot be hidden.
    The returned report is deliberately diagnostic and always sets
    ``safe_for_robot_use`` to false.
    """
    training_ids, prospective_ids = _validate_ids(
        dataset, training_sample_ids, prospective_sample_ids
    )
    limits = (maximum_translation_p95_mm, maximum_rotation_p95_deg)
    if any(not np.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("prospective P95 limits must be finite and positive")
    if len(input_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in input_sha256.lower()
    ):
        raise ValueError("input_sha256 must be a 64-character hexadecimal digest")
    if input_byte_count <= 0:
        raise ValueError("input_byte_count must be positive")

    config = (
        CalibrationConfig(holdout_fraction=0.0)
        if calibration_config is None
        else calibration_config
    )
    if config.holdout_fraction != 0.0:
        raise ValueError(
            "prospective training requires holdout_fraction=0 so all declared "
            "training samples reach the robust solver"
        )
    training_dataset = _subset_dataset(dataset, training_ids)
    training_report = calibrate_eye_in_hand(training_dataset, config)
    actual_training_ids = tuple(training_report.get("training_sample_ids", ()))
    actual_holdout_ids = tuple(training_report.get("holdout_sample_ids", ()))
    solver_protocol_ok = (
        bool(training_report["success"])
        and training_report.get("selected_method") == ROBUST_SOLVER_NAME
        and actual_training_ids == training_ids
        and not actual_holdout_ids
    )

    transform_value = training_report.get("T_link7_camera_optical")
    reference_value = training_report.get("estimated_T_base_checkerboard")
    prospective_metrics = _empty_metrics()
    if solver_protocol_ok and transform_value is not None and reference_value is not None:
        transform = validate_transform(
            np.asarray(transform_value, dtype=np.float64),
            "frozen_T_link7_camera_optical",
        )
        reference = validate_transform(
            np.asarray(reference_value, dtype=np.float64),
            "fixed_training_T_base_checkerboard",
        )
        by_id = {sample.sample_id: sample for sample in dataset.samples}
        translations = []
        rotations = []
        per_sample = []
        for sample_id in prospective_ids:
            sample = by_id[sample_id]
            observed_board = (
                sample.T_base_link7
                @ transform
                @ sample.T_camera_checkerboard
            )
            translation_mm, rotation_deg = transform_error(
                observed_board, reference
            )
            translations.append(translation_mm)
            rotations.append(rotation_deg)
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "translation_mm": translation_mm,
                    "rotation_deg": rotation_deg,
                }
            )
        prospective_metrics = {
            "sample_ids": list(prospective_ids),
            "translation_mm": _summary(translations),
            "rotation_deg": _summary(rotations),
            "per_sample": per_sample,
        }

    translation_p95 = prospective_metrics["translation_mm"]["p95"]
    rotation_p95 = prospective_metrics["rotation_deg"]["p95"]
    translation_passed = (
        translation_p95 is not None
        and translation_p95 <= maximum_translation_p95_mm
    )
    rotation_passed = (
        rotation_p95 is not None
        and rotation_p95 <= maximum_rotation_p95_deg
    )
    passed = solver_protocol_ok and translation_passed and rotation_passed
    failures = []
    if not solver_protocol_ok:
        failures.append(
            "training did not complete the required full-set "
            f"{ROBUST_SOLVER_NAME} protocol"
        )
    if translation_p95 is not None and not translation_passed:
        failures.append(
            f"prospective translation P95 {translation_p95:.3f} mm exceeds "
            f"{maximum_translation_p95_mm:.3f} mm"
        )
    if rotation_p95 is not None and not rotation_passed:
        failures.append(
            f"prospective rotation P95 {rotation_p95:.3f} deg exceeds "
            f"{maximum_rotation_p95_deg:.3f} deg"
        )
    reason = (
        "frozen prospective P95 gates passed"
        if passed
        else "; ".join(failures) or "prospective metrics are unavailable"
    )

    return {
        "report_version": PROSPECTIVE_REPORT_VERSION,
        "passed": passed,
        "status": "passed_diagnostic" if passed else "failed_prospective_gate",
        "reason": reason,
        "diagnostic_only": True,
        "diagnostic_note": DIAGNOSTIC_NOTE,
        "safe_for_robot_use": False,
        "input_dataset": {
            "path": input_path,
            "sha256": input_sha256.lower(),
            "byte_count": int(input_byte_count),
            "schema_version": SCHEMA_VERSION,
            "session_id": dataset.session_id,
            "sample_count": len(dataset.samples),
        },
        "transform_convention": dict(TRANSFORM_CONVENTION),
        "evaluation_protocol": {
            "solver": ROBUST_SOLVER_NAME,
            "training_holdout_fraction": 0.0,
            "all_declared_training_samples_given_to_solver": (
                actual_training_ids == training_ids and not actual_holdout_ids
            ),
            "test_samples_used_for_training": False,
            "transform_refit_with_test_samples": False,
            "checkerboard_reference_refit_with_test_samples": False,
            "checkerboard_reference_estimator": (
                "training-inlier robust center: componentwise median translation "
                "and chordal projected rotation"
            ),
            "acceptance_metric": "prospective P95",
            "maximum_errors_are_reported_but_not_used_as_the_gate_metric": True,
        },
        "training_sample_ids": list(training_ids),
        "prospective_sample_ids": list(prospective_ids),
        "training_fit": {
            "success": bool(training_report["success"]),
            "status": training_report["status"],
            "reason": training_report["reason"],
            "config": asdict(config),
            "input_sample_count": len(training_dataset.samples),
            "solver_training_sample_ids": list(actual_training_ids),
            "solver_holdout_sample_ids": list(actual_holdout_ids),
            "selected_method": training_report.get("selected_method"),
            "selected_initial_method": training_report.get(
                "selected_initial_method"
            ),
            "inlier_sample_ids": training_report.get("inlier_sample_ids", []),
            "outlier_sample_ids": training_report.get("outlier_sample_ids", []),
            "selected_solver_sample_ids": training_report.get(
                "selected_solver_sample_ids", []
            ),
            "single_fit_safe_for_robot_use": bool(
                training_report.get("safe_for_robot_use", False)
            ),
        },
        "frozen_T_link7_camera_optical": transform_value,
        "fixed_training_T_base_checkerboard": reference_value,
        "prospective_metrics": prospective_metrics,
        "acceptance_gate": {
            "metric": "P95",
            "maximum_translation_mm": float(maximum_translation_p95_mm),
            "maximum_rotation_deg": float(maximum_rotation_p95_deg),
            "translation_passed": bool(translation_passed),
            "rotation_passed": bool(rotation_passed),
            "passed": bool(passed),
        },
        "formal_robot_use_authority": (
            "strawberry_handeye_stability_report/v1 with all 20 splits and "
            "residual-scale trials passed"
        ),
    }


def validate_prospective_file(
    input_path: str | Path,
    training_sample_ids: Sequence[str],
    prospective_sample_ids: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Load one dataset, bind its exact SHA-256, and validate it offline."""
    source = Path(input_path)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    dataset = load_dataset(source)
    return validate_prospective_holdout(
        dataset,
        training_sample_ids,
        prospective_sample_ids,
        input_path=str(source.resolve()),
        input_sha256=digest.hexdigest(),
        input_byte_count=byte_count,
        **kwargs,
    )
