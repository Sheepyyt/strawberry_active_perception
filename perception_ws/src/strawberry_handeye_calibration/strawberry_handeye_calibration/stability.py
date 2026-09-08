"""Cross-split stability acceptance for offline eye-in-hand calibration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import itertools
from typing import Any

import numpy as np

from .calibration import CalibrationConfig, calibrate_eye_in_hand
from .schema import CalibrationDataset, TRANSFORM_CONVENTION
from .transforms import transform_error


STABILITY_REPORT_VERSION = "strawberry_handeye_stability_report/v1"


@dataclass(frozen=True)
class StabilityConfig:
    """Independent holdout repetitions and required estimate agreement."""

    split_seeds: tuple[int, ...] = tuple(range(20))
    minimum_success_fraction: float = 1.0
    maximum_pairwise_translation_mm: float = 10.0
    maximum_pairwise_rotation_deg: float = 2.0
    residual_scale_trials: tuple[tuple[float, float], ...] = (
        (1.0, 0.125),
        (1.0, 0.25),
        (1.0, 0.5),
        (2.0, 0.25),
        (2.0, 0.5),
        (3.0, 1.0),
    )
    maximum_scale_sensitivity_translation_mm: float = 10.0
    maximum_scale_sensitivity_rotation_deg: float = 2.0

    def __post_init__(self) -> None:
        seeds = tuple(int(value) for value in self.split_seeds)
        object.__setattr__(self, "split_seeds", seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("split_seeds must be a non-empty tuple of unique integers")
        if self.minimum_success_fraction != 1.0:
            raise ValueError("robot-use stability always requires every split to pass")
        scales = tuple(
            (float(translation), float(rotation))
            for translation, rotation in self.residual_scale_trials
        )
        object.__setattr__(self, "residual_scale_trials", scales)
        if not scales or len(set(scales)) != len(scales):
            raise ValueError("residual_scale_trials must be non-empty and unique")
        if any(
            not np.isfinite(value) or value <= 0.0
            for scale in scales
            for value in scale
        ):
            raise ValueError("all residual scale trials must be finite and positive")
        limits = (
            self.maximum_pairwise_translation_mm,
            self.maximum_pairwise_rotation_deg,
            self.maximum_scale_sensitivity_translation_mm,
            self.maximum_scale_sensitivity_rotation_deg,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in limits):
            raise ValueError("cross-split transform limits must be finite and positive")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _metric_p95(report: dict[str, Any], group: str, quantity: str) -> float | None:
    value = (
        report.get("metrics", {})
        .get(group, {})
        .get(quantity, {})
        .get("p95")
    )
    return None if value is None else float(value)


def _compact_run(seed: int, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "success": bool(report["success"]),
        "status": report["status"],
        "reason": report["reason"],
        "selected_method": report.get("selected_method"),
        "T_link7_camera_optical": report.get("T_link7_camera_optical"),
        "training_sample_ids": report.get("training_sample_ids", []),
        "holdout_sample_ids": report.get("holdout_sample_ids", []),
        "inlier_sample_ids": report.get("inlier_sample_ids", []),
        "outlier_sample_ids": report.get("outlier_sample_ids", []),
        "training_translation_p95_mm": _metric_p95(
            report, "training", "translation_mm"
        ),
        "training_rotation_p95_deg": _metric_p95(
            report, "training", "rotation_deg"
        ),
        "holdout_translation_p95_mm": _metric_p95(
            report, "holdout", "translation_mm"
        ),
        "holdout_rotation_p95_deg": _metric_p95(
            report, "holdout", "rotation_deg"
        ),
        "ax_xb_translation_p95_mm": _metric_p95(
            report, "ax_xb_training_inliers", "translation_mm"
        ),
        "ax_xb_rotation_p95_deg": _metric_p95(
            report, "ax_xb_training_inliers", "rotation_deg"
        ),
    }


def _pairwise_stability(runs: list[dict[str, Any]]) -> dict[str, Any]:
    finite = []
    for run in runs:
        value = run["T_link7_camera_optical"]
        if value is not None:
            finite.append((int(run["seed"]), str(run["selected_method"]), np.asarray(value)))
    translations = []
    rotations = []
    pairs = []
    for (first_seed, first_method, first), (second_seed, second_method, second) in (
        itertools.combinations(finite, 2)
    ):
        translation, rotation = transform_error(first, second)
        translations.append(translation)
        rotations.append(rotation)
        pairs.append(
            {
                "first_seed": first_seed,
                "first_method": first_method,
                "second_seed": second_seed,
                "second_method": second_method,
                "translation_mm": translation,
                "rotation_deg": rotation,
            }
        )
    worst_translation = (
        max(pairs, key=lambda item: item["translation_mm"]) if pairs else None
    )
    worst_rotation = max(pairs, key=lambda item: item["rotation_deg"]) if pairs else None
    return {
        "finite_transform_count": len(finite),
        "pair_count": len(pairs),
        "translation_mm": _summary(translations),
        "rotation_deg": _summary(rotations),
        "worst_translation_pair": worst_translation,
        "worst_rotation_pair": worst_rotation,
    }


def _medoid_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    finite = [run for run in runs if run["T_link7_camera_optical"] is not None]
    if not finite:
        return None
    best: tuple[float, int, dict[str, Any]] | None = None
    for candidate in finite:
        candidate_transform = np.asarray(candidate["T_link7_camera_optical"])
        cost = 0.0
        for other in finite:
            translation, rotation = transform_error(
                candidate_transform,
                np.asarray(other["T_link7_camera_optical"]),
            )
            cost += translation + 5.0 * rotation
        key = (cost, int(candidate["seed"]), candidate)
        if best is None or key[:2] < best[:2]:
            best = key
    return best[2] if best is not None else None


def _scale_sensitivity(
    dataset: CalibrationDataset,
    base: CalibrationConfig,
    settings: StabilityConfig,
) -> dict[str, Any]:
    """Solve the full dataset at every predeclared residual normalization."""
    runs = []
    finite = []
    for translation_mm, rotation_deg in settings.residual_scale_trials:
        report = calibrate_eye_in_hand(
            dataset,
            replace(
                base,
                holdout_fraction=0.0,
                pairwise_translation_scale_mm=translation_mm,
                pairwise_rotation_scale_deg=rotation_deg,
            ),
        )
        value = report.get("T_link7_camera_optical")
        run = {
            "translation_scale_mm": translation_mm,
            "rotation_scale_deg": rotation_deg,
            "success": bool(report["success"]),
            "status": report["status"],
            "reason": report["reason"],
            "T_link7_camera_optical": value,
        }
        runs.append(run)
        if value is not None:
            finite.append((translation_mm, rotation_deg, np.asarray(value)))
    pairs = []
    for first, second in itertools.combinations(finite, 2):
        translation, rotation = transform_error(first[2], second[2])
        pairs.append(
            {
                "first_translation_scale_mm": first[0],
                "first_rotation_scale_deg": first[1],
                "second_translation_scale_mm": second[0],
                "second_rotation_scale_deg": second[1],
                "translation_mm": translation,
                "rotation_deg": rotation,
            }
        )
    return {
        "trial_count": len(runs),
        "finite_transform_count": len(finite),
        "all_trials_passed": all(run["success"] for run in runs),
        "translation_mm": _summary(
            [float(pair["translation_mm"]) for pair in pairs]
        ),
        "rotation_deg": _summary(
            [float(pair["rotation_deg"]) for pair in pairs]
        ),
        "runs": runs,
        "pairs": pairs,
    }


def validate_cross_split_stability(
    dataset: CalibrationDataset,
    calibration_config: CalibrationConfig | None = None,
    stability_config: StabilityConfig | None = None,
) -> dict[str, Any]:
    """Require calibration to pass and agree across many independent holdouts."""
    base = CalibrationConfig() if calibration_config is None else calibration_config
    settings = StabilityConfig() if stability_config is None else stability_config
    full_reports = [
        calibrate_eye_in_hand(dataset, replace(base, random_seed=seed))
        for seed in settings.split_seeds
    ]
    runs = [
        _compact_run(seed, report)
        for seed, report in zip(settings.split_seeds, full_reports)
    ]
    success_count = sum(run["success"] for run in runs)
    success_fraction = success_count / len(runs)
    pairwise = _pairwise_stability(runs)
    scale_sensitivity = _scale_sensitivity(dataset, base, settings)
    failures = []
    if len(runs) < 20:
        failures.append(
            f"only {len(runs)} independent holdout splits were run; at least 20 are required"
        )
    if success_count != len(runs):
        failures.append(
            f"only {success_count}/{len(runs)} holdout splits passed "
            "(every split must pass)"
        )
    if pairwise["finite_transform_count"] != len(runs):
        failures.append("one or more holdout splits produced no finite transform")
    translation_max = pairwise["translation_mm"]["max"]
    rotation_max = pairwise["rotation_deg"]["max"]
    if (
        translation_max is not None
        and translation_max > settings.maximum_pairwise_translation_mm
    ):
        failures.append(
            f"cross-split transform translation spread {translation_max:.3f} mm "
            f"exceeds {settings.maximum_pairwise_translation_mm:.3f} mm"
        )
    if rotation_max is not None and rotation_max > settings.maximum_pairwise_rotation_deg:
        failures.append(
            f"cross-split transform rotation spread {rotation_max:.3f} deg "
            f"exceeds {settings.maximum_pairwise_rotation_deg:.3f} deg"
        )
    scale_translation_max = scale_sensitivity["translation_mm"]["max"]
    scale_rotation_max = scale_sensitivity["rotation_deg"]["max"]
    if not scale_sensitivity["all_trials_passed"]:
        failures.append("one or more residual-scale sensitivity trials failed")
    if scale_sensitivity["finite_transform_count"] != len(
        settings.residual_scale_trials
    ):
        failures.append("one or more residual-scale trials produced no finite transform")
    if (
        scale_translation_max is not None
        and scale_translation_max
        > settings.maximum_scale_sensitivity_translation_mm
    ):
        failures.append(
            "residual-scale transform translation spread "
            f"{scale_translation_max:.3f} mm exceeds "
            f"{settings.maximum_scale_sensitivity_translation_mm:.3f} mm"
        )
    if (
        scale_rotation_max is not None
        and scale_rotation_max > settings.maximum_scale_sensitivity_rotation_deg
    ):
        failures.append(
            "residual-scale transform rotation spread "
            f"{scale_rotation_max:.3f} deg exceeds "
            f"{settings.maximum_scale_sensitivity_rotation_deg:.3f} deg"
        )
    method_counts = Counter(
        str(run["selected_method"])
        for run in runs
        if run["selected_method"] is not None
    )
    outlier_counts = Counter(
        sample_id for run in runs for sample_id in run["outlier_sample_ids"]
    )
    medoid = _medoid_run(runs)
    success = not failures
    return {
        "report_version": STABILITY_REPORT_VERSION,
        "success": success,
        "status": "passed" if success else "failed_cross_split_stability",
        "reason": (
            "calibration passed all independent holdout and transform stability checks"
            if success
            else "; ".join(failures)
        ),
        "session_id": dataset.session_id,
        "sample_count": len(dataset.samples),
        "transform_convention": dict(TRANSFORM_CONVENTION),
        "calibration_config": asdict(base),
        "stability_config": asdict(settings),
        "successful_split_count": success_count,
        "split_count": len(runs),
        "successful_split_fraction": success_fraction,
        "selected_method_counts": dict(sorted(method_counts.items())),
        "outlier_occurrence_counts": dict(sorted(outlier_counts.items())),
        "cross_split_transform_stability": pairwise,
        "residual_scale_sensitivity": scale_sensitivity,
        "motion_diagnostics": full_reports[0]["motion_diagnostics"],
        "split_runs": runs,
        "recommended_seed": int(medoid["seed"]) if success and medoid else None,
        "recommended_method": medoid["selected_method"] if success and medoid else None,
        "T_link7_camera_optical": (
            medoid["T_link7_camera_optical"] if success and medoid else None
        ),
        "diagnostic_medoid_seed": int(medoid["seed"]) if medoid else None,
        "diagnostic_medoid_method": medoid["selected_method"] if medoid else None,
        "diagnostic_medoid_T_link7_camera_optical": (
            medoid["T_link7_camera_optical"] if medoid else None
        ),
        "physical_prior_used_for_acceptance": False,
        "safe_for_robot_use": success,
    }
