"""Robust, offline eye-in-hand calibration independent of ROS and robot I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .schema import CalibrationDataset, CalibrationSample, TRANSFORM_CONVENTION
from .transforms import (
    invert_transform,
    make_transform,
    mean_transform,
    rotation_angle_deg,
    rotation_vector,
    transform_error,
    validate_transform,
)


REPORT_VERSION = "strawberry_handeye_report/v1"
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class CalibrationConfig:
    """Deterministic solver, outlier, degeneracy, and acceptance settings."""

    methods: tuple[str, ...] = tuple(METHODS)
    holdout_fraction: float = 0.2
    random_seed: int = 0
    min_samples: int = 6
    min_solver_samples: int = 4
    min_rotation_span_deg: float = 8.0
    min_axis_diversity_ratio: float = 0.05
    min_translation_span_mm: float = 10.0
    ransac_subsets: int = 16
    outlier_translation_mm: float = 8.0
    outlier_rotation_deg: float = 1.5
    max_holdout_translation_p95_mm: float = 10.0
    max_holdout_rotation_p95_deg: float = 2.0
    min_method_consensus: int = 3
    method_consensus_translation_mm: float = 10.0
    method_consensus_rotation_deg: float = 2.0
    max_initial_ax_xb_translation_p95_mm: float = 10.0
    max_initial_ax_xb_rotation_p95_deg: float = 2.0
    pairwise_translation_scale_mm: float = 1.0
    pairwise_rotation_scale_deg: float = 0.25
    max_refinement_initialization_translation_mm: float = 1.0
    max_refinement_initialization_rotation_deg: float = 0.1
    refinement_max_nfev: int = 1000
    refinement_irls_iterations: int = 50

    def __post_init__(self) -> None:
        """Reject a configuration that could silently weaken validation."""
        methods = tuple(str(method).upper() for method in self.methods)
        object.__setattr__(self, "methods", methods)
        unknown = sorted(set(methods) - set(METHODS))
        if unknown:
            raise ValueError(f"unsupported hand-eye method(s): {', '.join(unknown)}")
        if not methods:
            raise ValueError("at least one hand-eye method is required")
        if not 0.0 <= self.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be in [0, 0.5)")
        if self.min_solver_samples < 3:
            raise ValueError("min_solver_samples must be at least 3")
        if self.min_samples < self.min_solver_samples + 1:
            raise ValueError("min_samples must leave room for an independent holdout")
        if self.min_method_consensus < 1:
            raise ValueError("min_method_consensus must be at least 1")
        if self.refinement_max_nfev < 1:
            raise ValueError("refinement_max_nfev must be at least 1")
        if self.refinement_irls_iterations < 1:
            raise ValueError("refinement_irls_iterations must be at least 1")
        positive = (
            self.min_rotation_span_deg,
            self.min_axis_diversity_ratio,
            self.min_translation_span_mm,
            self.outlier_translation_mm,
            self.outlier_rotation_deg,
            self.max_holdout_translation_p95_mm,
            self.max_holdout_rotation_p95_deg,
            self.method_consensus_translation_mm,
            self.method_consensus_rotation_deg,
            self.max_initial_ax_xb_translation_p95_mm,
            self.max_initial_ax_xb_rotation_p95_deg,
            self.pairwise_translation_scale_mm,
            self.pairwise_rotation_scale_deg,
            self.max_refinement_initialization_translation_mm,
            self.max_refinement_initialization_rotation_deg,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("all motion and residual thresholds must be positive")
        if self.ransac_subsets < 0:
            raise ValueError("ransac_subsets cannot be negative")


@dataclass(frozen=True)
class _Candidate:
    method: str
    indices: tuple[int, ...]
    transform: np.ndarray
    score: float
    metrics: dict[str, float | int | None]


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _motion_diagnostics(samples: tuple[CalibrationSample, ...]) -> dict[str, Any]:
    rotations = []
    translations_mm = []
    for first, second in itertools.combinations(samples, 2):
        relative = invert_transform(first.T_base_link7) @ second.T_base_link7
        vector = rotation_vector(relative[:3, :3])
        angle_deg = math.degrees(float(np.linalg.norm(vector)))
        translations_mm.append(float(np.linalg.norm(relative[:3, 3]) * 1000.0))
        if angle_deg >= 1.0:
            rotations.append(vector / np.linalg.norm(vector))
    if rotations:
        singular_values = np.linalg.svd(np.stack(rotations), compute_uv=False)
        axis_ratio = (
            float(singular_values[1] / singular_values[0])
            if len(singular_values) >= 2 and singular_values[0] > 0.0
            else 0.0
        )
    else:
        singular_values = np.zeros(3, dtype=np.float64)
        axis_ratio = 0.0
    rotation_angles = []
    for first, second in itertools.combinations(samples, 2):
        relative = invert_transform(first.T_base_link7) @ second.T_base_link7
        rotation_angles.append(rotation_angle_deg(relative[:3, :3]))
    return {
        "pair_count": len(rotation_angles),
        "informative_rotation_pair_count": len(rotations),
        "maximum_relative_rotation_deg": (
            float(max(rotation_angles)) if rotation_angles else 0.0
        ),
        "maximum_relative_translation_mm": (
            float(max(translations_mm)) if translations_mm else 0.0
        ),
        "rotation_axis_singular_values": singular_values.tolist(),
        "rotation_axis_diversity_ratio": axis_ratio,
    }


def _degeneracy_reasons(
    diagnostics: dict[str, Any],
    config: CalibrationConfig,
) -> list[str]:
    reasons = []
    if diagnostics["maximum_relative_rotation_deg"] < config.min_rotation_span_deg:
        reasons.append(
            "robot orientation span is too small "
            f"({diagnostics['maximum_relative_rotation_deg']:.3f} deg < "
            f"{config.min_rotation_span_deg:.3f} deg)"
        )
    if diagnostics["informative_rotation_pair_count"] < 2:
        reasons.append("fewer than two robot pose pairs contain a useful rotation")
    if (
        diagnostics["rotation_axis_diversity_ratio"]
        < config.min_axis_diversity_ratio
    ):
        reasons.append(
            "robot rotations are nearly all around one axis "
            f"(axis diversity {diagnostics['rotation_axis_diversity_ratio']:.4f} < "
            f"{config.min_axis_diversity_ratio:.4f})"
        )
    if diagnostics["maximum_relative_translation_mm"] < config.min_translation_span_mm:
        reasons.append(
            "robot translation span is too small "
            f"({diagnostics['maximum_relative_translation_mm']:.3f} mm < "
            f"{config.min_translation_span_mm:.3f} mm)"
        )
    return reasons


def _split_indices(count: int, config: CalibrationConfig) -> tuple[tuple[int, ...], ...]:
    if config.holdout_fraction <= 0.0:
        return tuple(range(count)), ()
    holdout_count = max(1, int(round(count * config.holdout_fraction)))
    holdout_count = min(holdout_count, count - config.min_solver_samples)
    generator = np.random.default_rng(config.random_seed)
    holdout = tuple(sorted(int(value) for value in generator.permutation(count)[:holdout_count]))
    held_out = set(holdout)
    training = tuple(index for index in range(count) if index not in held_out)
    return training, holdout


def _solve_opencv(
    samples: tuple[CalibrationSample, ...],
    indices: tuple[int, ...],
    method: str,
) -> np.ndarray:
    selected = [samples[index] for index in indices]
    rotations_gripper_to_base = [item.T_base_link7[:3, :3] for item in selected]
    translations_gripper_to_base = [
        item.T_base_link7[:3, 3].reshape(3, 1) for item in selected
    ]
    rotations_target_to_camera = [
        item.T_camera_checkerboard[:3, :3] for item in selected
    ]
    translations_target_to_camera = [
        item.T_camera_checkerboard[:3, 3].reshape(3, 1) for item in selected
    ]
    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=METHODS[method],
    )
    return make_transform(rotation, translation)


def _board_poses(
    samples: tuple[CalibrationSample, ...],
    transform_link7_camera: np.ndarray,
    indices: tuple[int, ...],
) -> list[np.ndarray]:
    transform = validate_transform(
        transform_link7_camera, "T_link7_camera_optical"
    )
    return [
        samples[index].T_base_link7
        @ transform
        @ samples[index].T_camera_checkerboard
        for index in indices
    ]


def _residuals_against_board(
    samples: tuple[CalibrationSample, ...],
    transform_link7_camera: np.ndarray,
    indices: tuple[int, ...],
    reference_board: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = _board_poses(samples, transform_link7_camera, indices)
    reference = mean_transform(poses) if reference_board is None else reference_board
    errors = [transform_error(pose, reference) for pose in poses]
    return (
        np.asarray([item[0] for item in errors], dtype=np.float64),
        np.asarray([item[1] for item in errors], dtype=np.float64),
        reference,
    )


def _candidate_score(
    samples: tuple[CalibrationSample, ...],
    transform: np.ndarray,
    scoring_indices: tuple[int, ...],
) -> tuple[float, dict[str, float | int | None]]:
    translation, rotation, _ = _residuals_against_board(
        samples, transform, scoring_indices
    )
    translation_summary = _summary(translation)
    rotation_summary = _summary(rotation)
    board_score = (
        float(translation_summary["median"])
        + 5.0 * float(rotation_summary["median"])
        + 0.25 * float(translation_summary["p95"])
        + 1.25 * float(rotation_summary["p95"])
    )
    # Absolute board-pose consistency alone can reward a physically different
    # hand-eye translation when robot rotations are small.  The independent
    # AX=XB equation is therefore part of the score, rather than a report-only
    # diagnostic.  This specifically prevents a divergent Andreff solution
    # from winning merely because its nuisance board center moved with it.
    ax_xb = _ax_xb_metrics(samples, transform, scoring_indices)
    ax_translation = ax_xb["translation_mm"]
    ax_rotation = ax_xb["rotation_deg"]
    ax_xb_score = (
        float(ax_translation["median"])
        + 5.0 * float(ax_rotation["median"])
        + 0.25 * float(ax_translation["p95"])
        + 1.25 * float(ax_rotation["p95"])
    )
    score = board_score + ax_xb_score
    return score, {
        "score": score,
        "board_score": board_score,
        "ax_xb_score": ax_xb_score,
        "translation_median_mm": translation_summary["median"],
        "translation_p95_mm": translation_summary["p95"],
        "rotation_median_deg": rotation_summary["median"],
        "rotation_p95_deg": rotation_summary["p95"],
        "ax_xb_translation_median_mm": ax_translation["median"],
        "ax_xb_translation_p95_mm": ax_translation["p95"],
        "ax_xb_rotation_median_deg": ax_rotation["median"],
        "ax_xb_rotation_p95_deg": ax_rotation["p95"],
    }


def _solver_subsets(
    training: tuple[int, ...],
    config: CalibrationConfig,
) -> tuple[tuple[int, ...], ...]:
    subsets: list[tuple[int, ...]] = [training]
    if len(training) <= 12 and len(training) - 1 >= config.min_solver_samples:
        subsets.extend(
            tuple(index for index in training if index != omitted)
            for omitted in training
        )
    subset_size = max(
        config.min_solver_samples,
        int(math.ceil(len(training) * 0.65)),
    )
    if subset_size < len(training):
        generator = np.random.default_rng(config.random_seed + 1009)
        for _ in range(config.ransac_subsets):
            chosen = tuple(
                sorted(
                    int(value)
                    for value in generator.choice(
                        training,
                        size=subset_size,
                        replace=False,
                    )
                )
            )
            subsets.append(chosen)
    return tuple(dict.fromkeys(subsets))


def _generate_candidates(
    samples: tuple[CalibrationSample, ...],
    solver_subsets: tuple[tuple[int, ...], ...],
    scoring_indices: tuple[int, ...],
    config: CalibrationConfig,
) -> tuple[list[_Candidate], list[dict[str, str]]]:
    candidates = []
    failures = []
    for indices in solver_subsets:
        subset_diagnostics = _motion_diagnostics(
            tuple(samples[index] for index in indices)
        )
        subset_degenerate = _degeneracy_reasons(subset_diagnostics, config)
        # A subset need not meet the translation preference, but it must have
        # non-parallel rotations for AX=XB to be observable.
        fatal = [
            reason
            for reason in subset_degenerate
            if not reason.startswith("robot translation span")
        ]
        if fatal:
            continue
        for method in config.methods:
            try:
                transform = _solve_opencv(samples, indices, method)
                score, metrics = _candidate_score(
                    samples, transform, scoring_indices
                )
                if not np.isfinite(score):
                    raise ValueError("candidate score is non-finite")
                candidates.append(
                    _Candidate(method, indices, transform, score, metrics)
                )
            except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
                failures.append(
                    {
                        "method": method,
                        "solver_sample_ids": ",".join(
                            samples[index].sample_id for index in indices
                        ),
                        "reason": str(error),
                    }
                )
    return candidates, failures


def _robust_limits(values: np.ndarray, minimum_limit: float) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(minimum_limit, median + 4.0 * 1.4826 * mad)


def _metric_group(
    samples: tuple[CalibrationSample, ...],
    transform: np.ndarray,
    indices: tuple[int, ...],
    reference_board: np.ndarray,
) -> dict[str, Any]:
    translation, rotation, _ = _residuals_against_board(
        samples,
        transform,
        indices,
        reference_board,
    )
    return {
        "sample_ids": [samples[index].sample_id for index in indices],
        "translation_mm": _summary(translation),
        "rotation_deg": _summary(rotation),
        "per_sample": [
            {
                "sample_id": samples[index].sample_id,
                "translation_mm": float(translation[offset]),
                "rotation_deg": float(rotation[offset]),
            }
            for offset, index in enumerate(indices)
        ],
    }


def _ax_xb_metrics(
    samples: tuple[CalibrationSample, ...],
    transform: np.ndarray,
    indices: tuple[int, ...],
) -> dict[str, Any]:
    translations = []
    rotations = []
    for first_index, second_index in itertools.combinations(indices, 2):
        for source_index, destination_index in (
            (first_index, second_index),
            (second_index, first_index),
        ):
            source = samples[source_index]
            destination = samples[destination_index]
            motion_link = (
                invert_transform(destination.T_base_link7)
                @ source.T_base_link7
            )
            motion_board = (
                destination.T_camera_checkerboard
                @ invert_transform(source.T_camera_checkerboard)
            )
            left = motion_link @ transform
            right = transform @ motion_board
            translation, rotation = transform_error(left, right)
            translations.append(translation)
            rotations.append(rotation)
    return {
        "pair_count": len(translations),
        "translation_mm": _summary(translations),
        "rotation_deg": _summary(rotations),
    }


def _transform_medoid(
    named_transforms: list[tuple[str, np.ndarray]],
    translation_scale_mm: float,
    rotation_scale_deg: float,
) -> tuple[str, np.ndarray]:
    """Return a deterministic observed transform nearest all the others."""
    if not named_transforms:
        raise ValueError("at least one named transform is required")
    best: tuple[float, str, np.ndarray] | None = None
    for name, transform in named_transforms:
        cost = 0.0
        for _, other in named_transforms:
            translation, rotation = transform_error(transform, other)
            cost += (
                translation / translation_scale_mm
                + rotation / rotation_scale_deg
            )
        candidate = (cost, name, transform)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def _largest_method_consensus(
    named_transforms: list[tuple[str, np.ndarray]],
    config: CalibrationConfig,
) -> tuple[str, ...]:
    """Find the largest all-pairs-consistent method clique."""
    by_name = dict(named_transforms)
    ordered = tuple(name for name in config.methods if name in by_name)
    best: tuple[int, float, tuple[str, ...]] | None = None
    for size in range(len(ordered), 0, -1):
        for names in itertools.combinations(ordered, size):
            cost = 0.0
            agrees = True
            for first, second in itertools.combinations(names, 2):
                translation, rotation = transform_error(
                    by_name[first], by_name[second]
                )
                if (
                    translation > config.method_consensus_translation_mm
                    or rotation > config.method_consensus_rotation_deg
                ):
                    agrees = False
                    break
                cost += (
                    translation / config.method_consensus_translation_mm
                    + rotation / config.method_consensus_rotation_deg
                )
            if agrees:
                candidate = (-size, cost, names)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            break
    return () if best is None else best[2]


def _method_consensus_initialization(
    samples: tuple[CalibrationSample, ...],
    indices: tuple[int, ...],
    config: CalibrationConfig,
) -> tuple[np.ndarray | None, list[dict[str, Any]], list[tuple[str, np.ndarray]]]:
    """Screen OpenCV initializers by AX=XB quality and geometric consensus."""
    finite: list[tuple[str, np.ndarray]] = []
    reports: list[dict[str, Any]] = []
    for method in config.methods:
        try:
            transform = _solve_opencv(samples, indices, method)
            metrics = _ax_xb_metrics(samples, transform, indices)
            translation_p95 = float(metrics["translation_mm"]["p95"])
            rotation_p95 = float(metrics["rotation_deg"]["p95"])
            ax_xb_warnings = []
            if translation_p95 > config.max_initial_ax_xb_translation_p95_mm:
                ax_xb_warnings.append(
                    "AX=XB translation P95 "
                    f"{translation_p95:.3f} mm exceeds "
                    f"{config.max_initial_ax_xb_translation_p95_mm:.3f} mm"
                )
            if rotation_p95 > config.max_initial_ax_xb_rotation_p95_deg:
                ax_xb_warnings.append(
                    "AX=XB rotation P95 "
                    f"{rotation_p95:.3f} deg exceeds "
                    f"{config.max_initial_ax_xb_rotation_p95_deg:.3f} deg"
                )
            finite.append((method, transform))
            reports.append(
                {
                    "method": method,
                    "solver_success": True,
                    "T_link7_camera_optical": transform.tolist(),
                    "ax_xb": metrics,
                    "passed_ax_xb_quality": not ax_xb_warnings,
                    "ax_xb_quality_warnings": ax_xb_warnings,
                    "accepted_for_consensus": False,
                    "rejection_reasons": [],
                }
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            reports.append(
                {
                    "method": method,
                    "solver_success": False,
                    "passed_ax_xb_quality": False,
                    "ax_xb_quality_warnings": [],
                    "accepted_for_consensus": False,
                    "rejection_reasons": [str(error)],
                }
            )
    # Geometric consensus is deliberately formed before hard AX=XB quality
    # screening.  A single bad capture contaminates every pair touching it and
    # can raise every method's P95; the robust refinement below identifies that
    # sample first.  Divergent methods are still both excluded geometrically
    # and diagnosed by their independent AX=XB metrics.
    consensus = _largest_method_consensus(finite, config)
    report_by_method = {report["method"]: report for report in reports}
    for method, _ in finite:
        report = report_by_method[method]
        if method in consensus:
            report["accepted_for_consensus"] = True
        else:
            report["rejection_reasons"].extend(
                report.get("ax_xb_quality_warnings", [])
            )
            report["rejection_reasons"].append(
                "outside the largest pairwise transform-consensus cluster"
            )
    required = min(config.min_method_consensus, len(config.methods))
    if len(consensus) < required:
        return None, reports, finite
    consensus_transforms = [item for item in finite if item[0] in consensus]
    initial_method, initial = _transform_medoid(
        consensus_transforms,
        config.method_consensus_translation_mm,
        config.method_consensus_rotation_deg,
    )
    report_by_method[initial_method]["selected_consensus_medoid"] = True
    return initial, reports, finite


def _vector_to_transform(value: np.ndarray) -> np.ndarray:
    """Map rotation-vector plus translation coordinates onto SE(3)."""
    vector = np.asarray(value, dtype=np.float64).reshape(6)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    result[:3, 3] = vector[3:]
    return validate_transform(result)


def _transform_to_vector(value: np.ndarray) -> np.ndarray:
    """Map one rigid transform to optimizer coordinates."""
    transform = validate_transform(value)
    return np.concatenate(
        (Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3])
    )


def _pairwise_motion_arrays(
    samples: tuple[CalibrationSample, ...],
    indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build A and B arrays for ``A X = X B`` over all selected pairs."""
    motions_link = []
    motions_board = []
    for first_index, second_index in itertools.combinations(indices, 2):
        # Both directed motions are required.  Keeping only the arbitrary
        # i<j direction makes a non-quadratic robust objective depend on file
        # ordering because inversion changes translation through the adjoint.
        for source_index, destination_index in (
            (first_index, second_index),
            (second_index, first_index),
        ):
            source = samples[source_index]
            destination = samples[destination_index]
            motions_link.append(
                invert_transform(destination.T_base_link7)
                @ source.T_base_link7
            )
            motions_board.append(
                destination.T_camera_checkerboard
                @ invert_transform(source.T_camera_checkerboard)
            )
    if not motions_link:
        raise ValueError("pairwise refinement needs at least two samples")
    return np.stack(motions_link), np.stack(motions_board)


def _refine_pairwise_ax_xb(
    samples: tuple[CalibrationSample, ...],
    indices: tuple[int, ...],
    initial: np.ndarray,
    config: CalibrationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robustly minimize normalized pairwise AX=XB residuals on SE(3)."""
    motions_link, motions_board = _pairwise_motion_arrays(samples, indices)
    rotations_link = motions_link[:, :3, :3]
    translations_link = motions_link[:, :3, 3]
    rotations_board = motions_board[:, :3, :3]
    translations_board = motions_board[:, :3, 3]
    translation_scale_m = config.pairwise_translation_scale_mm * 1.0e-3
    rotation_scale_rad = math.radians(config.pairwise_rotation_scale_deg)

    def residual_groups(value: np.ndarray) -> np.ndarray:
        transform = _vector_to_transform(value)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        left_rotation = rotations_link @ rotation
        left_translation = rotations_link @ translation + translations_link
        right_rotation = rotation @ rotations_board
        right_translation = (
            (rotation @ translations_board[..., None])[..., 0] + translation
        )
        error_rotation = left_rotation.transpose(0, 2, 1) @ right_rotation
        error_translation = (
            left_rotation.transpose(0, 2, 1)
            @ (right_translation - left_translation)[..., None]
        )[..., 0]
        rotation_vectors = Rotation.from_matrix(error_rotation).as_rotvec()
        return np.concatenate(
            (
                rotation_vectors / rotation_scale_rad,
                error_translation / translation_scale_m,
            ),
            axis=1,
        )

    # Group IRLS implements soft-L1 on each complete normalized SE(3)
    # residual: rho(||r||^2)=2(sqrt(1+||r||^2)-1).  SciPy's built-in
    # ``loss='soft_l1'`` acts on each scalar separately, which would make the
    # answer depend on the coordinate basis.  The group norm below is invariant
    # under a common rotation of the translation and rotation coordinates.
    value = _transform_to_vector(initial)
    weights = np.ones(len(motions_link), dtype=np.float64)
    result = None
    irls_iterations = 0
    for irls_iterations in range(1, config.refinement_irls_iterations + 1):
        previous = value.copy()

        def weighted_residual(candidate: np.ndarray) -> np.ndarray:
            return (residual_groups(candidate) * weights[:, None]).reshape(-1)

        result = least_squares(
            weighted_residual,
            value,
            loss="linear",
            max_nfev=config.refinement_max_nfev,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        value = result.x
        norms_squared = np.sum(residual_groups(value) ** 2, axis=1)
        updated_weights = np.power(1.0 + norms_squared, -0.25)
        step_translation, step_rotation = transform_error(
            _vector_to_transform(previous), _vector_to_transform(value)
        )
        weight_change = float(np.max(np.abs(updated_weights - weights)))
        weights = updated_weights
        if (
            step_translation <= 1.0e-7
            and step_rotation <= 1.0e-8
            and weight_change <= 1.0e-8
        ):
            break
    assert result is not None
    transform = _vector_to_transform(result.x)
    final_groups = residual_groups(result.x)
    final_norms_squared = np.sum(final_groups * final_groups, axis=1)
    if not result.success or not np.all(np.isfinite(final_groups)):
        raise ValueError(
            "robust pairwise refinement failed: "
            f"status={result.status}, message={result.message}"
        )
    return transform, {
        "estimator": "robust_pairwise_ax_xb_se3",
        "loss": "group_soft_l1_irls",
        "loss_f_scale": 1.0,
        "translation_scale_mm": config.pairwise_translation_scale_mm,
        "rotation_scale_deg": config.pairwise_rotation_scale_deg,
        "directed_pair_count": int(len(motions_link)),
        "unordered_pair_count": int(len(motions_link) // 2),
        "normalized_cost": float(
            np.sum(np.sqrt(1.0 + final_norms_squared) - 1.0)
        ),
        "normalized_residual_rms": float(
            np.sqrt(np.mean(final_groups * final_groups))
        ),
        "normalized_group_norm_p95": float(
            np.percentile(np.sqrt(final_norms_squared), 95.0)
        ),
        "irls_iterations": irls_iterations,
        "function_evaluations": int(result.nfev),
        "optimality": float(result.optimality),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
    }


def _refinement_initialization_trials(
    samples: tuple[CalibrationSample, ...],
    indices: tuple[int, ...],
    finite_initials: list[tuple[str, np.ndarray]],
    selected_initial: np.ndarray,
    config: CalibrationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine from every finite method and require one optimizer basin."""
    trials = []
    selected_transform, selected_metrics = _refine_pairwise_ax_xb(
        samples, indices, selected_initial, config
    )
    refined = []
    for method, initial in finite_initials:
        try:
            transform, metrics = _refine_pairwise_ax_xb(
                samples, indices, initial, config
            )
            refined.append((method, transform))
            trials.append(
                {
                    "initial_method": method,
                    "success": True,
                    "T_link7_camera_optical": transform.tolist(),
                    **metrics,
                }
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            trials.append(
                {
                    "initial_method": method,
                    "success": False,
                    "reason": str(error),
                }
            )
    if len(refined) != len(finite_initials):
        raise ValueError("one or more finite OpenCV initializations did not refine")
    comparisons = []
    for (first_method, first), (second_method, second) in itertools.combinations(
        refined, 2
    ):
        translation, rotation = transform_error(first, second)
        comparisons.append(
            {
                "first_method": first_method,
                "second_method": second_method,
                "translation_mm": translation,
                "rotation_deg": rotation,
            }
        )
    translation_max = max(
        (item["translation_mm"] for item in comparisons), default=0.0
    )
    rotation_max = max(
        (item["rotation_deg"] for item in comparisons), default=0.0
    )
    initialization_report = {
        "all_finite_method_initializations_tested": True,
        "trial_count": len(trials),
        "successful_trial_count": len(refined),
        "maximum_pairwise_translation_mm": translation_max,
        "maximum_pairwise_rotation_deg": rotation_max,
        "maximum_allowed_translation_mm": (
            config.max_refinement_initialization_translation_mm
        ),
        "maximum_allowed_rotation_deg": (
            config.max_refinement_initialization_rotation_deg
        ),
        "trials": trials,
    }
    if (
        translation_max > config.max_refinement_initialization_translation_mm
        or rotation_max > config.max_refinement_initialization_rotation_deg
    ):
        raise ValueError(
            "robust refinement depends on OpenCV initialization "
            f"({translation_max:.6f} mm, {rotation_max:.6f} deg)"
        )
    initialization_report["selected_run"] = selected_metrics
    return selected_transform, initialization_report


def _estimate_pairwise_transform(
    samples: tuple[CalibrationSample, ...],
    indices: tuple[int, ...],
    config: CalibrationConfig,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Form a method consensus and robustly refine its hand-eye transform."""
    initial, method_reports, finite_initials = _method_consensus_initialization(
        samples, indices, config
    )
    consensus = [
        report["method"]
        for report in method_reports
        if report.get("accepted_for_consensus")
    ]
    required = min(config.min_method_consensus, len(config.methods))
    if initial is None:
        raise ValueError(
            "OpenCV method consensus is insufficient "
            f"({len(consensus)} methods, need {required})"
        )
    transform, initialization_report = _refinement_initialization_trials(
        samples,
        indices,
        finite_initials,
        initial,
        config,
    )
    initialization_report["consensus_methods"] = consensus
    initialization_report["required_consensus_method_count"] = required
    return transform, method_reports, initialization_report


def _failure_report(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    status: str,
    reason: str,
    diagnostics: dict[str, Any],
    training: tuple[int, ...] = (),
    holdout: tuple[int, ...] = (),
) -> dict[str, Any]:
    return {
        "report_version": REPORT_VERSION,
        "success": False,
        "status": status,
        "reason": reason,
        "session_id": dataset.session_id,
        "transform_convention": dict(TRANSFORM_CONVENTION),
        "config": asdict(config),
        "sample_count": len(dataset.samples),
        "training_sample_ids": [dataset.samples[index].sample_id for index in training],
        "holdout_sample_ids": [dataset.samples[index].sample_id for index in holdout],
        "motion_diagnostics": diagnostics,
        "T_link7_camera_optical": None,
        "safe_for_robot_use": False,
    }


def calibrate_eye_in_hand(
    dataset: CalibrationDataset,
    config: CalibrationConfig | None = None,
) -> dict[str, Any]:
    """Estimate ``T_link7_camera_optical`` from a stationary checkerboard session.

    OpenCV receives ``T_base_link7`` as gripper-to-base and
    ``T_camera_checkerboard`` as target-to-camera.  Its camera-to-gripper output
    is therefore exactly the transform named by this function's result.
    Candidate methods and deterministic subsets are compared using the fact
    that ``T_base_checkerboard`` must remain constant for every sample.
    """
    settings = CalibrationConfig() if config is None else config
    samples = dataset.samples
    diagnostics = _motion_diagnostics(samples)
    if len(samples) < settings.min_samples:
        return _failure_report(
            dataset,
            settings,
            "failed_insufficient_samples",
            f"need at least {settings.min_samples} samples; received {len(samples)}",
            diagnostics,
        )
    reasons = _degeneracy_reasons(diagnostics, settings)
    if reasons:
        return _failure_report(
            dataset,
            settings,
            "failed_degenerate_motion",
            "; ".join(reasons),
            diagnostics,
        )

    training, holdout = _split_indices(len(samples), settings)
    training_diagnostics = _motion_diagnostics(
        tuple(samples[index] for index in training)
    )
    training_reasons = _degeneracy_reasons(training_diagnostics, settings)
    if training_reasons:
        return _failure_report(
            dataset,
            settings,
            "failed_degenerate_training_split",
            "; ".join(training_reasons),
            diagnostics,
            training,
            holdout,
        )

    initial, method_reports, finite_initials = _method_consensus_initialization(
        samples, training, settings
    )
    if initial is None:
        report = _failure_report(
            dataset,
            settings,
            "failed_method_consensus",
            "too few OpenCV methods passed AX=XB screening and agreed on a transform",
            diagnostics,
            training,
            holdout,
        )
        report["method_candidates"] = method_reports
        return report
    try:
        selected_transform, initialization_report = (
            _refinement_initialization_trials(
                samples,
                training,
                finite_initials,
                initial,
                settings,
            )
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        report = _failure_report(
            dataset,
            settings,
            "failed_pairwise_refinement",
            str(error),
            diagnostics,
            training,
            holdout,
        )
        report["method_candidates"] = method_reports
        return report

    translation, rotation, _ = _residuals_against_board(
        samples, selected_transform, training
    )
    translation_limit = _robust_limits(
        translation, settings.outlier_translation_mm
    )
    rotation_limit = _robust_limits(rotation, settings.outlier_rotation_deg)
    inliers = tuple(
        index
        for offset, index in enumerate(training)
        if translation[offset] <= translation_limit
        and rotation[offset] <= rotation_limit
    )
    outliers = tuple(index for index in training if index not in set(inliers))

    if len(inliers) >= settings.min_solver_samples and outliers:
        refined_initial, refined_method_reports, refined_finite = (
            _method_consensus_initialization(samples, inliers, settings)
        )
        if refined_initial is None:
            report = _failure_report(
                dataset,
                settings,
                "failed_method_consensus",
                "method consensus failed after excluding training outliers",
                diagnostics,
                training,
                holdout,
            )
            report["method_candidates"] = refined_method_reports
            return report
        try:
            selected_transform, initialization_report = (
                _refinement_initialization_trials(
                    samples,
                    inliers,
                    refined_finite,
                    refined_initial,
                    settings,
                )
            )
            method_reports = refined_method_reports
        except (ValueError, np.linalg.LinAlgError) as error:
            report = _failure_report(
                dataset,
                settings,
                "failed_pairwise_refinement",
                str(error),
                diagnostics,
                training,
                holdout,
            )
            report["method_candidates"] = refined_method_reports
            return report
    else:
        inliers = training
        outliers = ()

    initialization_report["consensus_methods"] = [
        report["method"]
        for report in method_reports
        if report.get("accepted_for_consensus")
    ]
    initialization_report["required_consensus_method_count"] = min(
        settings.min_method_consensus, len(settings.methods)
    )

    _, _, reference_board = _residuals_against_board(
        samples, selected_transform, inliers
    )
    training_metrics = _metric_group(
        samples, selected_transform, training, reference_board
    )
    inlier_metrics = _metric_group(
        samples, selected_transform, inliers, reference_board
    )
    holdout_metrics = _metric_group(
        samples, selected_transform, holdout, reference_board
    )

    holdout_translation_p95 = holdout_metrics["translation_mm"]["p95"]
    holdout_rotation_p95 = holdout_metrics["rotation_deg"]["p95"]
    validation_failures = []
    if holdout:
        if holdout_translation_p95 > settings.max_holdout_translation_p95_mm:
            validation_failures.append(
                f"holdout translation P95 {holdout_translation_p95:.3f} mm exceeds "
                f"{settings.max_holdout_translation_p95_mm:.3f} mm"
            )
        if holdout_rotation_p95 > settings.max_holdout_rotation_p95_deg:
            validation_failures.append(
                f"holdout rotation P95 {holdout_rotation_p95:.3f} deg exceeds "
                f"{settings.max_holdout_rotation_p95_deg:.3f} deg"
            )

    success = not validation_failures
    status = "passed" if success else "failed_holdout_validation"
    reason = (
        "calibration and independent holdout validation passed"
        if success and holdout
        else "calibration passed without a holdout set"
        if success
        else "; ".join(validation_failures)
    )
    return {
        "report_version": REPORT_VERSION,
        "success": success,
        "status": status,
        "reason": reason,
        "session_id": dataset.session_id,
        "transform_convention": dict(TRANSFORM_CONVENTION),
        "frames": {
            "base": dataset.base_frame,
            "link": dataset.link_frame,
            "camera_optical": dataset.camera_frame,
            "checkerboard": dataset.checkerboard_frame,
        },
        "config": asdict(settings),
        "sample_count": len(samples),
        "training_sample_ids": [samples[index].sample_id for index in training],
        "holdout_sample_ids": [samples[index].sample_id for index in holdout],
        "inlier_sample_ids": [samples[index].sample_id for index in inliers],
        "outlier_sample_ids": [samples[index].sample_id for index in outliers],
        "selected_method": "ROBUST_PAIRWISE_AX_XB",
        "selected_initial_method": next(
            (
                report["method"]
                for report in method_reports
                if report.get("selected_consensus_medoid")
            ),
            None,
        ),
        "selected_solver_sample_ids": [
            samples[index].sample_id for index in inliers
        ],
        "T_link7_camera_optical": selected_transform.tolist(),
        # A single split is diagnostic only.  The stability module is the sole
        # place that can mark a transform safe after all 20 splits and scale
        # sensitivity trials pass.
        "safe_for_robot_use": False,
        "requires_cross_split_stability": True,
        "estimated_T_base_checkerboard": reference_board.tolist(),
        "motion_diagnostics": diagnostics,
        "training_motion_diagnostics": training_diagnostics,
        "outlier_limits": {
            "translation_mm": translation_limit,
            "rotation_deg": rotation_limit,
        },
        "metrics": {
            "training": training_metrics,
            "training_inliers": inlier_metrics,
            "holdout": holdout_metrics,
            "ax_xb_training_inliers": _ax_xb_metrics(
                samples, selected_transform, inliers
            ),
        },
        "method_candidates": method_reports,
        "pairwise_refinement": initialization_report,
        "solver_failures": [
            {
                "method": report["method"],
                "reason": "; ".join(report.get("rejection_reasons", [])),
            }
            for report in method_reports
            if not report.get("accepted_for_consensus")
        ],
    }
