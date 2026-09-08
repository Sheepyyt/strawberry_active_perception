"""Strict disk-only reprocessing of captured checkerboard poses.

This module exists for one narrow recovery workflow: historical capture NPZs
contain valid image corners and robot poses, but their saved distortion vector
was incorrectly zero.  It re-runs planar PnP with an explicit, audited camera
model while leaving every source NPZ byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .npz_session import (
    CaptureSessionImportError,
    _load_arrays,
    capture_paths,
    load_capture_session,
)
from .schema import CalibrationDataset, CalibrationSample
from .transforms import make_transform, transform_error


CAMERA_MODEL_SCHEMA_VERSION = "strawberry_camera_model/v1"
REPROCESS_AUDIT_VERSION = "strawberry_camera_model_reprocessing/v1"
DEFAULT_K_ATOL = 1.0e-9
_CAMERA_MODEL_KEYS = {
    "schema_version",
    "source_name",
    "image_size",
    "K",
    "D",
    "distortion_model",
    "provenance",
}
_DISTORTION_LENGTHS = {
    "plumb_bob": 5,
    "rational_polynomial": 8,
}


class CameraModelReprocessError(ValueError):
    """The camera model or capture evidence failed a closed validation."""


def _error(reason: str) -> CameraModelReprocessError:
    return CameraModelReprocessError(reason)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error(f"camera model contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _error(f"camera model contains non-standard numeric constant {value!r}")


def _validated_matrix(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _error(f"camera model {name} must contain only numbers") from error
    if result.shape != shape:
        raise _error(f"camera model {name} has shape {result.shape}, expected {shape}")
    if not np.all(np.isfinite(result)):
        raise _error(f"camera model {name} contains NaN or Inf")
    return result


@dataclass(frozen=True)
class CameraModel:
    """One explicit OpenCV camera model with immutable file provenance."""

    source_name: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    distortion_model: str
    provenance: dict[str, Any]
    document: dict[str, Any]
    source_path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise _error("camera model source_name must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise _error("camera model image width and height must be positive")
        K = _validated_matrix(self.K, "K", (3, 3))
        D = np.asarray(self.D, dtype=np.float64)
        expected_length = _DISTORTION_LENGTHS.get(self.distortion_model)
        if expected_length is None:
            supported = ", ".join(sorted(_DISTORTION_LENGTHS))
            raise _error(
                f"unsupported distortion_model {self.distortion_model!r}; "
                f"expected one of {supported}"
            )
        if D.shape != (expected_length,):
            raise _error(
                f"camera model D has shape {D.shape}; {self.distortion_model} "
                f"requires exactly {expected_length} coefficients"
            )
        if not np.all(np.isfinite(D)):
            raise _error("camera model D contains NaN or Inf")
        if not np.any(D != 0.0):
            raise _error("camera model D must contain at least one non-zero coefficient")
        if (
            K[0, 0] <= 0.0
            or K[1, 1] <= 0.0
            or not np.allclose(K[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-12)
        ):
            raise _error("camera model K is not a valid pinhole matrix")
        if not isinstance(self.provenance, dict) or not self.provenance:
            raise _error("camera model provenance must be a non-empty JSON object")
        if len(self.sha256) != 64:
            raise _error("camera model file SHA-256 is invalid")
        object.__setattr__(self, "K", K.copy())
        object.__setattr__(self, "D", D.copy())


def load_camera_model(path: str | os.PathLike[str]) -> CameraModel:
    """Read and strictly validate one versioned camera-model JSON file."""
    source = Path(path).expanduser().resolve()
    try:
        raw_bytes = source.read_bytes()
    except OSError as error:
        raise _error(f"cannot read camera model {source}: {error}") from error
    try:
        value = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except CameraModelReprocessError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _error(f"camera model is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise _error("camera model root must be a JSON object")
    missing = sorted(_CAMERA_MODEL_KEYS - set(value))
    unknown = sorted(set(value) - _CAMERA_MODEL_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise _error("camera model fields are invalid: " + "; ".join(details))
    if value["schema_version"] != CAMERA_MODEL_SCHEMA_VERSION:
        raise _error(
            f"unsupported camera model schema_version {value['schema_version']!r}; "
            f"expected {CAMERA_MODEL_SCHEMA_VERSION!r}"
        )
    if not isinstance(value["source_name"], str):
        raise _error("camera model source_name must be a string")
    if not isinstance(value["distortion_model"], str):
        raise _error("camera model distortion_model must be a string")
    image_size = value["image_size"]
    if not isinstance(image_size, dict) or set(image_size) != {"width", "height"}:
        raise _error("camera model image_size must contain exactly width and height")
    width, height = image_size["width"], image_size["height"]
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise _error("camera model image width and height must be integers")
    try:
        distortion = np.asarray(value["D"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _error("camera model D must contain only numbers") from error
    return CameraModel(
        source_name=str(value["source_name"]),
        width=width,
        height=height,
        K=_validated_matrix(value["K"], "K", (3, 3)),
        D=distortion,
        distortion_model=value["distortion_model"],
        provenance=value["provenance"],
        document=value,
        source_path=str(source),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def ndarray_sha256(value: np.ndarray) -> str:
    """Hash an ndarray including its dtype, shape, and C-order bytes."""
    array = np.asarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"strawberry_ndarray/v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise _error(f"cannot hash capture {path}: {error}") from error


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise _error("internal summary input is empty or non-finite")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _recompute_pose(
    objects: np.ndarray,
    corners: np.ndarray,
    model: CameraModel,
) -> tuple[np.ndarray, float]:
    success, rotation_vector, translation = cv2.solvePnP(
        np.ascontiguousarray(objects, dtype=np.float64),
        np.ascontiguousarray(corners, dtype=np.float64),
        model.K,
        model.D,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success:
        raise _error("cv2.solvePnP(SOLVEPNP_IPPE) returned failure")
    if not (
        np.all(np.isfinite(rotation_vector)) and np.all(np.isfinite(translation))
    ):
        raise _error("cv2.solvePnP(SOLVEPNP_IPPE) returned NaN or Inf")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = make_transform(rotation, translation)
    camera_points = (
        transform[:3, :3] @ objects.astype(np.float64).T
        + transform[:3, 3].reshape(3, 1)
    )
    if np.any(camera_points[2] <= 0.0):
        raise _error("IPPE solution places one or more checkerboard points behind camera")
    projected, _ = cv2.projectPoints(
        objects.astype(np.float64),
        rotation_vector,
        translation,
        model.K,
        model.D,
    )
    residual = projected.reshape(-1, 2) - corners.astype(np.float64)
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if not np.isfinite(rms):
        raise _error("recomputed reprojection RMS is not finite")
    return transform, rms


def reprocess_capture_session(
    input_directory: str | os.PathLike[str],
    camera_model_path: str | os.PathLike[str],
    *,
    sample_ids: Sequence[str] | None = None,
    allow_k_change: bool = False,
    k_atol: float = DEFAULT_K_ATOL,
) -> CalibrationDataset:
    """Re-run IPPE PnP for an immutable selection of capture NPZ files.

    By default the external model must preserve the exact capture focal lengths
    and principal point within ``k_atol``.  ``allow_k_change`` is intentionally
    explicit and marks the resulting dataset as diagnostic-only.
    """
    if not np.isfinite(k_atol) or k_atol < 0.0:
        raise _error("k_atol must be finite and non-negative")
    model = load_camera_model(camera_model_path)
    directory = Path(input_directory).expanduser().resolve()
    try:
        selected_paths = capture_paths(directory, sample_ids)
        original = load_capture_session(directory, sample_ids=sample_ids)
    except CaptureSessionImportError as error:
        raise _error(f"capture validation failed: {error}") from error
    if len(selected_paths) != len(original.samples):
        raise _error("capture path and validated sample counts differ")

    original_source = str(original.metadata["source_name"])
    if model.source_name != original_source:
        raise _error(
            "camera model source_name/serial differs from captures: "
            f"{model.source_name!r} != {original_source!r}"
        )
    original_K = np.asarray(original.metadata["K"], dtype=np.float64)
    original_D = np.asarray(original.metadata["D"], dtype=np.float64)
    if not np.array_equal(original_D, np.zeros_like(original_D)):
        raise _error(
            "captures do not have the required original D=0 state; refusing "
            "to replace a non-zero distortion model"
        )
    maximum_k_difference = float(np.max(np.abs(model.K - original_K)))
    k_changed = not np.allclose(
        model.K,
        original_K,
        rtol=0.0,
        atol=k_atol,
    )
    if k_changed and not allow_k_change:
        raise _error(
            "camera model K differs from capture K "
            f"(maximum absolute difference {maximum_k_difference:.12g}, "
            f"allowed {k_atol:.12g}); pass allow_k_change=True or the CLI "
            "--allow-k-change only for an explicitly diagnostic re-fit"
        )

    capture_audit_by_name = {
        str(item["source_file"]): item for item in original.metadata["capture_audit"]
    }
    new_samples: list[CalibrationSample] = []
    sample_audits: list[dict[str, Any]] = []
    translation_changes: list[float] = []
    rotation_changes: list[float] = []
    old_rms_values: list[float] = []
    new_rms_values: list[float] = []

    for path, old_sample in zip(selected_paths, original.samples, strict=True):
        imported_audit = capture_audit_by_name.get(path.name)
        if imported_audit is None:
            raise _error(f"{path.name}: missing original capture hash audit")
        expected_file_hash = str(imported_audit["sha256"])
        before_hash = _file_sha256(path)
        if before_hash != expected_file_hash:
            raise _error(f"{path.name}: file SHA-256 changed after capture validation")
        try:
            values = _load_arrays(path)
        except CaptureSessionImportError as error:
            raise _error(f"capture validation failed: {error}") from error
        after_hash = _file_sha256(path)
        if after_hash != before_hash:
            raise _error(f"{path.name}: file changed while it was being reprocessed")

        source_name = str(values["source_name"].item())
        rgb = values["rgb"]
        K = values["K"].astype(np.float64)
        D = values["D"].astype(np.float64)
        corners = values["checkerboard_corners_px"].astype(np.float64)
        objects = values["checkerboard_object_points_m"].astype(np.float64)
        if source_name != model.source_name:
            raise _error(f"{path.name}: source_name/serial differs from camera model")
        if rgb.shape[:2] != (model.height, model.width):
            raise _error(
                f"{path.name}: image size {rgb.shape[1]}x{rgb.shape[0]} differs "
                f"from camera model {model.width}x{model.height}"
            )
        if not np.allclose(K, original_K, rtol=0.0, atol=1.0e-12):
            raise _error(f"{path.name}: K differs from the common capture K")
        if not np.array_equal(D, np.zeros_like(D)):
            raise _error(f"{path.name}: original D is not exactly zero")
        expected_count = original.checkerboard_columns * original.checkerboard_rows
        if corners.shape != (expected_count, 2) or objects.shape != (expected_count, 3):
            raise _error(f"{path.name}: checkerboard corner/object evidence shape changed")

        new_pose, new_rms = _recompute_pose(objects, corners, model)
        translation_mm, rotation_deg = transform_error(
            new_pose,
            old_sample.T_camera_checkerboard,
        )
        new_samples.append(
            CalibrationSample(
                sample_id=old_sample.sample_id,
                T_base_link7=old_sample.T_base_link7,
                T_camera_checkerboard=new_pose,
                timestamp_sec=old_sample.timestamp_sec,
                corner_count=old_sample.corner_count,
                reprojection_rms_px=new_rms,
            )
        )
        old_rms = float(old_sample.reprojection_rms_px)
        translation_changes.append(translation_mm)
        rotation_changes.append(rotation_deg)
        old_rms_values.append(old_rms)
        new_rms_values.append(new_rms)
        sample_audits.append(
            {
                "sample_id": old_sample.sample_id,
                "source_file": path.name,
                "source_npz_sha256": before_hash,
                "checkerboard_corners_sha256": ndarray_sha256(
                    values["checkerboard_corners_px"]
                ),
                "checkerboard_object_points_sha256": ndarray_sha256(
                    values["checkerboard_object_points_m"]
                ),
                "original_T_camera_checkerboard_sha256": ndarray_sha256(
                    values["T_camera_checkerboard"]
                ),
                "T_base_link7_sha256": ndarray_sha256(values["T_base_link7"]),
                "original_reprojection_rms_px": old_rms,
                "reprocessed_reprojection_rms_px": new_rms,
                "pose_change_translation_mm": translation_mm,
                "pose_change_rotation_deg": rotation_deg,
            }
        )

    metadata = dict(original.metadata)
    metadata["camera_model_reprocessing"] = {
        "audit_version": REPROCESS_AUDIT_VERSION,
        "operation": "recompute_T_camera_checkerboard_with_cv2_solvePnP_IPPE",
        "robot_contacted": False,
        "source_npz_modified": False,
        "camera_model_path": model.source_path,
        "camera_model_sha256": model.sha256,
        "camera_model": model.document,
        "input_directory": str(directory),
        "sample_ids": [sample.sample_id for sample in new_samples],
        "sample_count": len(new_samples),
        "original_camera_evidence": {
            "source_name": original_source,
            "image_size": {"width": model.width, "height": model.height},
            "K": original_K.tolist(),
            "D": original_D.tolist(),
            "D_was_exactly_zero": True,
        },
        "K_policy": {
            "allow_k_change": bool(allow_k_change),
            "K_changed_beyond_tolerance": k_changed,
            "absolute_tolerance": float(k_atol),
            "maximum_absolute_difference": maximum_k_difference,
            "diagnostic_only_due_to_K_change": k_changed,
        },
        "safety": {
            "safe_for_robot_use": False,
            "reason": (
                "diagnostic camera-intrinsic re-fit; must not be promoted to a "
                "robot transform"
                if k_changed
                else "dataset requires hand-eye solve and full cross-split stability "
                "validation before any robot use"
            ),
            "requires_handeye_solve": True,
            "requires_cross_split_stability_validation": True,
        },
        "opencv": {
            "version": cv2.__version__,
            "pnp_flag": "SOLVEPNP_IPPE",
        },
        "pose_change_summary": {
            "translation_mm": _summary(translation_changes),
            "rotation_deg": _summary(rotation_changes),
            "original_reprojection_rms_px": _summary(old_rms_values),
            "reprocessed_reprojection_rms_px": _summary(new_rms_values),
        },
        "sample_audit": sample_audits,
    }
    return CalibrationDataset(
        session_id=original.session_id,
        samples=tuple(new_samples),
        checkerboard_columns=original.checkerboard_columns,
        checkerboard_rows=original.checkerboard_rows,
        square_size_m=original.square_size_m,
        base_frame=original.base_frame,
        link_frame=original.link_frame,
        camera_frame=original.camera_frame,
        checkerboard_frame=original.checkerboard_frame,
        metadata=metadata,
    )
