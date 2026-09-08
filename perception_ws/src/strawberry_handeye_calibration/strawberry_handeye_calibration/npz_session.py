"""Fail-closed import of read-only field captures into the offline core schema."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .schema import CalibrationDataset, CalibrationSample


CAPTURE_SCHEMA_VERSION = "strawberry_handeye_capture/v1"
REAL_SOURCE_TYPE = 1


class CaptureSessionImportError(ValueError):
    """A saved capture is incomplete, inconsistent, or unsafe to import."""


@dataclass(frozen=True)
class _ImportedCapture:
    sample: CalibrationSample
    scene_id: str
    observation_id: str
    observation_stamp_ns: int
    base_frame: str
    link_frame: str
    camera_frame: str
    source_name: str
    K: np.ndarray
    D: np.ndarray
    checkerboard_columns: int
    checkerboard_rows: int
    square_size_m: float
    audit: dict[str, Any]


def _fail(path: Path, reason: str) -> CaptureSessionImportError:
    return CaptureSessionImportError(f"{path.name}: {reason}")


def _required(values: dict[str, np.ndarray], path: Path, key: str) -> np.ndarray:
    if key not in values:
        raise _fail(path, f"missing required field {key!r}")
    return values[key]


def _scalar(
    values: dict[str, np.ndarray],
    path: Path,
    key: str,
    *,
    kinds: str,
) -> Any:
    value = _required(values, path, key)
    if value.shape != () or value.dtype.kind not in kinds:
        raise _fail(path, f"{key} must be a scalar with dtype kind in {kinds!r}")
    return value.item()


def _text(values: dict[str, np.ndarray], path: Path, key: str) -> str:
    result = str(_scalar(values, path, key, kinds="US"))
    if not result.strip():
        raise _fail(path, f"{key} must be non-empty")
    return result


def _finite_scalar(
    values: dict[str, np.ndarray], path: Path, key: str
) -> float:
    result = float(_scalar(values, path, key, kinds="fiu"))
    if not np.isfinite(result):
        raise _fail(path, f"{key} must be finite")
    return result


def _array(
    values: dict[str, np.ndarray],
    path: Path,
    key: str,
    *,
    shape: tuple[int, ...] | None = None,
    kinds: str | None = None,
    finite: bool = False,
) -> np.ndarray:
    result = _required(values, path, key)
    if shape is not None and result.shape != shape:
        raise _fail(path, f"{key} has shape {result.shape}, expected {shape}")
    if kinds is not None and result.dtype.kind not in kinds:
        raise _fail(path, f"{key} has incompatible dtype {result.dtype}")
    if finite and not np.all(np.isfinite(result)):
        raise _fail(path, f"{key} contains NaN or Inf")
    return result


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {}
            for key in archive.files:
                value = np.array(archive[key], copy=True)
                if value.dtype.kind == "O":
                    raise _fail(path, f"field {key!r} uses forbidden object dtype")
                values[key] = value
            return values
    except CaptureSessionImportError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise _fail(path, f"cannot read safe non-pickle NPZ: {error}") from error


def _validate_image_evidence(values: dict[str, np.ndarray], path: Path) -> None:
    rgb = _array(values, path, "rgb", kinds="u")
    depth = _array(values, path, "depth_m", kinds="f")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise _fail(path, "rgb must be an HxWx3 uint8 array")
    if depth.dtype != np.float32 or depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise _fail(path, "depth_m must be an HxW float32 array matching rgb")
    valid_fraction = _finite_scalar(values, path, "valid_depth_fraction")
    measured = float(np.isfinite(depth).mean())
    if not 0.0 <= valid_fraction <= 1.0 or not np.isclose(
        valid_fraction, measured, rtol=0.0, atol=2.0e-6
    ):
        raise _fail(path, "valid_depth_fraction does not match depth_m")


def _validate_pnp_evidence(
    values: dict[str, np.ndarray],
    path: Path,
    *,
    columns: int,
    rows: int,
    square_size_m: float,
    K: np.ndarray,
    D: np.ndarray,
    T_camera_checkerboard: np.ndarray,
) -> tuple[int, float]:
    count = int(_scalar(values, path, "corner_count", kinds="iu"))
    expected_count = columns * rows
    if count != expected_count:
        raise _fail(path, f"corner_count is {count}, expected {expected_count}")
    corners = _array(
        values,
        path,
        "checkerboard_corners_px",
        shape=(count, 2),
        kinds="f",
        finite=True,
    ).astype(np.float64)
    objects = _array(
        values,
        path,
        "checkerboard_object_points_m",
        shape=(count, 3),
        kinds="f",
        finite=True,
    ).astype(np.float64)
    expected_objects = np.zeros((count, 3), dtype=np.float32)
    expected_objects[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    expected_objects[:, :2] *= np.float32(square_size_m)
    if not np.allclose(objects, expected_objects.astype(np.float64), atol=1.0e-9):
        raise _fail(path, "checkerboard object points differ from board dimensions")
    _array(
        values,
        path,
        "checkerboard_outer_corners_px",
        shape=(4, 2),
        kinds="f",
        finite=True,
    )
    margin = _finite_scalar(values, path, "checkerboard_minimum_border_margin_px")
    if margin < 0.0:
        raise _fail(path, "checkerboard border margin cannot be negative")
    stored_rms = _finite_scalar(values, path, "reprojection_rms_px")
    duplicate_rms = _finite_scalar(
        values, path, "checkerboard_reprojection_rms_px"
    )
    if stored_rms < 0.0 or not np.isclose(
        stored_rms, duplicate_rms, rtol=0.0, atol=1.0e-12
    ):
        raise _fail(path, "checkerboard reprojection RMS fields disagree")
    rotation_vector, _ = cv2.Rodrigues(T_camera_checkerboard[:3, :3])
    projected, _ = cv2.projectPoints(
        objects,
        rotation_vector,
        T_camera_checkerboard[:3, 3],
        K,
        D,
    )
    residual = projected.reshape(-1, 2) - corners
    measured_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    # Rodrigues matrix/vector round-tripping adds about 1e-6 px of harmless
    # floating-point variation with the system OpenCV build.
    if not np.isclose(measured_rms, stored_rms, rtol=0.0, atol=1.0e-5):
        raise _fail(
            path,
            "saved corners, intrinsics, pose, and reprojection RMS are inconsistent",
        )
    return count, stored_rms


def _read_capture(path: Path) -> _ImportedCapture:
    values = _load_arrays(path)
    if _text(values, path, "schema_version") != CAPTURE_SCHEMA_VERSION:
        raise _fail(path, "unsupported capture schema_version")
    sample_id = _text(values, path, "sample_id")
    if sample_id != path.stem:
        raise _fail(path, "sample_id does not match the NPZ filename")
    scene_id = _text(values, path, "scene_id")
    observation_id = _text(values, path, "observation_id")
    source_name = _text(values, path, "source_name")
    if int(_scalar(values, path, "source_type", kinds="iu")) != REAL_SOURCE_TYPE:
        raise _fail(path, "only SOURCE_REAL captures can enter a real session")
    if int(_scalar(values, path, "motion_commands_sent", kinds="iu")) != 0:
        raise _fail(path, "capture audit reports that it sent a motion command")

    stamp = _array(values, path, "stamp", shape=(2,), kinds="iu")
    seconds, nanoseconds = int(stamp[0]), int(stamp[1])
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise _fail(path, "stamp is not a normalized non-negative ROS time")
    stamp_ns = seconds * 1_000_000_000 + nanoseconds
    timestamp_sec = _finite_scalar(values, path, "timestamp_sec")
    if not np.isclose(
        timestamp_sec,
        stamp_ns * 1.0e-9,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise _fail(path, "timestamp_sec does not match stamp")

    base_frame = _text(values, path, "base_frame")
    link_frame = _text(values, path, "link_frame")
    camera_frame = _text(values, path, "frame_id")
    K = _array(values, path, "K", shape=(3, 3), kinds="f", finite=True).astype(
        np.float64
    )
    D = _array(values, path, "D", kinds="f", finite=True).astype(np.float64)
    if D.ndim != 1:
        raise _fail(path, "D must be one-dimensional")
    if (
        K[0, 0] <= 0.0
        or K[1, 1] <= 0.0
        or not np.allclose(K[2], (0.0, 0.0, 1.0), atol=1.0e-12)
    ):
        raise _fail(path, "K is not a valid pinhole camera matrix")

    inner = _array(values, path, "board_inner_corners", shape=(2,), kinds="iu")
    squares = _array(values, path, "board_squares", shape=(2,), kinds="iu")
    columns, rows = int(inner[0]), int(inner[1])
    if columns < 2 or rows < 2 or not np.array_equal(squares, inner + 1):
        raise _fail(path, "checkerboard square and inner-corner dimensions disagree")
    square_size_m = _finite_scalar(values, path, "square_size_m")
    if square_size_m <= 0.0:
        raise _fail(path, "square_size_m must be positive")

    _validate_image_evidence(values, path)
    try:
        sample = CalibrationSample(
            sample_id=sample_id,
            T_base_link7=_required(values, path, "T_base_link7"),
            T_camera_checkerboard=_required(
                values, path, "T_camera_checkerboard"
            ),
            timestamp_sec=timestamp_sec,
            corner_count=columns * rows,
            reprojection_rms_px=_finite_scalar(
                values, path, "reprojection_rms_px"
            ),
        )
    except ValueError as error:
        raise _fail(path, str(error)) from error
    count, reprojection_rms = _validate_pnp_evidence(
        values,
        path,
        columns=columns,
        rows=rows,
        square_size_m=square_size_m,
        K=K,
        D=D,
        T_camera_checkerboard=sample.T_camera_checkerboard,
    )

    joint_names = _array(values, path, "joint_names", shape=(7,), kinds="US")
    joint_positions = _array(
        values, path, "joint_positions", shape=(7,), kinds="f", finite=True
    )
    tf_lookup_mode = _text(values, path, "tf_lookup_mode")
    if tf_lookup_mode not in ("observation_stamp", "latest_while_stationary"):
        raise _fail(path, "tf_lookup_mode is not recognized")
    joint_selection_mode = _text(values, path, "joint_selection_mode")
    if joint_selection_mode not in (
        "nearest_observation_ros_stamp",
        "nearest_tf_stamp_while_stationary",
    ):
        raise _fail(path, "joint_selection_mode is not recognized")
    static_span = _finite_scalar(
        values, path, "joint_maximum_position_span_rad"
    )
    if static_span < 0.0:
        raise _fail(path, "joint static span cannot be negative")

    return _ImportedCapture(
        sample=sample,
        scene_id=scene_id,
        observation_id=observation_id,
        observation_stamp_ns=stamp_ns,
        base_frame=base_frame,
        link_frame=link_frame,
        camera_frame=camera_frame,
        source_name=source_name,
        K=K,
        D=D,
        checkerboard_columns=columns,
        checkerboard_rows=rows,
        square_size_m=square_size_m,
        audit={
            "source_file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "observation_id": observation_id,
            "observation_stamp_ns": stamp_ns,
            "corner_count": count,
            "reprojection_rms_px": reprojection_rms,
            "joint_names": [str(value) for value in joint_names],
            "joint_positions_rad": joint_positions.astype(float).tolist(),
            "joint_maximum_position_span_rad": static_span,
            "tf_lookup_mode": tf_lookup_mode,
            "joint_selection_mode": joint_selection_mode,
            "motion_commands_sent": 0,
        },
    )


def _same_array(first: np.ndarray, second: np.ndarray) -> bool:
    return first.shape == second.shape and np.allclose(
        first, second, rtol=0.0, atol=1.0e-12
    )


def _select_paths(
    directory: Path, sample_ids: Sequence[str] | None
) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise CaptureSessionImportError(f"input directory does not exist: {directory}")
    if sample_ids is None:
        paths = tuple(sorted(directory.glob("*.npz")))
    else:
        requested = tuple(str(value) for value in sample_ids)
        if len(set(requested)) != len(requested):
            raise CaptureSessionImportError("requested sample IDs must be unique")
        if any(
            not value.strip() or Path(value).name != value or value in (".", "..")
            for value in requested
        ):
            raise CaptureSessionImportError("sample IDs must be plain non-empty filenames")
        paths = tuple(directory / f"{value}.npz" for value in requested)
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise CaptureSessionImportError(
                "requested capture file(s) do not exist: " + ", ".join(missing)
            )
    if not paths:
        raise CaptureSessionImportError(f"no NPZ captures found in {directory}")
    return paths


def load_capture_session(
    input_directory: str | Path,
    *,
    sample_ids: Sequence[str] | None = None,
) -> CalibrationDataset:
    """Import one immutable selection of capture NPZs into the core dataset."""
    directory = Path(input_directory).expanduser().resolve()
    captures = tuple(_read_capture(path) for path in _select_paths(directory, sample_ids))
    first = captures[0]
    mismatches: list[str] = []
    for capture in captures[1:]:
        fields = (
            ("scene_id", capture.scene_id, first.scene_id),
            ("base_frame", capture.base_frame, first.base_frame),
            ("link_frame", capture.link_frame, first.link_frame),
            ("camera_frame", capture.camera_frame, first.camera_frame),
            ("source_name", capture.source_name, first.source_name),
            ("checkerboard_columns", capture.checkerboard_columns,
             first.checkerboard_columns),
            ("checkerboard_rows", capture.checkerboard_rows, first.checkerboard_rows),
            ("square_size_m", capture.square_size_m, first.square_size_m),
        )
        for name, actual, expected in fields:
            if actual != expected:
                mismatches.append(f"{capture.sample.sample_id}: {name} differs")
        if not _same_array(capture.K, first.K):
            mismatches.append(f"{capture.sample.sample_id}: K differs")
        if not _same_array(capture.D, first.D):
            mismatches.append(f"{capture.sample.sample_id}: D differs")
    observation_ids = [capture.observation_id for capture in captures]
    observation_stamps = [capture.observation_stamp_ns for capture in captures]
    if len(set(observation_ids)) != len(observation_ids):
        mismatches.append("observation_id values are not unique")
    if len(set(observation_stamps)) != len(observation_stamps):
        mismatches.append("observation stamps are not unique")
    if mismatches:
        raise CaptureSessionImportError(
            "session consistency check failed: " + "; ".join(mismatches)
        )

    return CalibrationDataset(
        session_id=first.scene_id,
        samples=tuple(capture.sample for capture in captures),
        checkerboard_columns=first.checkerboard_columns,
        checkerboard_rows=first.checkerboard_rows,
        square_size_m=first.square_size_m,
        base_frame=first.base_frame,
        link_frame=first.link_frame,
        camera_frame=first.camera_frame,
        checkerboard_frame="checkerboard",
        metadata={
            "importer": "strawberry_handeye_capture_npz/v1",
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "input_directory": str(directory),
            "source_name": first.source_name,
            "K": first.K.tolist(),
            "D": first.D.tolist(),
            "capture_audit": [capture.audit for capture in captures],
        },
    )


def capture_paths(
    input_directory: str | Path, sample_ids: Iterable[str] | None = None
) -> tuple[Path, ...]:
    """Return the deterministic NPZ selection used by the importer."""
    directory = Path(input_directory).expanduser().resolve()
    normalized = None if sample_ids is None else tuple(sample_ids)
    return _select_paths(directory, normalized)
