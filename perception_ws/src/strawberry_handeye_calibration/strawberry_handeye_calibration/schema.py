"""Versioned JSON schema for offline hand-eye calibration samples."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .transforms import validate_transform


SCHEMA_VERSION = "strawberry_handeye_samples/v1"
TRANSFORM_CONVENTION = {
    "matrix_layout": "row-major homogeneous 4x4",
    "T_base_link7": (
        "maps a point expressed in link7 into base_link; p_base = T_base_link7 p_link7"
    ),
    "T_camera_checkerboard": (
        "maps a checkerboard point into camera optical coordinates; "
        "p_camera = T_camera_checkerboard p_checkerboard"
    ),
    "result_T_link7_camera_optical": (
        "maps a camera optical point into link7; "
        "p_link7 = T_link7_camera_optical p_camera"
    ),
    "camera_optical_axes": "+X right, +Y down, +Z forward",
    "units": "metres and radians unless a field name states otherwise",
}


@dataclass(frozen=True)
class CalibrationSample:
    """One synchronized robot pose and checkerboard detection."""

    sample_id: str
    T_base_link7: np.ndarray
    T_camera_checkerboard: np.ndarray
    timestamp_sec: float | None = None
    corner_count: int | None = None
    reprojection_rms_px: float | None = None

    def __post_init__(self) -> None:
        """Validate data before it can reach a numerical solver."""
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        object.__setattr__(
            self,
            "T_base_link7",
            validate_transform(self.T_base_link7, "T_base_link7"),
        )
        object.__setattr__(
            self,
            "T_camera_checkerboard",
            validate_transform(
                self.T_camera_checkerboard,
                "T_camera_checkerboard",
            ),
        )
        if self.timestamp_sec is not None and not np.isfinite(self.timestamp_sec):
            raise ValueError("timestamp_sec must be finite")
        if self.corner_count is not None and self.corner_count <= 0:
            raise ValueError("corner_count must be positive when supplied")
        if self.reprojection_rms_px is not None:
            if not np.isfinite(self.reprojection_rms_px):
                raise ValueError("reprojection_rms_px must be finite")
            if self.reprojection_rms_px < 0.0:
                raise ValueError("reprojection_rms_px cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize one sample without losing transform direction labels."""
        result: dict[str, Any] = {
            "sample_id": self.sample_id,
            "T_base_link7": self.T_base_link7.tolist(),
            "T_camera_checkerboard": self.T_camera_checkerboard.tolist(),
        }
        if self.timestamp_sec is not None:
            result["timestamp_sec"] = float(self.timestamp_sec)
        detection: dict[str, Any] = {}
        if self.corner_count is not None:
            detection["corner_count"] = int(self.corner_count)
        if self.reprojection_rms_px is not None:
            detection["reprojection_rms_px"] = float(self.reprojection_rms_px)
        if detection:
            result["detection"] = detection
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationSample":
        """Parse one JSON sample and reject missing transform fields."""
        if not isinstance(value, dict):
            raise ValueError("each sample must be a JSON object")
        detection = value.get("detection", {})
        if not isinstance(detection, dict):
            raise ValueError("sample detection must be a JSON object")
        try:
            return cls(
                sample_id=str(value["sample_id"]),
                T_base_link7=np.asarray(value["T_base_link7"], dtype=np.float64),
                T_camera_checkerboard=np.asarray(
                    value["T_camera_checkerboard"], dtype=np.float64
                ),
                timestamp_sec=(
                    None
                    if "timestamp_sec" not in value
                    else float(value["timestamp_sec"])
                ),
                corner_count=(
                    None
                    if "corner_count" not in detection
                    else int(detection["corner_count"])
                ),
                reprojection_rms_px=(
                    None
                    if "reprojection_rms_px" not in detection
                    else float(detection["reprojection_rms_px"])
                ),
            )
        except KeyError as error:
            raise ValueError(f"sample is missing {error.args[0]}") from error


@dataclass(frozen=True)
class CalibrationDataset:
    """A complete stationary-checkerboard calibration session."""

    session_id: str
    samples: tuple[CalibrationSample, ...]
    checkerboard_columns: int
    checkerboard_rows: int
    square_size_m: float
    base_frame: str = "base_link"
    link_frame: str = "link7"
    camera_frame: str = "camera_color_optical_frame"
    checkerboard_frame: str = "checkerboard"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject mixed, ambiguous, or impossible sessions."""
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        object.__setattr__(self, "samples", tuple(self.samples))
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample_id values must be unique")
        if self.checkerboard_columns < 2 or self.checkerboard_rows < 2:
            raise ValueError("checkerboard internal-corner dimensions must be at least 2x2")
        if not np.isfinite(self.square_size_m) or self.square_size_m <= 0.0:
            raise ValueError("square_size_m must be finite and positive")
        frames = (
            self.base_frame,
            self.link_frame,
            self.camera_frame,
            self.checkerboard_frame,
        )
        if any(not item or not item.strip() for item in frames):
            raise ValueError("frame names must be non-empty")
        if len(set(frames)) != len(frames):
            raise ValueError("base, link, camera, and checkerboard frames must differ")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, versioned JSON representation."""
        return {
            "schema_version": SCHEMA_VERSION,
            "transform_convention": dict(TRANSFORM_CONVENTION),
            "session_id": self.session_id,
            "frames": {
                "base": self.base_frame,
                "link": self.link_frame,
                "camera_optical": self.camera_frame,
                "checkerboard": self.checkerboard_frame,
            },
            "checkerboard": {
                "columns": int(self.checkerboard_columns),
                "rows": int(self.checkerboard_rows),
                "square_size_m": float(self.square_size_m),
            },
            "samples": [sample.to_dict() for sample in self.samples],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationDataset":
        """Parse a v1 dataset while checking the declared convention."""
        if not isinstance(value, dict):
            raise ValueError("dataset must be a JSON object")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        convention = value.get("transform_convention")
        if convention != TRANSFORM_CONVENTION:
            raise ValueError("transform_convention is missing or differs from the v1 contract")
        try:
            frames = value["frames"]
            board = value["checkerboard"]
            raw_samples = value["samples"]
            if not isinstance(frames, dict) or not isinstance(board, dict):
                raise ValueError("frames and checkerboard must be JSON objects")
            if not isinstance(raw_samples, list):
                raise ValueError("samples must be a JSON array")
            return cls(
                session_id=str(value["session_id"]),
                samples=tuple(
                    CalibrationSample.from_dict(sample) for sample in raw_samples
                ),
                checkerboard_columns=int(board["columns"]),
                checkerboard_rows=int(board["rows"]),
                square_size_m=float(board["square_size_m"]),
                base_frame=str(frames["base"]),
                link_frame=str(frames["link"]),
                camera_frame=str(frames["camera_optical"]),
                checkerboard_frame=str(frames["checkerboard"]),
                metadata=dict(value.get("metadata", {})),
            )
        except KeyError as error:
            raise ValueError(f"dataset is missing {error.args[0]}") from error


def load_dataset(path: str | os.PathLike[str]) -> CalibrationDataset:
    """Load and validate one calibration dataset from disk."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return CalibrationDataset.from_dict(value)


def write_json_atomic(value: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Write a report or dataset atomically with stable formatting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_dataset(
    dataset: CalibrationDataset,
    path: str | os.PathLike[str],
) -> None:
    """Save a validated calibration dataset atomically."""
    write_json_atomic(dataset.to_dict(), path)
