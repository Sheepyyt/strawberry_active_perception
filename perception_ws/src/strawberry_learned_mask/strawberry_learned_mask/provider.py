"""Pure image and instance-selection logic for a learned mask provider."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class InstancePrediction:
    """One model instance on the canonical image grid."""

    class_id: int
    label: str
    confidence: float
    mask_probability: np.ndarray


@dataclass(frozen=True)
class MaskDecision:
    """Deterministic single-target mask plus auditable selection metadata."""

    mask: np.ndarray
    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    """Normalize an Ultralytics class-name table."""
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _ros_uint8(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 1:
            raise ValueError("ROS uint8 byte field must contain exactly one byte")
        return value[0]
    return int(value)


def decode_rgb8(image: Any) -> np.ndarray:
    """Decode a possibly padded ROS rgb8 Image without cv_bridge."""
    if str(image.encoding).lower() != "rgb8":
        raise ValueError(f"color encoding must be rgb8, got {image.encoding!r}")
    height, width, step = int(image.height), int(image.width), int(image.step)
    if height <= 0 or width <= 0 or step < width * 3:
        raise ValueError("rgb8 dimensions or row step are invalid")
    raw = np.frombuffer(bytes(image.data), dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError("rgb8 data length does not match height*step")
    return raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3).copy()


def decode_depth_32fc1(image: Any) -> np.ndarray:
    """Decode a possibly padded ROS 32FC1 Image into native float32."""
    if str(image.encoding).lower() != "32fc1":
        raise ValueError(f"depth encoding must be 32FC1, got {image.encoding!r}")
    height, width, step = int(image.height), int(image.width), int(image.step)
    if height <= 0 or width <= 0 or step < width * 4 or step % 4:
        raise ValueError("32FC1 dimensions or row step are invalid")
    endian = ">" if _ros_uint8(image.is_bigendian) else "<"
    raw = np.frombuffer(bytes(image.data), dtype=np.dtype(endian + "f4"))
    row_values = step // 4
    if raw.size != height * row_values:
        raise ValueError("32FC1 data length does not match height*step")
    return raw.reshape(height, row_values)[:, :width].astype(np.float32, copy=True)


def choose_instance_mask(
    predictions: Iterable[InstancePrediction],
    output_shape: tuple[int, int],
    *,
    allowed_labels: Iterable[str] = ("strawberry",),
    confidence_threshold: float = 0.70,
    mask_threshold: float = 0.50,
    policy: str = "highest_confidence",
) -> MaskDecision:
    """Filter classes/confidence and select one strawberry instance by default."""
    height, width = int(output_shape[0]), int(output_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("output_shape must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 < mask_threshold < 1.0:
        raise ValueError("mask_threshold must be in (0, 1)")
    if policy not in ("highest_confidence", "largest", "union"):
        raise ValueError("policy must be highest_confidence, largest, or union")
    allowed = {str(value).strip().casefold() for value in allowed_labels}
    if not allowed or "" in allowed:
        raise ValueError("allowed_labels must contain non-empty labels")

    candidates: list[tuple[InstancePrediction, np.ndarray, int]] = []
    rejected: list[dict[str, Any]] = []
    for prediction in predictions:
        probability = np.asarray(prediction.mask_probability, dtype=np.float32)
        reason = ""
        if probability.shape != (height, width):
            reason = "grid_mismatch"
        elif prediction.label.strip().casefold() not in allowed:
            reason = "class_not_allowed"
        elif not np.isfinite(prediction.confidence):
            reason = "nonfinite_confidence"
        elif prediction.confidence < confidence_threshold:
            reason = "confidence_below_threshold"
        elif not np.all(np.isfinite(probability)):
            reason = "nonfinite_mask"
        else:
            binary = probability >= mask_threshold
            pixels = int(np.count_nonzero(binary))
            if pixels:
                candidates.append((prediction, binary, pixels))
                continue
            reason = "empty_mask"
        rejected.append(
            {
                "class_id": int(prediction.class_id),
                "label": str(prediction.label),
                "confidence": float(prediction.confidence),
                "reason": reason,
            }
        )

    if policy == "highest_confidence":
        candidates.sort(key=lambda item: (-item[0].confidence, -item[2], item[0].class_id))
        selected = candidates[:1]
    elif policy == "largest":
        candidates.sort(key=lambda item: (-item[2], -item[0].confidence, item[0].class_id))
        selected = candidates[:1]
    else:
        selected = sorted(
            candidates,
            key=lambda item: (-item[0].confidence, -item[2], item[0].class_id),
        )

    output = np.zeros((height, width), dtype=np.uint8)
    accepted: list[dict[str, Any]] = []
    for prediction, binary, pixels in selected:
        output[binary] = 255
        accepted.append(
            {
                "class_id": int(prediction.class_id),
                "label": str(prediction.label),
                "confidence": float(prediction.confidence),
                "pixels_before_union": pixels,
            }
        )
    return MaskDecision(output, tuple(accepted), tuple(rejected))
