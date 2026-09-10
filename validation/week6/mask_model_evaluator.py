#!/usr/bin/env python3
"""Offline-only evaluation of an Ultralytics strawberry segmentation weight.

This tool deliberately has no ROS publisher, service client, or action client.
It turns one RGB image into a mono8 mask, an overlay, and an audit JSON so an
untrusted third-party checkpoint can be inspected before it is allowed near a
real active-perception session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


HF_DIAGNOSTIC_CHECKPOINT_SHA256 = (
    "e89a33b2b53c89fad0deac0d7922ce53cfe780c230c593f569d0dd4852062adc"
)
DEFAULT_FRUIT_LABELS = (
    "Healthy Strawberry",
    "anthracnose_fruit_rot",
    "gray_mold",
    "powdery_mildew_fruit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def union_selected_instance_masks(
    masks: np.ndarray,
    class_ids: Iterable[int],
    confidences: Iterable[float],
    names: Mapping[int, str] | Sequence[str],
    allowed_labels: Iterable[str],
    confidence_threshold: float,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Union accepted instance masks into one canonical uint8 0/255 mask."""
    mask_array = np.asarray(masks, dtype=np.float32)
    if mask_array.ndim != 3:
        raise ValueError("instance masks must have shape [N, H, W]")
    ids = np.asarray(tuple(class_ids), dtype=np.int64)
    scores = np.asarray(tuple(confidences), dtype=np.float64)
    if mask_array.shape[0] != ids.size or ids.size != scores.size:
        raise ValueError("mask, class-id, and confidence counts must match")
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    height, width = (int(output_shape[0]), int(output_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("output shape must be positive")

    name_map = normalize_names(names)
    allowed = {str(label) for label in allowed_labels}
    union = np.zeros((height, width), dtype=np.uint8)
    accepted: list[dict[str, object]] = []
    for index, (class_id, confidence) in enumerate(zip(ids, scores)):
        label = name_map.get(int(class_id), f"class_{int(class_id)}")
        if label not in allowed or not np.isfinite(confidence):
            continue
        if float(confidence) < float(confidence_threshold):
            continue
        instance = mask_array[index]
        if instance.shape != (height, width):
            instance = cv2.resize(
                instance, (width, height), interpolation=cv2.INTER_NEAREST
            )
        binary = instance >= 0.5
        pixels = int(np.count_nonzero(binary))
        if pixels == 0:
            continue
        union[binary] = 255
        accepted.append(
            {
                "instance_index": index,
                "class_id": int(class_id),
                "label": label,
                "confidence": float(confidence),
                "pixels_before_union": pixels,
            }
        )
    return union, accepted


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.checkpoint).resolve()
    image_path = Path(args.image).resolve()
    output_directory = Path(args.output_directory).resolve()
    if not args.accept_untrusted_pickle:
        raise RuntimeError(
            "Ultralytics .pt is a pickle-bearing third-party file. Inspect its "
            "source and pass --accept-untrusted-pickle for this offline test."
        )
    actual_sha = sha256_file(checkpoint)
    if actual_sha != args.expected_sha256.lower():
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.expected_sha256}, "
            f"got {actual_sha}"
        )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {image_path}")

    # Import only after the explicit trust and hash checks above.  Keeping this
    # dependency lazy lets the normal ROS/Placo environments remain untouched.
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(checkpoint))
    results = model.predict(
        source=str(image_path), conf=float(args.confidence), verbose=False
    )
    if len(results) != 1:
        raise RuntimeError(f"expected one result, got {len(results)}")
    result = results[0]
    if result.masks is None or result.boxes is None:
        masks = np.empty((0, image.shape[0], image.shape[1]), dtype=np.float32)
        class_ids: tuple[int, ...] = ()
        confidences: tuple[float, ...] = ()
    else:
        masks = result.masks.data.detach().cpu().numpy()
        class_ids = tuple(
            int(value) for value in result.boxes.cls.detach().cpu().numpy()
        )
        confidences = tuple(
            float(value) for value in result.boxes.conf.detach().cpu().numpy()
        )
    mask, accepted = union_selected_instance_masks(
        masks=masks,
        class_ids=class_ids,
        confidences=confidences,
        names=model.names,
        allowed_labels=args.labels,
        confidence_threshold=float(args.confidence),
        output_shape=image.shape[:2],
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = output_directory / f"{stem}_learned_mask.png"
    overlay_path = output_directory / f"{stem}_learned_overlay.png"
    report_path = output_directory / f"{stem}_learned_mask.json"
    overlay = image.copy()
    foreground = mask == 255
    overlay[foreground] = (
        0.45 * overlay[foreground] + 0.55 * np.array([0, 255, 0])
    ).astype(np.uint8)
    if not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError(f"failed to write {mask_path}")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"failed to write {overlay_path}")
    report: dict[str, object] = {
        "schema": "strawberry_learned_mask_offline_evaluation/v1",
        "control_capability": "none_offline_only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "input_image": str(image_path),
        "input_image_sha256": sha256_file(image_path),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "confidence_threshold": float(args.confidence),
        "allowed_labels": list(args.labels),
        "model_names": normalize_names(model.names),
        "accepted_instances": accepted,
        "mask_pixels": int(np.count_nonzero(foreground)),
        "mask_fraction": float(np.mean(foreground)),
        "passes_existing_minimum_200_pixels": bool(
            np.count_nonzero(foreground) >= 200
        ),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "safe_for_robot_use": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--expected-sha256", default=HF_DIAGNOSTIC_CHECKPOINT_SHA256
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_FRUIT_LABELS))
    parser.add_argument("--accept-untrusted-pickle", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        report = evaluate(parse_args())
    except Exception as error:  # CLI boundary must provide one clear reason.
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
