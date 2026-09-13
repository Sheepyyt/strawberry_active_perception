"""Offline YOLO11 evaluation on a canonical NPZ or ordinary image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .provider import choose_instance_mask, sha256_file
from .ultralytics_backend import UltralyticsBackend


def _load_input(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if path.suffix.casefold() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "rgb" not in payload.files:
                raise ValueError("canonical NPZ has no rgb array")
            rgb = np.asarray(payload["rgb"], dtype=np.uint8)
            reference = (
                np.asarray(payload["mask"], dtype=np.uint8)
                if "mask" in payload.files
                else None
            )
            metadata = {
                key: np.asarray(payload[key]).item()
                for key in ("scene_id", "observation_id", "source_name")
                if key in payload.files
            }
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        reference = None
        metadata = {}
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("input RGB must have shape HxWx3")
    if reference is not None and reference.shape != rgb.shape[:2]:
        raise ValueError("reference mask grid differs from RGB")
    return rgb, reference, metadata


def _overlay(rgb: np.ndarray, learned: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    result = rgb.copy()
    foreground = learned == 255
    result[foreground] = (
        0.45 * result[foreground] + 0.55 * np.array([0, 255, 0])
    ).astype(np.uint8)
    if reference is not None:
        reference_only = (reference == 255) & ~foreground
        result[reference_only] = (
            0.45 * result[reference_only] + 0.55 * np.array([255, 165, 0])
        ).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    input_path = Path(args.input).resolve()
    output_directory = Path(args.output_directory).resolve()
    if not args.accept_pickle_checkpoint:
        raise RuntimeError(
            "the .pt checkpoint is pickle-bearing; pass "
            "--accept-pickle-checkpoint only for the audited SHA"
        )
    backend = UltralyticsBackend(
        str(checkpoint),
        args.checkpoint_sha256,
        device=args.device,
        image_size=args.image_size,
    )
    rgb, reference, metadata = _load_input(input_path)
    predictions, backend_audit = backend.predict(rgb, args.confidence)
    decision = choose_instance_mask(
        predictions,
        rgb.shape[:2],
        allowed_labels=args.labels,
        confidence_threshold=args.confidence,
        mask_threshold=args.mask_threshold,
        policy=args.instance_policy,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    mask_path = output_directory / f"{stem}_yolo11_mask.png"
    overlay_path = output_directory / f"{stem}_yolo11_overlay.png"
    report_path = output_directory / f"{stem}_yolo11_report.json"
    cv2.imwrite(str(mask_path), decision.mask)
    cv2.imwrite(str(overlay_path), _overlay(rgb, decision.mask, reference))

    learned = decision.mask == 255
    comparison: dict[str, Any] | None = None
    if reference is not None:
        baseline = reference == 255
        intersection = int(np.count_nonzero(learned & baseline))
        union = int(np.count_nonzero(learned | baseline))
        comparison = {
            "reference": "stored HSV mask",
            "hsv_pixels": int(np.count_nonzero(baseline)),
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": float(intersection / union) if union else 1.0,
        }
    report = {
        "schema": "strawberry_yolo11_mask_offline_evaluation/v2",
        "control_capability": "none_offline_only",
        "safe_for_robot_use": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_metadata": metadata,
        "image_shape": list(rgb.shape),
        "confidence_threshold": float(args.confidence),
        "mask_threshold": float(args.mask_threshold),
        "allowed_labels": list(args.labels),
        "instance_policy": args.instance_policy,
        "accepted_instances": list(decision.accepted),
        "rejected_instances": list(decision.rejected),
        "mask_pixels": int(np.count_nonzero(learned)),
        "comparison_to_hsv": comparison,
        "backend": backend_audit,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--labels", nargs="+", default=["strawberry"])
    parser.add_argument(
        "--instance-policy",
        choices=("highest_confidence", "largest", "union"),
        default="highest_confidence",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--accept-pickle-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        report = evaluate(parse_args())
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
