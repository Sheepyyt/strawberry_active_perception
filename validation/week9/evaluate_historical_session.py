#!/usr/bin/env python3
"""Evaluate the pinned YOLO11 strawberry segmenter on real saved observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from strawberry_learned_mask.provider import choose_instance_mask, sha256_file
from strawberry_learned_mask.ultralytics_backend import UltralyticsBackend


MODEL_SHA256 = "7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357"


def _scalar(payload: np.lib.npyio.NpzFile, key: str) -> object:
    return np.asarray(payload[key]).item()


def _tint(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = rgb.copy()
    selected = mask > 0
    if np.any(selected):
        output[selected] = (
            0.42 * output[selected] + 0.58 * np.asarray(color)
        ).astype(np.uint8)
    return output


def _panel(rgb: np.ndarray, label: str, width: int = 320) -> np.ndarray:
    scale = width / rgb.shape[1]
    resized = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (width, int(round(rgb.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    cv2.rectangle(resized, (0, 0), (width, 28), (22, 25, 32), -1)
    cv2.putText(
        resized,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return resized


def _contact_sheet(rows: list[dict[str, object]], output: Path) -> None:
    rendered = []
    for row in rows:
        rgb = np.asarray(row.pop("_rgb"))
        hsv = np.asarray(row.pop("_hsv"))
        learned = np.asarray(row.pop("_learned"))
        hsv_view = _tint(rgb, hsv, (255, 145, 0))
        learned_view = _tint(rgb, learned, (0, 255, 80))
        learned_view = _panel(
            learned_view,
            "YOLO11  conf={:.3f}  pixels={}".format(
                row["confidence"], row["learned_mask_pixels"]
            ),
        )
        rendered.append(
            np.hstack(
                [
                    _panel(rgb, f"{row['observation_id']}  RGB"),
                    _panel(hsv_view, f"HSV baseline  pixels={row['hsv_mask_pixels']}"),
                    learned_view,
                ]
            )
        )
    sheet = np.vstack(rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to write {output}")


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    width, height = 1280, 720
    canvas = np.full((height, width, 3), (250, 250, 250), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width, 74), (28, 35, 48), -1)
    cv2.putText(
        canvas,
        "YOLO11m strawberry segmentation - historical real-camera validation",
        (34, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    left, right = 88, width - 42
    top, bottom = 118, 418
    cv2.rectangle(canvas, (left, top), (right, bottom), (220, 220, 220), 1)
    n = len(rows)
    xs = np.linspace(left + 35, right - 35, n).astype(int)
    series = [
        ("confidence", (35, 130, 35), [float(row["confidence"]) for row in rows]),
        ("IoU vs HSV", (210, 105, 25), [float(row["iou_vs_hsv"]) for row in rows]),
        (
            "valid-depth ratio in YOLO mask",
            (160, 65, 175),
            [float(row["learned_valid_depth_ratio"]) for row in rows],
        ),
    ]
    for y_value in (0.0, 0.25, 0.5, 0.7, 0.9, 1.0):
        y = int(bottom - y_value * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(
            canvas,
            f"{y_value:.2f}",
            (36, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
    for name, color, values in series:
        points = np.array(
            [
                (x, int(bottom - value * (bottom - top)))
                for x, value in zip(xs, values)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, tuple(point), 5, color, -1, cv2.LINE_AA)
    legend_x = left
    for name, color, _ in series:
        cv2.line(canvas, (legend_x, 455), (legend_x + 30, 455), color, 4)
        cv2.putText(
            canvas,
            name,
            (legend_x + 38, 461),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        legend_x += 360
    for x, row in zip(xs, rows):
        cv2.putText(
            canvas,
            str(row["sequence"]),
            (x - 7, bottom + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    confidence_values = [float(row["confidence"]) for row in rows]
    mask_values = [int(row["learned_mask_pixels"]) for row in rows]
    summary = [
        f"All {n}/{n} frames passed confidence >= 0.70 and mask >= 200 px",
        "confidence: min {:.3f}, median {:.3f}, max {:.3f}".format(
            min(confidence_values),
            float(np.median(confidence_values)),
            max(confidence_values),
        ),
        "learned mask pixels: min {}, median {:.0f}, max {}".format(
            min(mask_values), float(np.median(mask_values)), max(mask_values)
        ),
        "One highest-confidence instance is selected for one-target NBV.",
    ]
    y = 510
    for line in summary:
        cv2.putText(
            canvas,
            line,
            (90, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (40, 48, 60),
            1,
            cv2.LINE_AA,
        )
        y += 42
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"failed to write {output}")


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    input_directory = Path(args.input_directory).resolve()
    output_directory = Path(args.output_directory).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    paths = sorted(input_directory.glob("observation_*.npz"))
    paths = [path for path in paths if int(path.name.split("_")[1]) >= args.first]
    if not paths:
        raise RuntimeError("no observation NPZ files selected")
    backend = UltralyticsBackend(
        str(checkpoint), MODEL_SHA256, device="cpu", image_size=640
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            rgb = np.asarray(payload["rgb"], dtype=np.uint8)
            depth = np.asarray(payload["depth_m"], dtype=np.float32)
            hsv = np.asarray(payload["mask"], dtype=np.uint8)
            observation_id = str(_scalar(payload, "observation_id"))
        predictions, backend_audit = backend.predict(rgb, 0.05)
        decision = choose_instance_mask(
            predictions,
            rgb.shape[:2],
            confidence_threshold=args.confidence,
            mask_threshold=0.50,
            policy="highest_confidence",
        )
        learned = decision.mask
        learned_pixels = int(np.count_nonzero(learned))
        valid = np.isfinite(depth) & (depth > 0.0)
        valid_pixels = int(np.count_nonzero((learned > 0) & valid))
        intersection = int(np.count_nonzero((learned > 0) & (hsv > 0)))
        union = int(np.count_nonzero((learned > 0) | (hsv > 0)))
        confidence = (
            float(decision.accepted[0]["confidence"]) if decision.accepted else 0.0
        )
        row: dict[str, object] = {
            "sequence": int(path.name.split("_")[1]),
            "observation_id": observation_id,
            "input_file": str(path.relative_to(Path.cwd())),
            "input_sha256": sha256_file(path),
            "confidence": confidence,
            "learned_mask_pixels": learned_pixels,
            "learned_valid_depth_pixels": valid_pixels,
            "learned_valid_depth_ratio": (
                float(valid_pixels / learned_pixels) if learned_pixels else 0.0
            ),
            "hsv_mask_pixels": int(np.count_nonzero(hsv)),
            "iou_vs_hsv": float(intersection / union) if union else 1.0,
            "raw_instances": [
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "pixels": int(np.count_nonzero(item.mask_probability >= 0.50)),
                }
                for item in predictions
            ],
            "inference_ms": backend_audit["inference_ms"],
            "passed": confidence >= args.confidence
            and learned_pixels >= 200
            and valid_pixels >= 100,
            "_rgb": rgb,
            "_hsv": hsv,
            "_learned": learned,
        }
        rows.append(row)
    contact = output_directory / "yolo11_historical_contact_sheet.png"
    metrics = output_directory / "yolo11_historical_metrics.png"
    _contact_sheet(rows, contact)
    _plot(rows, metrics)
    report = {
        "schema": "strawberry_yolo11_historical_validation/v1",
        "control_capability": "none_offline_only",
        "safe_for_robot_use": False,
        "checkpoint_sha256": MODEL_SHA256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "class_table": backend.names,
        "confidence_threshold": args.confidence,
        "selection_policy": "highest_confidence_single_instance",
        "frame_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "all_passed": all(bool(row["passed"]) for row in rows),
        "contact_sheet": str(contact.relative_to(Path.cwd())),
        "metrics_plot": str(metrics.relative_to(Path.cwd())),
        "frames": rows,
    }
    report_path = output_directory / "yolo11_historical_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--first", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.70)
    args = parser.parse_args()
    try:
        report = evaluate(args)
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    print(
        f"PASS: {report['passed_count']}/{report['frame_count']} frames; "
        f"report={report['contact_sheet']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
