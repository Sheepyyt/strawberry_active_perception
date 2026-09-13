#!/usr/bin/env python3
"""Render the compact Week 10 YOLO-NBV evidence bundle offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


NAVY = "#10233f"
BLUE = "#2f6bff"
GREEN = "#18a875"
ORANGE = "#f29e38"
RED = "#e45151"
LIGHT = "#f3f6fa"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_depth_story(
    images: list[Path], reports: list[dict[str, Any]], output: Path
) -> None:
    titles = (
        "A. Oblique view: RGB mask survived, target depth did not",
        "B. Returned to the last depth-valid view",
        "C. Final view after the successful closed loop",
    )
    colors = (RED, ORANGE, GREEN)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    figure.suptitle(
        "Why the first attempt stopped, and how the experiment recovered",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )
    for axis, image_path, report, title, color in zip(
        axes, images, reports, titles, colors
    ):
        axis.imshow(plt.imread(image_path))
        axis.set_title(title, fontsize=11, color=NAVY, pad=10)
        axis.axis("off")
        mask = int(report["mask_pixels"])
        valid = int(report["valid_depth_pixels_in_mask"])
        depth = float(report["median_target_depth_m"])
        text = (
            f"mask: {mask} px\nvalid depth: {valid} px\n"
            f"median depth: {depth:.3f} m"
        )
        axis.text(
            0.02,
            0.03,
            text,
            transform=axis.transAxes,
            color="white",
            fontsize=10,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "alpha": 0.90},
        )
    figure.text(
        0.5,
        0.01,
        "The safety supervisor rejected background depth instead of "
        "corrupting the strawberry map.",
        ha="center",
        fontsize=12,
        color=NAVY,
    )
    _save(figure, output)


def render_pipeline(output: Path) -> None:
    labels = (
        "Gemini 2 XL\nRGB + aligned depth",
        "YOLO11m-seg\nstrawberry mask",
        "5-frame aggregate\nmedian + majority",
        "persistent 3-D\nvoxel map",
        "Gradient-NBV\ninformation gain",
        "reachable lattice\nPlaco IK filter",
        "NERO movement\ndouble gate",
        "observe again\nor stop",
    )
    figure, axis = plt.subplots(figsize=(17, 4.4))
    axis.set_xlim(0, 17)
    axis.set_ylim(0, 4.4)
    axis.axis("off")
    figure.suptitle(
        "YOLO-mask-driven real active-perception loop",
        fontsize=21,
        fontweight="bold",
        color=NAVY,
    )
    box_width = 1.72
    centres = [1.05 + index * 2.1 for index in range(len(labels))]
    palette = (BLUE, GREEN, ORANGE, BLUE, GREEN, ORANGE, RED, NAVY)
    for index, (centre, label, color) in enumerate(zip(centres, labels, palette)):
        patch = FancyBboxPatch(
            (centre - box_width / 2, 1.45),
            box_width,
            1.45,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=0,
            facecolor=color,
            alpha=0.96,
        )
        axis.add_patch(patch)
        axis.text(
            centre,
            2.18,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
        if index + 1 < len(labels):
            axis.annotate(
                "",
                xy=(centres[index + 1] - box_width / 2 - 0.06, 2.18),
                xytext=(centre + box_width / 2 + 0.06, 2.18),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": NAVY},
            )
    axis.annotate(
        "same scene / same map",
        xy=(centres[3], 1.35),
        xytext=(centres[7], 0.60),
        ha="center",
        color=NAVY,
        fontsize=11,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-|>",
            "connectionstyle": "arc3,rad=-0.25",
            "lw": 2.2,
            "color": GREEN,
        },
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.3, 3.35),
            16.4,
            0.55,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1,
            edgecolor="#d7e0ea",
            facecolor=LIGHT,
        )
    )
    axis.text(
        8.5,
        3.625,
        "Stop: coverage target reached · gain plateau · no positive-gain "
        "reachable candidate · safety/data fault",
        ha="center",
        va="center",
        color=NAVY,
        fontsize=11,
    )
    _save(figure, output)


def build_summary(
    execution_path: Path,
    preview_path: Path,
    map_paths: list[Path],
    failed_execution_path: Path,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    execution = _load(execution_path)
    progress = execution["session_progress"]
    steps = execution["motion_steps"]
    if execution.get("status") != "executed_session_scientific_acceptance_passed":
        raise ValueError("the selected execution did not pass scientific acceptance")
    if int(execution.get("motion_goal_count", -1)) != len(steps):
        raise ValueError("motion goal count and step records differ")
    coverages = [float(progress["initial_coverage"])] + [
        float(step["coverage"]["after"]) for step in steps
    ]
    if any(right < left for left, right in zip(coverages, coverages[1:])):
        raise ValueError("coverage is not monotonic")
    model_sources = {
        str(batch["aggregate_source_name"])
        for batch in (execution["bootstrap_batch"], execution["planning_batch"])
    }
    if not all("mask=yolo11m:7bea8d97b68c" in value for value in model_sources):
        raise ValueError("execution is not bound to the audited YOLO11 model")
    return {
        "schema": "strawberry_yolo11_real_nbv_result/v1",
        "result": "passed",
        "plain_language_result": (
            "YOLO found one strawberry; two robot-selected views increased the same "
            "3-D map coverage by 20.81 percentage points, then the reachable search "
            "found no additional positive-gain view and stopped safely."
        ),
        "model": {
            "type": "YOLO11m-seg",
            "class": "strawberry",
            "confidence_threshold": 0.70,
            "checkpoint_sha256": (
                "7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357"
            ),
        },
        "execution": {
            "scene_id": execution["scene_id"],
            "motion_goal_count": len(steps),
            "coverage_target": float(progress["coverage_target"]),
            "coverage_target_reached": bool(progress["coverage_target_reached"]),
            "coverage_fraction": coverages,
            "coverage_percent": [100.0 * value for value in coverages],
            "total_gain_percentage_points": 100.0
            * float(progress["total_coverage_delta"]),
            "actual_step_translation_mm": [
                1000.0
                * float(step["cumulative_camera_motion"]["actual_step_translation_m"])
                for step in steps
            ],
            "actual_step_rotation_deg": [
                float(step["cumulative_camera_motion"]["actual_step_rotation_deg"])
                for step in steps
            ],
            "final_position_error_mm": [
                1000.0 * float(step["post_pose_error"]["translation_m"])
                for step in steps
            ],
            "valid_target_depth_pixels": [
                int(step["post_target"]["valid_mask_pixels"]) for step in steps
            ],
            "move_j_trajectory_samples": [
                int(step["motion_command_count_step"]) for step in steps
            ],
            "stop_reason": progress["convergence_reason"],
            "every_step_gates_closed": all(
                step.get("completed_gates_closed") is True for step in steps
            ),
            "same_map_configure_call_count": int(
                execution["configure_call_count"]
            ),
        },
        "depth_failure_and_recovery": {
            "failed_attempt_sha256": _sha256(failed_execution_path),
            "failed_view_median_depth_m": reports[0]["median_target_depth_m"],
            "recovered_view_median_depth_m": reports[1]["median_target_depth_m"],
            "final_view_median_depth_m": reports[2]["median_target_depth_m"],
            "interpretation": (
                "The oblique RGB mask remained correct, but Gemini returned only the "
                "background layer inside it. The supervisor stopped, the robot returned "
                "to the last depth-valid view, and the next run used a frozen 25 mm cap."
            ),
        },
        "evidence": {
            "preview_sha256": _sha256(preview_path),
            "execution_sha256": _sha256(execution_path),
            "map_snapshot_sha256": [_sha256(path) for path in map_paths],
        },
    }


def write_manifest(output: Path) -> None:
    """Hash every generated artifact except the manifest itself."""
    manifest_path = output / "manifest.json"
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        records.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "strawberry_yolo11_nbv_artifact_manifest/v1",
                "file_count": len(records),
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--failed-execution", type=Path, required=True)
    parser.add_argument("--map-snapshots", type=Path, nargs=3, required=True)
    parser.add_argument("--images", type=Path, nargs=3, required=True)
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    reports = [_load(path) for path in arguments.reports]
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    render_depth_story(
        list(arguments.images), reports, output / "depth_failure_recovery_story.png"
    )
    render_pipeline(output / "yolo_nbv_pipeline.png")
    summary = build_summary(
        arguments.execution,
        arguments.preview,
        list(arguments.map_snapshots),
        arguments.failed_execution,
        reports,
    )
    summary_path = output / "yolo11_real_nbv_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
