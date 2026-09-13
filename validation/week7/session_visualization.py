#!/usr/bin/env python3
"""Create presentation-ready figures from one audited real NBV session.

The renderer is deliberately offline: it reads JSON/NPZ evidence and never
imports ROS or creates a hardware command surface.  Every generated file is
listed with a SHA-256 digest so the presentation can be reproduced later.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,I202
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402


NAVY = "#10233f"
BLUE = "#2f6bff"
CYAN = "#00a9ce"
GREEN = "#18a875"
ORANGE = "#f29e38"
RED = "#e45151"
GRAY = "#8793a1"
LIGHT = "#f3f6fa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    return matrix


def _steps(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = document.get("motion_steps")
    if not isinstance(raw, list) or not raw:
        raise ValueError("execution document has no motion_steps")
    steps = [step for step in raw if isinstance(step, Mapping)]
    if len(steps) != len(raw):
        raise ValueError("motion_steps contains a non-object")
    return steps


def _candidate_rounds(
    document: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]
) -> list[tuple[int, list[Mapping[str, Any]], Mapping[str, Any] | None]]:
    rounds: list[tuple[int, list[Mapping[str, Any]], Mapping[str, Any] | None]] = []
    first = document.get("ik_candidates")
    if isinstance(first, list) and first:
        rounds.append((1, first, steps[0].get("selected_candidate")))
    for step in steps[1:]:
        candidates = step.get("ik_candidates")
        if isinstance(candidates, list) and candidates:
            rounds.append(
                (int(step.get("step_index", len(rounds) + 1)), candidates, step)
            )
    terminal = steps[-1].get("next_view_convergence")
    if isinstance(terminal, Mapping):
        candidates = terminal.get("ik_candidates")
        if isinstance(candidates, list) and candidates:
            rounds.append((int(terminal["next_step_index"]), candidates, None))
    return rounds


def _candidate_counts(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {"IK rejected": 0, "reachable / no gain": 0, "useful": 0}
    for candidate in candidates:
        if not bool(candidate.get("ik_gate_passed", False)):
            result["IK rejected"] += 1
        elif (
            bool(candidate.get("gain_scored", False))
            and float(candidate.get("gain_improvement", 0.0)) > 0.0
        ):
            result["useful"] += 1
        else:
            result["reachable / no gain"] += 1
    return result


def _style(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.set_facecolor("white")
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(True, color="#dfe6ef", linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def render_coverage(
    document: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], output: Path
) -> Path:
    progress = document.get("session_progress", {})
    initial = float(progress.get("initial_coverage", steps[0]["coverage"]["before"]))
    after = [float(step["coverage"]["after"]) for step in steps]
    values = [initial, *after]
    deltas = [float(step["coverage"]["delta"]) for step in steps]
    target = progress.get("coverage_target")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    _style(axes[0])
    indices = np.arange(len(values))
    axes[0].plot(indices, np.asarray(values) * 100.0, "o-", color=BLUE, lw=3)
    axes[0].fill_between(indices, np.asarray(values) * 100.0, color=BLUE, alpha=0.10)
    if target is not None:
        axes[0].axhline(
            float(target) * 100.0, color=GREEN, ls="--", lw=2, label="stop target"
        )
        axes[0].legend(frameon=False)
    axes[0].set_xticks(
        indices, ["start", *[f"step {i}" for i in range(1, len(values))]]
    )
    axes[0].set_ylabel("ROI ray coverage (%)")
    axes[0].set_title("Persistent voxel-map coverage")
    for x, value in zip(indices, values):
        axes[0].annotate(
            f"{value * 100:.2f}%",
            (x, value * 100),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    _style(axes[1])
    colors = [GREEN if value >= 0.01 else ORANGE for value in deltas]
    axes[1].bar(np.arange(1, len(deltas) + 1), np.asarray(deltas) * 100.0, color=colors)
    axes[1].axhline(0.5, color=ORANGE, ls="--", label="plateau threshold (0.5 pp)")
    axes[1].axhline(1.0, color=GREEN, ls=":", label="strong-step threshold (1 pp)")
    axes[1].set_xlabel("motion step")
    axes[1].set_ylabel("coverage gain (percentage points)")
    axes[1].set_title("What each new view contributed")
    axes[1].legend(frameon=False, fontsize=8)
    for x, value in enumerate(deltas, start=1):
        axes[1].text(
            x,
            value * 100.0,
            f"+{value * 100:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.suptitle(
        "Active perception progress", fontsize=17, color=NAVY, weight="bold"
    )
    return _save(figure, output)


def render_motion(steps: Sequence[Mapping[str, Any]], output: Path) -> Path:
    labels = [
        int(step.get("step_index", index + 1)) for index, step in enumerate(steps)
    ]
    planned_mm = [
        float(step["cumulative_camera_motion"]["planned_step_translation_m"]) * 1000
        for step in steps
    ]
    actual_mm = [
        float(step["cumulative_camera_motion"]["actual_step_translation_m"]) * 1000
        for step in steps
    ]
    planned_deg = [
        float(step["cumulative_camera_motion"]["planned_step_rotation_deg"])
        for step in steps
    ]
    actual_deg = [
        float(step["cumulative_camera_motion"]["actual_step_rotation_deg"])
        for step in steps
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis in axes:
        _style(axis)
    axes[0].bar(x - 0.18, planned_mm, width=0.36, color=BLUE, label="planned")
    axes[0].bar(x + 0.18, actual_mm, width=0.36, color=CYAN, label="measured")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("motion step")
    axes[0].set_ylabel("camera translation (mm)")
    axes[0].set_title("Visible camera translation")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, planned_deg, width=0.36, color="#744fc6", label="planned")
    axes[1].bar(x + 0.18, actual_deg, width=0.36, color="#bd79d1", label="measured")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("motion step")
    axes[1].set_ylabel("camera rotation (deg)")
    axes[1].set_title("Camera reorientation")
    axes[1].legend(frameon=False)
    figure.suptitle(
        "Planned motion versus real motion", fontsize=17, color=NAVY, weight="bold"
    )
    return _save(figure, output)


def render_accuracy_and_safety(
    steps: Sequence[Mapping[str, Any]], output: Path
) -> Path:
    labels = np.arange(1, len(steps) + 1)
    pose_mm = [float(step["post_pose_error"]["translation_m"]) * 1000 for step in steps]
    pose_deg = [float(step["post_pose_error"]["rotation_deg"]) for step in steps]
    ik = [step.get("fresh_execution_ik", {}) for step in steps]
    sigma = [float(item.get("sigma_min", math.nan)) for item in ik]
    condition = [float(item.get("condition_number", math.nan)) for item in ik]
    delta_q = [
        float(item.get("independent_max_joint_delta_rad", math.nan)) for item in ik
    ]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis in axes.flat:
        _style(axis)
        axis.set_xticks(labels)
        axis.set_xlabel("motion step")
    axes[0, 0].plot(labels, pose_mm, "o-", color=BLUE, lw=2.5)
    axes[0, 0].axhline(5.0, color=RED, ls="--", label="limit 5 mm")
    axes[0, 0].set_ylabel("position error (mm)")
    axes[0, 0].set_title("Final camera position error")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(labels, pose_deg, "o-", color="#744fc6", lw=2.5)
    axes[0, 1].axhline(2.0, color=RED, ls="--", label="limit 2 deg")
    axes[0, 1].set_ylabel("orientation error (deg)")
    axes[0, 1].set_title("Final camera orientation error")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(labels, sigma, "o-", color=GREEN, lw=2.5, label="sigma min")
    axes[1, 0].axhline(0.10, color=RED, ls="--", label="minimum 0.10")
    axes[1, 0].set_ylabel("Jacobian sigma min")
    axes[1, 0].set_title("Distance from singularity")
    axes[1, 0].legend(frameon=False)
    axis2 = axes[1, 1]
    axis2.plot(labels, condition, "o-", color=ORANGE, lw=2.5, label="condition number")
    axis2.axhline(20.0, color=RED, ls="--", label="maximum 20")
    axis2.set_ylabel("condition number")
    axis2.set_title("IK conditioning / joint movement")
    twin = axis2.twinx()
    twin.plot(labels, delta_q, "s:", color=CYAN, lw=2, label="max joint delta")
    twin.axhline(0.35, color=GRAY, ls=":", label="max joint delta 0.35 rad")
    twin.set_ylabel("largest joint change (rad)")
    handles, names = axis2.get_legend_handles_labels()
    handles2, names2 = twin.get_legend_handles_labels()
    axis2.legend(handles + handles2, names + names2, frameon=False, fontsize=8)
    figure.suptitle(
        "Accuracy and robot safety margins", fontsize=17, color=NAVY, weight="bold"
    )
    return _save(figure, output)


def render_candidate_funnel(
    rounds: Sequence[tuple[int, list[Mapping[str, Any]], Mapping[str, Any] | None]],
    output: Path,
) -> Path:
    if not rounds:
        raise ValueError("no candidate round is available")
    labels = [round_[0] for round_ in rounds]
    counts = [_candidate_counts(round_[1]) for round_ in rounds]
    rejected = np.asarray([item["IK rejected"] for item in counts])
    no_gain = np.asarray([item["reachable / no gain"] for item in counts])
    useful = np.asarray([item["useful"] for item in counts])
    figure, axis = plt.subplots(figsize=(11, 5.3), constrained_layout=True)
    _style(axis)
    x = np.arange(len(labels))
    axis.bar(x, rejected, color="#b9c0c9", label="IK / safety rejected")
    axis.bar(x, no_gain, bottom=rejected, color=ORANGE, label="reachable, no gain")
    axis.bar(
        x, useful, bottom=rejected + no_gain, color=BLUE, label="reachable + useful"
    )
    totals = rejected + no_gain + useful
    for index, total in enumerate(totals):
        axis.text(index, total + 1, f"{int(total)} tested", ha="center", fontsize=9)
    axis.set_xticks(x, [f"step {value}" for value in labels])
    axis.set_ylabel("candidate camera poses")
    axis.set_title(
        "How reachability and information gain filter the search lattice",
        color=NAVY,
        weight="bold",
        fontsize=15,
    )
    axis.legend(frameon=False, ncol=3, loc="upper center")
    return _save(figure, output)


def render_target_and_map_quality(
    steps: Sequence[Mapping[str, Any]], output: Path
) -> Path:
    labels = np.arange(1, len(steps) + 1)
    mask_pixels = [
        int(step.get("post_target", {}).get("mask_pixels", 0)) for step in steps
    ]
    valid_pixels = [
        int(step.get("post_target", {}).get("valid_mask_pixels", 0)) for step in steps
    ]
    observed = [
        int(step.get("post_map_update", {}).get("voxel_counts", {}).get("observed", 0))
        for step in steps
    ]
    occupied = [
        int(step.get("post_map_update", {}).get("voxel_counts", {}).get("occupied", 0))
        for step in steps
    ]
    commands = [int(step.get("motion_command_count_step", 0)) for step in steps]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis in axes:
        _style(axis)
        axis.set_xticks(labels)
        axis.set_xlabel("motion step")
    axes[0].plot(labels, mask_pixels, "o-", color=RED, label="mask pixels")
    axes[0].plot(labels, valid_pixels, "s-", color=GREEN, label="mask + valid depth")
    axes[0].axhline(100, color=ORANGE, ls="--", label="minimum 100")
    axes[0].set_title("Strawberry evidence")
    axes[0].set_ylabel("pixels")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(labels, observed, "o-", color=BLUE, label="observed")
    axes[1].plot(labels, occupied, "s-", color=NAVY, label="depth surfaces")
    axes[1].set_title("Voxel-map growth")
    axes[1].set_ylabel("voxel count")
    axes[1].legend(frameon=False)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    axes[2].bar(labels, commands, color=CYAN)
    axes[2].set_title("Smooth trajectory samples")
    axes[2].set_ylabel("low-level setpoints")
    for x, value in zip(labels, commands):
        axes[2].text(x, value, str(value), ha="center", va="bottom")
    figure.suptitle(
        "Target visibility, map growth and controller output",
        fontsize=17,
        color=NAVY,
        weight="bold",
    )
    return _save(figure, output)


def _camera_matrix_for_step(step: Mapping[str, Any]) -> np.ndarray:
    post_map = step.get("post_map_update", {})
    value = post_map.get("T_base_camera_raw")
    if value is None:
        value = step.get("selected_candidate", {}).get("T_base_camera")
    if value is None:
        value = step.get("T_base_camera")
    return _matrix(value, "camera trajectory pose")


def render_camera_trajectory(
    document: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], output: Path
) -> Path:
    initial_value = steps[0].get("planned_start_camera")
    if initial_value is None:
        initial_value = document.get("final_tf", {}).get("T_base_camera_optical")
    poses = [_matrix(initial_value, "initial camera")]
    poses.extend(_camera_matrix_for_step(step) for step in steps)
    points = np.stack([pose[:3, 3] for pose in poses])
    target = np.asarray(
        document.get("final_target", {}).get(
            "base_xyz_m",
            document.get("configuration", {}).get("request", {}).get("target_center_m"),
        ),
        dtype=float,
    )
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        target = np.mean(points, axis=0)
    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        points[:, 0], points[:, 1], points[:, 2], "o-", color=GREEN, lw=3, markersize=7
    )
    for index, pose in enumerate(poses):
        origin = pose[:3, 3]
        optical_z = pose[:3, 2]
        axis.quiver(
            *origin,
            *optical_z,
            length=0.035,
            normalize=True,
            color=BLUE,
            arrow_length_ratio=0.25,
        )
        axis.text(*origin, f"  {index}", color=NAVY, weight="bold")
    axis.scatter(*target, marker="X", s=150, color=RED, label="target centre")
    axis.set_xlabel("base X (m)")
    axis.set_ylabel("base Y (m)")
    axis.set_zlabel("base Z (m)")
    axis.set_title(
        "Measured camera path and viewing directions",
        color=NAVY,
        weight="bold",
        fontsize=15,
    )
    axis.legend(frameon=False)
    spans = np.ptp(np.vstack([points, target]), axis=0)
    half = max(0.08, float(np.max(spans)) * 0.62)
    centre = 0.5 * (
        np.max(np.vstack([points, target]), axis=0)
        + np.min(np.vstack([points, target]), axis=0)
    )
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    return _save(figure, output)


def render_candidate_round(
    step_index: int,
    candidates: Sequence[Mapping[str, Any]],
    selected_record: Mapping[str, Any] | None,
    target: np.ndarray,
    output: Path,
) -> Path:
    figure = plt.figure(figsize=(8, 6.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    styles = {
        "IK rejected": (GRAY, "x", 18),
        "reachable / no gain": (ORANGE, ".", 24),
        "useful": (BLUE, "o", 26),
    }
    all_points: list[np.ndarray] = []
    for category in styles:
        points = []
        for candidate in candidates:
            if not bool(candidate.get("ik_gate_passed", False)):
                current_category = "IK rejected"
            elif (
                bool(candidate.get("gain_scored", False))
                and float(candidate.get("gain_improvement", 0.0)) > 0.0
            ):
                current_category = "useful"
            else:
                current_category = "reachable / no gain"
            if current_category == category:
                points.append(_matrix(candidate["T_base_camera"], "candidate")[:3, 3])
        if points:
            array = np.stack(points)
            all_points.extend(points)
            color, marker, size = styles[category]
            axis.scatter(
                array[:, 0],
                array[:, 1],
                array[:, 2],
                c=color,
                marker=marker,
                s=size,
                alpha=0.75,
                label=category,
            )
    if selected_record is not None:
        value = selected_record.get("T_base_camera")
        if value is None:
            value = selected_record.get("selected_candidate", {}).get("T_base_camera")
        if value is not None:
            selected = _matrix(value, "selected candidate")[:3, 3]
            axis.scatter(*selected, c=GREEN, marker="*", s=240, label="selected")
            all_points.append(selected)
    axis.scatter(*target, c=RED, marker="X", s=120, label="target centre")
    all_points.append(target)
    array = np.stack(all_points)
    centre = 0.5 * (array.min(axis=0) + array.max(axis=0))
    half = max(0.02, 0.58 * float(np.max(np.ptp(array, axis=0))))
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_xlabel("base X (m)")
    axis.set_ylabel("base Y (m)")
    axis.set_zlabel("base Z (m)")
    axis.set_title(
        f"Candidate search before motion step {step_index}", color=NAVY, weight="bold"
    )
    axis.legend(frameon=False, fontsize=8)
    return _save(figure, output)


def render_dashboard(
    document: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], output: Path
) -> Path:
    progress = document.get("session_progress", {})
    coverage = [float(progress.get("initial_coverage", steps[0]["coverage"]["before"]))]
    coverage.extend(float(step["coverage"]["after"]) for step in steps)
    deltas = [float(step["coverage"]["delta"]) for step in steps]
    motion = [
        float(step["cumulative_camera_motion"]["actual_step_translation_m"]) * 1000
        for step in steps
    ]
    errors = [float(step["post_pose_error"]["translation_m"]) * 1000 for step in steps]
    pixels = [
        int(step.get("post_target", {}).get("valid_mask_pixels", 0)) for step in steps
    ]
    x = np.arange(1, len(steps) + 1)
    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=[0.72, 1.4, 1.4])
    title = figure.add_subplot(grid[0, :])
    title.axis("off")
    title.text(
        0.0,
        0.86,
        "REAL ROBOT ACTIVE PERCEPTION",
        fontsize=12,
        color=CYAN,
        weight="bold",
    )
    title.text(
        0.0,
        0.52,
        "Next-Best-View closed-loop experiment",
        fontsize=27,
        color=NAVY,
        weight="bold",
    )
    reason = str(
        progress.get("convergence_reason")
        or progress.get("termination_reason")
        or "safety backstop reached"
    )
    title.text(
        0.0,
        0.16,
        f"Result: {document.get('status', 'unknown')}  |  Stop: {reason}",
        fontsize=11,
        color="#4a5868",
    )
    kpis = [
        ("MOTIONS", f"{len(steps)}"),
        ("COVERAGE", f"{coverage[-1] * 100:.2f}%"),
        ("GAIN", f"+{(coverage[-1] - coverage[0]) * 100:.2f} pp"),
        ("PATH", f"{sum(motion):.0f} mm"),
    ]
    for index, (name, value) in enumerate(kpis):
        left = 0.57 + index * 0.105
        patch = FancyBboxPatch(
            (left, 0.08),
            0.095,
            0.75,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=title.transAxes,
            facecolor=LIGHT,
            edgecolor="#d8e1ec",
        )
        title.add_patch(patch)
        title.text(
            left + 0.0475,
            0.59,
            name,
            ha="center",
            va="center",
            fontsize=8,
            color=GRAY,
            weight="bold",
        )
        title.text(
            left + 0.0475,
            0.31,
            value,
            ha="center",
            va="center",
            fontsize=15,
            color=NAVY,
            weight="bold",
        )

    axes = [
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2:4]),
        figure.add_subplot(grid[2, 0:2]),
        figure.add_subplot(grid[2, 2:4]),
    ]
    for axis in axes:
        _style(axis)
    obs_x = np.arange(len(coverage))
    axes[0].plot(obs_x, np.asarray(coverage) * 100, "o-", color=BLUE, lw=3)
    target = progress.get("coverage_target")
    if target is not None:
        axes[0].axhline(float(target) * 100, color=GREEN, ls="--", label="stop target")
        axes[0].legend(frameon=False)
    axes[0].set_title("Map coverage keeps growing")
    axes[0].set_ylabel("coverage (%)")
    axes[0].set_xticks(obs_x, ["start", *[str(i) for i in x]])
    axes[0].set_xlabel("observation / motion step")
    axes[1].bar(
        x,
        np.asarray(deltas) * 100,
        color=[GREEN if value >= 0.01 else ORANGE for value in deltas],
    )
    axes[1].axhline(0.5, color=RED, ls="--", lw=1.5)
    axes[1].set_title("Information added by every new view")
    axes[1].set_ylabel("gain (percentage points)")
    axes[1].set_xlabel("motion step")
    axes[2].bar(x - 0.18, motion, width=0.36, color=CYAN, label="real translation")
    axes[2].bar(x + 0.18, errors, width=0.36, color=RED, label="final error")
    axes[2].set_title("Big visible moves, small terminal error")
    axes[2].set_ylabel("millimetres")
    axes[2].set_xlabel("motion step")
    axes[2].legend(frameon=False)
    axes[3].plot(x, pixels, "o-", color=RED, lw=2.5, label="valid strawberry pixels")
    axes[3].axhline(100, color=ORANGE, ls="--", label="minimum")
    axes[3].set_title("Target remained visible with valid depth")
    axes[3].set_ylabel("pixels")
    axes[3].set_xlabel("motion step")
    axes[3].legend(frameon=False)
    return _save(figure, output)


def _write_reports(
    document: Mapping[str, Any], outputs: Sequence[Path], output_directory: Path
) -> tuple[Path, Path]:
    progress = document.get("session_progress", {})
    rows = []
    for step in _steps(document):
        row_template = (
            "| {i} | {move:.1f} mm | {rot:.2f}° | {coverage:.2f}% | "
            "+{gain:.2f} pp | {error:.2f} mm / {angle:.3f}° | {pixels} |"
        )
        rows.append(
            row_template.format(
                i=int(step.get("step_index", len(rows) + 1)),
                move=float(
                    step["cumulative_camera_motion"]["actual_step_translation_m"]
                )
                * 1000,
                rot=float(step["cumulative_camera_motion"]["actual_step_rotation_deg"]),
                coverage=float(step["coverage"]["after"]) * 100,
                gain=float(step["coverage"]["delta"]) * 100,
                error=float(step["post_pose_error"]["translation_m"]) * 1000,
                angle=float(step["post_pose_error"]["rotation_deg"]),
                pixels=int(step.get("post_target", {}).get("valid_mask_pixels", 0)),
            )
        )
    markdown = output_directory / "REPORT_CN.md"
    initial_coverage = float(progress.get("initial_coverage", 0)) * 100.0
    final_coverage = float(progress.get("final_coverage", 0)) * 100.0
    stop_reason = (
        progress.get("convergence_reason")
        or progress.get("termination_reason")
        or "达到预先设置的安全步数上限"
    )
    gate_state = "已关闭" if not progress.get("terminated") else "请查看审计 JSON"
    markdown.write_text(
        "# 真实 NBV 闭环展示报告\n\n"
        f"- 程序状态：`{document.get('status', 'unknown')}`\n"
        f"- 实际高层运动次数：{len(_steps(document))}\n"
        f"- coverage：{initial_coverage:.2f}% → {final_coverage:.2f}%\n"
        f"- 停止原因：{stop_reason}\n"
        f"- 运动结束时两道执行门：{gate_state}\n\n"
        "| 步骤 | 实际移动 | 实际转动 | coverage | 本步新增 | 最终误差 | 有效目标像素 |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n## 图片怎么读\n\n"
        "- `00_experiment_dashboard.png`：一页式总览，适合直接放汇报 PPT。\n"
        "- `01_coverage_curve.png`：蓝线越往上，代表更多目标附近空间被相机射线看过。\n"
        "- `02_camera_trajectory_3d.png`：绿线是相机真实走过的路，蓝箭头是每次朝向。\n"
        "- `03_candidate_funnel.png`：灰色是机械臂去不了的点，橙色是能去但没新信息，蓝色是能去且有用。\n"
        "- `04_motion_profile.png`：对比计划动作与真实动作。\n"
        "- `05_accuracy_safety.png`：显示到位误差、奇异性和关节变化是否在门槛内。\n"
        "- `06_target_map_quality.png`：显示草莓像素、体素地图大小和轨迹采样点。\n"
        "- `candidate_step_*.png`：每一步完整备选观察点的三维分布。\n"
        "- `map/nbv_map_progress.gif`：体素地图随观察次数增长的动画。\n",
        encoding="utf-8",
    )
    image_rows = "\n".join(
        "<section><h2>{title}</h2><img src=\"{source}\"></section>".format(
            title=html.escape(path.stem.replace("_", " ").title()),
            source=html.escape(path.name),
        )
        for path in outputs
        if path.suffix.lower() == ".png"
    )
    page = output_directory / "index.html"
    page_status = html.escape(str(document.get("status")))
    page.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        "<title>真实 NBV 实验展示</title>"
        "<style>body{margin:0;background:#eef2f7;color:#10233f;"
        "font-family:system-ui,sans-serif}header{padding:48px 7%;"
        "background:linear-gradient(120deg,#10233f,#245ea8);color:white}"
        "main{max-width:1400px;margin:auto;padding:28px}"
        "section{background:white;margin:24px 0;padding:22px;border-radius:18px;"
        "box-shadow:0 8px 28px #10233f18}img{width:100%;height:auto;"
        "border-radius:10px}h1{font-size:38px;margin:0 0 12px}</style>"
        "<header><h1>真实机器人 Next-Best-View 闭环</h1>"
        f"<p>状态：{page_status} · 运动 {len(_steps(document))} 次 · "
        f"最终 coverage {final_coverage:.2f}%</p></header>"
        f"<main>{image_rows}</main></html>\n",
        encoding="utf-8",
    )
    return markdown, page


def render_session(
    execution_path: str | Path,
    output_directory: str | Path,
    *,
    snapshot_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    execution = Path(execution_path).expanduser().resolve()
    document = json.loads(execution.read_text(encoding="utf-8"))
    steps = _steps(document)
    rounds = _candidate_rounds(document, steps)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = [
        render_dashboard(document, steps, output / "00_experiment_dashboard.png"),
        render_coverage(document, steps, output / "01_coverage_curve.png"),
        render_camera_trajectory(
            document, steps, output / "02_camera_trajectory_3d.png"
        ),
        render_candidate_funnel(rounds, output / "03_candidate_funnel.png"),
        render_motion(steps, output / "04_motion_profile.png"),
        render_accuracy_and_safety(steps, output / "05_accuracy_safety.png"),
        render_target_and_map_quality(steps, output / "06_target_map_quality.png"),
    ]
    target = np.asarray(
        document.get("final_target", {}).get(
            "base_xyz_m",
            document.get("configuration", {}).get("request", {}).get("target_center_m"),
        ),
        dtype=float,
    )
    if target.shape == (3,) and np.all(np.isfinite(target)):
        for step_index, candidates, selected in rounds:
            generated.append(
                render_candidate_round(
                    step_index,
                    candidates,
                    selected,
                    target,
                    output / f"candidate_step_{step_index:03d}.png",
                )
            )
    snapshots = [Path(path).expanduser().resolve() for path in snapshot_paths]
    map_manifest: Mapping[str, Any] | None = None
    if snapshots:
        from strawberry_gradient_nbv.map_visualization import render_snapshot_sequence

        map_manifest = render_snapshot_sequence(snapshots, output / "map")
    markdown, page = _write_reports(document, generated, output)
    generated.extend((markdown, page))
    manifest = {
        "schema": "strawberry_nbv_presentation_bundle/v1",
        "execution_json": str(execution),
        "execution_sha256": _sha256(execution),
        "status": document.get("status"),
        "motion_goal_count": len(steps),
        "session_progress": document.get("session_progress"),
        "candidate_round_count": len(rounds),
        "generated_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in generated
        ],
        "map_visualization": map_manifest,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("execution", type=Path, help="audited execution JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, nargs="*", default=())
    arguments = parser.parse_args()
    manifest = render_session(
        arguments.execution,
        arguments.output_dir,
        snapshot_paths=arguments.snapshots,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
