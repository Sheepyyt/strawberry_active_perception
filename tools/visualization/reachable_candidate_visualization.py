#!/usr/bin/env python3
"""Render the audited finite NBV/IK candidate lattice from a preview JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _position(matrix: Any, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    return value[:3, 3]


def render_candidate_lattice(document: dict[str, Any], output: Path) -> dict[str, int]:
    """Save a 3-D candidate plot and return auditable category counts."""
    candidates = document.get("ik_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("preview has no ik_candidates list")
    current = _position(
        document.get("final_tf", {}).get("T_base_camera_optical"),
        "current camera",
    )
    selected_matrix = document.get("selected_candidate", {}).get("T_base_camera")
    selected = _position(selected_matrix, "selected camera")
    target = np.asarray(
        document.get("final_target", {}).get("base_xyz_m"), dtype=float
    )
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("preview target centre is invalid")

    categories: dict[str, list[np.ndarray]] = {
        "IK rejected": [],
        "reachable, no gain": [],
        "reachable + useful": [],
    }
    for index, candidate in enumerate(candidates):
        point = _position(candidate.get("T_base_camera"), f"candidate {index}")
        if not candidate.get("ik_gate_passed", False):
            category = "IK rejected"
        elif candidate.get("gain_scored") and float(
            candidate.get("gain_improvement", 0.0)
        ) > 0.0:
            category = "reachable + useful"
        else:
            category = "reachable, no gain"
        categories[category].append(point)

    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    styles = {
        "IK rejected": ("#a9a9a9", "x", 18),
        "reachable, no gain": ("#f2a541", ".", 22),
        "reachable + useful": ("#3274a1", "o", 24),
    }
    for label, points in categories.items():
        if not points:
            continue
        values = np.stack(points)
        color, marker, size = styles[label]
        axis.scatter(
            values[:, 0], values[:, 1], values[:, 2],
            c=color, marker=marker, s=size, alpha=0.75, label=label,
        )
    axis.scatter(*current, c="black", marker="s", s=90, label="current camera")
    axis.scatter(*selected, c="#20a64a", marker="*", s=220, label="selected")
    axis.scatter(*target, c="#d62728", marker="X", s=120, label="target centre")
    axis.plot(
        (current[0], selected[0]),
        (current[1], selected[1]),
        (current[2], selected[2]),
        color="#20a64a",
        linewidth=2.5,
    )
    all_points = np.vstack(
        [current, selected, target]
        + [np.stack(points) for points in categories.values() if points]
    )
    centre = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    half = max(0.01, 0.55 * float(np.max(np.ptp(all_points, axis=0))))
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_xlabel("base X (m)")
    axis.set_ylabel("base Y (m)")
    axis.set_zlabel("base Z (m)")
    axis.set_title("Reachability-aware NBV candidates")
    axis.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {label: len(points) for label, points in categories.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="large-step preview JSON")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    document = json.loads(arguments.input.read_text(encoding="utf-8"))
    counts = render_candidate_lattice(document, arguments.output)
    print(json.dumps({"output": str(arguments.output), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
