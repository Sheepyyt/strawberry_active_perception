#!/usr/bin/env python3
"""Render presentation-ready 3-D voxel clouds from audited NBV snapshots.

This module is deliberately offline.  It reads immutable NPZ map snapshots,
creates no ROS entity, and has no robot or camera command surface.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,I202
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from strawberry_gradient_nbv.map_visualization import load_map_snapshot  # noqa: E402


NAVY = "#10233f"
BLUE = "#3694e6"
GREEN = "#19a875"
RED = "#e45151"
CHARCOAL = "#30343b"
ORANGE = "#f39b35"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(points: np.ndarray, limit: int) -> np.ndarray:
    """Deterministically thin a cloud without hiding either end of it."""
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def _voxel_centres(snapshot: Mapping[str, Any], mask: np.ndarray) -> np.ndarray:
    indices = np.argwhere(mask)
    return (
        np.asarray(snapshot["origin_m"], dtype=float)
        + (indices.astype(float) + 0.5) * float(snapshot["voxel_size_m"])
    )


def _clouds(snapshot: Mapping[str, Any]) -> dict[str, np.ndarray]:
    target_mask = np.asarray(snapshot["target"], dtype=bool)
    occupied_mask = np.asarray(snapshot["occupied"], dtype=bool) & ~target_mask
    observed_mask = (
        np.asarray(snapshot["observed"], dtype=bool)
        & ~occupied_mask
        & ~target_mask
    )
    return {
        "observed": _sample(_voxel_centres(snapshot, observed_mask), 18000),
        "occupied": _sample(_voxel_centres(snapshot, occupied_mask), 12000),
        "target": _sample(_voxel_centres(snapshot, target_mask), 8000),
    }


def _draw_box(axis: plt.Axes, centre: np.ndarray, size: np.ndarray) -> None:
    low = centre - 0.5 * size
    high = centre + 0.5 * size
    corners = np.asarray(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )
    edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    for first, second in edges:
        axis.plot(*corners[[first, second]].T, color=GREEN, lw=1.6, alpha=0.85)


def _set_bounds(axis: plt.Axes, snapshot: Mapping[str, Any]) -> None:
    low = np.asarray(snapshot["origin_m"], dtype=float)
    high = low + np.asarray(snapshot["dimensions"], dtype=float) * float(
        snapshot["voxel_size_m"]
    )
    centre = 0.5 * (low + high)
    half = 0.53 * float(np.max(high - low))
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_box_aspect((1, 1, 1))


def _draw_cloud_scene(
    axis: plt.Axes,
    snapshot: Mapping[str, Any],
    *,
    elevation: float = 24.0,
    azimuth: float = -56.0,
) -> None:
    clouds = _clouds(snapshot)
    observed = clouds["observed"]
    occupied = clouds["occupied"]
    target = clouds["target"]
    if len(observed):
        axis.scatter(*observed.T, s=1.2, c=BLUE, alpha=0.075, edgecolors="none", label="observed free/unknown boundary")
    if len(occupied):
        axis.scatter(*occupied.T, s=5, c=CHARCOAL, alpha=0.38, edgecolors="none", label="measured surface")
    if len(target):
        axis.scatter(*target.T, s=13, c=RED, alpha=0.9, edgecolors="none", label="mask-supported target")
    history = np.asarray(snapshot["camera_pose_history"], dtype=float)
    positions = history[:, :3, 3]
    axis.plot(*positions.T, "o-", color=GREEN, lw=3, ms=5, label="camera path")
    for index, pose in enumerate(history):
        origin = pose[:3, 3]
        optical_z = pose[:3, 2]
        axis.quiver(*origin, *optical_z, length=0.028, normalize=True,
                    color=ORANGE, arrow_length_ratio=0.28)
        axis.text(*origin, f" {index + 1}", color=NAVY, fontsize=8, weight="bold")
    target_centre = np.asarray(snapshot["target_center_m"], dtype=float)
    axis.scatter(*target_centre, marker="X", s=130, c=RED, edgecolors="white", linewidths=0.8)
    _draw_box(
        axis,
        target_centre,
        np.asarray(snapshot["target_roi_size_m"], dtype=float),
    )
    _set_bounds(axis, snapshot)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlabel("base X (m)")
    axis.set_ylabel("base Y (m)")
    axis.set_zlabel("base Z (m)")
    axis.grid(True, alpha=0.18)


def _counts(snapshot: Mapping[str, Any]) -> tuple[int, int, int]:
    return tuple(
        int(np.count_nonzero(snapshot[name]))
        for name in ("observed", "occupied", "target")
    )


def render_final_dashboard(snapshot: Mapping[str, Any], output: Path) -> Path:
    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(2.25, 1, 1))
    axis = figure.add_subplot(grid[:, :2], projection="3d")
    _draw_cloud_scene(axis, snapshot)
    axis.set_title("3-D voxel map after active observation", fontsize=18, color=NAVY, weight="bold")
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc="upper left", fontsize=8, frameon=True)

    count_axis = figure.add_subplot(grid[0, 2])
    count_axis.spines[["top", "right"]].set_visible(False)
    observed, occupied, target = _counts(snapshot)
    names = ("Observed", "Surface", "Target")
    values = (observed, occupied, target)
    bars = count_axis.bar(names, values, color=(BLUE, CHARCOAL, RED))
    count_axis.set_title("What the map contains", color=NAVY, weight="bold")
    count_axis.set_ylabel("voxel count")
    count_axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        count_axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                        ha="center", va="bottom", fontsize=9)

    info = figure.add_subplot(grid[1, 2])
    info.axis("off")
    info.text(0, 0.94, "AUDITED MAP", color=GREEN, fontsize=11, weight="bold")
    rows = (
        ("Coverage", f"{float(snapshot['coverage']) * 100:.2f}%"),
        ("Views fused", str(len(snapshot["camera_pose_history"]))),
        ("Voxel size", f"{float(snapshot['voxel_size_m']) * 1000:.1f} mm"),
        ("Grid", " × ".join(str(int(x)) for x in snapshot["dimensions"])),
        ("Scene", str(snapshot["scene_id"])),
    )
    y = 0.78
    for label, value in rows:
        info.text(0, y, label.upper(), color="#8190a3", fontsize=8, weight="bold")
        info.text(0, y - 0.09, value, color=NAVY, fontsize=13, weight="bold", wrap=True)
        y -= 0.18
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _scene_frame(snapshot: Mapping[str, Any], elevation: float, azimuth: float) -> Image.Image:
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    _draw_cloud_scene(axis, snapshot, elevation=elevation, azimuth=azimuth)
    axis.set_title(
        f"Coverage {float(snapshot['coverage']) * 100:.2f}% · {len(snapshot['camera_pose_history'])} view(s)",
        color=NAVY,
        weight="bold",
    )
    buffer = BytesIO()
    # Keep every animation frame on the same fixed canvas.  A tight bounding
    # box changes with the 3-D azimuth and produces unequal GIF frame sizes.
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=105, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def render_growth(snapshots: Sequence[Mapping[str, Any]], output: Path) -> Path:
    frames = [_scene_frame(snapshot, 24.0, -56.0) for snapshot in snapshots]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1200,
                   loop=0, optimize=True)
    return output


def render_spin(snapshot: Mapping[str, Any], output: Path) -> Path:
    frames = [_scene_frame(snapshot, 24.0, float(angle)) for angle in range(-75, 286, 20)]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=130,
                   loop=0, optimize=True)
    return output


def render_bundle(snapshot_paths: Sequence[str | Path], output_directory: str | Path) -> dict[str, Any]:
    if not snapshot_paths:
        raise ValueError("at least one snapshot is required")
    paths = [Path(path).expanduser().resolve() for path in snapshot_paths]
    snapshots = [load_map_snapshot(path) for path in paths]
    scenes = {snapshot["scene_id"] for snapshot in snapshots}
    if len(scenes) != 1:
        raise ValueError("all snapshots must belong to the same scene")
    output = Path(output_directory).expanduser().resolve()
    generated = (
        render_final_dashboard(snapshots[-1], output / "voxel_cloud_final_3d.png"),
        render_growth(snapshots, output / "voxel_cloud_growth.gif"),
        render_spin(snapshots[-1], output / "voxel_cloud_spin.gif"),
    )
    manifest = {
        "schema": "strawberry_voxel_cloud_visualization/v1",
        "scene_id": snapshots[-1]["scene_id"],
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in paths],
        "coverage": [float(snapshot["coverage"]) for snapshot in snapshots],
        "voxel_counts": [dict(zip(("observed", "occupied", "target"), _counts(snapshot))) for snapshot in snapshots],
        "outputs": [{"path": str(path), "sha256": _sha256(path)} for path in generated],
    }
    manifest_path = output / "voxel_cloud_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    print(json.dumps(render_bundle(arguments.snapshots, arguments.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
