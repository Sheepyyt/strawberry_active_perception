#!/usr/bin/env python3
"""Render canonical RGB-D observations as photos, overlays and point clouds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,I202
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_observation(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "schema",
            "scene_id",
            "observation_id",
            "stamp_ns",
            "rgb",
            "depth_m",
            "mask",
            "K",
            "T_world_camera_optical",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"observation snapshot is missing {sorted(missing)}")
        if (
            str(np.asarray(archive["schema"]).item())
            != "strawberry_canonical_observation_snapshot/v1"
        ):
            raise ValueError("unsupported observation snapshot schema")
        result = {name: np.asarray(archive[name]).copy() for name in archive.files}
    rgb = result["rgb"]
    depth = result["depth_m"].astype(np.float32)
    mask = result["mask"]
    K = result["K"].astype(np.float64)
    pose = result["T_world_camera_optical"].astype(np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB snapshot must be HxWx3 uint8")
    if depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
        raise ValueError("RGB, depth and mask grids differ")
    if K.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("camera K or pose has the wrong shape")
    result["path"] = source
    return result


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32)
    foreground = mask > 0
    tint = np.zeros_like(result)
    tint[..., 1] = 255.0
    tint[..., 2] = 110.0
    result[foreground] = 0.48 * result[foreground] + 0.52 * tint[foreground]
    return np.clip(result, 0, 255).astype(np.uint8)


def _point_cloud(
    observation: dict[str, Any], stride: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = observation["depth_m"]
    mask = observation["mask"] > 0
    K = observation["K"]
    row_indices = np.arange(0, depth.shape[0], stride)
    column_indices = np.arange(0, depth.shape[1], stride)
    vv, uu = np.meshgrid(row_indices, column_indices, indexing="ij")
    zz = depth[::stride, ::stride]
    valid = np.isfinite(zz)
    x = (uu - K[0, 2]) / K[0, 0] * zz
    y = (vv - K[1, 2]) / K[1, 1] * zz
    points = np.stack((x, y, zz), axis=-1)[valid]
    colors = observation["rgb"][::stride, ::stride][valid].astype(np.float32) / 255.0
    target = mask[::stride, ::stride][valid]
    return points, colors, target


def render_observation(
    observation: dict[str, Any], output: Path, *, index: int
) -> Path:
    rgb = observation["rgb"]
    depth = observation["depth_m"]
    mask = observation["mask"]
    finite = np.isfinite(depth)
    low, high = (0.2, 2.5)
    if np.any(finite):
        low, high = np.nanpercentile(depth, (2, 98))
        if high <= low:
            high = low + 0.01
    points, colors, target = _point_cloud(observation)
    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
    ]
    cloud = figure.add_subplot(grid[1, 1], projection="3d")
    axes[0].imshow(rgb)
    axes[0].set_title("Rectified RGB")
    axes[1].imshow(_overlay(rgb, mask))
    axes[1].set_title(f"Strawberry mask overlay · {np.count_nonzero(mask):,} pixels")
    depth_image = axes[2].imshow(depth, cmap="turbo", vmin=low, vmax=high)
    axes[2].set_title("Registered depth (metres)")
    figure.colorbar(depth_image, ax=axes[2], fraction=0.046, pad=0.04, label="m")
    for axis in axes:
        axis.axis("off")
    if points.size:
        context = ~target
        if np.any(context):
            cloud.scatter(
                points[context, 0],
                points[context, 2],
                -points[context, 1],
                c=colors[context],
                s=1,
                alpha=0.12,
            )
        if np.any(target):
            cloud.scatter(
                points[target, 0],
                points[target, 2],
                -points[target, 1],
                c="#ff3158",
                s=11,
                alpha=0.95,
                label="mask target",
            )
    cloud.set_xlabel("camera X / right (m)")
    cloud.set_ylabel("camera Z / forward (m)")
    cloud.set_zlabel("camera -Y / up (m)")
    cloud.set_title("RGB-D point cloud")
    if np.any(target):
        cloud.legend(frameon=False)
    scene = str(np.asarray(observation["scene_id"]).item())
    obs_id = str(np.asarray(observation["observation_id"]).item())
    figure.suptitle(
        f"Observation {index:02d} · {scene}\n{obs_id}",
        fontsize=16,
        weight="bold",
        color="#10233f",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _make_contact_sheet(observations: Sequence[dict[str, Any]], output: Path) -> Path:
    thumbnails: list[Image.Image] = []
    for index, observation in enumerate(observations, start=1):
        image = Image.fromarray(_overlay(observation["rgb"], observation["mask"]))
        image.thumbnail((480, 300), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (500, 340), "white")
        canvas.paste(image, ((500 - image.width) // 2, 18))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (18, 315),
            f"view {index:02d} · mask {np.count_nonzero(observation['mask']):,} px",
            fill=(16, 35, 63),
        )
        thumbnails.append(canvas)
    columns = min(3, len(thumbnails))
    rows = int(np.ceil(len(thumbnails) / columns))
    sheet = Image.new("RGB", (columns * 500, rows * 340), (238, 243, 249))
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * 500, (index // columns) * 340))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    return output


def render_sequence(
    paths: Sequence[str | Path], output_directory: str | Path
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one observation snapshot is required")
    observations = [load_observation(path) for path in paths]
    observations.sort(key=lambda item: int(np.asarray(item["stamp_ns"]).item()))
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rendered = [
        render_observation(
            observation, output / f"observation_{index:03d}.png", index=index
        )
        for index, observation in enumerate(observations, start=1)
    ]
    contact = _make_contact_sheet(
        observations, output / "observation_contact_sheet.png"
    )
    gif_frames = []
    for path in rendered:
        frame = Image.open(path).convert("RGB")
        frame.thumbnail((1100, 700), Image.Resampling.LANCZOS)
        gif_frames.append(frame.copy())
        frame.close()
    gif_path = output / "observation_progress.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1100,
        loop=0,
        optimize=True,
    )
    mask_pixels = [int(np.count_nonzero(item["mask"])) for item in observations]
    valid_mask_pixels = [
        int(np.count_nonzero((item["mask"] > 0) & np.isfinite(item["depth_m"])))
        for item in observations
    ]
    manifest = {
        "schema": "strawberry_observation_visualization/v1",
        "observation_count": len(observations),
        "inputs": [
            {"path": str(item["path"]), "sha256": _hash(item["path"])}
            for item in observations
        ],
        "mask_pixels": mask_pixels,
        "valid_mask_depth_pixels": valid_mask_pixels,
        "rendered": [
            {"path": str(path), "sha256": _hash(path)}
            for path in [*rendered, contact, gif_path]
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            render_sequence(arguments.snapshots, arguments.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
