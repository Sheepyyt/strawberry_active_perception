"""Portable Gradient-NBV voxel snapshots and dependency-light rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


SNAPSHOT_SCHEMA = "strawberry_gradient_nbv_map/v1"
_COLORS = {
    "unknown": (238, 238, 238),
    "observed": (92, 166, 230),
    "occupied": (70, 70, 70),
    "target": (225, 55, 55),
    "roi": (35, 150, 80),
    "camera": (20, 175, 90),
    "next": (245, 170, 20),
}


def _json_compatible(value: Any) -> Any:
    """Convert NumPy-heavy configuration data to deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"configuration contains unsupported type {type(value).__name__}")


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def _rigid_pose(value: Any, name: str) -> np.ndarray:
    pose = _array(value, (4, 4), name).astype(np.float64, copy=True)
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return pose


def _validate_state(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "dimensions",
        "origin_m",
        "voxel_size_m",
        "target_center_m",
        "target_roi_size_m",
        "observed",
        "occupied",
        "target",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"map state is missing {sorted(missing)}")
    dimensions_array = _array(state["dimensions"], (3,), "dimensions").astype(
        np.int64, copy=True
    )
    if np.any(dimensions_array <= 0):
        raise ValueError("dimensions must be positive")
    dimensions = tuple(int(value) for value in dimensions_array)
    voxel_size = float(np.asarray(state["voxel_size_m"]).item())
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    output = {
        "dimensions": dimensions_array.astype(np.int32),
        "origin_m": _array(state["origin_m"], (3,), "origin_m").astype(
            np.float64, copy=True
        ),
        "voxel_size_m": np.asarray(voxel_size, dtype=np.float64),
        "target_center_m": _array(
            state["target_center_m"], (3,), "target_center_m"
        ).astype(np.float64, copy=True),
        "target_roi_size_m": _array(
            state["target_roi_size_m"], (3,), "target_roi_size_m"
        ).astype(np.float64, copy=True),
    }
    for name in ("observed", "occupied", "target"):
        output[name] = _array(state[name], dimensions, name).astype(bool, copy=True)
    if not np.all(np.isfinite(output["origin_m"])):
        raise ValueError("origin_m must be finite")
    if not np.all(np.isfinite(output["target_center_m"])):
        raise ValueError("target_center_m must be finite")
    if np.any(output["target_roi_size_m"] <= 0.0):
        raise ValueError("target_roi_size_m must be positive")
    if np.any(output["occupied"] & ~output["observed"]):
        raise ValueError("occupied voxels must also be observed")
    if np.any(output["target"] & ~output["observed"]):
        raise ValueError("target voxels must also be observed")
    return output


def save_map_snapshot(
    path: str | Path,
    state: Mapping[str, Any],
    *,
    scene_id: str,
    observation_id: str,
    world_frame: str,
    coverage: float,
    current_camera_pose: np.ndarray,
    next_camera_pose: np.ndarray,
    camera_pose_history: np.ndarray,
    configuration: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save one complete categorical map state without pickle."""
    clean = _validate_state(state)
    coverage_value = float(coverage)
    if not math.isfinite(coverage_value) or not 0.0 <= coverage_value <= 1.0:
        raise ValueError("coverage must be finite and in [0, 1]")
    current = _rigid_pose(current_camera_pose, "current_camera_pose")
    planned = _rigid_pose(next_camera_pose, "next_camera_pose")
    history = np.asarray(camera_pose_history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (4, 4) or history.shape[0] < 1:
        raise ValueError("camera_pose_history must be a non-empty Nx4x4 array")
    for index, pose in enumerate(history):
        _rigid_pose(pose, f"camera_pose_history[{index}]")
    if not str(scene_id).strip() or not str(observation_id).strip():
        raise ValueError("scene_id and observation_id must be non-empty")
    if not str(world_frame).strip():
        raise ValueError("world_frame must be non-empty")
    configuration_json = json.dumps(
        _json_compatible(dict(configuration or {})),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=output.parent, delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez_compressed(
                handle,
                schema=np.asarray(SNAPSHOT_SCHEMA),
                scene_id=np.asarray(str(scene_id)),
                observation_id=np.asarray(str(observation_id)),
                world_frame=np.asarray(str(world_frame)),
                coverage=np.asarray(coverage_value, dtype=np.float64),
                current_camera_pose=current,
                next_camera_pose=planned,
                camera_pose_history=history,
                configuration_json=np.asarray(configuration_json),
                **clean,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output


def load_map_snapshot(path: str | Path) -> dict[str, Any]:
    """Load and validate a snapshot with pickle explicitly disabled."""
    snapshot_path = Path(path).expanduser().resolve()
    with np.load(snapshot_path, allow_pickle=False) as archive:
        required = {
            "schema",
            "scene_id",
            "observation_id",
            "world_frame",
            "coverage",
            "current_camera_pose",
            "next_camera_pose",
            "camera_pose_history",
            "configuration_json",
            "dimensions",
            "origin_m",
            "voxel_size_m",
            "target_center_m",
            "target_roi_size_m",
            "observed",
            "occupied",
            "target",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"snapshot is missing {sorted(missing)}")
        if str(np.asarray(archive["schema"]).item()) != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported map snapshot schema")
        state_names = {
            "dimensions",
            "origin_m",
            "voxel_size_m",
            "target_center_m",
            "target_roi_size_m",
            "observed",
            "occupied",
            "target",
        }
        state = _validate_state(
            {name: archive[name] for name in required if name in state_names}
        )
        result: dict[str, Any] = {
            **state,
            "path": snapshot_path,
            "scene_id": str(np.asarray(archive["scene_id"]).item()),
            "observation_id": str(np.asarray(archive["observation_id"]).item()),
            "world_frame": str(np.asarray(archive["world_frame"]).item()),
            "coverage": float(np.asarray(archive["coverage"]).item()),
            "current_camera_pose": _rigid_pose(
                archive["current_camera_pose"], "current_camera_pose"
            ),
            "next_camera_pose": _rigid_pose(
                archive["next_camera_pose"], "next_camera_pose"
            ),
            "camera_pose_history": np.asarray(
                archive["camera_pose_history"], dtype=np.float64
            ).copy(),
            "configuration": json.loads(
                str(np.asarray(archive["configuration_json"]).item())
            ),
        }
    if not 0.0 <= result["coverage"] <= 1.0:
        raise ValueError("snapshot coverage is outside [0, 1]")
    return result


def _project_categories(snapshot: Mapping[str, Any], axis: int) -> np.ndarray:
    observed = np.any(snapshot["observed"], axis=axis)
    occupied = np.any(snapshot["occupied"], axis=axis)
    target = np.any(snapshot["target"], axis=axis)
    categorical = np.zeros(observed.shape, dtype=np.uint8)
    categorical[observed] = 1
    categorical[occupied] = 2
    categorical[target] = 3
    # Array dimensions are [x,y,z]. The remaining first coordinate should be
    # horizontal and the second vertical, with positive vertical drawn upward.
    return np.flip(categorical.T, axis=0)


def _category_image(categorical: np.ndarray, size: int) -> Image.Image:
    palette = np.asarray(
        [
            _COLORS["unknown"],
            _COLORS["observed"],
            _COLORS["occupied"],
            _COLORS["target"],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(palette[categorical], mode="RGB")
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def _world_to_panel(
    point: np.ndarray,
    snapshot: Mapping[str, Any],
    axes: tuple[int, int],
    left: int,
    top: int,
    size: int,
) -> tuple[int, int]:
    indices = (
        (np.asarray(point, dtype=np.float64) - snapshot["origin_m"])
        / float(snapshot["voxel_size_m"])
    )
    dimensions = np.asarray(snapshot["dimensions"], dtype=np.float64)
    x = left + int(round(float(indices[axes[0]] / dimensions[axes[0]]) * size))
    y = top + size - int(
        round(float(indices[axes[1]] / dimensions[axes[1]]) * size)
    )
    return x, y


def _draw_roi(
    draw: ImageDraw.ImageDraw,
    snapshot: Mapping[str, Any],
    axes: tuple[int, int],
    left: int,
    top: int,
    size: int,
) -> None:
    centre = np.asarray(snapshot["target_center_m"], dtype=np.float64)
    half = np.asarray(snapshot["target_roi_size_m"], dtype=np.float64) / 2.0
    minimum = centre - half
    maximum = centre + half
    first = _world_to_panel(minimum, snapshot, axes, left, top, size)
    second = _world_to_panel(maximum, snapshot, axes, left, top, size)
    x0, x1 = sorted((first[0], second[0]))
    y0, y1 = sorted((first[1], second[1]))
    draw.rectangle((x0, y0, x1, y1), outline=_COLORS["roi"], width=2)


def render_snapshot(
    snapshot: Mapping[str, Any],
    coverage_history: Sequence[float],
    *,
    width: int = 1280,
    height: int = 620,
) -> Image.Image:
    """Render three orthographic map views plus the coverage curve."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panel_size = 330
    panel_top = 60
    panels = (
        ("Top: X-Y", 2, (0, 1)),
        ("Front: X-Z", 1, (0, 2)),
        ("Side: Y-Z", 0, (1, 2)),
    )
    lefts = (35, 445, 855)
    history = np.asarray(snapshot["camera_pose_history"], dtype=np.float64)
    for (label, collapse_axis, axes), left in zip(panels, lefts):
        panel = _category_image(
            _project_categories(snapshot, collapse_axis), panel_size
        )
        image.paste(panel, (left, panel_top))
        draw.rectangle(
            (left, panel_top, left + panel_size, panel_top + panel_size),
            outline=(110, 110, 110),
            width=1,
        )
        draw.text((left, panel_top - 22), label, fill=(20, 20, 20))
        _draw_roi(draw, snapshot, axes, left, panel_top, panel_size)
        previous: tuple[int, int] | None = None
        for index, pose in enumerate(history):
            point = _world_to_panel(
                pose[:3, 3], snapshot, axes, left, panel_top, panel_size
            )
            if previous is not None:
                draw.line((*previous, *point), fill=_COLORS["camera"], width=2)
            draw.ellipse(
                (point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4),
                fill=_COLORS["camera"],
            )
            draw.text((point[0] + 5, point[1] - 8), str(index + 1), fill=(0, 80, 35))
            previous = point
        planned = _world_to_panel(
            snapshot["next_camera_pose"][:3, 3],
            snapshot,
            axes,
            left,
            panel_top,
            panel_size,
        )
        draw.ellipse(
            (planned[0] - 6, planned[1] - 6, planned[0] + 6, planned[1] + 6),
            outline=_COLORS["next"],
            width=3,
        )

    coverage_values = [float(value) for value in coverage_history]
    chart_left, chart_top = 70, 455
    chart_width, chart_height = 760, 115
    draw.rectangle(
        (chart_left, chart_top, chart_left + chart_width, chart_top + chart_height),
        outline=(120, 120, 120),
    )
    points: list[tuple[int, int]] = []
    for index, value in enumerate(coverage_values):
        x = chart_left + int(
            round(index / max(1, len(coverage_values) - 1) * chart_width)
        )
        y = chart_top + chart_height - int(round(value * chart_height))
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(30, 100, 210), width=3)
    for index, point in enumerate(points):
        draw.ellipse(
            (point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4),
            fill=(30, 100, 210),
        )
        draw.text((point[0] - 6, chart_top + chart_height + 5), str(index + 1), fill=(0, 0, 0))
    draw.text((chart_left, chart_top - 20), "ROI coverage by observation", fill=(0, 0, 0))
    draw.text(
        (860, 445),
        f"Scene: {snapshot['scene_id']}\n"
        f"Observation: {snapshot['observation_id']}\n"
        f"Coverage: {snapshot['coverage'] * 100.0:.2f}%\n"
        f"Observed voxels: {int(np.count_nonzero(snapshot['observed'])):,}\n"
        f"Occupied voxels: {int(np.count_nonzero(snapshot['occupied'])):,}\n"
        f"Target voxels: {int(np.count_nonzero(snapshot['target'])):,}",
        fill=(20, 20, 20),
        spacing=6,
    )
    legend = (
        ("unknown", "Unknown"),
        ("observed", "Observed free/seen"),
        ("occupied", "Depth surface"),
        ("target", "Mask-supported target"),
        ("camera", "Camera path"),
        ("next", "Planned next view"),
        ("roi", "Target ROI"),
    )
    legend_x = 35
    for key, label in legend:
        draw.rectangle((legend_x, 20, legend_x + 14, 34), fill=_COLORS[key])
        draw.text((legend_x + 19, 20), label, fill=(20, 20, 20))
        legend_x += 155
    return image


def render_snapshot_sequence(
    snapshot_paths: Sequence[str | Path], output_directory: str | Path
) -> dict[str, Any]:
    """Render one PNG per map update, a final PNG, and an animated GIF."""
    if not snapshot_paths:
        raise ValueError("at least one snapshot is required")
    snapshots = [load_map_snapshot(path) for path in snapshot_paths]
    scenes = {snapshot["scene_id"] for snapshot in snapshots}
    if len(scenes) != 1:
        raise ValueError("all snapshots must belong to the same scene")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    coverage_history: list[float] = []
    frames: list[Image.Image] = []
    png_paths: list[Path] = []
    for index, snapshot in enumerate(snapshots, start=1):
        coverage_history.append(float(snapshot["coverage"]))
        frame = render_snapshot(snapshot, coverage_history)
        frame_path = output / f"map_step_{index:03d}.png"
        frame.save(frame_path, format="PNG", optimize=True)
        frames.append(frame)
        png_paths.append(frame_path)
    final_path = output / "nbv_map_final.png"
    frames[-1].save(final_path, format="PNG", optimize=True)
    gif_path = output / "nbv_map_progress.gif"
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )
    manifest = {
        "schema": "strawberry_gradient_nbv_visualization/v1",
        "scene_id": snapshots[0]["scene_id"],
        "snapshot_count": len(snapshots),
        "coverage": coverage_history,
        "snapshot_sha256": [
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in snapshot_paths
        ],
        "step_png": [str(path) for path in png_paths],
        "final_png": str(final_path),
        "animated_gif": str(gif_path),
        "legend": {
            "unknown": "gray: no valid ray has touched this voxel",
            "observed": "blue: seen by at least one valid depth ray",
            "occupied": "dark: a measured depth surface endpoint",
            "target": "red: a mask-supported target endpoint",
            "camera": "green: measured camera path",
            "next": "orange ring: proposed next camera position",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_demo(output_directory: str | Path, device: str | None = None) -> dict[str, Any]:
    """Run the deterministic five-view fixture and render map evolution."""
    from .core import GradientNBVCore
    from .fixtures import make_multiview_fixture

    fixture = make_multiview_fixture()
    core = GradientNBVCore(device=device)
    configuration = {**fixture.config, "scene_id": fixture.scene_id}
    core.configure(configuration)
    output = Path(output_directory).expanduser().resolve()
    snapshot_directory = output / "snapshots"
    camera_history: list[np.ndarray] = []
    paths: list[Path] = []
    for index, observation in enumerate(fixture.observations(), start=1):
        result = core.update_and_plan(
            observation.depth,
            observation.mask,
            observation.K,
            observation.pose,
        )
        camera_history.append(observation.pose.copy())
        path = snapshot_directory / f"demo_step_{index:03d}.npz"
        save_map_snapshot(
            path,
            core.visualization_state(),
            scene_id=fixture.scene_id,
            observation_id=observation.observation_id,
            world_frame=str(configuration["world_frame"]),
            coverage=result.coverage,
            current_camera_pose=observation.pose,
            next_camera_pose=result.pose,
            camera_pose_history=np.stack(camera_history),
            configuration=configuration,
        )
        paths.append(path)
    return render_snapshot_sequence(paths, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save/render human-readable Gradient-NBV voxel-map progress"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic five-view demo")
    demo.add_argument("--output-dir", required=True)
    demo.add_argument("--device", choices=("cpu", "cuda"), default=None)
    render = subparsers.add_parser("render", help="render existing NPZ snapshots")
    render.add_argument("--output-dir", required=True)
    render.add_argument("snapshot", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "demo":
        manifest = generate_demo(arguments.output_dir, arguments.device)
    else:
        manifest = render_snapshot_sequence(arguments.snapshot, arguments.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
