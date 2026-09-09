"""Tests for portable voxel snapshots and their human-readable rendering."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from strawberry_gradient_nbv.map_visualization import (
    load_map_snapshot,
    render_snapshot_sequence,
    save_map_snapshot,
)


def _state() -> dict[str, np.ndarray]:
    observed = np.zeros((5, 4, 3), dtype=bool)
    observed[1:4, 1:3, :] = True
    occupied = np.zeros_like(observed)
    occupied[2, 2, 1] = True
    target = np.zeros_like(observed)
    target[3, 1, 1] = True
    return {
        "dimensions": np.asarray(observed.shape, dtype=np.int32),
        "origin_m": np.array([-0.05, -0.04, 0.30]),
        "voxel_size_m": np.asarray(0.02),
        "target_center_m": np.array([0.0, 0.0, 0.33]),
        "target_roi_size_m": np.array([0.06, 0.06, 0.04]),
        "observed": observed,
        "occupied": occupied,
        "target": target,
    }


def _pose(x: float = 0.0) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = x
    return result


def test_snapshot_round_trip_is_pickle_free_and_copies_configuration(tmp_path) -> None:
    path = save_map_snapshot(
        tmp_path / "step.npz",
        _state(),
        scene_id="scene/a",
        observation_id="obs:1",
        world_frame="world",
        coverage=0.4,
        current_camera_pose=_pose(),
        next_camera_pose=_pose(0.01),
        camera_pose_history=np.stack((_pose(),)),
        configuration={"target": np.array([1.0, 2.0, 3.0])},
    )
    loaded = load_map_snapshot(path)
    assert loaded["coverage"] == pytest.approx(0.4)
    assert loaded["configuration"] == {"target": [1.0, 2.0, 3.0]}
    assert np.array_equal(loaded["target"], _state()["target"])
    with np.load(path, allow_pickle=False) as archive:
        assert not any(archive[name].dtype == object for name in archive.files)


def test_invalid_category_subset_is_rejected(tmp_path) -> None:
    state = _state()
    state["occupied"][0, 0, 0] = True
    with pytest.raises(ValueError, match="occupied voxels"):
        save_map_snapshot(
            tmp_path / "invalid.npz",
            state,
            scene_id="scene",
            observation_id="obs",
            world_frame="world",
            coverage=0.0,
            current_camera_pose=_pose(),
            next_camera_pose=_pose(),
            camera_pose_history=np.stack((_pose(),)),
        )
    assert not (tmp_path / "invalid.npz").exists()


def test_sequence_renderer_writes_png_gif_and_manifest(tmp_path) -> None:
    snapshots: list[Path] = []
    for index, coverage in enumerate((0.2, 0.45), start=1):
        snapshots.append(
            save_map_snapshot(
                tmp_path / f"snapshot_{index}.npz",
                _state(),
                scene_id="scene",
                observation_id=f"obs-{index}",
                world_frame="world",
                coverage=coverage,
                current_camera_pose=_pose(index * 0.005),
                next_camera_pose=_pose((index + 1) * 0.005),
                camera_pose_history=np.stack(
                    tuple(_pose(step * 0.005) for step in range(1, index + 1))
                ),
            )
        )
    output = tmp_path / "rendered"
    manifest = render_snapshot_sequence(snapshots, output)
    assert manifest["coverage"] == [0.2, 0.45]
    assert len(manifest["snapshot_sha256"]) == 2
    assert (output / "manifest.json").is_file()
    assert (output / "nbv_map_final.png").is_file()
    assert (output / "nbv_map_progress.gif").is_file()
    with Image.open(output / "nbv_map_progress.gif") as animation:
        assert animation.n_frames == 2
