from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perception_ws/src/strawberry_gradient_nbv"))

from strawberry_gradient_nbv.map_visualization import save_map_snapshot  # noqa: E402
from tools.visualization.voxel_cloud_visualization import render_bundle  # noqa: E402


def _snapshot(tmp_path: Path, observation_id: str = "view_001") -> Path:
    shape = (10, 10, 10)
    observed = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    target = np.zeros(shape, dtype=bool)
    observed[2:8, 2:8, 2:8] = True
    occupied[4:7, 4:7, 5] = True
    target[5:7, 5:7, 5] = True
    pose = np.eye(4)
    pose[:3, 3] = (0.1, 0.1, 0.2)
    return save_map_snapshot(
        tmp_path / f"{observation_id}.npz",
        {
            "dimensions": shape,
            "origin_m": (0.0, 0.0, 0.0),
            "voxel_size_m": 0.01,
            "target_center_m": (0.05, 0.05, 0.05),
            "target_roi_size_m": (0.04, 0.04, 0.04),
            "observed": observed,
            "occupied": occupied,
            "target": target,
        },
        scene_id="fixture",
        observation_id=observation_id,
        world_frame="base_link",
        coverage=0.25,
        current_camera_pose=pose,
        next_camera_pose=pose,
        camera_pose_history=np.stack((pose,)),
    )


def test_voxel_cloud_bundle_is_created_from_audited_snapshot(tmp_path: Path) -> None:
    manifest = render_bundle([_snapshot(tmp_path)], tmp_path / "figures")
    assert manifest["scene_id"] == "fixture"
    assert manifest["coverage"] == [0.25]
    assert manifest["voxel_counts"][0]["target"] == 4
    for name in (
        "voxel_cloud_final_3d.png",
        "voxel_cloud_growth.gif",
        "voxel_cloud_spin.gif",
        "voxel_cloud_manifest.json",
    ):
        assert (tmp_path / "figures" / name).is_file()


@pytest.mark.parametrize("observation_count", [1, 2])
def test_compact_voxel_cloud_omits_redundant_animations(tmp_path: Path, observation_count: int) -> None:
    snapshots = [_snapshot(tmp_path, f"view_{index:03d}") for index in range(observation_count)]
    output = tmp_path / "compact"
    manifest = render_bundle(snapshots, output, compact=True)
    expected = {"voxel_cloud_final_3d.png", "voxel_cloud_manifest.json"}
    if observation_count > 1:
        expected.add("voxel_cloud_growth.gif")
    assert {path.name for path in output.iterdir()} == expected
    assert manifest["render_mode"] == "compact"
    assert len(manifest["inputs"]) == observation_count
    assert len(manifest["outputs"]) == observation_count
