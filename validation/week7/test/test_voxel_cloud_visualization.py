from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perception_ws/src/strawberry_gradient_nbv"))

from strawberry_gradient_nbv.map_visualization import save_map_snapshot  # noqa: E402
from validation.week7.voxel_cloud_visualization import render_bundle  # noqa: E402


def test_voxel_cloud_bundle_is_created_from_audited_snapshot(
    tmp_path: Path,
) -> None:
    shape = (10, 10, 10)
    observed = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    target = np.zeros(shape, dtype=bool)
    observed[2:8, 2:8, 2:8] = True
    occupied[4:7, 4:7, 5] = True
    target[5:7, 5:7, 5] = True
    pose = np.eye(4)
    pose[:3, 3] = (0.1, 0.1, 0.2)
    snapshot = save_map_snapshot(
        tmp_path / "map.npz",
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
        observation_id="view_001",
        world_frame="base_link",
        coverage=0.25,
        current_camera_pose=pose,
        next_camera_pose=pose,
        camera_pose_history=np.stack((pose,)),
    )
    manifest = render_bundle([snapshot], tmp_path / "figures")
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
