from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from validation.week7.observation_recorder import (
    observation_arrays,
    save_observation_snapshot,
)
from validation.week7.observation_visualization import render_sequence
from validation.week7.session_visualization import render_session


def _stamp(seconds: int, nanoseconds: int = 0) -> SimpleNamespace:
    return SimpleNamespace(sec=seconds, nanosec=nanoseconds)


def _image(array: np.ndarray, encoding: str, stamp: SimpleNamespace) -> SimpleNamespace:
    channels = 3 if array.ndim == 3 else 1
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id="camera_color_optical_frame"),
        height=array.shape[0],
        width=array.shape[1],
        encoding=encoding,
        is_bigendian=b"\x00",
        step=array.shape[1] * channels * array.dtype.itemsize,
        data=array.tobytes(),
    )


def _observation() -> SimpleNamespace:
    stamp = _stamp(42, 10)
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[7:17, 12:22] = (220, 30, 40)
    depth = np.full((24, 32), 0.7, dtype=np.float32)
    depth[:2] = np.nan
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[7:17, 12:22] = 255
    pose = SimpleNamespace(
        position=SimpleNamespace(x=0.1, y=-0.2, z=0.3),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id="camera_color_optical_frame"),
        scene_id="fixture_scene",
        observation_id="fixture_observation",
        source_type=b"\x01",
        source_name="fixture_camera",
        color=_image(rgb, "rgb8", stamp),
        depth=_image(depth, "32FC1", stamp),
        target_mask=_image(mask, "mono8", stamp),
        camera_info=SimpleNamespace(
            k=[30.0, 0.0, 15.5, 0.0, 30.0, 11.5, 0.0, 0.0, 1.0],
            d=[0.0] * 8,
        ),
        camera_pose=SimpleNamespace(
            header=SimpleNamespace(stamp=stamp, frame_id="base_link"),
            pose=pose,
        ),
        pose_valid=True,
        valid_depth_fraction=22 / 24,
        color_depth_skew_sec=0.0,
    )


def _pose(x: float, y: float, z: float) -> list[list[float]]:
    result = np.eye(4)
    result[:3, 3] = (x, y, z)
    return result.tolist()


def _candidate(x: float, useful: bool, reachable: bool = True) -> dict:
    return {
        "T_base_camera": _pose(x, 0.1, 0.5),
        "ik_gate_passed": reachable,
        "gain_scored": reachable,
        "gain_improvement": 0.01 if useful else 0.0,
    }


def _execution() -> dict:
    steps = []
    coverages = ((0.10, 0.14), (0.14, 0.21))
    for index, (before, after) in enumerate(coverages, start=1):
        target = _pose(index * 0.05, 0.1, 0.5)
        step = {
            "step_index": index,
            "planned_start_camera": _pose((index - 1) * 0.05, 0.1, 0.5),
            "T_base_camera": target,
            "selected_candidate": {"T_base_camera": target},
            "ik_candidates": [
                _candidate(index * 0.05, True),
                _candidate(index * 0.04, False),
                _candidate(index * 0.06, False, reachable=False),
            ],
            "coverage": {"before": before, "after": after, "delta": after - before},
            "cumulative_camera_motion": {
                "planned_step_translation_m": 0.05,
                "actual_step_translation_m": 0.049,
                "planned_step_rotation_deg": 4.0,
                "actual_step_rotation_deg": 3.9,
            },
            "post_pose_error": {"translation_m": 0.0012, "rotation_deg": 0.04},
            "fresh_execution_ik": {
                "sigma_min": 0.18,
                "condition_number": 9.5,
                "independent_max_joint_delta_rad": 0.21,
            },
            "post_target": {"mask_pixels": 500, "valid_mask_pixels": 420},
            "post_map_update": {
                "T_base_camera_raw": target,
                "voxel_counts": {"observed": 10000 * index, "occupied": 500 * index},
            },
            "motion_command_count_step": 55 + index,
        }
        steps.append(step)
    return {
        "status": "executed_session_scientific_acceptance_passed",
        "final_target": {"base_xyz_m": [0.0, 0.0, 0.5]},
        "final_tf": {"T_base_camera_optical": _pose(0.0, 0.1, 0.5)},
        "ik_candidates": steps[0]["ik_candidates"],
        "motion_steps": steps,
        "session_progress": {
            "initial_coverage": 0.10,
            "final_coverage": 0.21,
            "coverage_target": 0.20,
            "convergence_reason": "coverage target 0.200000 reached",
        },
    }


def test_observation_recording_and_visualization(tmp_path: Path) -> None:
    arrays = observation_arrays(_observation())
    assert arrays["rgb"].shape == (24, 32, 3)
    assert arrays["mask"].sum() == 100 * 255
    assert np.count_nonzero(np.isfinite(arrays["depth_m"])) == 22 * 32
    assert arrays["T_world_camera_optical"][0, 3] == 0.1
    snapshot = save_observation_snapshot(arrays, tmp_path / "observation.npz")
    manifest = render_sequence([snapshot], tmp_path / "observation_figures")
    assert manifest["observation_count"] == 1
    assert (tmp_path / "observation_figures/observation_001.png").is_file()
    assert (tmp_path / "observation_figures/observation_contact_sheet.png").is_file()
    assert (tmp_path / "observation_figures/observation_progress.gif").is_file()


def test_session_dashboard_creates_all_presentation_views(tmp_path: Path) -> None:
    execution = tmp_path / "execution.json"
    import json

    execution.write_text(json.dumps(_execution()), encoding="utf-8")
    manifest = render_session(execution, tmp_path / "dashboard")
    assert manifest["motion_goal_count"] == 2
    assert manifest["candidate_round_count"] == 2
    for name in (
        "00_experiment_dashboard.png",
        "01_coverage_curve.png",
        "02_camera_trajectory_3d.png",
        "03_candidate_funnel.png",
        "04_motion_profile.png",
        "05_accuracy_safety.png",
        "06_target_map_quality.png",
        "candidate_step_001.png",
        "candidate_step_002.png",
        "REPORT_CN.md",
        "index.html",
        "manifest.json",
    ):
        assert (tmp_path / "dashboard" / name).is_file(), name


def test_observation_recorder_has_no_command_surface() -> None:
    source = Path("validation/week7/observation_recorder.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "create_subscription" in attributes
    assert "create_publisher" not in attributes
    assert "create_client" not in attributes
    assert "create_service" not in attributes
    assert "create_action_client" not in attributes
