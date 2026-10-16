"""Exercise empty/replayed observations and variable starts without ROS or hardware."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("demo_initialization", ROOT / "tools/demo/initialization.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_empty_mask_is_a_readiness_failure_with_json_safe_values():
    result = MODULE.analyze_target(np.full((30, 30), np.nan), np.zeros((30, 30), dtype=np.uint8), np.eye(3))
    assert result["passed"] is False
    assert result["valid_depth_ratio_in_mask"] == 0.0
    assert result["median_target_depth_m"] is None
    assert result["bbox_xyxy"] is None


def test_nearest_supported_target_layer_is_used_instead_of_majority_background():
    depth = np.full((30, 30), 1.5)
    depth[:10, :20] = 0.45
    mask = np.full((30, 30), 255, dtype=np.uint8)
    intrinsic = np.array([[500, 0, 15], [0, 500, 15], [0, 0, 1]])
    result = MODULE.analyze_target(depth, mask, intrinsic)
    assert result["passed"] is True
    assert result["median_target_depth_m"] == pytest.approx(0.45)
    assert result["retained_target_pixels"] == 200
    assert result["depth_layer_count"] == 2


def test_depth_missing_target_does_not_pass_just_from_large_mask():
    depth = np.full((30, 30), np.nan)
    depth[:4, :20] = 0.45
    result = MODULE.analyze_target(depth, np.full((30, 30), 255, dtype=np.uint8), np.eye(3))
    assert not result["passed"]
    assert result["valid_depth_pixels_in_mask"] == 80


def test_arbitrary_joint_start_is_reordered_and_limits_are_checked():
    names = list(reversed(MODULE.NERO_JOINT_NAMES))
    positions = [0.1, 0.1, 0.1, 1.5, 0.1, -1.2, 0.1]
    result = MODULE.analyze_joints(names, positions)
    assert result["passed"]
    assert result["positions_rad"] == list(reversed(positions))
    positions[0] = 2.0
    failed = MODULE.analyze_joints(names, positions)
    assert not failed["passed"]
    assert failed["limiting_joint"] == "joint7"
    assert failed["minimum_margin_rad"] < 0


def test_missing_joint_is_rejected():
    with pytest.raises(ValueError, match="joint1 through joint7"):
        MODULE.analyze_joints(["joint1"], [0.0])


def test_stale_latched_or_wrong_yolo_result_is_rejected():
    message = SimpleNamespace(scene_id="new", observation_id="capture42", source_name="gemini|mask=yolo11m:abc", header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=1)))
    assert MODULE.observation_matches(message, "new", "capture42", 10_000_000_000)
    assert not MODULE.observation_matches(message, "old", "capture42", 10_000_000_000)
    assert not MODULE.observation_matches(message, "new", "capture41", 10_000_000_000)
    assert not MODULE.observation_matches(message, "new", "capture42", 11_000_000_000)
    message.source_name = "gemini"
    assert not MODULE.observation_matches(message, "new", "capture42", 10_000_000_000)
