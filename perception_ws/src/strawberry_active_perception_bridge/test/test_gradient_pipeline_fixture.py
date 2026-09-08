"""Tests for the full synthetic Gradient-NBV to Placo fixture payload."""

from builtin_interfaces.msg import Time
import numpy as np
import pytest

from strawberry_active_perception_bridge.gradient_pipeline_fixture import (
    _configure_request,
    _diagnostic_boolean,
    make_pipeline_observation,
)


def test_pipeline_observation_has_one_canonical_grid_and_pose() -> None:
    current = np.eye(4)
    observation, target = make_pipeline_observation(current, Time(sec=7))
    assert observation.color.encoding == "rgb8"
    assert observation.depth.encoding == "32FC1"
    assert observation.target_mask.encoding == "mono8"
    assert observation.color.width == observation.depth.width
    assert observation.color.height == observation.target_mask.height
    assert observation.depth.header.stamp == observation.header.stamp
    assert observation.camera_info.header.stamp == observation.header.stamp
    assert observation.pose_valid
    assert observation.camera_pose.header.frame_id == "base_link"
    assert observation.valid_depth_fraction == 1.0
    np.testing.assert_allclose(target, (0.0, 0.0, 0.35))
    depth = np.frombuffer(observation.depth.data, dtype="<f4")
    mask = np.frombuffer(observation.target_mask.data, dtype=np.uint8)
    assert np.all(np.isfinite(depth))
    assert np.count_nonzero(mask == 255) >= 200
    assert np.all(depth[mask == 255] == np.float32(0.35))


def test_pipeline_configuration_uses_approved_geometry() -> None:
    current = np.eye(4)
    request = _configure_request(current, np.array((0.0, 0.0, 0.35)), "base_link")
    assert request.scene_id
    assert request.world_frame == "base_link"
    assert np.isclose(request.voxel_size, np.float32(0.003))
    assert np.isclose(request.depth_min, np.float32(0.10))
    assert np.isclose(request.depth_max, np.float32(0.75))
    assert request.samples_per_ray == 128
    assert request.optimization_steps == 10
    assert np.isclose(request.max_step, np.float32(0.10))
    assert request.random_seed == 0


def test_controller_diagnostic_boolean_is_strict() -> None:
    assert _diagnostic_boolean(False, "gate") is False
    assert _diagnostic_boolean("False", "gate") is False
    assert _diagnostic_boolean("true", "gate") is True
    with pytest.raises(RuntimeError, match="not boolean"):
        _diagnostic_boolean("disabled", "gate")
