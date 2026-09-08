"""Numerical, transaction and independence tests for the pure NBV core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from strawberry_gradient_nbv.core import (
    GradientNBVCore,
    NBVConfig,
    NBVInputError,
    look_at_optical,
)
from strawberry_gradient_nbv.fixtures import make_multiview_fixture


def _configured_core(*, device="cpu", **overrides):
    fixture = make_multiview_fixture(64, 48)
    values = dict(fixture.config)
    values.update(
        scene_id=fixture.scene_id,
        voxel_size=0.006,
        samples_per_ray=16,
        optimization_steps=2,
        downsample=1,
    )
    values.update(overrides)
    core = GradientNBVCore(device)
    core.configure(values)
    return core, fixture


def _state(core):
    return (
        core._log_odds.detach().cpu().clone(),
        core._semantic_log_odds.detach().cpu().clone(),
        core._ever_observed.detach().cpu().clone(),
    )


def test_look_at_optical_z_points_to_target_and_handles_parallel_hint() -> None:
    eye = np.array((0.0, -1.0, 0.0))
    target = np.zeros(3)
    rotation = look_at_optical(eye, target)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(rotation[:, 2], target - eye, atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_look_at_preserves_supplied_camera_down_without_world_frame_roll() -> None:
    eye = np.array((0.26410652, 0.04458210, 0.33846686))
    target = np.array((0.20575674, -0.57255830, 0.24502241))
    current_rotation = np.array(
        (
            (-0.69971266, -0.70666585, 0.10500276),
            (-0.30360302, 0.16107818, -0.93908414),
            (0.64670504, -0.68896822, -0.32725433),
        )
    )

    rotation = look_at_optical(eye, target, current_rotation[:, 1])
    expected_forward = target - eye
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(rotation[:, 2], expected_forward, atol=1.0e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    relative = current_rotation.T @ rotation
    angle = np.degrees(
        np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    )
    assert 10.0 < angle < 25.0


def test_look_at_with_down_hint_is_world_frame_equivariant() -> None:
    eye = np.array((0.2, -0.1, 0.4))
    target = np.array((-0.3, 0.5, 0.8))
    down_hint = np.array((0.1, 0.95, -0.2))
    angle = 0.73
    world_rotation = np.array(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )

    original = look_at_optical(eye, target, down_hint)
    transformed = look_at_optical(
        world_rotation @ eye,
        world_rotation @ target,
        world_rotation @ down_hint,
    )
    np.testing.assert_allclose(transformed, world_rotation @ original, atol=1.0e-12)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"voxel_size": 0.0}, "voxel"),
        ({"depth_min": 1.0, "depth_max": 0.5}, "depth"),
        ({"observation_min": [1, 0, 0]}, "observation_min"),
        ({"target_roi_size": [1, 1, 1]}, "target ROI"),
        ({"samples_per_ray": 1}, "ray samples"),
    ],
)
def test_bad_config_is_rejected(change, match) -> None:
    fixture = make_multiview_fixture(64, 48)
    values = dict(fixture.config)
    values.update(scene_id=fixture.scene_id)
    values.update(change)
    with pytest.raises(NBVInputError, match=match):
        NBVConfig.from_mapping(values)


@pytest.mark.parametrize(
    "kind",
    (
        "empty_mask",
        "bad_depth",
        "wrong_depth_dtype",
        "bad_k",
        "bad_size",
        "bad_pose",
        "pose_outside_bounds",
        "target_depth_invalid",
    ),
)
def test_bad_observation_does_not_modify_map(kind: str) -> None:
    core, fixture = _configured_core()
    observation = next(fixture.observations())
    depth = observation.depth.copy()
    mask = observation.mask.copy()
    K = observation.K.copy()
    pose = observation.pose.copy()
    if kind == "empty_mask":
        mask.fill(0)
    elif kind == "bad_depth":
        depth.fill(np.nan)
    elif kind == "wrong_depth_dtype":
        depth = depth.astype(np.float64)
    elif kind == "bad_k":
        K[0, 0] = 0.0
    elif kind == "bad_size":
        mask = mask[:-1]
    elif kind == "bad_pose":
        pose[0, 0] = 2.0
    elif kind == "target_depth_invalid":
        depth[mask == 255] = np.nan
    else:
        pose[0, 3] = 10.0
    before = _state(core)
    with pytest.raises(NBVInputError):
        core.update_and_plan(depth, mask, K, pose)
    after = _state(core)
    for first, second in zip(before, after):
        assert torch.equal(first, second)


def test_planning_exception_rolls_back_fused_map(monkeypatch) -> None:
    core, fixture = _configured_core()
    observation = next(fixture.observations())
    before = _state(core)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(core, "_plan", fail)
    with pytest.raises(RuntimeError, match="optimizer"):
        core.update_and_plan(
            observation.depth, observation.mask, observation.K, observation.pose
        )
    for first, second in zip(before, _state(core)):
        assert torch.equal(first, second)


def test_configuration_allocation_failure_keeps_previous_session(monkeypatch) -> None:
    core, fixture = _configured_core()
    observation = next(fixture.observations())
    core.update_and_plan(observation.depth, observation.mask, observation.K, observation.pose)
    before = _state(core)
    scene_before = core.config.scene_id

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("synthetic device allocation failure")

    monkeypatch.setattr(torch, "zeros", fail_allocation)
    replacement = dict(fixture.config)
    replacement.update(scene_id="replacement_scene", voxel_size=0.006)
    with pytest.raises(RuntimeError, match="allocation"):
        core.configure(replacement)
    assert core.config.scene_id == scene_before
    for first, second in zip(before, _state(core)):
        assert torch.equal(first, second)


def test_configuration_owns_immutable_geometry_arrays() -> None:
    fixture = make_multiview_fixture(64, 48)
    values = dict(fixture.config)
    mutable_target = np.asarray(values["target_center"], dtype=np.float64)
    values.update(
        scene_id=fixture.scene_id,
        target_center=mutable_target,
        voxel_size=0.006,
        samples_per_ray=16,
        optimization_steps=2,
    )
    core = GradientNBVCore("cpu")
    core.configure(values)
    mutable_target[:] = 42.0
    np.testing.assert_array_equal(core.config.target_center, np.zeros(3))
    assert not core.config.target_center.flags.writeable


def test_five_views_increase_ever_observed_roi_coverage() -> None:
    fixture = make_multiview_fixture(160, 100)
    values = dict(fixture.config)
    values.update(scene_id=fixture.scene_id)
    core = GradientNBVCore("cuda" if torch.cuda.is_available() else "cpu")
    core.configure(values)
    coverage = []
    results = []
    for observation in fixture.observations():
        result = core.update_and_plan(
            observation.depth, observation.mask, observation.K, observation.pose
        )
        coverage.append(result.coverage)
        results.append(result)
    growth = np.diff(coverage)
    assert np.all(growth >= 0.0)
    assert np.count_nonzero(growth > 0.0) >= 3
    assert coverage[-1] - coverage[0] >= 0.20
    for observation, result in zip(fixture.observations(), results):
        displacement = np.linalg.norm(result.pose[:3, 3] - observation.pose[:3, 3])
        assert displacement <= values["max_step"] + 1.0e-6
        assert np.all(result.pose[:3, 3] >= np.asarray(values["observation_min"]) - 1e-6)
        assert np.all(result.pose[:3, 3] <= np.asarray(values["observation_max"]) + 1e-6)
        direction = np.asarray(values["target_center"]) - result.pose[:3, 3]
        direction /= np.linalg.norm(direction)
        angle = np.degrees(np.arccos(np.clip(np.dot(result.pose[:3, 2], direction), -1, 1)))
        assert angle <= 1.0
        assert np.isfinite(result.gain)
        assert result.gain == result.planned_gain
        assert result.gain_improvement == pytest.approx(
            result.planned_gain - result.current_gain
        )
        assert result.gain_improvement > 0.0
        assert displacement >= 1.0e-4
        assert np.all(np.isfinite(result.pose))


def test_target_mask_location_changes_gain_field() -> None:
    fixture = make_multiview_fixture(160, 100)
    observation = next(fixture.observations())
    values = dict(fixture.config)
    values.update(scene_id=fixture.scene_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def plan(mask):
        core = GradientNBVCore(device)
        core.configure(values)
        return core.update_and_plan(
            observation.depth, mask, observation.K, observation.pose
        )

    target_result = plan(observation.mask)
    # Keep the same mask area and pixel values but move it away from the known
    # target surface.  This tests semantic location, not merely pixel count.
    shifted_mask = np.roll(observation.mask, observation.mask.shape[1] // 5, axis=1)
    shifted_result = plan(shifted_mask)
    assert np.count_nonzero(shifted_mask) == np.count_nonzero(observation.mask)
    assert abs(target_result.current_gain - shifted_result.current_gain) >= 0.05
    assert target_result.current_gain >= shifted_result.current_gain * 1.15


def test_target_mask_must_survive_strided_downsampling() -> None:
    core, fixture = _configured_core(downsample=2)
    observation = next(fixture.observations())
    mask = np.zeros_like(observation.mask)
    valid = np.argwhere(np.isfinite(observation.depth))
    row, column = next(
        (int(row), int(column))
        for row, column in valid
        if row % 2 == 1 or column % 2 == 1
    )
    mask[row, column] = 255
    before = _state(core)
    with pytest.raises(NBVInputError, match="strided downsampling"):
        core.update_and_plan(observation.depth, mask, observation.K, observation.pose)
    for first, second in zip(before, _state(core)):
        assert torch.equal(first, second)


def test_fixed_seed_result_is_exactly_reproducible() -> None:
    first, fixture = _configured_core()
    second, _ = _configured_core()
    observation = next(fixture.observations())
    first_result = first.update_and_plan(
        observation.depth, observation.mask, observation.K, observation.pose
    )
    second_result = second.update_and_plan(
        observation.depth, observation.mask, observation.K, observation.pose
    )
    np.testing.assert_array_equal(first_result.pose, second_result.pose)
    assert first_result.gain == second_result.gain
    assert first_result.coverage == second_result.coverage


def test_reset_retains_configuration_and_clears_state() -> None:
    core, fixture = _configured_core()
    observation = next(fixture.observations())
    core.update_and_plan(observation.depth, observation.mask, observation.K, observation.pose)
    assert core.reset() is True
    assert core.coverage == 0.0
    assert core.reset() is False
    assert core.config.scene_id == fixture.scene_id


def test_snapshot_restores_complete_map_transaction() -> None:
    core, fixture = _configured_core()
    observations = list(fixture.observations())
    core.update_and_plan(
        observations[0].depth,
        observations[0].mask,
        observations[0].K,
        observations[0].pose,
    )
    snapshot = core.snapshot()
    before = _state(core)
    core.update_and_plan(
        observations[1].depth,
        observations[1].mask,
        observations[1].K,
        observations[1].pose,
    )
    core.restore(snapshot)
    for first, second in zip(before, _state(core)):
        assert torch.equal(first, second)


def test_core_source_has_no_forbidden_runtime_dependencies() -> None:
    package = Path(__file__).parents[1] / "strawberry_gradient_nbv"
    source = (package / "core.py").read_text(encoding="utf-8")
    forbidden_imports = (
        "import rclpy",
        "import rospy",
        "import moveit",
        "from moveit",
        "import abb_control",
        "open3d",
        "matplotlib",
        "cv_bridge",
    )
    assert all(value not in source.lower() for value in forbidden_imports)
