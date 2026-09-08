"""Contract tests for deterministic Gradient-NBV fixtures and replay."""

from __future__ import annotations

import json

import numpy as np
import pytest

from strawberry_gradient_nbv.fixtures import (
    load_fixture_npz,
    look_at_optical,
    make_multiview_fixture,
    make_plane_fixture,
    save_fixture_npz,
)
from strawberry_gradient_nbv.replay import NPZReplay, main


def _assert_contract(fixture) -> None:
    assert fixture.color.dtype == np.uint8
    assert fixture.depth.dtype == np.float32
    assert fixture.mask.dtype == np.uint8
    assert fixture.color.shape[:3] == fixture.depth.shape == fixture.mask.shape
    assert fixture.K.shape == (len(fixture), 3, 3)
    assert fixture.pose.shape == (len(fixture), 4, 4)
    assert set(np.unique(fixture.mask)).issubset({0, 255})
    assert np.all(fixture.depth[np.isfinite(fixture.depth)] > 0.0)
    assert np.allclose(fixture.pose[:, 3], (0.0, 0.0, 0.0, 1.0))


def test_known_plane_has_metre_depth_and_canonical_invalids() -> None:
    fixture = make_plane_fixture(distance_m=1.0)
    _assert_contract(fixture)
    assert len(fixture) == 1
    assert np.nanmedian(fixture.depth[0]) == pytest.approx(1.0, abs=1e-7)
    assert np.isnan(fixture.depth[0, 0, 0])
    assert np.count_nonzero(fixture.mask[0]) >= 200
    assert np.array_equal(fixture.pose[0], np.eye(4))


def test_multiview_has_five_known_look_at_poses_and_visible_target() -> None:
    fixture = make_multiview_fixture()
    _assert_contract(fixture)
    assert len(fixture) == 5
    assert len(set(fixture.observation_ids)) == 5
    target = np.asarray(fixture.config["target_center"])
    for transform, mask in zip(fixture.pose, fixture.mask):
        optical_z_world = transform[:3, 2]
        target_direction = target - transform[:3, 3]
        target_direction /= np.linalg.norm(target_direction)
        assert np.dot(optical_z_world, target_direction) > 1.0 - 1e-12
        assert np.count_nonzero(mask) >= 200
    # Occlusion is viewpoint dependent, so views are not duplicate images.
    assert len({view.tobytes() for view in fixture.mask}) >= 3


def test_look_at_handles_reference_axis_degeneracy() -> None:
    transform = look_at_optical(np.array([0.0, -1.0, 0.0]), np.zeros(3))
    assert np.all(np.isfinite(transform))
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)
    assert np.allclose(transform[:3, 2], (0.0, 1.0, 0.0))


def test_npz_round_trip_is_semantically_exact(tmp_path) -> None:
    original = make_multiview_fixture(96, 64)
    path = save_fixture_npz(original, tmp_path / "five_views.npz")
    loaded = load_fixture_npz(path)
    assert loaded.scene_id == original.scene_id
    assert loaded.observation_ids == original.observation_ids
    assert loaded.config == original.config
    for name in ("color", "mask", "K", "pose", "stamp"):
        assert np.array_equal(getattr(loaded, name), getattr(original, name))
    assert np.array_equal(loaded.depth, original.depth, equal_nan=True)
    assert not loaded.depth.flags.writeable


def test_npz_generation_is_byte_reproducible(tmp_path) -> None:
    fixture = make_multiview_fixture(64, 48)
    first = save_fixture_npz(fixture, tmp_path / "first.npz")
    second = save_fixture_npz(fixture, tmp_path / "second.npz")
    assert first.read_bytes() == second.read_bytes()


def test_replay_preserves_ids_pose_intrinsics_and_metre_depth(tmp_path) -> None:
    fixture = make_multiview_fixture(96, 64)
    path = save_fixture_npz(fixture, tmp_path / "replay.npz")
    observations = list(NPZReplay(path))
    assert [item.observation_id for item in observations] == list(fixture.observation_ids)
    for index, item in enumerate(observations):
        assert np.array_equal(item.K, fixture.K[index])
        assert np.array_equal(item.pose, fixture.pose[index])
        assert np.array_equal(item.depth, fixture.depth[index], equal_nan=True)


def test_loader_rejects_incomplete_archive(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, depth=np.ones((1, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="missing fields"):
        load_fixture_npz(path)


def test_cli_generate_inspect_and_replay(tmp_path, capsys) -> None:
    path = tmp_path / "plane.npz"
    assert main(["generate", "plane", str(path), "--width", "64", "--height", "48"]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["observations"] == 1
    assert main(["inspect", str(path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["scene_id"] == "known_distance_plane_v1"
    assert main(["replay", str(path)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["observation_id"] == "known_distance_plane_v1:000"
