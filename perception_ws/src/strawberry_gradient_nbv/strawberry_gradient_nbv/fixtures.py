"""Deterministic, ROS-free RGB-D fixtures for Gradient-NBV acceptance tests.

All poses in this module are ``T_world_camera_optical``.  Camera coordinates
follow REP-103 optical convention: +X right, +Y down and +Z forward.  Depth is
optical-axis Z in metres (not Euclidean range along a normalized ray).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


FIXTURE_SCHEMA_VERSION = 1


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FixtureObservation:
    """One canonical observation, represented without ROS message classes."""

    scene_id: str
    observation_id: str
    color: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    pose: np.ndarray
    stamp: float
    config: Mapping[str, Any]


@dataclass(frozen=True)
class ObservationFixture:
    """A validated sequence suitable for synthetic and NPZ replay paths."""

    scene_id: str
    observation_ids: tuple[str, ...]
    color: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    pose: np.ndarray
    stamp: np.ndarray
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.observation_ids)
        if not self.scene_id or count == 0 or len(set(self.observation_ids)) != count:
            raise ValueError("fixture needs a scene ID and unique observation IDs")
        if self.color.ndim != 4 or self.color.shape[-1] != 3:
            raise ValueError("color must have shape (N,H,W,3)")
        expected_image_shape = self.color.shape[:3]
        if self.depth.shape != expected_image_shape or self.mask.shape != expected_image_shape:
            raise ValueError("color, depth, and mask must share one pixel grid")
        if self.K.shape != (count, 3, 3) or self.pose.shape != (count, 4, 4):
            raise ValueError("K and pose must have shapes (N,3,3) and (N,4,4)")
        if self.stamp.shape != (count,) or self.color.shape[0] != count:
            raise ValueError("all fixture arrays must have the same observation count")
        if self.color.dtype != np.uint8 or self.depth.dtype != np.float32:
            raise TypeError("color must be uint8 and depth must be float32 metres")
        if self.mask.dtype != np.uint8 or not np.all(np.isin(self.mask, (0, 255))):
            raise ValueError("mask must be uint8 with values 0 or 255")
        finite_depth = np.isfinite(self.depth)
        if np.any(self.depth[finite_depth] <= 0.0):
            raise ValueError("finite depth samples must be positive metres")
        if not np.all(np.isfinite(self.K)) or not np.all(np.isfinite(self.pose)):
            raise ValueError("K and pose must be finite")
        if np.any(self.K[:, 0, 0] <= 0.0) or np.any(self.K[:, 1, 1] <= 0.0):
            raise ValueError("focal lengths must be positive")
        rigid_tail = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (count, 4))
        if not np.allclose(self.pose[:, 3, :], rigid_tail, atol=1e-12):
            raise ValueError("pose must be a homogeneous transform")
        rotations = self.pose[:, :3, :3]
        identities = np.einsum("nji,njk->nik", rotations, rotations)
        if not np.allclose(identities, np.eye(3), atol=1e-9):
            raise ValueError("pose rotations must be orthonormal")
        if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-9):
            raise ValueError("pose rotations must be right handed")
        if not np.all(np.isfinite(self.stamp)) or np.any(np.diff(self.stamp) <= 0.0):
            raise ValueError("stamps must be finite and strictly increasing")

    def __len__(self) -> int:
        return len(self.observation_ids)

    def observations(self) -> Iterator[FixtureObservation]:
        """Yield views in deterministic exposure-time order."""
        for index, observation_id in enumerate(self.observation_ids):
            yield FixtureObservation(
                scene_id=self.scene_id,
                observation_id=observation_id,
                color=self.color[index],
                depth=self.depth[index],
                mask=self.mask[index],
                K=self.K[index],
                pose=self.pose[index],
                stamp=float(self.stamp[index]),
                config=self.config,
            )


def _intrinsics(width: int, height: int, focal_length: float | None = None) -> np.ndarray:
    if width < 16 or height < 16:
        raise ValueError("fixture images must be at least 16 by 16 pixels")
    focal = float(focal_length if focal_length is not None else 0.875 * width)
    if not np.isfinite(focal) or focal <= 0.0:
        raise ValueError("focal length must be finite and positive")
    return np.array(
        [[focal, 0.0, (width - 1.0) / 2.0],
         [0.0, focal, (height - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def look_at_optical(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Construct ``T_world_camera_optical`` with optical +Z aimed at target."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z_axis = target - eye
    norm = np.linalg.norm(z_axis)
    if eye.shape != (3,) or target.shape != (3,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("eye and target must be distinct finite 3-vectors")
    z_axis /= norm
    # Prefer world +Y as image-down.  Switch references near the singularity.
    down_hint = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(down_hint, z_axis))) > 0.98:
        down_hint = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(down_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = eye
    return transform


def _scene_config(target: np.ndarray, depth_max: float) -> dict[str, Any]:
    return {
        "world_frame": "fixture_world",
        "target_center": np.asarray(target, dtype=float).tolist(),
        "map_center": np.asarray(target, dtype=float).tolist(),
        "map_size": [0.30, 0.30, 0.30],
        "target_roi_size": [0.15, 0.15, 0.15],
        "observation_min": [-0.12, -0.12, -0.45],
        "observation_max": [0.12, 0.12, -0.25],
        "voxel_size": 0.003,
        "depth_min": 0.10,
        "depth_max": float(depth_max),
        "samples_per_ray": 128,
        "optimization_steps": 10,
        "max_step": 0.10,
        "random_seed": 0,
    }


def make_plane_fixture(
    width: int = 160,
    height: int = 100,
    distance_m: float = 1.0,
) -> ObservationFixture:
    """Create one fronto-parallel plane with a known optical-Z distance."""
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("plane distance must be finite and positive")
    K = _intrinsics(width, height)
    color = np.full((1, height, width, 3), (38, 55, 72), dtype=np.uint8)
    mask = np.zeros((1, height, width), dtype=np.uint8)
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    color[0, y0:y1, x0:x1] = (220, 20, 20)
    mask[0, y0:y1, x0:x1] = 255
    depth = np.full((1, height, width), distance_m, dtype=np.float32)
    # A small deterministic invalid border exercises canonical NaN handling.
    depth[:, :2, :] = np.nan
    depth[:, -2:, :] = np.nan
    depth[:, :, :2] = np.nan
    depth[:, :, -2:] = np.nan
    pose = np.eye(4, dtype=np.float64)[None, ...]
    config = _scene_config(np.array([0.0, 0.0, distance_m]), max(2.5, distance_m + 0.1))
    # The plane camera is at the world origin, unlike the object fixture whose
    # observation shell is behind a target at the origin.
    config["observation_min"] = [-0.10, -0.10, -0.10]
    config["observation_max"] = [0.10, 0.10, 0.10]
    return ObservationFixture(
        scene_id="known_distance_plane_v1",
        observation_ids=("known_distance_plane_v1:000",),
        color=_readonly(color, np.uint8),
        depth=_readonly(depth, np.float32),
        mask=_readonly(mask, np.uint8),
        K=_readonly(K[None, ...], np.float64),
        pose=_readonly(pose, np.float64),
        stamp=_readonly(np.array([1_700_000_000.0]), np.float64),
        config=config,
    )


def _positive_sphere_depth(
    origin: np.ndarray,
    directions: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    offset = origin - center
    a = np.einsum("...i,...i->...", directions, directions)
    b = 2.0 * np.einsum("...i,i->...", directions, offset)
    c = float(np.dot(offset, offset) - radius * radius)
    discriminant = b * b - 4.0 * a * c
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = (-b - root) / (2.0 * a)
    far = (-b + root) / (2.0 * a)
    result = np.where((discriminant >= 0.0) & (near > 0.0), near, far)
    return np.where((discriminant >= 0.0) & (result > 0.0), result, np.inf)


def _render_occluded_target(
    width: int,
    height: int,
    K: np.ndarray,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    camera_directions = np.stack(
        ((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)),
        axis=-1,
    )
    directions = camera_directions @ pose[:3, :3].T
    origin = pose[:3, 3]

    target_depth = _positive_sphere_depth(origin, directions, np.zeros(3), 0.055)
    occluder_depth = _positive_sphere_depth(
        origin, directions, np.array([0.025, 0.0, -0.155]), 0.028
    )
    plane_depth = (0.09 - origin[2]) / directions[..., 2]
    plane_depth = np.where(plane_depth > 0.0, plane_depth, np.inf)
    stacked = np.stack((plane_depth, target_depth, occluder_depth), axis=0)
    labels = np.argmin(stacked, axis=0)
    depth = np.min(stacked, axis=0)
    invalid = ~np.isfinite(depth) | (depth < 0.10) | (depth > 0.75)
    depth = depth.astype(np.float32)
    depth[invalid] = np.nan
    labels[invalid] = -1

    color = np.full((height, width, 3), (35, 68, 45), dtype=np.uint8)
    color[labels == 0] = (52, 82, 61)
    color[labels == 1] = (225, 24, 20)
    color[labels == 2] = (115, 118, 122)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[labels == 1] = 255
    return color, depth, mask


def make_multiview_fixture(width: int = 160, height: int = 100) -> ObservationFixture:
    """Render five known poses viewing a red sphere behind a grey occluder."""
    K = _intrinsics(width, height)
    target = np.zeros(3, dtype=np.float64)
    eyes = np.array(
        [[0.0, 0.0, -0.35],
         [0.075, 0.0, -0.342],
         [-0.075, 0.0, -0.342],
         [0.0, 0.065, -0.344],
         [0.0, -0.065, -0.344]],
        dtype=np.float64,
    )
    poses = np.stack([look_at_optical(eye, target) for eye in eyes])
    rendered = [_render_occluded_target(width, height, K, pose) for pose in poses]
    color = np.stack([item[0] for item in rendered])
    depth = np.stack([item[1] for item in rendered])
    mask = np.stack([item[2] for item in rendered])
    scene_id = "occluded_red_target_5view_v1"
    return ObservationFixture(
        scene_id=scene_id,
        observation_ids=tuple(f"{scene_id}:{index:03d}" for index in range(5)),
        color=_readonly(color, np.uint8),
        depth=_readonly(depth, np.float32),
        mask=_readonly(mask, np.uint8),
        K=_readonly(np.repeat(K[None, ...], 5, axis=0), np.float64),
        pose=_readonly(poses, np.float64),
        stamp=_readonly(1_700_000_100.0 + np.arange(5, dtype=np.float64) * 0.1, np.float64),
        config=_scene_config(target, 0.75),
    )


# Concise aliases for callers that name fixtures by their acceptance scenario.
known_distance_plane = make_plane_fixture
occluded_red_target_multiview = make_multiview_fixture


def save_fixture_npz(fixture: ObservationFixture, path: str | Path) -> Path:
    """Write a portable, deterministic uncompressed NPZ without pickle.

    ``numpy.savez_compressed`` embeds implementation-dependent DEFLATE output;
    the canonical fixture is deliberately uncompressed so identical arrays
    remain byte-for-byte stable across generation runs on the same Python/NumPy
    stack and are cheap to memory-map after extraction.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        schema_version=np.asarray(FIXTURE_SCHEMA_VERSION, dtype=np.uint32),
        scene_id=np.asarray(fixture.scene_id),
        observation_id=np.asarray(fixture.observation_ids),
        color=fixture.color,
        depth=fixture.depth,
        mask=fixture.mask,
        K=fixture.K,
        pose=fixture.pose,
        stamp=fixture.stamp,
        config=np.asarray(json.dumps(fixture.config, sort_keys=True, separators=(",", ":"))),
    )
    return output


def load_fixture_npz(path: str | Path) -> ObservationFixture:
    """Load and validate an NPZ fixture with pickle explicitly disabled."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "schema_version", "scene_id", "observation_id", "color", "depth",
            "mask", "K", "pose", "stamp", "config",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"fixture is missing fields: {sorted(missing)}")
        version = int(np.asarray(archive["schema_version"]).item())
        if version != FIXTURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported fixture schema version: {version}")
        config = json.loads(str(np.asarray(archive["config"]).item()))
        return ObservationFixture(
            scene_id=str(np.asarray(archive["scene_id"]).item()),
            observation_ids=tuple(str(value) for value in archive["observation_id"].tolist()),
            color=_readonly(archive["color"], np.uint8),
            depth=_readonly(archive["depth"], np.float32),
            mask=_readonly(archive["mask"], np.uint8),
            K=_readonly(archive["K"], np.float64),
            pose=_readonly(archive["pose"], np.float64),
            stamp=_readonly(archive["stamp"], np.float64),
            config=config,
        )
