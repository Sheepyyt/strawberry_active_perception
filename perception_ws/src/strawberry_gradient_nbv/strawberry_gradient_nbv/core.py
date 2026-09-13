"""Pure NumPy/PyTorch Gradient-NBV implementation.

The core consumes metric optical-Z depth, a binary target mask, a registered
pinhole matrix and ``T_world_camera_optical``.  It intentionally imports no ROS,
robot, visualization or camera-driver modules.

The gain model follows Burusa et al., *Gradient-based local next-best-view
planning for improved perception of targeted plant nodes*, ICRA 2024.  This is
a clean-room implementation informed by the public paper and upstream commit
``81b501defc117732f66478ed01cb528b15140208``; this is an independent
reimplementation and no upstream source is copied verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as torch_functional


_EPSILON = 1.0e-7
_SEMANTIC_PRIOR_PROBABILITY = 0.10
_SEMANTIC_PRIOR_LOG_ODDS = math.log(
    _SEMANTIC_PRIOR_PROBABILITY / (1.0 - _SEMANTIC_PRIOR_PROBABILITY)
)


class NBVInputError(ValueError):
    """An input is invalid and the voxel map has not been modified."""


def _voxel_dimensions(map_size: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    ratios = np.asarray(map_size, dtype=np.float64) / float(voxel_size)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise NBVInputError("voxel grid dimensions are invalid")
    # The product cap implies no individual dimension may exceed this value.
    # Check before converting to int64 so extreme finite ratios cannot overflow.
    if np.any(ratios > 20_000_000.0):
        raise NBVInputError("voxel grid dimensions exceed 20M cells")
    dimensions = tuple(int(math.ceil(float(value))) for value in ratios)
    if math.prod(dimensions) > 20_000_000:
        raise NBVInputError("voxel grid dimensions exceed 20M cells")
    return dimensions


@dataclass(frozen=True)
class NBVConfig:
    """All scene geometry and numerical settings, in SI units."""

    scene_id: str
    world_frame: str
    target_center: np.ndarray
    map_size: np.ndarray
    target_roi_size: np.ndarray
    observation_min: np.ndarray
    observation_max: np.ndarray
    voxel_size: float = 0.003
    depth_min: float = 0.10
    depth_max: float = 0.75
    samples_per_ray: int = 128
    optimization_steps: int = 10
    max_step: float = 0.10
    random_seed: int = 0
    downsample: int = 2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NBVConfig":
        """Build and validate a configuration from a JSON/ROS-like mapping."""
        required = (
            "scene_id",
            "world_frame",
            "target_center",
            "map_size",
            "target_roi_size",
            "observation_min",
            "observation_max",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise NBVInputError(f"configuration is missing {missing}")
        config = cls(
            scene_id=str(values["scene_id"]),
            world_frame=str(values["world_frame"]),
            target_center=np.asarray(values["target_center"], dtype=np.float64),
            map_size=np.asarray(values["map_size"], dtype=np.float64),
            target_roi_size=np.asarray(values["target_roi_size"], dtype=np.float64),
            observation_min=np.asarray(values["observation_min"], dtype=np.float64),
            observation_max=np.asarray(values["observation_max"], dtype=np.float64),
            voxel_size=float(values.get("voxel_size", 0.003)),
            depth_min=float(values.get("depth_min", 0.10)),
            depth_max=float(values.get("depth_max", 0.75)),
            samples_per_ray=int(values.get("samples_per_ray", 128)),
            optimization_steps=int(values.get("optimization_steps", 10)),
            max_step=float(values.get("max_step", 0.10)),
            random_seed=int(values.get("random_seed", 0)),
            downsample=int(values.get("downsample", 2)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject configurations that cannot define one finite voxel session."""
        if not self.scene_id.strip() or not self.world_frame.strip():
            raise NBVInputError("scene_id and world_frame must be non-empty")
        vector_fields = {
            "target_center": self.target_center,
            "map_size": self.map_size,
            "target_roi_size": self.target_roi_size,
            "observation_min": self.observation_min,
            "observation_max": self.observation_max,
        }
        for name, value in vector_fields.items():
            if np.asarray(value).shape != (3,) or not np.all(np.isfinite(value)):
                raise NBVInputError(f"{name} must contain three finite values")
        if np.any(self.map_size <= 0.0) or np.any(self.target_roi_size <= 0.0):
            raise NBVInputError("map_size and target_roi_size must be positive")
        if np.any(self.target_roi_size > self.map_size):
            raise NBVInputError("target ROI must fit inside the map")
        if np.any(self.observation_min >= self.observation_max):
            raise NBVInputError("observation_min must be below observation_max")
        scalars = (self.voxel_size, self.depth_min, self.depth_max, self.max_step)
        if not all(math.isfinite(value) for value in scalars):
            raise NBVInputError("voxel/depth/step values must be finite")
        if self.voxel_size <= 0.0 or not 0.0 <= self.depth_min < self.depth_max:
            raise NBVInputError("invalid voxel size or depth interval")
        if self.max_step <= 0.0:
            raise NBVInputError("max_step must be positive")
        if self.samples_per_ray < 2 or self.optimization_steps < 1:
            raise NBVInputError("ray samples >=2 and optimization steps >=1 are required")
        if self.downsample < 1:
            raise NBVInputError("downsample must be at least one")
        _voxel_dimensions(self.map_size, self.voxel_size)


@dataclass(frozen=True)
class NBVResult:
    """One map update plus the optimized next optical camera pose."""

    pose: np.ndarray
    # ``gain`` remains the public/ROS-compatible name for the planned gain.
    gain: float
    current_gain: float
    planned_gain: float
    gain_improvement: float
    coverage: float
    total_voxel_count: int
    observed_voxel_count: int
    occupied_voxel_count: int
    unknown_voxel_count: int
    optimization_iterations: int
    compute_time_ms: float
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class NBVMapSnapshot:
    """Opaque deep snapshot used by the ROS wrapper for full transactions."""

    log_odds: torch.Tensor
    semantic_log_odds: torch.Tensor
    ever_observed: torch.Tensor


def _validate_rigid_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise NBVInputError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise NBVInputError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise NBVInputError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise NBVInputError(f"{name} rotation must have determinant +1")
    return transform


def look_at_optical_torch(
    eye: torch.Tensor,
    target: torch.Tensor,
    down_hint: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a differentiable optical look-at rotation.

    ``down_hint`` is the desired optical +Y direction in the world frame.  A
    planner must pass the current camera +Y axis here so looking at a nearby
    target preserves camera roll.  Falling back to a fixed world axis is useful
    for standalone callers, but is not invariant to the chosen world frame.
    """
    forward = target - eye
    norm = torch.linalg.vector_norm(forward)
    if not bool(torch.isfinite(norm)) or float(norm.detach()) <= 1.0e-9:
        raise NBVInputError("camera position must differ from target center")
    forward = forward / norm
    if down_hint is None:
        hint = torch.tensor(
            (0.0, 1.0, 0.0), dtype=eye.dtype, device=eye.device
        )
    else:
        hint = torch.as_tensor(down_hint, dtype=eye.dtype, device=eye.device)
        if hint.shape != (3,) or not bool(torch.all(torch.isfinite(hint))):
            raise NBVInputError("camera down hint must be a finite 3-vector")
        hint_norm = torch.linalg.vector_norm(hint)
        if float(hint_norm.detach()) <= 1.0e-9:
            raise NBVInputError("camera down hint must be non-zero")
        hint = hint / hint_norm

    # If +Y is parallel to the new viewing direction, roll is mathematically
    # undefined.  Select the canonical axis least aligned with +Z to keep the
    # basis finite.  Otherwise projecting the current +Y axis onto the image
    # plane yields the minimum-roll look-at orientation.
    if float(torch.abs(torch.dot(hint, forward)).detach()) > 0.98:
        axes = torch.eye(3, dtype=eye.dtype, device=eye.device)
        hint = axes[int(torch.argmin(torch.abs(axes @ forward)).detach())]
    right = torch.linalg.cross(hint, forward, dim=0)
    right = right / torch.linalg.vector_norm(right).clamp_min(_EPSILON)
    down = torch.linalg.cross(forward, right, dim=0)
    return torch.stack((right, down, forward), dim=1)


def look_at_optical(
    eye: np.ndarray,
    target: np.ndarray,
    down_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Return the stable optical look-at rotation as a NumPy array."""
    return look_at_optical_torch(
        torch.tensor(np.asarray(eye), dtype=torch.float64),
        torch.tensor(np.asarray(target), dtype=torch.float64),
        None
        if down_hint is None
        else torch.tensor(np.asarray(down_hint), dtype=torch.float64),
    ).detach().cpu().numpy()


def _rigid_pose(value: Any, name: str) -> np.ndarray:
    """Return one finite proper 4x4 transform for read-only view scoring."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise NBVInputError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise NBVInputError(f"{name} has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise NBVInputError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise NBVInputError(f"{name} rotation must have determinant +1")
    return pose.copy()


class GradientNBVCore:
    """Stateful voxel fusion and local differentiable next-view planner."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        requested = torch.device(device) if device is not None else None
        self.device = requested or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.config: NBVConfig | None = None
        self._origin: torch.Tensor | None = None
        self._dimensions: tuple[int, int, int] = (0, 0, 0)
        self._log_odds: torch.Tensor | None = None
        self._semantic_log_odds: torch.Tensor | None = None
        self._ever_observed: torch.Tensor | None = None
        self._roi_mask: torch.Tensor | None = None

    def configure(self, config: NBVConfig | Mapping[str, Any]) -> None:
        """Atomically replace the scene configuration and clear the map."""
        candidate = config if isinstance(config, NBVConfig) else NBVConfig.from_mapping(config)
        # A frozen dataclass does not freeze the contents of NumPy arrays.  Own
        # read-only copies so caller mutation cannot desynchronize bounds from an
        # already allocated voxel map.
        normalized = NBVConfig(
            scene_id=str(candidate.scene_id),
            world_frame=str(candidate.world_frame),
            target_center=np.array(candidate.target_center, dtype=np.float64, copy=True),
            map_size=np.array(candidate.map_size, dtype=np.float64, copy=True),
            target_roi_size=np.array(candidate.target_roi_size, dtype=np.float64, copy=True),
            observation_min=np.array(candidate.observation_min, dtype=np.float64, copy=True),
            observation_max=np.array(candidate.observation_max, dtype=np.float64, copy=True),
            voxel_size=float(candidate.voxel_size),
            depth_min=float(candidate.depth_min),
            depth_max=float(candidate.depth_max),
            samples_per_ray=int(candidate.samples_per_ray),
            optimization_steps=int(candidate.optimization_steps),
            max_step=float(candidate.max_step),
            random_seed=int(candidate.random_seed),
            downsample=int(candidate.downsample),
        )
        normalized.validate()
        for value in (
            normalized.target_center,
            normalized.map_size,
            normalized.target_roi_size,
            normalized.observation_min,
            normalized.observation_max,
        ):
            value.setflags(write=False)
        dimensions = _voxel_dimensions(normalized.map_size, normalized.voxel_size)
        dimensions_array = np.asarray(dimensions, dtype=np.float64)
        origin = normalized.target_center - dimensions_array * normalized.voxel_size / 2.0
        indices = torch.meshgrid(
            *(torch.arange(size, device=self.device, dtype=torch.float32) for size in dimensions),
            indexing="ij",
        )
        centers = torch.stack(indices, dim=-1)
        centers = torch.as_tensor(origin, device=self.device, dtype=torch.float32) + (
            centers + 0.5
        ) * normalized.voxel_size
        roi_half = torch.as_tensor(
            normalized.target_roi_size / 2.0,
            device=self.device,
            dtype=torch.float32,
        )
        target = torch.tensor(normalized.target_center, device=self.device, dtype=torch.float32)
        roi_mask = torch.all(torch.abs(centers - target) <= roi_half + 1.0e-7, dim=-1)
        # Allocate every replacement first.  A device allocation failure must not
        # leave a new configuration paired with an old/partially cleared map.
        origin_tensor = torch.as_tensor(origin, device=self.device, dtype=torch.float32)
        log_odds = torch.zeros(dimensions, device=self.device, dtype=torch.float32)
        semantic_log_odds = torch.full(
            dimensions,
            _SEMANTIC_PRIOR_LOG_ODDS,
            device=self.device,
            dtype=torch.float32,
        )
        ever_observed = torch.zeros(dimensions, device=self.device, dtype=torch.bool)
        # Assign state only after every allocation/validation succeeds.
        self.config = normalized
        self._dimensions = dimensions
        self._origin = origin_tensor
        self._log_odds = log_odds
        self._semantic_log_odds = semantic_log_odds
        self._ever_observed = ever_observed
        self._roi_mask = roi_mask

    def reset(self) -> bool:
        """Clear map state while retaining configuration; return whether data existed."""
        if self.config is None or self._log_odds is None:
            raise NBVInputError("NBV core is not configured")
        had_data = bool(torch.any(self._ever_observed).item())
        self._log_odds.zero_()
        self._semantic_log_odds.fill_(_SEMANTIC_PRIOR_LOG_ODDS)
        self._ever_observed.zero_()
        return had_data

    def snapshot(self) -> NBVMapSnapshot:
        """Clone the configured map for an enclosing adapter transaction."""
        if self.config is None or self._log_odds is None:
            raise NBVInputError("NBV core is not configured")
        return NBVMapSnapshot(
            self._log_odds.clone(),
            self._semantic_log_odds.clone(),
            self._ever_observed.clone(),
        )

    def restore(self, snapshot: NBVMapSnapshot) -> None:
        """Restore a compatible snapshot after an adapter-side failure."""
        if not isinstance(snapshot, NBVMapSnapshot) or self._log_odds is None:
            raise NBVInputError("invalid NBV map snapshot")
        expected = self._log_odds.shape
        tensors = (
            snapshot.log_odds,
            snapshot.semantic_log_odds,
            snapshot.ever_observed,
        )
        if any(value.shape != expected or value.device != self.device for value in tensors):
            raise NBVInputError("NBV map snapshot is incompatible with current configuration")
        self._log_odds.copy_(snapshot.log_odds)
        self._semantic_log_odds.copy_(snapshot.semantic_log_odds)
        self._ever_observed.copy_(snapshot.ever_observed)

    def visualization_state(self) -> dict[str, np.ndarray]:
        """Return a compact CPU copy suitable for NPZ/PNG diagnostics.

        This deliberately exports categorical fields instead of the internal
        optimizer tensors.  ``observed`` means a valid ray touched the voxel,
        ``occupied`` means a depth endpoint increased occupancy, and
        ``target`` means mask-supported evidence made that voxel more likely
        to belong to the target.  Mutating the returned arrays cannot change
        the live map.
        """
        if (
            self.config is None
            or self._origin is None
            or self._log_odds is None
            or self._semantic_log_odds is None
            or self._ever_observed is None
        ):
            raise NBVInputError("NBV core is not configured")
        return {
            "dimensions": np.asarray(self._dimensions, dtype=np.int32),
            "origin_m": self._origin.detach().cpu().numpy().astype(
                np.float64, copy=True
            ),
            "voxel_size_m": np.asarray(self.config.voxel_size, dtype=np.float64),
            "target_center_m": np.asarray(
                self.config.target_center, dtype=np.float64
            ).copy(),
            "target_roi_size_m": np.asarray(
                self.config.target_roi_size, dtype=np.float64
            ).copy(),
            "observed": self._ever_observed.detach().cpu().numpy().copy(),
            "occupied": (
                self._log_odds > 0.0
            ).detach().cpu().numpy().copy(),
            "target": (
                self._semantic_log_odds > 0.0
            ).detach().cpu().numpy().copy(),
        }

    @property
    def coverage(self) -> float:
        """Fraction of target-ROI voxels ever touched by a valid ray."""
        if self._roi_mask is None or self._ever_observed is None:
            return 0.0
        denominator = int(torch.count_nonzero(self._roi_mask).item())
        if denominator == 0:
            return 0.0
        numerator = int(torch.count_nonzero(self._ever_observed & self._roi_mask).item())
        return numerator / denominator

    def _validate_observation(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.config is None:
            raise NBVInputError("NBV core is not configured")
        depth_array = np.asarray(depth)
        mask_array = np.asarray(mask)
        intrinsics = np.asarray(K, dtype=np.float64)
        transform = _validate_rigid_transform(pose, "T_world_camera_optical")
        if depth_array.ndim != 2 or mask_array.shape != depth_array.shape:
            raise NBVInputError("depth and mask must share a non-empty HxW grid")
        if depth_array.size == 0:
            raise NBVInputError("depth image is empty")
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise NBVInputError("K must be a finite 3x3 matrix")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise NBVInputError("K focal lengths must be positive")
        height, width = depth_array.shape
        principal_point_valid = (
            0.0 <= intrinsics[0, 2] < width
            and 0.0 <= intrinsics[1, 2] < height
        )
        if not principal_point_valid:
            raise NBVInputError("K principal point must lie in the registered image grid")
        if not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1.0e-9):
            raise NBVInputError("K has an invalid homogeneous row")
        if not np.allclose((intrinsics[0, 1], intrinsics[1, 0]), (0.0, 0.0), atol=1.0e-9):
            raise NBVInputError("K skew terms are unsupported by the pinhole backprojection")
        if depth_array.dtype != np.float32:
            raise NBVInputError("depth must be float32 metres")
        if mask_array.dtype != np.uint8 or not np.all(np.isin(mask_array, (0, 255))):
            raise NBVInputError("mask must be uint8 with values 0 or 255")
        if not np.any(mask_array == 255):
            raise NBVInputError("target mask is empty")
        valid = (
            np.isfinite(depth_array)
            & (depth_array >= self.config.depth_min)
            & (depth_array <= self.config.depth_max)
        )
        if not np.any(valid):
            raise NBVInputError("depth contains no valid samples in the configured range")
        target_valid = valid & (mask_array == 255)
        if not np.any(target_valid):
            raise NBVInputError("target mask has no valid metric depth samples")
        factor = self.config.downsample
        if not np.any(target_valid[::factor, ::factor]):
            raise NBVInputError(
                "target mask has no valid depth ray after configured strided downsampling"
            )
        position = transform[:3, 3]
        if np.any(position < self.config.observation_min - 1.0e-9) or np.any(
            position > self.config.observation_max + 1.0e-9
        ):
            raise NBVInputError("camera position is outside configured observation bounds")
        clean_depth = np.asarray(depth_array, dtype=np.float32).copy()
        clean_depth[~valid] = np.nan
        return clean_depth, mask_array.copy(), intrinsics, transform

    def _downsample_observation(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        factor = self.config.downsample
        if factor == 1:
            return depth, mask, K
        sampled_depth = depth[::factor, ::factor]
        sampled_mask = mask[::factor, ::factor]
        sampled_K = K.copy()
        # Exact strided pixels are u=factor*u', v=factor*v'.  The principal
        # point therefore scales directly with the focal lengths.
        sampled_K[0, 0] /= factor
        sampled_K[1, 1] /= factor
        sampled_K[0, 2] /= factor
        sampled_K[1, 2] /= factor
        return sampled_depth, sampled_mask, sampled_K

    def _linear_indices(self, world_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = torch.floor(
            (world_points - self._origin) / self.config.voxel_size
        ).to(torch.int64)
        dimensions = torch.tensor(self._dimensions, device=self.device, dtype=torch.int64)
        valid = torch.all((coordinates >= 0) & (coordinates < dimensions), dim=-1)
        valid_coordinates = coordinates[valid]
        linear = (
            valid_coordinates[:, 0] * (self._dimensions[1] * self._dimensions[2])
            + valid_coordinates[:, 1] * self._dimensions[2]
            + valid_coordinates[:, 2]
        )
        return linear, valid

    def _fuse(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        pose: np.ndarray,
    ) -> None:
        depth, mask, K = self._downsample_observation(depth, mask, K)
        depth_tensor = torch.tensor(depth, device=self.device, dtype=torch.float32)
        mask_tensor = torch.tensor(mask == 255, device=self.device)
        valid_pixel = torch.isfinite(depth_tensor)
        rows, columns = torch.nonzero(valid_pixel, as_tuple=True)
        z_values = depth_tensor[rows, columns]
        x_values = (columns.to(torch.float32) - float(K[0, 2])) / float(K[0, 0]) * z_values
        y_values = (rows.to(torch.float32) - float(K[1, 2])) / float(K[1, 1]) * z_values
        endpoints_camera = torch.stack((x_values, y_values, z_values), dim=-1)
        rotation = torch.tensor(pose[:3, :3], device=self.device, dtype=torch.float32)
        translation = torch.tensor(pose[:3, 3], device=self.device, dtype=torch.float32)
        endpoints_world = endpoints_camera @ rotation.T + translation
        # Samples use optical-Z scaling: point(z_s)=K^-1 pixel*z_s, not a
        # normalized Euclidean ray multiplied by the depth value.
        interpolation = torch.linspace(
            0.0,
            1.0,
            self.config.samples_per_ray,
            device=self.device,
            dtype=torch.float32,
        )
        start_fractions = (self.config.depth_min / z_values).clamp(0.0, 1.0)
        fractions = start_fractions[:, None] + (
            1.0 - start_fractions[:, None]
        ) * interpolation[None, :]
        points = translation + (
            endpoints_world - translation
        )[:, None, :] * fractions[..., None]
        linear, valid_sample = self._linear_indices(points.reshape(-1, 3))
        if linear.numel() == 0:
            raise NBVInputError("valid depth rays do not intersect the configured map")
        ray_count = rows.shape[0]
        endpoint_mask = torch.zeros(
            (ray_count, self.config.samples_per_ray), device=self.device, dtype=torch.bool
        )
        endpoint_mask[:, -1] = True
        endpoint_flat = endpoint_mask.reshape(-1)[valid_sample]
        target_flat = mask_tensor[rows, columns, None].expand(-1, self.config.samples_per_ray)
        target_flat = target_flat.reshape(-1)[valid_sample] & endpoint_flat
        if not bool(torch.any(target_flat)):
            raise NBVInputError("target depth endpoints do not intersect the configured voxel map")

        free_updates = torch.full_like(linear, -0.40, dtype=torch.float32)
        occupancy_updates = torch.where(endpoint_flat, 2.20, free_updates)
        # Semantics describe the measured surface only.  Marking every free-space
        # ray sample as background overwhelms the comparatively few target
        # endpoints and makes the planner effectively insensitive to ``mask``.
        # A target endpoint receives enough evidence to move from the 0.1 prior
        # to a confident posterior in one canonical observation; non-target
        # endpoints receive conservative negative evidence.  Interior samples
        # remain at their prior and are still represented by ``ever_observed``.
        semantic_updates = torch.zeros_like(linear, dtype=torch.float32)
        semantic_updates = torch.where(
            target_flat,
            torch.full_like(linear, 4.00, dtype=torch.float32),
            semantic_updates,
        )
        background_endpoint = endpoint_flat & ~target_flat
        semantic_updates = torch.where(
            background_endpoint,
            torch.full_like(linear, -1.00, dtype=torch.float32),
            semantic_updates,
        )
        # Correct duplicate semantics: aggregate all repeated voxel hits before
        # applying one indexed addition.  Advanced-index ``+=`` would lose hits.
        unique, inverse = torch.unique(linear, sorted=False, return_inverse=True)
        occupancy_sum = torch.zeros(unique.shape, device=self.device, dtype=torch.float32)
        semantic_sum = torch.zeros_like(occupancy_sum)
        occupancy_sum.scatter_add_(0, inverse, occupancy_updates)
        semantic_sum.scatter_add_(0, inverse, semantic_updates)
        flat_occ = self._log_odds.view(-1)
        flat_sem = self._semantic_log_odds.view(-1)
        flat_observed = self._ever_observed.view(-1)
        flat_occ[unique] = torch.clamp(flat_occ[unique] + occupancy_sum, -12.0, 12.0)
        flat_sem[unique] = torch.clamp(flat_sem[unique] + semantic_sum, -12.0, 12.0)
        flat_observed[unique] = True

    def _planning_features(self) -> torch.Tensor:
        """Build the fixed differentiable-planning volume for the current map.

        Unknown occupancy is intentionally transparent.  Target evidence is a
        short-range dilation of posterior evidence above the explicit 0.1 prior;
        this makes unseen voxels immediately around a detected target valuable
        without inventing a learned segmentation or changing coverage semantics.
        """
        occupancy = torch.sigmoid(self._log_odds)
        semantic = torch.sigmoid(self._semantic_log_odds)
        observed = self._ever_observed.to(torch.float32)
        roi = self._roi_mask.to(torch.float32)
        target_evidence = (
            (semantic - _SEMANTIC_PRIOR_PROBABILITY)
            / (1.0 - _SEMANTIC_PRIOR_PROBABILITY)
        ).clamp(0.0, 1.0)
        target_support = torch_functional.max_pool3d(
            target_evidence[None, None], kernel_size=5, stride=1, padding=2
        )[0, 0]
        return torch.stack((occupancy, semantic, observed, roi, target_support), dim=0)

    def _candidate_gain(
        self,
        position: torch.Tensor,
        ray_directions_camera: torch.Tensor,
        down_hint: torch.Tensor,
        planning_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = torch.tensor(
            self.config.target_center, device=self.device, dtype=torch.float32
        )
        rotation = look_at_optical_torch(position, target, down_hint)
        return self._candidate_gain_with_rotation(
            position,
            rotation,
            ray_directions_camera,
            planning_features,
        )

    def _candidate_gain_with_rotation(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        ray_directions_camera: torch.Tensor,
        planning_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate one explicit optical pose without changing map state."""
        if self.config is None or self._origin is None:
            raise NBVInputError("NBV core is not configured")
        directions_world = ray_directions_camera @ rotation.T
        z_samples = torch.linspace(
            self.config.depth_min,
            self.config.depth_max,
            self.config.samples_per_ray,
            device=self.device,
            dtype=torch.float32,
        )
        points = position + directions_world[:, None, :] * z_samples[None, :, None]
        lower = self._origin
        extent = torch.as_tensor(
            np.asarray(self._dimensions) * self.config.voxel_size,
            device=self.device,
            dtype=torch.float32,
        )
        normalized_xyz = 2.0 * (points - lower) / extent - 1.0
        in_bounds = torch.all((normalized_xyz >= -1.0) & (normalized_xyz <= 1.0), dim=-1)
        # grid_sample consumes coordinates x=W, y=H, z=D.  Our dense tensor is
        # indexed [world X, world Y, world Z], so sample coordinates are [Z,Y,X].
        sample_grid = normalized_xyz[..., (2, 1, 0)].reshape(1, -1, 1, 1, 3)
        features = (
            self._planning_features()
            if planning_features is None
            else planning_features
        ).unsqueeze(0)
        sampled = torch_functional.grid_sample(
            features, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).reshape(5, ray_directions_camera.shape[0], self.config.samples_per_ray)
        (
            occupancy_values,
            semantic_values,
            observed_values,
            roi_values,
            target_support_values,
        ) = sampled.unbind(0)
        occupancy_values = occupancy_values.clamp(_EPSILON, 1.0 - _EPSILON)
        semantic_values = semantic_values.clamp(_EPSILON, 1.0 - _EPSILON)
        entropy = -semantic_values * torch.log2(semantic_values) - (
            1.0 - semantic_values
        ) * torch.log2(1.0 - semantic_values)
        # Unknown cells have occupancy p=0.5, but must not occlude every ray at
        # the map entrance.  Only observed cells can be opaque; a moderate slope
        # retains spatial gradients at reconstructed surfaces.
        opacity = observed_values.clamp(0.0, 1.0) * torch.sigmoid(
            20.0 * (occupancy_values - 0.60)
        )
        transmittance = torch.cumprod(1.0 - opacity + _EPSILON, dim=-1)
        transmittance = torch.cat(
            (torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]), dim=-1
        )
        novelty = (1.0 - observed_values).clamp(0.0, 1.0)
        roi_relevance = roi_values.clamp(0.0, 1.0)
        target_support = target_support_values.clamp(0.0, 1.0)
        valid = in_bounds.to(torch.float32)
        # The base novelty term explores the explicitly configured target ROI.
        # Mask-derived target support dominates immediately around target
        # evidence, especially on its unobserved side.  The semantic-entropy term
        # keeps uncertain measured boundaries informative.  Consequently changing
        # the target mask measurably changes the gain field and optimized view.
        information = roi_relevance * (
            0.10 * novelty
            + target_support * (0.25 + 0.65 * novelty)
            + 0.10 * entropy * (0.25 + 0.75 * novelty)
        )
        gain = transmittance * information * valid
        return gain.sum(dim=-1).mean()

    def _planning_ray_directions(
        self,
        K: np.ndarray,
        height: int,
        width: int,
        *,
        maximum_rays: int = 64_000,
    ) -> torch.Tensor:
        """Build deterministic camera rays shared by planning and scoring."""
        intrinsics = np.asarray(K, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise NBVInputError("K must be a finite 3x3 matrix")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise NBVInputError("K focal lengths must be positive")
        if isinstance(height, bool) or isinstance(width, bool):
            raise NBVInputError("image dimensions must be positive integers")
        image_height = int(height)
        image_width = int(width)
        if image_height < 1 or image_width < 1:
            raise NBVInputError("image dimensions must be positive integers")
        if not 1 <= int(maximum_rays) <= 64_000:
            raise NBVInputError("maximum_rays must be in [1, 64000]")

        factor = self.config.downsample
        sampled_height = len(range(0, image_height, factor))
        sampled_width = len(range(0, image_width, factor))
        sampled_K = intrinsics.copy()
        sampled_K[0, 0] /= factor
        sampled_K[0, 2] /= factor
        sampled_K[1, 1] /= factor
        sampled_K[1, 2] /= factor
        rows = torch.arange(
            sampled_height, device=self.device, dtype=torch.float32
        )
        columns = torch.arange(
            sampled_width, device=self.device, dtype=torch.float32
        )
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        rays = torch.stack(
            (
                (column_grid - float(sampled_K[0, 2]))
                / float(sampled_K[0, 0]),
                (row_grid - float(sampled_K[1, 2]))
                / float(sampled_K[1, 1]),
                torch.ones_like(row_grid),
            ),
            dim=-1,
        ).reshape(-1, 3)
        if rays.shape[0] > int(maximum_rays):
            indices = torch.linspace(
                0,
                rays.shape[0] - 1,
                int(maximum_rays),
                device=self.device,
            ).round().to(torch.int64)
            rays = rays[indices]
        return rays

    def evaluate_candidate_poses(
        self,
        reference_pose: np.ndarray,
        candidate_poses: np.ndarray,
        K: np.ndarray,
        height: int,
        width: int,
        *,
        maximum_rays: int = 4096,
    ) -> tuple[float, np.ndarray]:
        """Score explicit optical poses against the current map, read-only.

        The smaller deterministic ray set keeps a reachability search over many
        candidates fast.  Scores are comparable within one call; the method
        never fuses observations and never changes coverage or map tensors.
        """
        if self.config is None or self._origin is None:
            raise NBVInputError("NBV core is not configured")
        reference = _rigid_pose(reference_pose, "reference_pose")
        candidates = np.asarray(candidate_poses, dtype=np.float64)
        if candidates.ndim != 3 or candidates.shape[1:] != (4, 4):
            raise NBVInputError("candidate_poses must have shape Nx4x4")
        if not 1 <= candidates.shape[0] <= 256:
            raise NBVInputError("candidate_poses count must be in [1, 256]")
        checked = [
            _rigid_pose(pose, f"candidate_poses[{index}]")
            for index, pose in enumerate(candidates)
        ]
        lower = np.asarray(self.config.observation_min, dtype=np.float64)
        upper = np.asarray(self.config.observation_max, dtype=np.float64)
        for index, pose in enumerate(checked):
            if np.any(pose[:3, 3] < lower - 1.0e-9) or np.any(
                pose[:3, 3] > upper + 1.0e-9
            ):
                raise NBVInputError(
                    f"candidate_poses[{index}] lies outside observation bounds"
                )

        rays = self._planning_ray_directions(
            K, height, width, maximum_rays=maximum_rays
        )
        features = self._planning_features()
        with torch.no_grad():
            reference_gain = float(
                self._candidate_gain_with_rotation(
                    torch.tensor(
                        reference[:3, 3], device=self.device, dtype=torch.float32
                    ),
                    torch.tensor(
                        reference[:3, :3], device=self.device, dtype=torch.float32
                    ),
                    rays,
                    features,
                )
            )
            gains = np.asarray(
                [
                    float(
                        self._candidate_gain_with_rotation(
                            torch.tensor(
                                pose[:3, 3],
                                device=self.device,
                                dtype=torch.float32,
                            ),
                            torch.tensor(
                                pose[:3, :3],
                                device=self.device,
                                dtype=torch.float32,
                            ),
                            rays,
                            features,
                        )
                    )
                    for pose in checked
                ],
                dtype=np.float64,
            )
        if not math.isfinite(reference_gain) or not np.all(np.isfinite(gains)):
            raise NBVInputError("candidate view gain is non-finite")
        return reference_gain, gains

    def _plan(
        self,
        pose: np.ndarray,
        K: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, float, float, tuple[float, ...]]:
        # Bound runtime for unusual inputs while retaining the exact 320x200
        # grid for the approved 640x400 default.
        ray_directions = self._planning_ray_directions(
            K, height, width, maximum_rays=64_000
        )

        current_position = torch.tensor(
            pose[:3, 3], device=self.device, dtype=torch.float32
        )
        current_down = torch.tensor(
            pose[:3, 1], device=self.device, dtype=torch.float32
        )
        lower = torch.maximum(
            torch.tensor(self.config.observation_min, device=self.device, dtype=torch.float32),
            current_position - self.config.max_step,
        )
        upper = torch.minimum(
            torch.tensor(self.config.observation_max, device=self.device, dtype=torch.float32),
            current_position + self.config.max_step,
        )
        if bool(torch.any(lower > upper)):
            raise NBVInputError("current camera pose lies too far outside observation bounds")
        planning_features = self._planning_features()
        position = current_position.detach().clone()
        loss_history: list[float] = []
        best_position = position.detach().clone()
        with torch.no_grad():
            current_gain = float(
                self._candidate_gain(
                    position, ray_directions, current_down, planning_features
                )
            )
        if not math.isfinite(current_gain) or current_gain <= 0.0:
            raise NBVInputError("current-view gain is non-finite or non-positive")
        best_gain = current_gain
        # Start each line search at the configured physical motion radius and
        # backtrack only when that distance does not improve gain.  The former
        # voxel-sized (3 mm) start made ``max_step=0.10`` mostly cosmetic: ten
        # iterations could not reach a useful 5--10 cm viewpoint even when the
        # full-radius candidate had substantially higher gain.  This remains a
        # monotone optimizer because no candidate is accepted unless its gain
        # is strictly greater than the current candidate's gain.
        initial_step = self.config.max_step
        gain_tolerance = max(1.0e-7, abs(current_gain) * 1.0e-6)
        for _ in range(self.config.optimization_steps):
            evaluated_position = position.detach().clone().requires_grad_(True)
            gain = self._candidate_gain(
                evaluated_position, ray_directions, current_down, planning_features
            )
            if not bool(torch.isfinite(gain)):
                raise NBVInputError("next-view gain became non-finite")
            loss = -torch.log(gain + _EPSILON)
            loss.backward()
            gradient = evaluated_position.grad
            if gradient is None or not bool(torch.all(torch.isfinite(gradient))):
                raise NBVInputError("next-view gradient became non-finite")
            numeric_gain = float(gain.detach())
            loss_history.append(float(loss.detach()))
            if numeric_gain > best_gain:
                best_gain = numeric_gain
                best_position = evaluated_position.detach().clone()
            gradient_norm = torch.linalg.vector_norm(gradient)
            if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) <= 1.0e-10:
                break
            direction = -gradient.detach() / gradient_norm
            accepted = False
            step_length = initial_step
            # Twelve halvings retain the old sub-voxel recovery path for
            # nearly-converged maps while still testing 10 cm and 5 cm first
            # in the large-motion profile.
            for _backtrack in range(12):
                with torch.no_grad():
                    candidate = position + direction * step_length
                    candidate.clamp_(lower, upper)
                    displacement = candidate - current_position
                    distance = torch.linalg.vector_norm(displacement)
                    if float(distance) > self.config.max_step:
                        candidate = (
                            current_position
                            + displacement / distance * self.config.max_step
                        )
                    candidate_gain = float(
                        self._candidate_gain(
                            candidate,
                            ray_directions,
                            current_down,
                            planning_features,
                        )
                    )
                if (
                    math.isfinite(candidate_gain)
                    and candidate_gain > numeric_gain + gain_tolerance
                ):
                    position = candidate.detach().clone()
                    accepted = True
                    if candidate_gain > best_gain:
                        best_gain = candidate_gain
                        best_position = position.detach().clone()
                    break
                step_length *= 0.5
            if not accepted:
                break
        target = torch.tensor(
            self.config.target_center, device=self.device, dtype=torch.float32
        )
        rotation = look_at_optical_torch(best_position, target, current_down)
        next_pose = np.eye(4, dtype=np.float64)
        next_pose[:3, :3] = rotation.detach().cpu().numpy()
        # Optimisation runs in float32 for GPU efficiency, while the public pose
        # is float64.  Re-anchor the final translation to the exact input pose so
        # float32 rounding cannot make a nominal 5 mm step exceed its contract
        # by a few tens of nanometres at the ROS safety boundary.
        exact_current_position = np.asarray(pose[:3, 3], dtype=np.float64)
        planned_position = np.asarray(
            best_position.detach().cpu().numpy(), dtype=np.float64
        )
        planned_position = np.clip(
            planned_position,
            np.asarray(self.config.observation_min, dtype=np.float64),
            np.asarray(self.config.observation_max, dtype=np.float64),
        )
        exact_displacement = planned_position - exact_current_position
        exact_distance = float(np.linalg.norm(exact_displacement))
        if exact_distance > self.config.max_step:
            bounded_step = np.nextafter(float(self.config.max_step), 0.0)
            planned_position = (
                exact_current_position
                + exact_displacement / exact_distance * bounded_step
            )
        next_pose[:3, 3] = planned_position
        if not np.all(np.isfinite(next_pose)) or not math.isfinite(best_gain):
            raise NBVInputError("optimized next camera pose is non-finite")
        return next_pose, current_gain, best_gain, tuple(loss_history)

    def update_and_plan(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        T_world_camera_optical: np.ndarray,
    ) -> NBVResult:
        """Validate, fuse exactly once, and optimize the next camera position."""
        started = time.perf_counter()
        depth_array, mask_array, intrinsics, pose = self._validate_observation(
            depth, mask, K, T_world_camera_optical
        )
        # A planning/numerical failure must not leave a partially fused map.
        # Cloning three dense fields is bounded by the configured 20M-cell cap
        # and keeps retry/idempotency semantics unambiguous at the ROS boundary.
        snapshot = (
            self._log_odds.clone(),
            self._semantic_log_odds.clone(),
            self._ever_observed.clone(),
        )
        try:
            self._fuse(depth_array, mask_array, intrinsics, pose)
            next_pose, current_gain, planned_gain, loss_history = self._plan(
                pose, intrinsics, depth_array.shape[0], depth_array.shape[1]
            )
        except Exception:
            self._log_odds.copy_(snapshot[0])
            self._semantic_log_odds.copy_(snapshot[1])
            self._ever_observed.copy_(snapshot[2])
            raise
        observed = int(torch.count_nonzero(self._ever_observed).item())
        occupied = int(torch.count_nonzero(self._log_odds > 0.0).item())
        total = int(np.prod(self._dimensions))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return NBVResult(
            pose=next_pose,
            gain=float(planned_gain),
            current_gain=float(current_gain),
            planned_gain=float(planned_gain),
            gain_improvement=float(planned_gain - current_gain),
            coverage=float(self.coverage),
            total_voxel_count=total,
            observed_voxel_count=observed,
            occupied_voxel_count=occupied,
            unknown_voxel_count=total - observed,
            optimization_iterations=self.config.optimization_steps,
            compute_time_ms=elapsed_ms,
            loss_history=loss_history,
        )
