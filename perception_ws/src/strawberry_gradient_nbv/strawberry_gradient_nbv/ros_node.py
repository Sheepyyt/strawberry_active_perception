"""ROS 2 boundary for the stateful, MoveIt-free Gradient-NBV core.

The node deliberately keeps ROS image handling at the NumPy buffer boundary in
``image_codec``.  It never imports ``cv_bridge`` and it validates the complete
canonical Observation before allowing the stateful backend to update its map.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from strawberry_perception_interfaces.action import ComputeNextView
from strawberry_perception_interfaces.msg import NextView, Observation
from strawberry_perception_interfaces.srv import ConfigureNBV, ResetNBVMap

from .core import GradientNBVCore, NBVConfig, NBVInputError
from .image_codec import (
    ObservationDecodeError,
    camera_matrix,
    decode_depth_32fc1_m,
    decode_mono8,
    decode_rgb8,
    pose_matrix,
)


_OBSERVATION_CACHE_CAPACITY = 32
_MAX_IMAGE_SKEW_SEC = 0.005
_FRACTION_TOLERANCE = 1.0e-5
_SKEW_TOLERANCE_SEC = 1.0e-9


def _safe_path_component(value: str) -> str:
    """Return a short filename component without allowing path traversal."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return (normalized or "unnamed")[:80]


@dataclass(frozen=True)
class _DecodedObservation:
    depth: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray
    pose: np.ndarray


class _ConfigurationError(ValueError):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = int(code)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _device_argument(value: Any) -> str | None:
    normalized = str(value).strip()
    return None if normalized.lower() in ("", "auto") else normalized


def _xyz(value: Any) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float64)


def _finite_vector(value: np.ndarray) -> bool:
    return value.shape == (3,) and bool(np.all(np.isfinite(value)))


def _stamp_nanoseconds(stamp: Any, name: str) -> int:
    """Return a non-zero ROS exposure stamp after strict wire validation."""
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ObservationDecodeError(
            f"{name} must be a valid non-negative ROS timestamp"
        )
    total = seconds * 1_000_000_000 + nanoseconds
    if total == 0:
        raise ObservationDecodeError(f"{name} must not be the zero timestamp")
    return total


def _configuration_mapping(request: ConfigureNBV.Request) -> dict[str, Any]:
    """Validate a service request and return a core-ready mapping.

    Validation is intentionally performed before ``GradientNBVCore.configure``
    so every public error has the most specific ConfigureNBV status code and a
    rejected request cannot disturb the active scene.
    """
    response_type = ConfigureNBV.Response
    scene_id = str(request.scene_id)
    world_frame = str(request.world_frame)
    if not _nonempty(scene_id):
        raise _ConfigurationError(
            response_type.INVALID_SCENE_ID, "scene_id must be non-empty"
        )
    if not _nonempty(world_frame):
        raise _ConfigurationError(
            response_type.INVALID_WORLD_FRAME, "world_frame must be non-empty"
        )

    target_center = _xyz(request.target_center)
    map_size = _xyz(request.map_size)
    target_roi_size = _xyz(request.target_roi_size)
    observation_min = _xyz(request.observation_min)
    observation_max = _xyz(request.observation_max)
    if not _finite_vector(target_center):
        raise _ConfigurationError(
            response_type.INVALID_MAP_BOUNDS,
            "target_center must contain three finite coordinates",
        )
    if not _finite_vector(map_size) or np.any(map_size <= 0.0):
        raise _ConfigurationError(
            response_type.INVALID_MAP_BOUNDS,
            "map_size must contain three finite positive extents",
        )
    if (
        not _finite_vector(target_roi_size)
        or np.any(target_roi_size <= 0.0)
        or np.any(target_roi_size > map_size)
    ):
        raise _ConfigurationError(
            response_type.INVALID_TARGET_ROI,
            "target_roi_size must be finite, positive, and fit inside map_size",
        )
    if (
        not _finite_vector(observation_min)
        or not _finite_vector(observation_max)
        or np.any(observation_min >= observation_max)
    ):
        raise _ConfigurationError(
            response_type.INVALID_OBSERVATION_BOUNDS,
            "observation_min must be finite and strictly below observation_max",
        )

    voxel_size = float(request.voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise _ConfigurationError(
            response_type.INVALID_VOXEL_SIZE,
            "voxel_size must be finite and positive",
        )
    dimension_ratios = map_size / voxel_size
    if (
        not np.all(np.isfinite(dimension_ratios))
        or np.any(dimension_ratios > 20_000_000)
    ):
        raise _ConfigurationError(
            response_type.INVALID_MAP_BOUNDS,
            "voxel grid dimensions are invalid or exceed 20M cells",
        )
    dimensions = tuple(math.ceil(float(value)) for value in dimension_ratios)
    if any(value < 1 for value in dimensions) or math.prod(dimensions) > 20_000_000:
        raise _ConfigurationError(
            response_type.INVALID_MAP_BOUNDS,
            "voxel grid dimensions are invalid or exceed 20M cells",
        )

    depth_min = float(request.depth_min)
    depth_max = float(request.depth_max)
    if (
        not math.isfinite(depth_min)
        or not math.isfinite(depth_max)
        or depth_min < 0.0
        or depth_min >= depth_max
    ):
        raise _ConfigurationError(
            response_type.INVALID_DEPTH_RANGE,
            "depth_min and depth_max must define a finite increasing interval",
        )
    samples_per_ray = int(request.samples_per_ray)
    optimization_steps = int(request.optimization_steps)
    max_step = float(request.max_step)
    if (
        samples_per_ray < 2
        or optimization_steps < 1
        or not math.isfinite(max_step)
        or max_step <= 0.0
    ):
        raise _ConfigurationError(
            response_type.INVALID_OPTIMIZER_CONFIG,
            "samples_per_ray >= 2, optimization_steps >= 1, and max_step > 0 are required",
        )

    return {
        "scene_id": scene_id,
        "world_frame": world_frame,
        "target_center": target_center,
        "map_size": map_size,
        "target_roi_size": target_roi_size,
        "observation_min": observation_min,
        "observation_max": observation_max,
        "voxel_size": voxel_size,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "samples_per_ray": samples_per_ray,
        "optimization_steps": optimization_steps,
        "max_step": max_step,
        "random_seed": int(request.random_seed),
    }


def _validate_result_pose(transform: Any) -> np.ndarray:
    result = np.asarray(transform, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError("backend next-view pose must be a finite 4x4 matrix")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError("backend next-view pose has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("backend next-view pose rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError("backend next-view pose rotation must have determinant +1")
    return result


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to a normalized ROS xyzw quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                (
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ),
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ),
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array(
                (
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ),
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("backend next-view pose produced an invalid quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _pose_stamped(transform: Any, frame_id: str, stamp: Any) -> PoseStamped:
    matrix = _validate_result_pose(transform)
    quaternion = _rotation_to_quaternion(matrix[:3, :3])
    message = PoseStamped()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.pose.position.x = float(matrix[0, 3])
    message.pose.position.y = float(matrix[1, 3])
    message.pose.position.z = float(matrix[2, 3])
    message.pose.orientation.x = quaternion[0]
    message.pose.orientation.y = quaternion[1]
    message.pose.orientation.z = quaternion[2]
    message.pose.orientation.w = quaternion[3]
    return message


def _result_value(result: Any, *names: str) -> Any:
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
        if isinstance(result, Mapping) and name in result:
            return result[name]
    raise ValueError(f"backend result is missing {names[0]}")


class GradientNBVNode(Node):
    """Cache canonical observations and expose one atomic NBV map session."""

    def __init__(self, backend: Any | None = None) -> None:
        """Create the ROS entities and an optional injected numerical backend."""
        super().__init__("gradient_nbv")
        self._callbacks = ReentrantCallbackGroup()
        self._state_lock = threading.RLock()
        self._observations: OrderedDict[tuple[str, str], Observation] = OrderedDict()
        # Successful results live for the complete configured-map generation.
        # Keeping the message, rather than only a processed-ID set, lets every
        # duplicate action return the exact original diagnostics without fusion.
        self._results: dict[tuple[str, str], NextView] = {}
        self._configuration: dict[str, Any] | None = None
        self._map_generation = 0
        self._map_step = 0
        self._camera_pose_history: list[np.ndarray] = []

        self.declare_parameter(
            "observation_topic", "/strawberry/perception/observation"
        )
        self.declare_parameter("next_view_topic", "/strawberry/nbv/next_view")
        self.declare_parameter("configure_service", "/strawberry/nbv/configure")
        self.declare_parameter("reset_service", "/strawberry/nbv/reset_map")
        self.declare_parameter(
            "compute_action", "/strawberry/nbv/compute_next_view"
        )
        self.declare_parameter("device", "")
        self.declare_parameter(
            "observation_cache_size", _OBSERVATION_CACHE_CAPACITY
        )
        self.declare_parameter("map_snapshot_directory", "")

        self._observation_cache_capacity = int(
            self.get_parameter("observation_cache_size").value
        )
        if not 1 <= self._observation_cache_capacity <= _OBSERVATION_CACHE_CAPACITY:
            raise ValueError("observation_cache_size must be between 1 and 32")

        if backend is None:
            backend = GradientNBVCore(
                device=_device_argument(self.get_parameter("device").value)
            )
        self._backend = backend

        self._observation_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self._observation_cache_capacity,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._next_view_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._observation_subscription = self.create_subscription(
            Observation,
            str(self.get_parameter("observation_topic").value),
            self._on_observation,
            self._observation_qos,
            callback_group=self._callbacks,
        )
        self._next_view_publisher = self.create_publisher(
            NextView,
            str(self.get_parameter("next_view_topic").value),
            self._next_view_qos,
        )
        self._configure_service = self.create_service(
            ConfigureNBV,
            str(self.get_parameter("configure_service").value),
            self._on_configure,
            callback_group=self._callbacks,
        )
        self._reset_service = self.create_service(
            ResetNBVMap,
            str(self.get_parameter("reset_service").value),
            self._on_reset,
            callback_group=self._callbacks,
        )
        self._compute_server = ActionServer(
            self,
            ComputeNextView,
            str(self.get_parameter("compute_action").value),
            execute_callback=self._execute_compute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callbacks,
        )

    # These three methods intentionally contain the entire core API adaptation.
    # Tests inject a lightweight backend, while production uses GradientNBVCore.
    def _backend_configure(self, config_dict: Mapping[str, Any]) -> None:
        self._backend.configure(NBVConfig.from_mapping(config_dict))

    def _backend_reset(self) -> bool:
        return bool(self._backend.reset())

    def _backend_snapshot(self) -> Any:
        return self._backend.snapshot()

    def _backend_restore(self, snapshot: Any) -> None:
        self._backend.restore(snapshot)

    def _backend_update_and_plan(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        pose: np.ndarray,
    ) -> Any:
        return self._backend.update_and_plan(depth, mask, intrinsics, pose)

    def _save_map_snapshot(
        self,
        scene_id: str,
        observation_id: str,
        decoded: _DecodedObservation,
        backend_result: Any,
        output: NextView,
    ) -> tuple[Path | None, list[np.ndarray]]:
        """Persist an optional human-readable map input after one update."""
        configured_directory = str(
            self.get_parameter("map_snapshot_directory").value
        ).strip()
        history = [*self._camera_pose_history, decoded.pose.copy()]
        if not configured_directory:
            return None, history
        visualization_state = getattr(self._backend, "visualization_state", None)
        if visualization_state is None:
            raise RuntimeError(
                "map_snapshot_directory requires a backend with visualization_state()"
            )
        from .map_visualization import save_map_snapshot

        directory = (
            Path(configured_directory).expanduser().resolve()
            / _safe_path_component(scene_id)
            / f"generation_{self._map_generation:03d}"
        )
        filename = (
            f"map_step_{self._map_step + 1:03d}_"
            f"{_safe_path_component(observation_id)}.npz"
        )
        path = save_map_snapshot(
            directory / filename,
            visualization_state(),
            scene_id=scene_id,
            observation_id=observation_id,
            world_frame=str(self._configuration["world_frame"]),
            coverage=float(output.coverage),
            current_camera_pose=decoded.pose,
            next_camera_pose=_validate_result_pose(
                _result_value(backend_result, "pose", "T_world_camera")
            ),
            camera_pose_history=np.stack(history),
            configuration=self._configuration,
        )
        return path, history

    def _restore_transaction(self, snapshot: Any) -> str | None:
        """Best-effort rollback; return a diagnostic only if rollback failed."""
        try:
            self._backend_restore(snapshot)
        except Exception as error:
            self.get_logger().error(f"NBV map transaction rollback failed: {error}")
            return str(error)
        return None

    def _on_observation(self, message: Observation) -> None:
        scene_id = str(message.scene_id)
        observation_id = str(message.observation_id)
        if not _nonempty(scene_id) or not _nonempty(observation_id):
            self.get_logger().warning(
                "Discarding Observation with an empty scene_id or observation_id"
            )
            return
        key = (scene_id, observation_id)
        with self._state_lock:
            if key in self._observations:
                self.get_logger().warning(
                    "Discarding duplicate Observation ID; retaining first payload "
                    f"for {scene_id}/{observation_id}"
                )
                return
            self._observations[key] = deepcopy(message)
            while len(self._observations) > self._observation_cache_capacity:
                self._observations.popitem(last=False)

    def _on_configure(
        self, request: ConfigureNBV.Request, response: ConfigureNBV.Response
    ) -> ConfigureNBV.Response:
        try:
            config = _configuration_mapping(request)
        except _ConfigurationError as error:
            response.success = False
            response.code = error.code
            response.reason = str(error)
            return response

        # The core builds all replacement tensors before assigning them.  The
        # wrapper similarly swaps its public session state only after success.
        with self._state_lock:
            try:
                self._backend_configure(config)
            except Exception as error:  # backend allocation/runtime failures
                response.success = False
                response.code = ConfigureNBV.Response.INTERNAL_ERROR
                response.reason = f"failed to configure NBV backend: {error}"
                self.get_logger().error(response.reason)
                return response
            self._configuration = config
            self._results.clear()
            self._observations.clear()
            self._map_generation += 1
            self._map_step = 0
            self._camera_pose_history.clear()
        response.success = True
        response.code = ConfigureNBV.Response.SUCCESS
        response.reason = "configuration applied atomically and map reset"
        return response

    def _on_reset(
        self, request: ResetNBVMap.Request, response: ResetNBVMap.Response
    ) -> ResetNBVMap.Response:
        scene_id = str(request.scene_id)
        if not _nonempty(scene_id):
            response.success = False
            response.code = ResetNBVMap.Response.INVALID_SCENE_ID
            response.reason = "scene_id must be non-empty"
            return response
        with self._state_lock:
            if self._configuration is None:
                response.success = False
                response.code = ResetNBVMap.Response.NOT_CONFIGURED
                response.reason = "NBV map is not configured"
                return response
            configured_scene = str(self._configuration["scene_id"])
            if scene_id != configured_scene:
                response.success = False
                response.code = ResetNBVMap.Response.SCENE_MISMATCH
                response.reason = (
                    f"scene_id mismatch: configured {configured_scene!r}, "
                    f"requested {scene_id!r}"
                )
                return response
            try:
                snapshot = self._backend_snapshot()
            except Exception as error:
                response.success = False
                response.code = ResetNBVMap.Response.INTERNAL_ERROR
                response.reason = f"failed to begin atomic map reset: {error}"
                self.get_logger().error(response.reason)
                return response
            try:
                had_data = self._backend_reset()
            except Exception as error:
                rollback_error = self._restore_transaction(snapshot)
                rollback_suffix = (
                    "" if rollback_error is None
                    else f"; map rollback also failed: {rollback_error}"
                )
                response.success = False
                response.code = ResetNBVMap.Response.INTERNAL_ERROR
                response.reason = (
                    f"failed to reset NBV backend: {error}{rollback_suffix}"
                )
                self.get_logger().error(response.reason)
                return response
            self._results.clear()
            self._observations.clear()
            self._map_generation += 1
            self._map_step = 0
            self._camera_pose_history.clear()
        response.success = True
        response.code = (
            ResetNBVMap.Response.SUCCESS
            if had_data
            else ResetNBVMap.Response.ALREADY_EMPTY
        )
        response.reason = "map reset" if had_data else "map was already empty"
        return response

    @staticmethod
    def _on_goal(_goal_request: ComputeNextView.Goal) -> GoalResponse:
        # Accept malformed references so the caller receives a structured
        # NextView INVALID_REQUEST instead of an opaque action rejection.
        return GoalResponse.ACCEPT

    @staticmethod
    def _on_cancel(_goal_handle: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _decode_observation(
        self, message: Observation, scene_id: str, observation_id: str
    ) -> _DecodedObservation:
        if message.scene_id != scene_id or message.observation_id != observation_id:
            raise ObservationDecodeError(
                "cached Observation identifiers do not match the action goal"
            )
        source_type = int(message.source_type)
        supported_sources = {
            Observation.SOURCE_REAL,
            Observation.SOURCE_OFFLINE,
            Observation.SOURCE_SYNTHETIC,
            Observation.SOURCE_REPLAY,
        }
        if source_type not in supported_sources:
            raise ObservationDecodeError(
                f"source_type {source_type} must identify a known Observation source"
            )
        if not _nonempty(str(message.source_name)):
            raise ObservationDecodeError("source_name must be non-empty")
        optical_frame = str(message.header.frame_id)
        if not _nonempty(optical_frame):
            raise ObservationDecodeError(
                "Observation header.frame_id must name a non-empty optical frame"
            )
        framed_fields = (
            ("color", message.color),
            ("depth", message.depth),
            ("target_mask", message.target_mask),
            ("camera_info", message.camera_info),
        )
        for name, field in framed_fields:
            if str(field.header.frame_id) != optical_frame:
                raise ObservationDecodeError(
                    f"{name}.header.frame_id must equal Observation frame "
                    f"{optical_frame!r}"
                )
        if not bool(message.pose_valid):
            raise ObservationDecodeError("camera pose is explicitly invalid")
        world_frame = str(self._configuration["world_frame"])
        if str(message.camera_pose.header.frame_id) != world_frame:
            raise ObservationDecodeError(
                "camera_pose.header.frame_id must equal configured world frame "
                f"{world_frame!r}"
            )

        observation_stamp = _stamp_nanoseconds(
            message.header.stamp, "Observation.header.stamp"
        )
        depth_stamp = _stamp_nanoseconds(message.depth.header.stamp, "depth stamp")
        color_stamp = _stamp_nanoseconds(message.color.header.stamp, "color stamp")
        mask_stamp = _stamp_nanoseconds(
            message.target_mask.header.stamp, "target_mask stamp"
        )
        camera_info_stamp = _stamp_nanoseconds(
            message.camera_info.header.stamp, "camera_info stamp"
        )
        pose_stamp = _stamp_nanoseconds(
            message.camera_pose.header.stamp, "camera_pose stamp"
        )
        if observation_stamp != depth_stamp:
            raise ObservationDecodeError(
                "Observation.header.stamp must equal the depth exposure stamp"
            )
        if camera_info_stamp != depth_stamp or pose_stamp != depth_stamp:
            raise ObservationDecodeError(
                "CameraInfo and camera_pose stamps must equal the depth exposure stamp"
            )
        color_skew_sec = (color_stamp - depth_stamp) / 1_000_000_000.0
        mask_skew_sec = (mask_stamp - depth_stamp) / 1_000_000_000.0
        if (
            abs(color_skew_sec) > _MAX_IMAGE_SKEW_SEC
            or abs(mask_skew_sec) > _MAX_IMAGE_SKEW_SEC
        ):
            raise ObservationDecodeError(
                "color and target_mask stamps must be within 5 ms of depth"
            )
        declared_skew = float(message.color_depth_skew_sec)
        if not math.isfinite(declared_skew) or not math.isclose(
            declared_skew,
            color_skew_sec,
            rel_tol=0.0,
            abs_tol=_SKEW_TOLERANCE_SEC,
        ):
            raise ObservationDecodeError(
                "color_depth_skew_sec must be finite and match the header stamps"
            )

        grids = {
            "color": (int(message.color.width), int(message.color.height)),
            "depth": (int(message.depth.width), int(message.depth.height)),
            "target_mask": (
                int(message.target_mask.width),
                int(message.target_mask.height),
            ),
            "camera_info": (
                int(message.camera_info.width),
                int(message.camera_info.height),
            ),
        }
        expected_grid = grids["depth"]
        if expected_grid[0] <= 0 or expected_grid[1] <= 0:
            raise ObservationDecodeError("registered image dimensions must be positive")
        mismatched = [name for name, grid in grids.items() if grid != expected_grid]
        if mismatched:
            raise ObservationDecodeError(
                "registered grid mismatch for " + ", ".join(mismatched)
            )

        # Decode color even though the numerical core does not consume it: a
        # canonical Observation with a malformed color buffer is not accepted.
        decode_rgb8(message.color)
        depth = decode_depth_32fc1_m(message.depth)
        mask = decode_mono8(message.target_mask)
        if not np.any(mask == 255):
            raise ObservationDecodeError("target mask contains no target pixels")
        intrinsics = camera_matrix(message.camera_info)

        orientation = message.camera_pose.pose.orientation
        quaternion = np.asarray(
            (orientation.x, orientation.y, orientation.z, orientation.w),
            dtype=np.float64,
        )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(quaternion_norm) or not math.isclose(
            quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3
        ):
            raise ObservationDecodeError("camera pose quaternion must have unit norm")
        pose = pose_matrix(message.camera_pose)

        depth_min = float(self._configuration["depth_min"])
        depth_max = float(self._configuration["depth_max"])
        valid_depth = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
        if not np.any(valid_depth):
            raise ObservationDecodeError(
                "depth contains no finite samples in the configured interval"
            )
        if not np.any(valid_depth & (mask == 255)):
            raise ObservationDecodeError(
                "target mask has no overlap with depth valid for this NBV session"
            )
        declared_fraction = float(message.valid_depth_fraction)
        actual_fraction = float(np.count_nonzero(np.isfinite(depth)) / depth.size)
        if (
            not math.isfinite(declared_fraction)
            or not 0.0 <= declared_fraction <= 1.0
            or not math.isclose(
                declared_fraction,
                actual_fraction,
                rel_tol=0.0,
                abs_tol=_FRACTION_TOLERANCE,
            )
        ):
            raise ObservationDecodeError(
                "valid_depth_fraction must match the canonical finite-depth pixel ratio"
            )
        return _DecodedObservation(depth, mask, intrinsics, pose)

    @staticmethod
    def _failure(scene_id: str, observation_id: str, code: int, reason: str) -> NextView:
        message = NextView()
        message.scene_id = scene_id
        message.observation_id = observation_id
        message.success = False
        message.code = int(code)
        message.reason = reason
        return message

    def _success_message(
        self,
        scene_id: str,
        observation_id: str,
        observation: Observation,
        result: Any,
    ) -> NextView:
        message = NextView()
        message.scene_id = scene_id
        message.observation_id = observation_id
        message.success = True
        message.code = NextView.SUCCESS
        message.reason = "next view computed after one idempotent map update"
        transform = _result_value(result, "pose", "T_world_camera")
        message.pose = _pose_stamped(
            transform,
            str(self._configuration["world_frame"]),
            observation.header.stamp,
        )
        message.gain = float(_result_value(result, "gain"))
        message.coverage = float(_result_value(result, "coverage"))
        message.total_voxel_count = int(
            _result_value(result, "total_voxel_count", "total")
        )
        message.observed_voxel_count = int(
            _result_value(result, "observed_voxel_count", "observed")
        )
        message.occupied_voxel_count = int(
            _result_value(result, "occupied_voxel_count", "occupied")
        )
        message.unknown_voxel_count = int(
            _result_value(result, "unknown_voxel_count", "unknown")
        )
        message.optimization_iterations = int(
            _result_value(result, "optimization_iterations", "iterations")
        )
        message.compute_time_ms = float(_result_value(result, "compute_time_ms"))
        numeric = (message.gain, message.coverage, message.compute_time_ms)
        counts = (
            message.total_voxel_count,
            message.observed_voxel_count,
            message.occupied_voxel_count,
            message.unknown_voxel_count,
            message.optimization_iterations,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("backend result contains non-finite diagnostics")
        if not 0.0 <= message.coverage <= 1.0 or message.compute_time_ms < 0.0:
            raise ValueError("backend result coverage or compute time is outside its range")
        if any(value < 0 for value in counts):
            raise ValueError("backend result contains negative voxel/iteration counts")
        if message.observed_voxel_count + message.unknown_voxel_count != message.total_voxel_count:
            raise ValueError("backend voxel counts are inconsistent")
        if message.occupied_voxel_count > message.observed_voxel_count:
            raise ValueError("occupied voxel count exceeds observed voxel count")
        return message

    def _publish_feedback(
        self,
        goal_handle: Any,
        phase: int,
        started: float,
        coverage: float = 0.0,
        iteration: int = 0,
    ) -> None:
        feedback = ComputeNextView.Feedback()
        feedback.phase = int(phase)
        feedback.coverage = float(np.clip(coverage, 0.0, 1.0))
        elapsed_ns = max(0, int((time.perf_counter() - started) * 1_000_000_000))
        feedback.elapsed = Duration(nanoseconds=elapsed_ns).to_msg()
        feedback.iteration = max(0, int(iteration))
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _backend_error(error: NBVInputError) -> tuple[int, str]:
        reason = str(error)
        lowered = reason.lower()
        if "gradient" in lowered or "non-finite" in lowered:
            return NextView.OPTIMIZATION_FAILED, f"next-view optimization failed: {reason}"
        if (
            "candidate" in lowered
            or "observation bounds" in lowered
            or "camera position" in lowered
            or "gain" in lowered
        ):
            return NextView.NO_VALID_CANDIDATE, f"no valid next-view candidate: {reason}"
        return NextView.MAP_UPDATE_FAILED, f"map update failed: {reason}"

    def _execute_compute(self, goal_handle: Any) -> ComputeNextView.Result:
        started = time.perf_counter()
        scene_id = str(goal_handle.request.scene_id)
        observation_id = str(goal_handle.request.observation_id)
        self._publish_feedback(
            goal_handle, ComputeNextView.Feedback.PHASE_VALIDATING, started
        )

        with self._state_lock:
            if goal_handle.is_cancel_requested:
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.CANCELED,
                    "request canceled before map update",
                )
                return self._complete_result(goal_handle, output, started, False)
            if not _nonempty(scene_id) or not _nonempty(observation_id):
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.INVALID_REQUEST,
                    "scene_id and observation_id must be non-empty",
                )
                return self._complete_result(goal_handle, output, started, False)
            if self._configuration is None:
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.NOT_CONFIGURED,
                    "NBV map is not configured",
                )
                return self._complete_result(goal_handle, output, started, False)
            configured_scene = str(self._configuration["scene_id"])
            if scene_id != configured_scene:
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.SCENE_MISMATCH,
                    f"scene_id mismatch: configured {configured_scene!r}, requested {scene_id!r}",
                )
                return self._complete_result(goal_handle, output, started, False)

            key = (scene_id, observation_id)
            cached = self._results.get(key)
            if cached is not None:
                return self._complete_result(
                    goal_handle, deepcopy(cached), started, True
                )
            observation = self._observations.get(key)
            if observation is None:
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.OBSERVATION_NOT_FOUND,
                    "referenced canonical Observation is not present in the "
                    f"{self._observation_cache_capacity}-entry cache",
                )
                return self._complete_result(goal_handle, output, started, False)

            try:
                decoded = self._decode_observation(
                    observation, scene_id, observation_id
                )
            except (ObservationDecodeError, ValueError, TypeError) as error:
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.INVALID_REQUEST,
                    f"invalid canonical Observation: {error}",
                )
                return self._complete_result(goal_handle, output, started, False)
            if goal_handle.is_cancel_requested:
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.CANCELED,
                    "request canceled before map update",
                )
                return self._complete_result(goal_handle, output, started, False)

            self._publish_feedback(
                goal_handle, ComputeNextView.Feedback.PHASE_UPDATING_MAP, started
            )
            self._publish_feedback(
                goal_handle, ComputeNextView.Feedback.PHASE_OPTIMIZING, started
            )
            try:
                snapshot = self._backend_snapshot()
            except Exception as error:
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.INTERNAL_ERROR,
                    f"failed to begin atomic NBV map transaction: {error}",
                )
                self.get_logger().error(output.reason)
                return self._complete_result(goal_handle, output, started, False)
            try:
                backend_result = self._backend_update_and_plan(
                    decoded.depth,
                    decoded.mask,
                    decoded.intrinsics,
                    decoded.pose,
                )
                if goal_handle.is_cancel_requested:
                    rollback_error = self._restore_transaction(snapshot)
                    if rollback_error is None:
                        output = self._failure(
                            scene_id,
                            observation_id,
                            NextView.CANCELED,
                            "request canceled; completed map update was rolled back",
                        )
                    else:
                        output = self._failure(
                            scene_id,
                            observation_id,
                            NextView.INTERNAL_ERROR,
                            "request canceled but map rollback failed: "
                            f"{rollback_error}",
                        )
                    return self._complete_result(
                        goal_handle, output, started, False
                    )
                output = self._success_message(
                    scene_id, observation_id, observation, backend_result
                )
                snapshot_path, camera_history = self._save_map_snapshot(
                    scene_id,
                    observation_id,
                    decoded,
                    backend_result,
                    output,
                )
                # Cache construction is part of the transaction: a message
                # that cannot be retained must not commit a fused map update.
                self._results[key] = deepcopy(output)
                self._camera_pose_history = camera_history
                self._map_step += 1
                if snapshot_path is not None:
                    self.get_logger().info(
                        f"Saved Gradient-NBV map snapshot: {snapshot_path}"
                    )
            except NBVInputError as error:
                rollback_error = self._restore_transaction(snapshot)
                if rollback_error is None:
                    code, reason = self._backend_error(error)
                else:
                    code = NextView.INTERNAL_ERROR
                    reason = (
                        f"NBV operation failed ({error}) and map rollback failed: "
                        f"{rollback_error}"
                    )
                output = self._failure(scene_id, observation_id, code, reason)
                return self._complete_result(goal_handle, output, started, False)
            except Exception as error:
                rollback_error = self._restore_transaction(snapshot)
                rollback_suffix = (
                    "" if rollback_error is None
                    else f"; map rollback also failed: {rollback_error}"
                )
                output = self._failure(
                    scene_id,
                    observation_id,
                    NextView.INTERNAL_ERROR,
                    f"unexpected NBV backend/result failure: {error}{rollback_suffix}",
                )
                self.get_logger().error(output.reason)
                return self._complete_result(goal_handle, output, started, False)

            # The lock covers lookup, fusion, and insertion.  Concurrent goals
            # for the same ID therefore cannot both enter the backend.
            return self._complete_result(goal_handle, output, started, True)

    def _complete_result(
        self,
        goal_handle: Any,
        output: NextView,
        started: float,
        success: bool,
    ) -> ComputeNextView.Result:
        self._publish_feedback(
            goal_handle,
            ComputeNextView.Feedback.PHASE_FINALIZING,
            started,
            output.coverage,
            output.optimization_iterations,
        )
        self._next_view_publisher.publish(deepcopy(output))
        if success:
            goal_handle.succeed()
        elif output.code == NextView.CANCELED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result = ComputeNextView.Result()
        result.next_view = deepcopy(output)
        return result


def main(args: list[str] | None = None) -> None:
    """Run the wrapper with concurrent topic, service, and action callbacks."""
    rclpy.init(args=args)
    node = GradientNBVNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
