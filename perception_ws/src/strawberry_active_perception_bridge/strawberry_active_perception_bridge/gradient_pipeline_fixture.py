"""Exercise synthetic Observation -> Gradient-NBV -> Placo SolveIK, without motion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

from diagnostic_msgs.msg import DiagnosticArray
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from strawberry_perception_interfaces.action import ComputeNextView
from strawberry_perception_interfaces.msg import Observation
from strawberry_perception_interfaces.srv import ConfigureNBV
from tf2_ros import Buffer, TransformException, TransformListener

from .simulation_fixture import (
    _diagnostic_values,
    _fill_pose,
    _transform_message_to_matrix,
)


SCENE_ID = "g4_gradient_pipeline_v1"
OBSERVATION_ID = "synthetic_ready_view_000"


def _fill_image(
    message: Image,
    array: np.ndarray,
    encoding: str,
    frame_id: str,
    stamp,
) -> None:
    """Serialize one tightly packed, little-endian canonical image."""
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.height = height
    message.width = width
    message.encoding = encoding
    message.is_bigendian = False
    message.step = width * channels * array.dtype.itemsize
    message.data = np.ascontiguousarray(array).tobytes()


def make_pipeline_observation(
    current_base_camera: np.ndarray,
    stamp,
    camera_frame: str = "camera_sim_optical_frame",
    base_frame: str = "base_link",
    width: int = 160,
    height: int = 100,
) -> tuple[Observation, np.ndarray]:
    """Create a red circular target 350 mm along the current optical +Z axis."""
    current = np.asarray(current_base_camera, dtype=np.float64)
    if current.shape != (4, 4) or width < 32 or height < 24:
        raise ValueError("invalid current pose or fixture image dimensions")
    target_center = current[:3, 3] + current[:3, 2] * 0.35

    rows, columns = np.mgrid[:height, :width]
    radius = max(8, min(width, height) // 7)
    target_pixels = (
        (columns - (width - 1) / 2.0) ** 2
        + (rows - (height - 1) / 2.0) ** 2
        <= radius**2
    )
    color = np.full((height, width, 3), 48, dtype=np.uint8)
    color[target_pixels] = (255, 0, 0)
    depth = np.full((height, width), 0.48, dtype="<f4")
    depth[target_pixels] = 0.35
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[target_pixels] = 255

    focal = float(width) * 0.9
    intrinsics = np.array(
        (
            (focal, 0.0, (width - 1) / 2.0),
            (0.0, focal, (height - 1) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )

    observation = Observation()
    observation.header.frame_id = camera_frame
    observation.header.stamp = stamp
    observation.scene_id = SCENE_ID
    observation.observation_id = OBSERVATION_ID
    observation.source_type = Observation.SOURCE_SYNTHETIC
    observation.source_name = "gradient_placo_pipeline_fixture"
    _fill_image(observation.color, color, "rgb8", camera_frame, stamp)
    _fill_image(observation.depth, depth, "32FC1", camera_frame, stamp)
    _fill_image(observation.target_mask, mask, "mono8", camera_frame, stamp)

    info = CameraInfo()
    info.header.frame_id = camera_frame
    info.header.stamp = stamp
    info.height = height
    info.width = width
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = intrinsics.reshape(-1).tolist()
    info.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
    info.p = [
        focal,
        0.0,
        (width - 1) / 2.0,
        0.0,
        0.0,
        focal,
        (height - 1) / 2.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    observation.camera_info = info
    _fill_pose(observation.camera_pose, current, base_frame, stamp)
    observation.pose_valid = True
    observation.valid_depth_fraction = 1.0
    observation.color_depth_skew_sec = 0.0
    return observation, target_center


def _configure_request(
    current_base_camera: np.ndarray,
    target_center: np.ndarray,
    base_frame: str,
) -> ConfigureNBV.Request:
    request = ConfigureNBV.Request()
    request.scene_id = SCENE_ID
    request.world_frame = base_frame
    request.target_center.x, request.target_center.y, request.target_center.z = (
        float(value) for value in target_center
    )
    request.map_size.x = request.map_size.y = request.map_size.z = 0.30
    request.target_roi_size.x = 0.15
    request.target_roi_size.y = 0.15
    request.target_roi_size.z = 0.15
    camera_position = current_base_camera[:3, 3]
    lower = camera_position - 0.10
    upper = camera_position + 0.10
    request.observation_min.x, request.observation_min.y, request.observation_min.z = (
        float(value) for value in lower
    )
    request.observation_max.x, request.observation_max.y, request.observation_max.z = (
        float(value) for value in upper
    )
    request.voxel_size = 0.003
    request.depth_min = 0.10
    request.depth_max = 0.75
    request.samples_per_ray = 128
    request.optimization_steps = 10
    request.max_step = 0.10
    request.random_seed = 0
    return request


def _wait_for_camera_transform(
    node: Node,
    buffer: Buffer,
    base_frame: str,
    camera_frame: str,
    deadline: float,
) -> np.ndarray:
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            transform = buffer.lookup_transform(
                base_frame,
                camera_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
            return _transform_message_to_matrix(transform)
        except TransformException:
            continue
    raise RuntimeError(f"TF {base_frame} <- {camera_frame} was unavailable")


def _compute_next_view(
    node: Node,
    client: ActionClient,
    deadline: float,
):
    """Request the configured observation and return its structured NextView."""
    goal = ComputeNextView.Goal()
    goal.scene_id = SCENE_ID
    goal.observation_id = OBSERVATION_ID
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(
        node,
        send_future,
        timeout_sec=max(0.0, deadline - time.monotonic()),
    )
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError("ComputeNextView goal was rejected")
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(
        node,
        result_future,
        timeout_sec=max(0.0, deadline - time.monotonic()),
    )
    wrapped_result = result_future.result()
    if wrapped_result is None:
        raise RuntimeError("ComputeNextView did not finish before timeout")
    next_view = wrapped_result.result.next_view
    if not next_view.success:
        raise RuntimeError(
            f"ComputeNextView failed ({next_view.code}): {next_view.reason}"
        )
    return next_view


def _next_view_signature(message) -> tuple[object, ...]:
    """Return every public result field in a directly comparable form."""
    pose = message.pose.pose
    return (
        message.scene_id,
        message.observation_id,
        message.success,
        message.code,
        message.reason,
        message.pose.header.frame_id,
        message.pose.header.stamp.sec,
        message.pose.header.stamp.nanosec,
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
        message.gain,
        message.coverage,
        message.total_voxel_count,
        message.observed_voxel_count,
        message.occupied_voxel_count,
        message.unknown_voxel_count,
        message.optimization_iterations,
        message.compute_time_ms,
    )


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    """Persist one complete evidence document without exposing partial JSON."""
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _diagnostic_boolean(value: object, field: str) -> bool:
    """Decode the controller's human-readable diagnostic boolean."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"controller diagnostic {field} is not boolean: {value!r}")


def main(args=None) -> None:
    """Run the complete synthetic pipeline and print its read-only evidence."""
    rclpy.init(args=args)
    node = Node("gradient_placo_pipeline_fixture")
    node.declare_parameter("base_frame", "base_link")
    node.declare_parameter("camera_frame", "camera_sim_optical_frame")
    node.declare_parameter("timeout_sec", 30.0)
    node.declare_parameter("output_path", "")
    base_frame = str(node.get_parameter("base_frame").value)
    camera_frame = str(node.get_parameter("camera_frame").value)
    timeout_sec = float(node.get_parameter("timeout_sec").value)
    output_path = str(node.get_parameter("output_path").value).strip()
    deadline = time.monotonic() + timeout_sec

    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    observation_publisher = node.create_publisher(
        Observation,
        "/strawberry/perception/observation",
        qos,
    )
    diagnostics: list[dict[str, object]] = []
    controller_diagnostics: list[dict[str, object]] = []
    command_messages: list[int] = []

    def diagnostic_callback(message: DiagnosticArray) -> None:
        for status in message.status:
            values = _diagnostic_values(status)
            if (
                values.get("scene_id") == SCENE_ID
                and values.get("observation_id") == OBSERVATION_ID
            ):
                diagnostics.append(
                    {
                        "name": status.name,
                        "message": status.message,
                        "hardware_id": status.hardware_id,
                        "values": values,
                    }
                )

    node.create_subscription(
        DiagnosticArray,
        "/strawberry/active_perception/ik_preview",
        diagnostic_callback,
        qos,
    )
    live_qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    def controller_diagnostic_callback(message: DiagnosticArray) -> None:
        for status in message.status:
            if status.hardware_id != "nero" or not status.name.endswith(": safety"):
                continue
            controller_diagnostics.append(
                {
                    "name": status.name,
                    "message": status.message,
                    "values": _diagnostic_values(status),
                }
            )

    def command_callback(_message: JointState) -> None:
        command_messages.append(time.monotonic_ns())

    node.create_subscription(
        DiagnosticArray,
        "/strawberry_nero/diagnostics",
        controller_diagnostic_callback,
        live_qos,
    )
    node.create_subscription(
        JointState,
        "/control/move_j",
        command_callback,
        live_qos,
    )
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    configure = node.create_client(ConfigureNBV, "/strawberry/nbv/configure")
    compute = ActionClient(
        node,
        ComputeNextView,
        "/strawberry/nbv/compute_next_view",
    )

    try:
        safety_discovery_deadline = min(deadline, time.monotonic() + 3.0)
        while time.monotonic() < safety_discovery_deadline and (
            not controller_diagnostics
            or node.count_publishers("/control/move_j") == 0
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if not controller_diagnostics:
            raise RuntimeError("NERO controller safety diagnostic was unavailable")
        command_publishers_before = node.count_publishers("/control/move_j")
        monitor_started = time.monotonic()
        current = _wait_for_camera_transform(
            node, tf_buffer, base_frame, camera_frame, deadline
        )
        stamp = node.get_clock().now().to_msg()
        observation, target_center = make_pipeline_observation(
            current, stamp, camera_frame, base_frame
        )
        if not configure.wait_for_service(
            timeout_sec=max(0.0, deadline - time.monotonic())
        ):
            raise RuntimeError("ConfigureNBV service is unavailable")
        configure_future = configure.call_async(
            _configure_request(current, target_center, base_frame)
        )
        rclpy.spin_until_future_complete(
            node,
            configure_future,
            timeout_sec=max(0.0, deadline - time.monotonic()),
        )
        configure_result = configure_future.result()
        if configure_result is None or not configure_result.success:
            reason = "no response" if configure_result is None else configure_result.reason
            raise RuntimeError(f"ConfigureNBV failed: {reason}")

        discovery_deadline = min(deadline, time.monotonic() + 3.0)
        while (
            observation_publisher.get_subscription_count() < 2
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        observation_publisher.publish(observation)
        # Let both the NBV wrapper and IK bridge cache the same canonical object.
        ready_at = time.monotonic() + 0.25
        while time.monotonic() < ready_at:
            rclpy.spin_once(node, timeout_sec=0.05)

        if not compute.wait_for_server(
            timeout_sec=max(0.0, deadline - time.monotonic())
        ):
            raise RuntimeError("ComputeNextView action is unavailable")
        next_view = _compute_next_view(node, compute, deadline)

        while time.monotonic() < deadline and not diagnostics:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not diagnostics:
            raise RuntimeError("no matching read-only IK preview diagnostic was received")
        diagnostic = diagnostics[-1]
        repeated_view = _compute_next_view(node, compute, deadline)
        idempotent_result = _next_view_signature(repeated_view) == _next_view_signature(
            next_view
        )
        if not idempotent_result:
            raise RuntimeError("duplicate action did not return the exact cached NextView")

        # Observe another live controller status after both Action requests.
        # The command subscriber was established before either request, so its
        # count is independent evidence for the zero-motion claim.
        diagnostic_count_before = len(controller_diagnostics)
        safety_deadline = min(deadline, time.monotonic() + 1.0)
        while time.monotonic() < safety_deadline and (
            len(controller_diagnostics) == diagnostic_count_before
            or time.monotonic() - monitor_started < 0.5
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        latest_controller = controller_diagnostics[-1]
        controller_values = latest_controller["values"]
        controller_simulation = str(controller_values.get("mode", "")).lower() == "sim"
        controller_execution_enabled = _diagnostic_boolean(
            controller_values.get("execution_enabled"), "execution_enabled"
        )
        command_publishers_after = node.count_publishers("/control/move_j")
        graph_nodes = sorted(
            (
                f"{namespace.rstrip('/')}/{name}"
                if namespace != "/"
                else f"/{name}"
            )
            for name, namespace in node.get_node_names_and_namespaces()
        )
        known_vendor_driver_names = {
            "agx_arm_ctrl",
            "agx_arm_driver",
            "nero_driver",
        }
        vendor_driver_nodes = [
            name
            for name in graph_nodes
            if name.rsplit("/", 1)[-1] in known_vendor_driver_names
        ]
        safety_passed = bool(
            controller_simulation
            and not controller_execution_enabled
            and command_publishers_before >= 1
            and command_publishers_after >= 1
            and not command_messages
            and not vendor_driver_nodes
        )

        candidate_position = np.array(
            (
                next_view.pose.pose.position.x,
                next_view.pose.pose.position.y,
                next_view.pose.pose.position.z,
            ),
            dtype=np.float64,
        )
        optical_z = np.array(
            (
                2.0
                * (
                    next_view.pose.pose.orientation.x
                    * next_view.pose.pose.orientation.z
                    + next_view.pose.pose.orientation.y
                    * next_view.pose.pose.orientation.w
                ),
                2.0
                * (
                    next_view.pose.pose.orientation.y
                    * next_view.pose.pose.orientation.z
                    - next_view.pose.pose.orientation.x
                    * next_view.pose.pose.orientation.w
                ),
                1.0
                - 2.0
                * (
                    next_view.pose.pose.orientation.x**2
                    + next_view.pose.pose.orientation.y**2
                ),
            ),
            dtype=np.float64,
        )
        target_direction = target_center - candidate_position
        target_direction /= np.linalg.norm(target_direction)
        look_at_error_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(optical_z, target_direction), -1.0, 1.0)
                )
            )
        )
        report = {
            "schema_version": 1,
            "artifact": "g4_gradient_to_placo_pipeline",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "status": "passed" if safety_passed else "failed",
            "scene_id": SCENE_ID,
            "observation_id": OBSERVATION_ID,
            "source_path": (
                "canonical synthetic Observation -> ComputeNextView.action -> "
                "NextView topic -> Placo SolveIK.srv"
            ),
            "next_view": {
                "gain": float(next_view.gain),
                "coverage": float(next_view.coverage),
                "compute_time_ms": float(next_view.compute_time_ms),
                "optimization_iterations": int(next_view.optimization_iterations),
                "world_frame": next_view.pose.header.frame_id,
                "position": candidate_position.tolist(),
                "current_position": current[:3, 3].tolist(),
                "translation_step_m": float(
                    np.linalg.norm(candidate_position - current[:3, 3])
                ),
                "target_center": target_center.tolist(),
                "look_at_error_deg": look_at_error_deg,
            },
            "idempotency": {
                "duplicate_action_exact_cached_result": idempotent_result,
                "repeat_gain": float(repeated_view.gain),
                "repeat_coverage": float(repeated_view.coverage),
                "repeat_compute_time_ms": float(repeated_view.compute_time_ms),
            },
            "ik_preview": diagnostic,
            "safety": {
                "evidence_kind": (
                    "live controller diagnostics + live /control/move_j "
                    "subscription + ROS graph audit"
                ),
                "controller_diagnostic": latest_controller,
                "controller_simulation_mode": controller_simulation,
                "controller_execution_enabled": controller_execution_enabled,
                "command_topic": "/control/move_j",
                "command_publisher_count_before": command_publishers_before,
                "command_publisher_count_after": command_publishers_after,
                "motion_command_count_observed": len(command_messages),
                "monitor_duration_sec": time.monotonic() - monitor_started,
                "known_vendor_driver_nodes_detected": vendor_driver_nodes,
                "runtime_graph_nodes": graph_nodes,
                "move_action_used_by_fixture": False,
                "bridge_motion_surface_guard": (
                    "test_no_motion_surface.py rejects motion action/CAN surfaces"
                ),
                "passed": safety_passed,
            },
        }
        if output_path:
            _write_json_atomic(Path(output_path), report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if not str(diagnostic["name"]).endswith("ik_preview_success"):
            raise RuntimeError(str(diagnostic["message"]))
        if not safety_passed:
            raise RuntimeError("live no-motion safety evidence did not pass")
    finally:
        del tf_listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
