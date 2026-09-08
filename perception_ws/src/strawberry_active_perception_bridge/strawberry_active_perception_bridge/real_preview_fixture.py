"""Preview the current or a 5 mm NBV-style pose using formal hand-eye data.

This executable publishes only an offline pose fixture and listens for the
bridge's SolveIK diagnostic.  It has no motion Action client and no command
publisher.  Both it and the bridge independently count live command messages.
"""

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
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from strawberry_perception_interfaces.msg import NextView, Observation
from tf2_ros import Buffer, TransformException, TransformListener

from .calibration_report import load_verified_handeye_report
from .transforms import matrix_to_pose_components, pose_components_to_matrix


DEFAULT_REPORT_PATH = (
    "/home/yyt/strawberry_active_perception/artifacts/week3/"
    "handeye_session_001/stability_pose001_030_factory_raw_D.json"
)
DEFAULT_REPORT_SHA256 = (
    "31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f"
)
SCENE_ID = "real_handeye_no_motion_preview"


def _transform_message_to_matrix(message) -> np.ndarray:
    translation = message.transform.translation
    rotation = message.transform.rotation
    return pose_components_to_matrix(
        (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )


def _fill_pose(message, transform: np.ndarray, frame_id: str, stamp) -> None:
    position, quaternion = matrix_to_pose_components(transform)
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.pose.position.x = float(position[0])
    message.pose.position.y = float(position[1])
    message.pose.position.z = float(position[2])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])


def make_small_nbv_target(
    current_base_camera: np.ndarray,
    displacement_m: float,
    target_distance_m: float,
) -> np.ndarray:
    """Translate along optical +X and keep optical +Z on the same target."""
    current = np.asarray(current_base_camera, dtype=float)
    # Validate through the public conversion and retain a defensive copy.
    position, quaternion = matrix_to_pose_components(current)
    current = pose_components_to_matrix(position, quaternion)
    if not np.isfinite(displacement_m) or not 0.0 < displacement_m <= 0.01:
        raise ValueError("small_nbv_step_m must be in (0, 0.01] metres")
    if not np.isfinite(target_distance_m) or target_distance_m <= 0.0:
        raise ValueError("target_distance_m must be positive")
    target_center = current[:3, 3] + current[:3, 2] * target_distance_m
    candidate_position = current[:3, 3] + current[:3, 0] * displacement_m
    forward = target_center - candidate_position
    forward /= np.linalg.norm(forward)
    optical_up = -current[:3, 1]
    right = np.cross(forward, optical_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-12:
        raise ValueError("small NBV look-at construction is degenerate")
    right /= right_norm
    down = np.cross(forward, right)
    candidate = np.eye(4, dtype=float)
    candidate[:3, :3] = np.column_stack((right, down, forward))
    candidate[:3, 3] = candidate_position
    # Reject any numerical reflection before it crosses the ROS boundary.
    matrix_to_pose_components(candidate)
    return candidate


def _diagnostic_values(status) -> dict[str, object]:
    values: dict[str, object] = {}
    for entry in status.values:
        try:
            values[entry.key] = json.loads(entry.value)
        except json.JSONDecodeError:
            values[entry.key] = entry.value
    return values


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
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


def _boolean(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"diagnostic field {field!r} is not boolean: {value!r}")


def _wait_for_transform(
    node: Node,
    buffer: Buffer,
    base_frame: str,
    link_frame: str,
    deadline: float,
) -> np.ndarray:
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            transform = buffer.lookup_transform(
                base_frame,
                link_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
            return _transform_message_to_matrix(transform)
        except TransformException:
            continue
    raise RuntimeError(f"TF {base_frame} <- {link_frame} was unavailable")


def main(args=None) -> None:
    """Publish one no-motion candidate and emit independently audited JSON."""
    rclpy.init(args=args)
    node = Node("real_handeye_preview_fixture")
    node.declare_parameter("handeye_report_path", DEFAULT_REPORT_PATH)
    node.declare_parameter("handeye_report_sha256", DEFAULT_REPORT_SHA256)
    node.declare_parameter("minimum_calibration_samples", 30)
    node.declare_parameter("base_frame", "base_link")
    node.declare_parameter("link_frame", "link7")
    node.declare_parameter("camera_frame", "camera_color_optical_frame")
    node.declare_parameter("preview_mode", "current")
    node.declare_parameter("small_nbv_step_m", 0.005)
    node.declare_parameter("target_distance_m", 0.35)
    node.declare_parameter("motion_command_topic", "/control/move_j")
    node.declare_parameter("timeout_sec", 15.0)
    node.declare_parameter("output_path", "")

    report = load_verified_handeye_report(
        str(node.get_parameter("handeye_report_path").value),
        str(node.get_parameter("handeye_report_sha256").value),
        int(node.get_parameter("minimum_calibration_samples").value),
    )
    base_frame = str(node.get_parameter("base_frame").value).strip()
    link_frame = str(node.get_parameter("link_frame").value).strip()
    camera_frame = str(node.get_parameter("camera_frame").value).strip()
    mode = str(node.get_parameter("preview_mode").value).strip()
    step = float(node.get_parameter("small_nbv_step_m").value)
    target_distance = float(node.get_parameter("target_distance_m").value)
    command_topic = str(node.get_parameter("motion_command_topic").value).strip()
    timeout_sec = float(node.get_parameter("timeout_sec").value)
    output_path = str(node.get_parameter("output_path").value).strip()
    if (base_frame, link_frame, camera_frame) != (
        "base_link",
        "link7",
        "camera_color_optical_frame",
    ):
        raise RuntimeError(
            "real preview frames must be base_link/link7/"
            "camera_color_optical_frame"
        )
    if mode not in {"current", "small_nbv"}:
        raise RuntimeError("preview_mode must be 'current' or 'small_nbv'")
    if timeout_sec <= 0.0 or not command_topic:
        raise RuntimeError("timeout_sec and motion_command_topic are invalid")
    deadline = time.monotonic() + timeout_sec

    state_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    live_qos = QoSProfile(
        depth=32,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    observations = node.create_publisher(
        Observation,
        "/strawberry/perception/observation",
        state_qos,
    )
    next_views = node.create_publisher(
        NextView,
        "/strawberry/nbv/next_view",
        state_qos,
    )
    preview_diagnostics: list[dict[str, object]] = []
    controller_diagnostics: list[dict[str, object]] = []
    command_stamps_ns: list[int] = []
    observation_id = ""

    def preview_callback(message: DiagnosticArray) -> None:
        for status in message.status:
            values = _diagnostic_values(status)
            if (
                values.get("scene_id") == SCENE_ID
                and values.get("observation_id") == observation_id
            ):
                preview_diagnostics.append(
                    {
                        "name": status.name,
                        "message": status.message,
                        "hardware_id": status.hardware_id,
                        "values": values,
                    }
                )

    def controller_callback(message: DiagnosticArray) -> None:
        for status in message.status:
            if status.hardware_id == "nero" and status.name.endswith(": safety"):
                controller_diagnostics.append(
                    {
                        "name": status.name,
                        "message": status.message,
                        "received_monotonic_ns": time.monotonic_ns(),
                        "values": _diagnostic_values(status),
                    }
                )

    def command_callback(_message: JointState) -> None:
        command_stamps_ns.append(time.monotonic_ns())

    node.create_subscription(
        DiagnosticArray,
        "/strawberry/active_perception/ik_preview",
        preview_callback,
        state_qos,
    )
    node.create_subscription(
        DiagnosticArray,
        "/strawberry_nero/diagnostics",
        controller_callback,
        live_qos,
    )
    node.create_subscription(
        JointState,
        command_topic,
        command_callback,
        live_qos,
    )
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    monitor_started_ns = time.monotonic_ns()

    try:
        safety_deadline = min(deadline, time.monotonic() + 3.0)
        while time.monotonic() < safety_deadline and not controller_diagnostics:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not controller_diagnostics:
            raise RuntimeError("NERO safety diagnostic was unavailable")
        safety_values = controller_diagnostics[-1]["values"]
        controller_execution_enabled = _boolean(
            safety_values.get("execution_enabled"),
            "execution_enabled",
        )
        if controller_execution_enabled:
            raise RuntimeError("NERO execution gate is open; refusing read-only preview")
        if command_stamps_ns:
            raise RuntimeError("motion command observed before preview request")
        controller_diagnostic_count_before = len(controller_diagnostics)

        current_base_link7 = _wait_for_transform(
            node,
            tf_buffer,
            base_frame,
            link_frame,
            deadline,
        )
        current_base_camera = (
            current_base_link7 @ report.transform_link7_camera_optical
        )
        target_base_camera = current_base_camera.copy()
        if mode == "small_nbv":
            target_base_camera = make_small_nbv_target(
                current_base_camera,
                step,
                target_distance,
            )
        stamp = node.get_clock().now().to_msg()
        observation_id = f"{mode}_{stamp.sec}_{stamp.nanosec:09d}"
        observation = Observation()
        observation.header.frame_id = camera_frame
        observation.header.stamp = stamp
        observation.scene_id = SCENE_ID
        observation.observation_id = observation_id
        observation.source_type = Observation.SOURCE_OFFLINE
        observation.source_name = "validated_handeye_no_motion_pose_fixture"
        _fill_pose(observation.camera_pose, current_base_camera, base_frame, stamp)
        observation.pose_valid = True

        next_view = NextView()
        next_view.scene_id = SCENE_ID
        next_view.observation_id = observation_id
        next_view.success = True
        next_view.code = NextView.SUCCESS
        next_view.reason = (
            "current calibrated camera pose"
            if mode == "current"
            else f"deterministic {step * 1000.0:.1f} mm NBV-style candidate"
        )
        _fill_pose(next_view.pose, target_base_camera, base_frame, stamp)

        discovery_deadline = min(deadline, time.monotonic() + 3.0)
        while time.monotonic() < discovery_deadline and (
            observations.get_subscription_count() == 0
            or next_views.get_subscription_count() == 0
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if (
            observations.get_subscription_count() == 0
            or next_views.get_subscription_count() == 0
        ):
            raise RuntimeError("report-bound preview bridge subscriptions unavailable")
        command_publishers_before = node.count_publishers(command_topic)
        observations.publish(observation)
        cache_deadline = min(deadline, time.monotonic() + 0.25)
        while time.monotonic() < cache_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        next_views.publish(next_view)
        while time.monotonic() < deadline and not preview_diagnostics:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not preview_diagnostics:
            raise RuntimeError("no matching SolveIK preview diagnostic was received")

        audit_started = time.monotonic()
        audit_until = min(deadline, audit_started + 2.0)
        while time.monotonic() < audit_until and (
            len(controller_diagnostics) == controller_diagnostic_count_before
            or time.monotonic() - audit_started < 0.25
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        controller_fresh_after_request_started = (
            len(controller_diagnostics) > controller_diagnostic_count_before
        )
        latest_safety_values = controller_diagnostics[-1]["values"]
        controller_execution_enabled = _boolean(
            latest_safety_values.get("execution_enabled"),
            "execution_enabled",
        )
        diagnostic = preview_diagnostics[-1]
        diagnostic_values = diagnostic["values"]
        transform_matches = np.array_equal(
            np.asarray(
                diagnostic_values.get("T_link7_camera_optical"),
                dtype=float,
            ),
            report.transform_link7_camera_optical,
        )
        bridge_motion_count = diagnostic_values.get(
            "motion_command_count_observed"
        )
        safety_passed = bool(
            str(diagnostic["name"]).endswith("ik_preview_success")
            and diagnostic["hardware_id"] == "validated_handeye_no_motion"
            and diagnostic_values.get("calibration_source") == "validated_report"
            and diagnostic_values.get("handeye_report_sha256") == report.sha256
            and transform_matches
            and bridge_motion_count == 0
            and not command_stamps_ns
            and controller_fresh_after_request_started
            and not controller_execution_enabled
        )
        result = {
            "schema_version": 1,
            "artifact": "real_handeye_solve_ik_no_motion_preview",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "status": "passed" if safety_passed else "failed",
            "preview_mode": mode,
            "scene_id": SCENE_ID,
            "observation_id": observation_id,
            "frames": {
                "base": base_frame,
                "controlled": link_frame,
                "camera": camera_frame,
            },
            "handeye_report": {
                "path": str(report.path),
                "sha256": report.sha256,
                "version": report.report_version,
                "session_id": report.session_id,
                "sample_count": report.sample_count,
                "transform_matches_bridge_diagnostic": transform_matches,
            },
            "T_base_link7_current": current_base_link7.tolist(),
            "T_base_camera_current": current_base_camera.tolist(),
            "T_base_camera_target": target_base_camera.tolist(),
            "translation_step_m": float(
                np.linalg.norm(
                    target_base_camera[:3, 3] - current_base_camera[:3, 3]
                )
            ),
            "ik_preview": diagnostic,
            "safety": {
                "controller_diagnostic": controller_diagnostics[-1],
                "controller_diagnostic_fresh_after_request_started": (
                    controller_fresh_after_request_started
                ),
                "controller_execution_enabled": controller_execution_enabled,
                "command_topic": command_topic,
                "command_publisher_count_before": command_publishers_before,
                "command_publisher_count_after": node.count_publishers(
                    command_topic
                ),
                "motion_command_count_observed": len(command_stamps_ns),
                "bridge_motion_command_count_observed": bridge_motion_count,
                "monitor_duration_sec": (
                    time.monotonic_ns() - monitor_started_ns
                )
                / 1.0e9,
                "move_action_client_created": False,
                "command_publisher_created": False,
                "passed": safety_passed,
            },
        }
        if output_path:
            _write_json_atomic(Path(output_path), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if not safety_passed:
            raise RuntimeError("real hand-eye no-motion preview safety audit failed")
    finally:
        del tf_listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
