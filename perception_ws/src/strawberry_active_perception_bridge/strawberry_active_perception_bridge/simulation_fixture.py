"""Publish the fixed G4 camera candidate and report its read-only IK result."""

from __future__ import annotations

import json
import time

from diagnostic_msgs.msg import DiagnosticArray
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from strawberry_perception_interfaces.msg import NextView, Observation
from tf2_ros import Buffer, TransformException, TransformListener

from .transforms import pose_components_to_matrix


SCENE_ID = "g4_fixed_scene"
OBSERVATION_ID = "nbv_ready_lateral_100mm"


def _transform_message_to_matrix(message) -> np.ndarray:
    translation = message.transform.translation
    rotation = message.transform.rotation
    return pose_components_to_matrix(
        (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )


def _fill_pose(message, transform: np.ndarray, frame_id: str, stamp) -> None:
    from .transforms import matrix_to_pose_components

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


def make_lateral_camera_target(
    current_base_camera: np.ndarray,
    displacement_m: float = 0.10,
    target_distance_m: float = 0.35,
) -> np.ndarray:
    """Move along optical +X and keep optical +Z aimed at the scene target."""
    current = np.asarray(current_base_camera, dtype=float)
    if current.shape != (4, 4):
        raise ValueError("current_base_camera must be 4x4")
    if displacement_m <= 0.0 or target_distance_m <= 0.0:
        raise ValueError("fixture distances must be positive")
    target_center = current[:3, 3] + current[:3, 2] * target_distance_m
    position = current[:3, 3] + current[:3, 0] * displacement_m
    forward = target_center - position
    forward /= np.linalg.norm(forward)
    up = -current[:3, 1]
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    target = current.copy()
    target[:3, :3] = np.column_stack((right, down, forward))
    target[:3, 3] = position
    return target


def _diagnostic_values(status) -> dict[str, object]:
    values: dict[str, object] = {}
    for entry in status.values:
        try:
            values[entry.key] = json.loads(entry.value)
        except json.JSONDecodeError:
            values[entry.key] = entry.value
    return values


def main(args=None) -> None:
    """Run the deterministic 100 mm request and exit after the G4 result."""
    rclpy.init(args=args)
    node = Node("nbv_ik_preview_fixture")
    node.declare_parameter("base_frame", "base_link")
    node.declare_parameter("camera_frame", "camera_sim_optical_frame")
    node.declare_parameter("timeout_sec", 15.0)
    base_frame = str(node.get_parameter("base_frame").value)
    camera_frame = str(node.get_parameter("camera_frame").value)
    timeout_sec = float(node.get_parameter("timeout_sec").value)
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
    next_view_publisher = node.create_publisher(
        NextView,
        "/strawberry/nbv/next_view",
        qos,
    )
    results: list[dict[str, object]] = []

    def diagnostic_callback(message: DiagnosticArray) -> None:
        for status in message.status:
            if not status.name.startswith("strawberry_active_perception_bridge/"):
                continue
            values = _diagnostic_values(status)
            if (
                values.get("scene_id") == SCENE_ID
                and values.get("observation_id") == OBSERVATION_ID
            ):
                results.append(
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
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    deadline = time.monotonic() + timeout_sec
    current = None
    while time.monotonic() < deadline and current is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            transform = tf_buffer.lookup_transform(
                base_frame,
                camera_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
            current = _transform_message_to_matrix(transform)
        except TransformException:
            continue
    if current is None:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError(f"TF {base_frame} <- {camera_frame} was unavailable")

    stamp = node.get_clock().now().to_msg()
    observation = Observation()
    observation.header.stamp = stamp
    observation.header.frame_id = camera_frame
    observation.scene_id = SCENE_ID
    observation.observation_id = OBSERVATION_ID
    observation.source_type = Observation.SOURCE_SYNTHETIC
    observation.source_name = "g4_tf_fixture"
    _fill_pose(observation.camera_pose, current, base_frame, stamp)
    observation.pose_valid = True

    target = make_lateral_camera_target(current)
    next_view = NextView()
    next_view.scene_id = SCENE_ID
    next_view.observation_id = OBSERVATION_ID
    next_view.success = True
    next_view.code = NextView.SUCCESS
    next_view.reason = "deterministic 100 mm lateral NBV-style candidate"
    _fill_pose(next_view.pose, target, base_frame, stamp)

    discovery_deadline = min(deadline, time.monotonic() + 3.0)
    while time.monotonic() < discovery_deadline and (
        observation_publisher.get_subscription_count() == 0
        or next_view_publisher.get_subscription_count() == 0
    ):
        rclpy.spin_once(node, timeout_sec=0.1)
    observation_publisher.publish(observation)
    time.sleep(0.25)
    next_view_publisher.publish(next_view)
    while time.monotonic() < deadline and not results:
        rclpy.spin_once(node, timeout_sec=0.1)

    del tf_listener
    node.destroy_node()
    rclpy.shutdown()
    if not results:
        raise RuntimeError("no matching read-only IK preview diagnostic was received")
    print(json.dumps(results[-1], ensure_ascii=False, indent=2, sort_keys=True))
    if not results[-1]["name"].endswith("ik_preview_success"):
        raise RuntimeError(str(results[-1]["message"]))


if __name__ == "__main__":
    main()
