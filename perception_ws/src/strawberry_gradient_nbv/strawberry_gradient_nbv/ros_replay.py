"""Publish a deterministic NPZ fixture as canonical ROS 2 Observations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from strawberry_perception_interfaces.msg import Observation

from .fixtures import FixtureObservation, load_fixture_npz


def _stamp(value: float):
    """Convert a finite non-negative epoch value to builtin ROS time."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("fixture stamp must be finite and non-negative")
    seconds = math.floor(value)
    nanoseconds = int(round((value - seconds) * 1_000_000_000.0))
    if nanoseconds == 1_000_000_000:
        seconds += 1
        nanoseconds = 0
    from builtin_interfaces.msg import Time

    return Time(sec=int(seconds), nanosec=nanoseconds)


def _header(stamp, frame_id: str) -> Header:
    result = Header()
    result.stamp = stamp
    result.frame_id = frame_id
    return result


def _image(array: np.ndarray, encoding: str, frame_id: str, stamp) -> Image:
    contiguous = np.ascontiguousarray(array)
    if contiguous.ndim not in (2, 3):
        raise ValueError("fixture image must be HxW or HxWxC")
    channels = 1 if contiguous.ndim == 2 else contiguous.shape[2]
    result = Image()
    result.header = _header(stamp, frame_id)
    result.height = int(contiguous.shape[0])
    result.width = int(contiguous.shape[1])
    result.encoding = encoding
    result.is_bigendian = False
    result.step = int(result.width * channels * contiguous.dtype.itemsize)
    result.data = contiguous.tobytes()
    return result


def _rotation_to_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert one proper rotation matrix to a normalized ROS quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("fixture rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def canonical_observation_from_fixture(
    item: FixtureObservation,
    *,
    source_type: int,
    optical_frame: str = "fixture_camera_optical_frame",
    source_name: str,
) -> Observation:
    """Serialize one canonical fixture record for a declared source kind.

    This is the shared *post-normalization* wire factory.  In particular,
    ``SOURCE_REAL`` means that ``item`` already represents the output of the
    C++ Gemini adapter (rgb8/32FC1 metres/mono8); this helper neither talks to
    hardware nor exercises the adapter's 16UC1 millimetre conversion.
    """
    if not optical_frame or not source_name:
        raise ValueError("optical_frame and source_name must be non-empty")
    supported_sources = {
        Observation.SOURCE_REAL,
        Observation.SOURCE_OFFLINE,
        Observation.SOURCE_SYNTHETIC,
        Observation.SOURCE_REPLAY,
    }
    if source_type not in supported_sources:
        raise ValueError("source_type must be REAL, OFFLINE, SYNTHETIC, or REPLAY")
    stamp = _stamp(float(item.stamp))
    height, width = item.depth.shape
    message = Observation()
    message.header = _header(stamp, optical_frame)
    message.scene_id = item.scene_id
    message.observation_id = item.observation_id
    message.source_type = source_type
    message.source_name = source_name
    message.color = _image(item.color, "rgb8", optical_frame, stamp)
    message.depth = _image(
        np.asarray(item.depth, dtype="<f4"), "32FC1", optical_frame, stamp
    )
    message.target_mask = _image(item.mask, "mono8", optical_frame, stamp)

    info = CameraInfo()
    info.header = _header(stamp, optical_frame)
    info.height = height
    info.width = width
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = np.asarray(item.K, dtype=np.float64).reshape(-1).tolist()
    info.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
    info.p = [
        float(item.K[0, 0]),
        0.0,
        float(item.K[0, 2]),
        0.0,
        0.0,
        float(item.K[1, 1]),
        float(item.K[1, 2]),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    message.camera_info = info

    message.camera_pose.header = _header(
        stamp, str(item.config.get("world_frame", "fixture_world"))
    )
    translation = item.pose[:3, 3]
    quaternion = _rotation_to_xyzw(item.pose[:3, :3])
    message.camera_pose.pose.position.x = float(translation[0])
    message.camera_pose.pose.position.y = float(translation[1])
    message.camera_pose.pose.position.z = float(translation[2])
    message.camera_pose.pose.orientation.x = quaternion[0]
    message.camera_pose.pose.orientation.y = quaternion[1]
    message.camera_pose.pose.orientation.z = quaternion[2]
    message.camera_pose.pose.orientation.w = quaternion[3]
    message.pose_valid = True
    message.valid_depth_fraction = float(
        np.count_nonzero(np.isfinite(item.depth)) / item.depth.size
    )
    message.color_depth_skew_sec = 0.0
    return message


def fixture_to_observation(
    item: FixtureObservation,
    *,
    optical_frame: str = "fixture_camera_optical_frame",
    source_name: str = "deterministic_npz_replay",
) -> Observation:
    """Convert one fixture record for the public NPZ/ROS replay entry point."""
    return canonical_observation_from_fixture(
        item,
        source_type=Observation.SOURCE_REPLAY,
        optical_frame=optical_frame,
        source_name=source_name,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="fixture NPZ created by replay.py")
    parser.add_argument(
        "--topic", default="/strawberry/perception/observation"
    )
    parser.add_argument("--optical-frame", default="fixture_camera_optical_frame")
    parser.add_argument("--interval-sec", type=float, default=0.25)
    parser.add_argument("--discovery-timeout-sec", type=float, default=3.0)
    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Publish every fixture observation once using the canonical Topic QoS."""
    parsed, ros_args = _parser().parse_known_args(args)
    if parsed.interval_sec < 0.0 or parsed.discovery_timeout_sec < 0.0:
        raise ValueError("interval and discovery timeout must be non-negative")
    fixture = load_fixture_npz(parsed.path)
    rclpy.init(args=list(ros_args))
    node = Node("gradient_nbv_ros_replay")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    publisher = node.create_publisher(Observation, parsed.topic, qos)
    try:
        discovery_deadline = time.monotonic() + parsed.discovery_timeout_sec
        while (
            publisher.get_subscription_count() == 0
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        for index, item in enumerate(fixture.observations()):
            message = fixture_to_observation(
                item, source_name=f"npz:{parsed.path.name}"
            )
            publisher.publish(message)
            node.get_logger().info(
                f"published {message.scene_id}/{message.observation_id} "
                f"on {parsed.topic}"
            )
            if index + 1 < len(fixture) and parsed.interval_sec > 0.0:
                deadline = time.monotonic() + parsed.interval_sec
                while time.monotonic() < deadline:
                    rclpy.spin_once(
                        node,
                        timeout_sec=min(0.05, deadline - time.monotonic()),
                    )
        # Give reliable DDS delivery a short opportunity before process teardown.
        settle_deadline = time.monotonic() + 0.1
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
