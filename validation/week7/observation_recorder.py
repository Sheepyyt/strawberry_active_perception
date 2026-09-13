#!/usr/bin/env python3
"""Read-only recorder for canonical observations used by a real NBV session.

The node has one subscription and no publisher, service client, action client,
or robot-control import.  It stores rectified RGB, metre depth, mask, K and the
exposure-time camera pose as deterministic NPZ evidence for later figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import numpy as np


SCHEMA = "strawberry_canonical_observation_snapshot/v1"


def _uint8(value: Any) -> int:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) != 1:
            raise ValueError("ROS uint8 field must contain exactly one byte")
        return raw[0]
    result = int(value)
    if not 0 <= result <= 255:
        raise ValueError("ROS uint8 field is outside [0, 255]")
    return result


def _stamp_ns(stamp: Any) -> int:
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("invalid ROS timestamp")
    return seconds * 1_000_000_000 + nanoseconds


def _decode_image(
    message: Any, encoding: str, dtype: np.dtype, channels: int = 1
) -> np.ndarray:
    if str(message.encoding).lower() != encoding.lower():
        raise ValueError(f"expected {encoding}, got {message.encoding!r}")
    height = int(message.height)
    width = int(message.width)
    itemsize = int(np.dtype(dtype).itemsize)
    packed = width * channels * itemsize
    step = int(message.step)
    if height <= 0 or width <= 0 or step < packed:
        raise ValueError("invalid image shape or row step")
    data = memoryview(message.data)
    if data.nbytes < step * height:
        raise ValueError("image data is shorter than step * height")
    shape = (height, width, channels) if channels > 1 else (height, width)
    strides = (
        (step, channels * itemsize, itemsize) if channels > 1 else (step, itemsize)
    )
    return np.array(
        np.ndarray(shape, dtype=dtype, buffer=data, strides=strides),
        copy=True,
        order="C",
    )


def _pose_matrix(pose_stamped: Any) -> np.ndarray:
    position = pose_stamped.pose.position
    orientation = pose_stamped.pose.orientation
    xyz = np.asarray((position.x, position.y, position.z), dtype=np.float64)
    quaternion = np.asarray(
        (orientation.x, orientation.y, orientation.z, orientation.w),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("camera pose is non-finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("camera pose quaternion is zero")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    matrix[:3, 3] = xyz
    return matrix


def observation_arrays(message: Any) -> dict[str, np.ndarray]:
    """Validate and decode one canonical Observation message-like object."""
    stamp_ns = _stamp_ns(message.header.stamp)
    frame = str(message.header.frame_id)
    if not frame or any(
        str(image.header.frame_id) != frame
        for image in (message.color, message.depth, message.target_mask)
    ):
        raise ValueError("canonical images do not share the observation frame")
    if _stamp_ns(message.depth.header.stamp) != stamp_ns:
        raise ValueError("depth stamp differs from Observation stamp")
    rgb = _decode_image(message.color, "rgb8", np.uint8, channels=3)
    mask = _decode_image(message.target_mask, "mono8", np.uint8)
    endian = ">f4" if _uint8(message.depth.is_bigendian) else "<f4"
    depth = _decode_image(message.depth, "32FC1", np.dtype(endian)).astype(np.float32)
    depth[(~np.isfinite(depth)) | (depth <= 0.0)] = np.nan
    if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError("canonical RGB, depth and mask grids differ")
    if not np.all(np.isin(np.unique(mask), (0, 255))):
        raise ValueError("target mask is not binary")
    K = np.asarray(message.camera_info.k, dtype=np.float64).reshape(3, 3)
    D = np.asarray(message.camera_info.d, dtype=np.float64)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("camera intrinsics are invalid")
    if D.size == 0 or not np.all(np.isfinite(D)) or not np.allclose(D, 0.0, atol=1e-12):
        raise ValueError("canonical CameraInfo must describe a rectified grid")
    if not bool(message.pose_valid):
        raise ValueError("canonical observation has no valid camera pose")
    pose = _pose_matrix(message.camera_pose)
    return {
        "schema": np.asarray(SCHEMA),
        "scene_id": np.asarray(str(message.scene_id)),
        "observation_id": np.asarray(str(message.observation_id)),
        "source_type": np.asarray(_uint8(message.source_type), dtype=np.uint8),
        "source_name": np.asarray(str(message.source_name)),
        "stamp_ns": np.asarray(stamp_ns, dtype=np.int64),
        "camera_frame": np.asarray(frame),
        "world_frame": np.asarray(str(message.camera_pose.header.frame_id)),
        "rgb": rgb,
        "depth_m": depth,
        "mask": mask,
        "K": K,
        "D": D,
        "T_world_camera_optical": pose,
        "valid_depth_fraction": np.asarray(
            float(message.valid_depth_fraction), dtype=np.float64
        ),
        "color_depth_skew_sec": np.asarray(
            float(message.color_depth_skew_sec), dtype=np.float64
        ),
    }


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return normalized[:64] or "observation"


def save_observation_snapshot(arrays: dict[str, np.ndarray], path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--topic", default="/strawberry/perception/real_nbv_observation"
    )
    parser.add_argument(
        "--scene-id", default="", help="ignore observations from other scenes"
    )
    parser.add_argument(
        "--max-count", type=int, default=0, help="0 means run until timeout/Ctrl-C"
    )
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    arguments = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from strawberry_perception_interfaces.msg import Observation

    if arguments.max_count < 0 or arguments.timeout_sec <= 0:
        raise SystemExit("max-count must be >=0 and timeout-sec must be >0")
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": "strawberry_observation_recording/v1",
        "topic": arguments.topic,
        "scene_filter": arguments.scene_id or None,
        "read_only": True,
        "records": [],
        "errors": [],
    }
    seen: set[tuple[str, str]] = set()
    rclpy.init()
    node = Node("strawberry_observation_evidence_recorder")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def callback(message: Observation) -> None:
        if arguments.scene_id and message.scene_id != arguments.scene_id:
            return
        identity = (str(message.scene_id), str(message.observation_id))
        if identity in seen:
            return
        try:
            arrays = observation_arrays(message)
            index = len(manifest["records"]) + 1
            filename = f"observation_{index:03d}_{_safe_name(identity[1])}.npz"
            path = save_observation_snapshot(arrays, output / filename)
            record = {
                "index": index,
                "scene_id": identity[0],
                "observation_id": identity[1],
                "stamp_ns": int(arrays["stamp_ns"]),
                "path": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rgb_shape": list(arrays["rgb"].shape),
                "finite_depth_fraction": float(
                    np.count_nonzero(np.isfinite(arrays["depth_m"]))
                    / arrays["depth_m"].size
                ),
                "mask_pixels": int(np.count_nonzero(arrays["mask"])),
                "valid_mask_depth_pixels": int(
                    np.count_nonzero(
                        (arrays["mask"] > 0) & np.isfinite(arrays["depth_m"])
                    )
                ),
            }
            manifest["records"].append(record)
            seen.add(identity)
            _atomic_json(manifest_path, manifest)
            node.get_logger().info(f"saved {filename}")
        except Exception as error:
            manifest["errors"].append(
                {
                    "scene_id": identity[0],
                    "observation_id": identity[1],
                    "reason": str(error),
                }
            )
            _atomic_json(manifest_path, manifest)
            node.get_logger().error(f"refused observation {identity}: {error}")

    node.create_subscription(Observation, arguments.topic, callback, qos)
    started = time.monotonic()
    try:
        while rclpy.ok():
            if arguments.max_count and len(manifest["records"]) >= arguments.max_count:
                break
            if time.monotonic() - started >= arguments.timeout_sec:
                break
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        manifest["elapsed_sec"] = time.monotonic() - started
        manifest["record_count"] = len(manifest["records"])
        manifest["finished_cleanly"] = True
        _atomic_json(manifest_path, manifest)
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
