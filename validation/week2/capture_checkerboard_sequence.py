#!/usr/bin/env python3
"""顺序采集 canonical Observation，并原子保存为单个无 pickle 的 NPZ。"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import numpy as np


OBSERVATION_TOPIC = "/strawberry/perception/observation"
CAPTURE_SERVICE = "/strawberry/perception/capture_observation"


class CaptureError(RuntimeError):
    """采集、消息契约或保存失败。"""


@dataclass(frozen=True)
class DecodedFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    stamp: np.ndarray
    T_world_camera_optical: np.ndarray
    scene_id: str
    observation_id: str
    optical_frame: str
    world_frame: str
    source_name: str
    source_type: int
    valid_depth_fraction: float
    color_depth_skew_sec: float


def _image_view(message: Any, channels: int, dtype: np.dtype[Any]) -> np.ndarray:
    height, width, step = int(message.height), int(message.width), int(message.step)
    itemsize = np.dtype(dtype).itemsize
    packed = width * channels * itemsize
    if height <= 0 or width <= 0 or step < packed:
        raise CaptureError("图像尺寸或 row step 非法")
    data = memoryview(message.data)
    if data.nbytes < step * height:
        raise CaptureError("图像数据长度不足")
    shape = (height, width, channels) if channels > 1 else (height, width)
    strides = (step, channels * itemsize, itemsize) if channels > 1 else (step, itemsize)
    return np.ndarray(shape, dtype=dtype, buffer=data, strides=strides)


def _decode_rgb(message: Any) -> np.ndarray:
    if str(message.encoding).lower() != "rgb8":
        raise CaptureError(f"彩色编码必须是 rgb8，实际为 {message.encoding!r}")
    return np.array(_image_view(message, 3, np.dtype("u1")), copy=True, order="C")


def _decode_depth(message: Any) -> np.ndarray:
    if str(message.encoding).upper() != "32FC1":
        raise CaptureError(f"深度编码必须是 32FC1 米，实际为 {message.encoding!r}")
    dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    result = np.asarray(_image_view(message, 1, dtype), dtype=np.float32).copy(order="C")
    result[(~np.isfinite(result)) | (result <= 0.0)] = np.nan
    return result


def _pose_matrix(pose_stamped: Any) -> np.ndarray:
    pose = pose_stamped.pose
    translation = np.array(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    quaternion = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise CaptureError("相机位姿包含 NaN 或 Inf")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise CaptureError("相机位姿四元数为零")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = translation
    return matrix


def _stamp_array(stamp: Any) -> np.ndarray:
    sec, nanosec = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise CaptureError("Observation 时间戳非法")
    return np.array([sec, nanosec], dtype=np.int64)


def decode_observation(
    message: Any, *, expected_scene: str, expected_observation_id: str
) -> DecodedFrame:
    """严格解码一条 canonical Observation（纯函数，不需要 ROS 运行环境）。"""
    scene_id, observation_id = str(message.scene_id), str(message.observation_id)
    if scene_id != expected_scene:
        raise CaptureError(f"场景不匹配：期望 {expected_scene!r}，收到 {scene_id!r}")
    if observation_id != expected_observation_id:
        raise CaptureError(
            f"Observation ID 不匹配：期望 {expected_observation_id!r}，收到 {observation_id!r}"
        )
    if not bool(message.pose_valid):
        raise CaptureError("Observation 未提供有效相机位姿")

    rgb, depth = _decode_rgb(message.color), _decode_depth(message.depth)
    if rgb.shape[:2] != depth.shape:
        raise CaptureError("彩色图和深度图尺寸不一致")
    info = message.camera_info
    if (int(info.height), int(info.width)) != depth.shape:
        raise CaptureError("CameraInfo 尺寸与图像不一致")
    K = np.asarray(info.k, dtype=np.float64)
    if K.shape != (9,):
        raise CaptureError("CameraInfo.k 必须有 9 个数")
    K = K.reshape(3, 3).copy()
    if (
        not np.all(np.isfinite(K))
        or K[0, 0] <= 0.0
        or K[1, 1] <= 0.0
        or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1.0e-9)
    ):
        raise CaptureError("CameraInfo.k 非法")

    frame = str(message.header.frame_id)
    fields = (message.color, message.depth, message.camera_info)
    if not frame or any(str(item.header.frame_id) != frame for item in fields):
        raise CaptureError("图像、内参和 Observation 的光学坐标系不一致")
    stamp = _stamp_array(message.header.stamp)
    if not np.array_equal(stamp, _stamp_array(message.depth.header.stamp)):
        raise CaptureError("Observation 时间戳不是深度曝光时间")
    valid_fraction = float(message.valid_depth_fraction)
    skew = float(message.color_depth_skew_sec)
    if not 0.0 <= valid_fraction <= 1.0 or not np.isfinite(skew):
        raise CaptureError("Observation 质量字段非法")

    return DecodedFrame(
        rgb=rgb,
        depth_m=depth,
        K=K,
        stamp=stamp,
        T_world_camera_optical=_pose_matrix(message.camera_pose),
        scene_id=scene_id,
        observation_id=observation_id,
        optical_frame=frame,
        world_frame=str(message.camera_pose.header.frame_id),
        source_name=str(message.source_name),
        source_type=int(message.source_type),
        valid_depth_fraction=valid_fraction,
        color_depth_skew_sec=skew,
    )


def sequence_arrays(frames: Sequence[DecodedFrame]) -> dict[str, np.ndarray]:
    """验证一组帧并转换为只含数值/Unicode dtype 的 NPZ 字段。"""
    if not frames:
        raise CaptureError("没有可保存的 Observation")
    shape = frames[0].depth_m.shape
    scene = frames[0].scene_id
    ids = [frame.observation_id for frame in frames]
    if len(ids) != len(set(ids)):
        raise CaptureError("采集结果含重复 Observation ID")
    if any(frame.depth_m.shape != shape or frame.rgb.shape != (*shape, 3) for frame in frames):
        raise CaptureError("多帧图像尺寸不一致")
    if any(frame.scene_id != scene for frame in frames):
        raise CaptureError("多帧 scene_id 不一致")
    stamp_ns = [int(f.stamp[0]) * 1_000_000_000 + int(f.stamp[1]) for f in frames]
    if any(current <= previous for previous, current in zip(stamp_ns, stamp_ns[1:])):
        raise CaptureError("多帧时间戳没有严格递增")
    return {
        "schema_version": np.array(1, dtype=np.int32),
        "rgb": np.stack([f.rgb for f in frames]).astype(np.uint8, copy=False),
        "depth_m": np.stack([f.depth_m for f in frames]).astype(np.float32, copy=False),
        "K": np.stack([f.K for f in frames]).astype(np.float64, copy=False),
        "stamp": np.stack([f.stamp for f in frames]).astype(np.int64, copy=False),
        "T_world_camera_optical": np.stack(
            [f.T_world_camera_optical for f in frames]
        ).astype(np.float64, copy=False),
        "scene_id": np.asarray([f.scene_id for f in frames], dtype=np.str_),
        "observation_id": np.asarray(ids, dtype=np.str_),
        "optical_frame": np.asarray([f.optical_frame for f in frames], dtype=np.str_),
        "world_frame": np.asarray([f.world_frame for f in frames], dtype=np.str_),
        "source_name": np.asarray([f.source_name for f in frames], dtype=np.str_),
        "source_type": np.asarray([f.source_type for f in frames], dtype=np.uint8),
        "valid_depth_fraction": np.asarray(
            [f.valid_depth_fraction for f in frames], dtype=np.float32
        ),
        "color_depth_skew_sec": np.asarray(
            [f.color_depth_skew_sec for f in frames], dtype=np.float64
        ),
    }


def save_sequence(frames: Sequence[DecodedFrame], output: str | Path) -> Path:
    """完整验证后原子保存；失败时不留下目标文件或临时文件。"""
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise CaptureError(f"输出文件已存在，拒绝覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = sequence_arrays(frames)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def _run_ros_capture(scene: str, count: int, interval: float) -> list[DecodedFrame]:
    """ROS 2 薄层；模块导入时不会依赖 ROS。"""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from strawberry_perception_interfaces.msg import Observation
        from strawberry_perception_interfaces.srv import CaptureObservation
    except ImportError as error:
        raise CaptureError("找不到系统 ROS 2/rclpy；请先 source ROS 和 perception_ws") from error

    rclpy.init()
    node = Node("capture_checkerboard_sequence")
    cache: OrderedDict[tuple[str, str], Any] = OrderedDict()

    def receive(message: Any) -> None:
        key = (str(message.scene_id), str(message.observation_id))
        cache[key] = message
        cache.move_to_end(key)
        while len(cache) > 32:
            cache.popitem(last=False)

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    subscription = node.create_subscription(Observation, OBSERVATION_TOPIC, receive, qos)
    client = node.create_client(CaptureObservation, CAPTURE_SERVICE)
    frames: list[DecodedFrame] = []
    try:
        if not client.wait_for_service(timeout_sec=10.0):
            raise CaptureError(f"10 秒内未找到采集服务 {CAPTURE_SERVICE}")
        for index in range(count):
            response = None
            for attempt in range(1, 4):
                request = CaptureObservation.Request()
                request.scene_id = scene
                request.not_before = node.get_clock().now().to_msg()
                request.timeout.sec = 8
                request.timeout.nanosec = 0
                request.discard_frames = 3
                request.require_color = True
                request.require_mask = False
                request.require_pose = True
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
                if not future.done() or future.result() is None:
                    reason = "服务调用在 10 秒内未完成"
                else:
                    response = future.result()
                    if bool(response.success):
                        break
                    reason = f"code={response.code}, {response.reason}"
                if attempt < 3:
                    print(
                        f"第 {index + 1}/{count} 帧暂未取得（{reason}），"
                        f"自动重试 {attempt}/2……",
                        flush=True,
                    )
            if response is None or not bool(response.success):
                raise CaptureError(
                    f"第 {index + 1}/{count} 帧连续 3 次失败：{reason}"
                )
            observation_id = str(response.observation_id)
            key = (scene, observation_id)
            deadline = time.monotonic() + 2.0
            while key not in cache and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            if key not in cache:
                raise CaptureError(f"服务返回 {observation_id!r}，但 2 秒内未收到对应 Topic 消息")
            message = cache.pop(key)
            response_stamp = _stamp_array(response.stamp)
            frame = decode_observation(
                message, expected_scene=scene, expected_observation_id=observation_id
            )
            if not np.array_equal(frame.stamp, response_stamp):
                raise CaptureError("服务响应时间戳与 Observation 不一致")
            frames.append(frame)
            print(f"已采集 {index + 1}/{count}：{observation_id}", flush=True)
            if index + 1 < count and interval > 0.0:
                end = time.monotonic() + interval
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=min(0.05, end - time.monotonic()))
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
    return frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="连续采集标定板 RGB-D Observation 到一个 NPZ")
    parser.add_argument("--scene", default="checkerboard_sequence", help="本次采集的场景 ID")
    parser.add_argument("--count", type=int, default=20, help="采集帧数（默认 20）")
    parser.add_argument("--output", required=True, help="输出 .npz 文件；已存在时拒绝覆盖")
    parser.add_argument("--interval", type=float, default=0.1, help="相邻请求间隔秒数")
    args = parser.parse_args(argv)
    if not args.scene.strip():
        parser.error("--scene 不能为空")
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    if args.interval < 0.0:
        parser.error("--interval 不能小于 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise CaptureError(f"输出文件已存在，拒绝覆盖：{destination}")
    frames = _run_ros_capture(args.scene, args.count, args.interval)
    saved = save_sequence(frames, destination)
    print(f"完成：{len(frames)} 帧已保存到 {saved}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CaptureError as error:
        print(f"采集失败：{error}", file=__import__("sys").stderr)
        raise SystemExit(1)
