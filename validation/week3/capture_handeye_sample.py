#!/usr/bin/env python3
"""只读采集一个静止姿态的手眼标定样本；绝不发布运动命令。"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


OBSERVATION_TOPIC = "/strawberry/perception/observation"
CAPTURE_SERVICE = "/strawberry/perception/capture_observation"
JOINT_TOPIC = "/feedback/joint_states"
BASE_FRAME = "base_link"
LINK_FRAME = "link7"
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
BOARD_INNER_CORNERS = (11, 8)
BOARD_SQUARES = (12, 9)
SQUARE_SIZE_M = 0.030
SCHEMA_VERSION = "strawberry_handeye_capture/v1"


class CaptureError(RuntimeError):
    """现场采集、消息契约或保存失败。"""


@dataclass(frozen=True)
class JointSample:
    stamp_ns: int
    received_monotonic: float
    positions: np.ndarray
    velocities: np.ndarray


@dataclass(frozen=True)
class StationaryEvidence:
    joint_names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    stamps_ns: np.ndarray
    maximum_position_span_rad: float
    maximum_absolute_velocity_rad_s: float
    maximum_feedback_gap_sec: float
    window_duration_sec: float


@dataclass(frozen=True)
class DecodedObservation:
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    D: np.ndarray
    stamp: np.ndarray
    scene_id: str
    observation_id: str
    optical_frame: str
    source_name: str
    source_type: int
    valid_depth_fraction: float
    color_depth_skew_sec: float


@dataclass(frozen=True)
class CheckerboardDetection:
    corners_px: np.ndarray
    object_points_m: np.ndarray
    T_camera_checkerboard: np.ndarray
    reprojection_rms_px: float
    outer_corners_px: np.ndarray
    minimum_border_margin_px: float


def stamp_to_ns(stamp: Any) -> int:
    """严格转换 ROS Time-like 对象。"""
    sec, nanosec = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise CaptureError("ROS 时间戳非法")
    return sec * 1_000_000_000 + nanosec


def stamp_array(stamp: Any) -> np.ndarray:
    stamp_ns = stamp_to_ns(stamp)
    return np.array(divmod(stamp_ns, 1_000_000_000), dtype=np.int64)


def normalized_duration_fields(seconds: float) -> tuple[int, int]:
    """把有限正秒数转换为规范化 ROS Duration sec/nanosec。"""
    value = float(seconds)
    if not np.isfinite(value) or value <= 0.0:
        raise CaptureError("Duration 必须为有限正数")
    sec = math.floor(value)
    nanosec = int(round((value - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    if sec > 2_147_483_647:
        raise CaptureError("Duration 超过 ROS int32 sec 范围")
    return int(sec), nanosec


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
    strides = (
        (step, channels * itemsize, itemsize)
        if channels > 1
        else (step, itemsize)
    )
    return np.ndarray(shape, dtype=dtype, buffer=data, strides=strides)


def decode_rgb(message: Any) -> np.ndarray:
    """解码 canonical rgb8，不依赖 cv_bridge。"""
    if str(message.encoding).lower() != "rgb8":
        raise CaptureError(f"彩色编码必须是 rgb8，实际为 {message.encoding!r}")
    return np.array(_image_view(message, 3, np.dtype("u1")), copy=True, order="C")


def decode_depth(message: Any) -> np.ndarray:
    """解码 canonical 32FC1 米并统一坏值为 NaN。"""
    if str(message.encoding).upper() != "32FC1":
        raise CaptureError(f"深度编码必须是 32FC1 米，实际为 {message.encoding!r}")
    dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    result = np.asarray(_image_view(message, 1, dtype), dtype=np.float32).copy(order="C")
    result[(~np.isfinite(result)) | (result <= 0.0)] = np.nan
    return result


def validate_transform(matrix: np.ndarray, name: str) -> np.ndarray:
    """验证并复制一个 T_parent_child 刚体矩阵。"""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise CaptureError(f"{name} 必须是有限 4x4 矩阵")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise CaptureError(f"{name} 齐次末行非法")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise CaptureError(f"{name} 旋转矩阵不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise CaptureError(f"{name} 旋转行列式不为 +1")
    return value.copy()


def transform_message_to_matrix(transform: Any) -> np.ndarray:
    """把 geometry_msgs/Transform-like 的 T_parent_child 转为矩阵。"""
    translation = transform.translation
    rotation = transform.rotation
    xyz = np.array((translation.x, translation.y, translation.z), dtype=np.float64)
    quaternion = np.array((rotation.x, rotation.y, rotation.z, rotation.w), dtype=np.float64)
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quaternion)):
        raise CaptureError("TF 含 NaN 或 Inf")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise CaptureError("TF 四元数为零")
    x, y, z, w = quaternion / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    result[:3, 3] = xyz
    return validate_transform(result, "T_base_link7")


def decode_observation(
    message: Any, *, expected_scene: str, expected_observation_id: str
) -> DecodedObservation:
    """严格解码服务所指向的 canonical Observation。"""
    scene_id, observation_id = str(message.scene_id), str(message.observation_id)
    if not expected_scene.strip() or not expected_observation_id.strip():
        raise CaptureError("期望的 scene_id/observation_id 不能为空")
    if not scene_id.strip() or not observation_id.strip():
        raise CaptureError("Observation 的 scene_id/observation_id 不能为空")
    if scene_id != expected_scene or observation_id != expected_observation_id:
        raise CaptureError("Observation 的 scene_id/observation_id 与服务响应不匹配")
    if int(message.source_type) != 1:
        raise CaptureError("手眼现场采集只接受 SOURCE_REAL Observation")
    if not str(message.source_name).strip():
        raise CaptureError("Observation source_name 为空")

    rgb, depth = decode_rgb(message.color), decode_depth(message.depth)
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
        or not np.allclose(K[2], (0.0, 0.0, 1.0), atol=1.0e-9)
    ):
        raise CaptureError("CameraInfo K 非法")
    D = np.asarray(info.d, dtype=np.float64).copy()
    if D.ndim != 1 or not np.all(np.isfinite(D)):
        raise CaptureError("CameraInfo D 非法")

    frame_id = str(message.header.frame_id)
    if not frame_id or any(
        str(field.header.frame_id) != frame_id
        for field in (message.color, message.depth, message.camera_info)
    ):
        raise CaptureError("Observation/color/depth/CameraInfo frame_id 不一致")
    authoritative = stamp_array(message.header.stamp)
    for name, field in (("depth", message.depth), ("CameraInfo", message.camera_info)):
        if not np.array_equal(authoritative, stamp_array(field.header.stamp)):
            raise CaptureError(f"{name} stamp 不等于 Observation 深度曝光 stamp")
    color_delta = (
        stamp_to_ns(message.color.header.stamp)
        - stamp_to_ns(message.header.stamp)
    ) * 1.0e-9
    skew = float(message.color_depth_skew_sec)
    if not np.isfinite(skew) or abs(skew) > 0.005:
        raise CaptureError("color_depth_skew_sec 非法或超过 5 ms")
    if not math.isclose(color_delta, skew, rel_tol=0.0, abs_tol=2.0e-6):
        raise CaptureError("color header stamp 与 color_depth_skew_sec 不一致")
    valid_fraction = float(message.valid_depth_fraction)
    measured_fraction = float(np.isfinite(depth).mean())
    if not 0.0 <= valid_fraction <= 1.0:
        raise CaptureError("valid_depth_fraction 不在 [0,1]")
    if not math.isclose(valid_fraction, measured_fraction, rel_tol=0.0, abs_tol=2.0e-6):
        raise CaptureError("valid_depth_fraction 与深度 payload 不一致")

    return DecodedObservation(
        rgb=rgb,
        depth_m=depth,
        K=K,
        D=D,
        stamp=authoritative,
        scene_id=scene_id,
        observation_id=observation_id,
        optical_frame=frame_id,
        source_name=str(message.source_name),
        source_type=int(message.source_type),
        valid_depth_fraction=valid_fraction,
        color_depth_skew_sec=skew,
    )


def ordered_joint_sample(message: Any, received_monotonic: float) -> JointSample:
    """提取完整、无重复的 joint1..joint7 反馈。"""
    names = tuple(str(value) for value in message.name)
    if len(names) != len(set(names)):
        raise CaptureError("JointState 含重复关节名")
    if len(message.position) != len(names):
        raise CaptureError("JointState name/position 长度不一致")
    position_by_name = dict(zip(names, message.position))
    missing = [name for name in JOINT_NAMES if name not in position_by_name]
    if missing:
        raise CaptureError("JointState 缺少 " + ", ".join(missing))
    positions = np.array([position_by_name[name] for name in JOINT_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(positions)):
        raise CaptureError("关节位置含 NaN 或 Inf")
    if len(message.velocity) != len(names):
        raise CaptureError("严格静止检查要求 velocity 与 name 等长")
    velocity_by_name = dict(zip(names, message.velocity))
    velocities = np.array([velocity_by_name[name] for name in JOINT_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(velocities)):
        raise CaptureError("关节速度含 NaN 或 Inf")
    return JointSample(
        stamp_ns=stamp_to_ns(message.header.stamp),
        received_monotonic=float(received_monotonic),
        positions=positions,
        velocities=velocities,
    )


def stationary_evidence(
    samples: Sequence[JointSample],
    *,
    minimum_duration_sec: float,
    maximum_position_span_rad: float,
    maximum_absolute_velocity_rad_s: float,
    maximum_feedback_age_sec: float,
    now_monotonic: float,
    maximum_feedback_gap_sec: float = 0.25,
    minimum_sample_count: int = 5,
) -> StationaryEvidence:
    """严格证明当前关节在窗口内静止且反馈新鲜。"""
    if minimum_sample_count < 2 or len(samples) < minimum_sample_count:
        raise CaptureError(f"静止检查至少需要 {minimum_sample_count} 条完整关节反馈")
    ordered = sorted(samples, key=lambda item: item.received_monotonic)
    received_times = np.asarray(
        [item.received_monotonic for item in ordered], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(received_times))
        or not np.isfinite(maximum_feedback_gap_sec)
        or maximum_feedback_gap_sec <= 0.0
    ):
        raise CaptureError("关节反馈接收时间或最大间隙门限非法")
    duration = float(received_times[-1] - received_times[0])
    age = now_monotonic - ordered[-1].received_monotonic
    if duration < minimum_duration_sec:
        raise CaptureError("关节静止观察窗口不足")
    if age < 0.0 or age > maximum_feedback_age_sec:
        raise CaptureError("最新关节反馈已过期")
    gap = float(np.max(np.diff(received_times)))
    if gap > maximum_feedback_gap_sec:
        raise CaptureError(
            f"关节反馈不连续：最大接收间隙 {gap:.6g} s"
        )
    positions = np.stack([sample.positions for sample in ordered])
    span = float(np.max(np.ptp(positions, axis=0)))
    if span > maximum_position_span_rad:
        raise CaptureError(
            f"机械臂未静止：窗口内最大关节跨度 {span:.6g} rad"
        )
    velocities = np.stack([sample.velocities for sample in ordered])
    velocity_max = float(np.max(np.abs(velocities)))
    if velocity_max > maximum_absolute_velocity_rad_s:
        raise CaptureError(
            f"机械臂未静止：最大反馈速度 {velocity_max:.6g} rad/s"
        )
    return StationaryEvidence(
        joint_names=JOINT_NAMES,
        positions=positions,
        velocities=velocities,
        stamps_ns=np.asarray([sample.stamp_ns for sample in ordered], dtype=np.int64),
        maximum_position_span_rad=span,
        maximum_absolute_velocity_rad_s=velocity_max,
        maximum_feedback_gap_sec=gap,
        window_duration_sec=duration,
    )


def recent_joint_samples(
    samples: Iterable[JointSample], *, now_monotonic: float, duration_sec: float
) -> list[JointSample]:
    """选择恰好覆盖当前静止门限的最近反馈，避免旧运动污染窗口。"""
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise CaptureError("关节窗口时长必须为有限正数")
    ordered = sorted(samples, key=lambda item: item.received_monotonic)
    lower = now_monotonic - duration_sec
    selected = [item for item in ordered if item.received_monotonic >= lower]
    if selected and selected[0].received_monotonic > lower:
        previous = [item for item in ordered if item.received_monotonic < lower]
        if previous:
            selected.insert(0, previous[-1])
    return selected


def nearest_joint_sample(
    samples: Sequence[JointSample], observation_stamp_ns: int, maximum_delta_sec: float
) -> tuple[JointSample, float]:
    """选取最接近相机曝光时间的关节向量并限制跨时钟误差。"""
    if not samples:
        raise CaptureError("没有关节反馈可与相机曝光时间配对")
    selected = min(samples, key=lambda item: abs(item.stamp_ns - observation_stamp_ns))
    delta_sec = abs(selected.stamp_ns - observation_stamp_ns) * 1.0e-9
    if delta_sec > maximum_delta_sec:
        raise CaptureError(
            f"最近 JointState 与相机曝光相差 {delta_sec:.6f} s，超过门限"
        )
    return selected, delta_sec


def checkerboard_object_points() -> np.ndarray:
    """返回 11x8 内角的 checkerboard-frame 坐标。"""
    columns, rows = BOARD_INNER_CORNERS
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= np.float32(SQUARE_SIZE_M)
    return points


def _grid_homography(corners: np.ndarray) -> np.ndarray:
    """拟合棋盘网格坐标到 RGB 像素的单应性。"""
    columns, rows = BOARD_INNER_CORNERS
    source = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
    homography, _ = cv2.findHomography(source, corners.astype(np.float32), method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise CaptureError("无法由内角点估计棋盘单应性")
    return homography


def _sample_gray(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    """双线性采样若干亚像素灰度点。"""
    values = cv2.remap(
        gray.astype(np.float32),
        points[:, 0].astype(np.float32),
        points[:, 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(values, dtype=np.float64).reshape(-1)


def canonicalize_checkerboard_corners(
    gray: np.ndarray, corners: np.ndarray, *, minimum_square_contrast: float
) -> np.ndarray:
    """用坐标原点候选侧外格为黑色的约定消除 180° 棋盘方向歧义。"""
    columns, rows = BOARD_INNER_CORNERS
    value = np.asarray(corners, dtype=np.float64).reshape(rows, columns, 2)
    homography = _grid_homography(value.reshape(-1, 2))
    grid_centres = np.array(
        [[-0.5, -0.5], [columns - 0.5, rows - 0.5]], dtype=np.float32
    )
    pixels = cv2.perspectiveTransform(grid_centres[:, None, :], homography)[:, 0, :]
    intensity = _sample_gray(gray, pixels)
    contrast = float(abs(intensity[0] - intensity[1]))
    if contrast < minimum_square_contrast:
        raise CaptureError(
            f"无法消除棋盘 180° 歧义：两端基准格对比仅 {contrast:.2f}"
        )
    # 12x9 格棋盘两端候选外格颜色相反；把黑格一端定义为物理坐标原点。
    if intensity[0] > intensity[1]:
        value = value[::-1, ::-1]
    return value.reshape(-1, 2).copy()


def _outer_board_corners(corners: np.ndarray) -> np.ndarray:
    """由内角单应性外推完整 12x9 格标定板四角。"""
    columns, rows = BOARD_INNER_CORNERS
    homography = _grid_homography(corners)
    outer = np.array(
        [[-1.0, -1.0], [columns, -1.0], [columns, rows], [-1.0, rows]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(outer[:, None, :], homography)[:, 0, :]


def detect_checkerboard(
    rgb: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    *,
    minimum_border_margin_px: float,
    maximum_reprojection_rms_px: float,
    minimum_square_contrast: float = 20.0,
) -> CheckerboardDetection:
    """检测完整 12x9 格/11x8 内角板并估计 T_camera_checkerboard。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CaptureError("棋盘检测输入必须是 uint8 RGB 图")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, detected = cv2.findChessboardCornersSB(gray, BOARD_INNER_CORNERS, flags=flags)
    expected = BOARD_INNER_CORNERS[0] * BOARD_INNER_CORNERS[1]
    if not found or detected is None or len(detected) != expected:
        raise CaptureError(f"未完整检测到 11x8={expected} 个棋盘内角")
    corners = canonicalize_checkerboard_corners(
        gray,
        np.asarray(detected, dtype=np.float64).reshape(expected, 2),
        minimum_square_contrast=minimum_square_contrast,
    )
    outer = _outer_board_corners(corners)
    height, width = rgb.shape[:2]
    margins = np.concatenate(
        (outer[:, 0], outer[:, 1], (width - 1) - outer[:, 0], (height - 1) - outer[:, 1])
    )
    margin = float(np.min(margins))
    if margin < minimum_border_margin_px:
        raise CaptureError(
            f"标定板不完整或太贴近画面边缘：最小外边界裕量 {margin:.2f} px"
        )
    object_points = checkerboard_object_points()
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        corners,
        K,
        D,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success:
        raise CaptureError("solvePnP 失败")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation_vector).reshape(3)
    transform = validate_transform(transform, "T_camera_checkerboard")
    if transform[2, 3] <= 0.0:
        raise CaptureError("棋盘 PnP 位于相机后方")
    projected, _ = cv2.projectPoints(
        object_points, rotation_vector, translation_vector, K, D
    )
    residual = projected.reshape(-1, 2) - corners
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if not np.isfinite(rms) or rms > maximum_reprojection_rms_px:
        raise CaptureError(
            f"棋盘重投影 RMS {rms:.3f} px 超过 {maximum_reprojection_rms_px:.3f} px"
        )
    return CheckerboardDetection(
        corners_px=corners,
        object_points_m=object_points.astype(np.float64),
        T_camera_checkerboard=transform,
        reprojection_rms_px=rms,
        outer_corners_px=outer.astype(np.float64),
        minimum_border_margin_px=margin,
    )


def assert_unique_sample(
    output_directory: str | Path,
    *,
    sample_id: str,
    observation_id: str,
    observation_stamp_ns: int,
    positions: np.ndarray,
    K: np.ndarray,
    optical_frame: str,
    source_name: str,
    minimum_joint_separation_rad: float,
) -> None:
    """扫描既有 NPZ，拒绝 ID、stamp、Observation 或近重复姿态。"""
    destination = Path(output_directory)
    for path in sorted(destination.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "schema_version" not in archive:
                    continue
                if str(archive["schema_version"].item()) != SCHEMA_VERSION:
                    continue
                if str(archive["sample_id"].item()) == sample_id:
                    raise CaptureError(f"sample_id 已存在于 {path.name}")
                if str(archive["observation_id"].item()) == observation_id:
                    raise CaptureError(f"Observation ID 已采集于 {path.name}")
                stamp = np.asarray(archive["stamp"], dtype=np.int64)
                if (
                    stamp.shape != (2,)
                    or int(stamp[0]) < 0
                    or not 0 <= int(stamp[1]) < 1_000_000_000
                ):
                    raise CaptureError(f"既有样本 {path.name} 的 stamp 非法")
                existing_ns = int(stamp[0]) * 1_000_000_000 + int(stamp[1])
                if existing_ns == observation_stamp_ns:
                    raise CaptureError(f"曝光时间戳已采集于 {path.name}")
                existing_K = np.asarray(archive["K"], dtype=np.float64)
                if existing_K.shape != (3, 3) or not np.allclose(
                    existing_K, K, rtol=0.0, atol=1.0e-9
                ):
                    raise CaptureError(f"相机内参 K 与既有样本 {path.name} 不一致")
                if str(archive["frame_id"].item()) != optical_frame:
                    raise CaptureError(f"相机光学 frame 与既有样本 {path.name} 不一致")
                if str(archive["source_name"].item()) != source_name:
                    raise CaptureError(f"相机 source_name 与既有样本 {path.name} 不一致")
                existing_positions = np.asarray(archive["joint_positions"], dtype=np.float64)
                if existing_positions.shape != (7,) or not np.all(
                    np.isfinite(existing_positions)
                ):
                    raise CaptureError(f"既有样本 {path.name} 的 joint_positions 非法")
                delta = float(np.max(np.abs(existing_positions - positions)))
                if delta < minimum_joint_separation_rad:
                    raise CaptureError(
                        f"当前姿态与 {path.name} 近重复：最大关节差仅 {delta:.6g} rad"
                    )
        except CaptureError:
            raise
        except (OSError, ValueError, KeyError) as error:
            raise CaptureError(f"无法验证既有样本 {path.name}: {error}") from error


def sample_arrays(
    *,
    sample_id: str,
    observation: DecodedObservation,
    detection: CheckerboardDetection,
    stationary: StationaryEvidence,
    T_base_link7: np.ndarray,
    tf_lookup_mode: str,
    tf_stamp_ns: int,
    selected_joint: JointSample,
    joint_observation_delta_sec: float,
    joint_tf_delta_sec: float,
    joint_selection_mode: str,
) -> dict[str, np.ndarray]:
    """构造不含 object/pickle 的单样本 NPZ 字段。"""
    if not sample_id.strip():
        raise CaptureError("sample_id 不能为空")
    T_base_link7 = validate_transform(T_base_link7, "T_base_link7")
    T_camera_checkerboard = validate_transform(
        detection.T_camera_checkerboard, "T_camera_checkerboard"
    )
    timestamp_sec = float(observation.stamp[0]) + float(observation.stamp[1]) * 1.0e-9
    positions = selected_joint.positions.copy()
    return {
        "schema_version": np.array(SCHEMA_VERSION),
        "sample_id": np.array(sample_id),
        "timestamp_sec": np.array(timestamp_sec, dtype=np.float64),
        "scene_id": np.array(observation.scene_id),
        "observation_id": np.array(observation.observation_id),
        "source_name": np.array(observation.source_name),
        "source_type": np.array(observation.source_type, dtype=np.uint8),
        "stamp": observation.stamp.astype(np.int64, copy=True),
        "frame_id": np.array(observation.optical_frame),
        "rgb": observation.rgb,
        "depth_m": observation.depth_m,
        "K": observation.K,
        "D": observation.D,
        "valid_depth_fraction": np.array(observation.valid_depth_fraction, dtype=np.float32),
        "color_depth_skew_sec": np.array(observation.color_depth_skew_sec, dtype=np.float64),
        "board_squares": np.asarray(BOARD_SQUARES, dtype=np.int32),
        "board_inner_corners": np.asarray(BOARD_INNER_CORNERS, dtype=np.int32),
        "square_size_m": np.array(SQUARE_SIZE_M, dtype=np.float64),
        "checkerboard_corners_px": detection.corners_px.astype(np.float64),
        "checkerboard_object_points_m": detection.object_points_m.astype(np.float64),
        "checkerboard_outer_corners_px": detection.outer_corners_px.astype(np.float64),
        "checkerboard_minimum_border_margin_px": np.array(
            detection.minimum_border_margin_px, dtype=np.float64
        ),
        "checkerboard_reprojection_rms_px": np.array(
            detection.reprojection_rms_px, dtype=np.float64
        ),
        "corner_count": np.array(len(detection.corners_px), dtype=np.int32),
        "reprojection_rms_px": np.array(detection.reprojection_rms_px, dtype=np.float64),
        "T_camera_checkerboard": T_camera_checkerboard,
        "base_frame": np.array(BASE_FRAME),
        "link_frame": np.array(LINK_FRAME),
        "T_base_link7": T_base_link7,
        "tf_lookup_mode": np.array(tf_lookup_mode),
        "tf_stamp_ns": np.array(tf_stamp_ns, dtype=np.int64),
        "joint_names": np.asarray(stationary.joint_names, dtype=np.str_),
        "joint_positions": positions.astype(np.float64),
        "joint_selected_stamp_ns": np.array(selected_joint.stamp_ns, dtype=np.int64),
        "joint_selection_mode": np.array(joint_selection_mode),
        "joint_observation_delta_sec": np.array(
            joint_observation_delta_sec, dtype=np.float64
        ),
        "joint_tf_delta_sec": np.array(joint_tf_delta_sec, dtype=np.float64),
        "joint_window_positions": stationary.positions.astype(np.float64),
        "joint_window_velocities": stationary.velocities.astype(np.float64),
        "joint_window_stamps_ns": stationary.stamps_ns.astype(np.int64),
        "joint_window_duration_sec": np.array(stationary.window_duration_sec, dtype=np.float64),
        "joint_maximum_position_span_rad": np.array(
            stationary.maximum_position_span_rad, dtype=np.float64
        ),
        "joint_maximum_absolute_velocity_rad_s": np.array(
            stationary.maximum_absolute_velocity_rad_s, dtype=np.float64
        ),
        "joint_maximum_feedback_gap_sec": np.array(
            stationary.maximum_feedback_gap_sec, dtype=np.float64
        ),
        "motion_commands_sent": np.array(0, dtype=np.uint8),
    }


def save_npz_atomic(arrays: dict[str, np.ndarray], output: str | Path) -> Path:
    """原子保存无 object NPZ；目标存在时拒绝覆盖。"""
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise CaptureError(f"输出已存在，拒绝覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise CaptureError("NPZ 字段不得使用 object dtype")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # hard-link 在同一目录内原子地实现 no-replace；目标若在检查后由另一进程
            # 创建，link 会失败而不会覆盖已有有效样本。
            os.link(temporary, destination)
        except FileExistsError as error:
            raise CaptureError(f"输出已存在，拒绝覆盖：{destination}") from error
        temporary.unlink()
        temporary = None
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def _run_ros_capture(arguments: argparse.Namespace) -> Path:
    """ROS 薄层：只订阅、查询 TF、调用 Capture 服务，不创建 publisher/action。"""
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from rclpy.time import Time
        from sensor_msgs.msg import JointState
        from strawberry_perception_interfaces.msg import Observation
        from strawberry_perception_interfaces.srv import CaptureObservation
        from tf2_ros import Buffer, TransformException, TransformListener
    except ImportError as error:
        raise CaptureError("找不到系统 ROS 2 接口；请 source ROS、nero_ws、perception_ws") from error

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("capture_handeye_sample_read_only")
            self.joints: deque[JointSample] = deque(maxlen=500)
            self.observations: OrderedDict[tuple[str, str], Any] = OrderedDict()
            self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.joint_subscription = self.create_subscription(
                JointState, arguments.joint_topic, self.on_joint, qos_profile_sensor_data
            )
            observation_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.observation_subscription = self.create_subscription(
                Observation,
                arguments.observation_topic,
                self.on_observation,
                observation_qos,
            )
            self.capture_client = self.create_client(
                CaptureObservation, arguments.capture_service
            )

        def on_joint(self, message: Any) -> None:
            try:
                self.joints.append(ordered_joint_sample(message, time.monotonic()))
            except CaptureError as error:
                self.get_logger().warning(f"忽略不完整 JointState：{error}")

        def on_observation(self, message: Any) -> None:
            key = (str(message.scene_id), str(message.observation_id))
            if key not in self.observations:
                self.observations[key] = message
            while len(self.observations) > 32:
                self.observations.popitem(last=False)

    rclpy.init(args=None)
    node = CaptureNode()
    try:
        if not node.capture_client.wait_for_service(timeout_sec=arguments.wait_timeout_sec):
            raise CaptureError(f"未找到采集服务 {arguments.capture_service}")

        joint_deadline = time.monotonic() + arguments.wait_timeout_sec
        before: StationaryEvidence | None = None
        while time.monotonic() < joint_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                now = time.monotonic()
                before = stationary_evidence(
                    recent_joint_samples(
                        node.joints,
                        now_monotonic=now,
                        duration_sec=arguments.stationary_window_sec,
                    ),
                    minimum_duration_sec=arguments.stationary_window_sec,
                    maximum_position_span_rad=arguments.max_joint_span_rad,
                    maximum_absolute_velocity_rad_s=arguments.max_joint_velocity_rad_s,
                    maximum_feedback_age_sec=arguments.max_feedback_age_sec,
                    now_monotonic=now,
                    maximum_feedback_gap_sec=arguments.max_feedback_gap_sec,
                    minimum_sample_count=arguments.min_joint_samples,
                )
                break
            except CaptureError:
                continue
        if before is None:
            raise CaptureError("等待超时：未取得新鲜、完整且满足静止门限的 joint1..joint7")

        capture_begin_monotonic = time.monotonic()
        request = CaptureObservation.Request()
        request.scene_id = arguments.scene
        request.not_before = node.get_clock().now().to_msg()
        not_before_ns = stamp_to_ns(request.not_before)
        request.timeout.sec, request.timeout.nanosec = normalized_duration_fields(
            arguments.capture_timeout_sec
        )
        request.discard_frames = arguments.discard_frames
        request.require_color = True
        request.require_mask = False
        request.require_pose = False
        future = node.capture_client.call_async(request)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=arguments.capture_timeout_sec + 2.0
        )
        if not future.done() or future.result() is None:
            raise CaptureError("CaptureObservation 服务未在期限内返回")
        response = future.result()
        if not bool(response.success):
            raise CaptureError(f"CaptureObservation 失败：code={response.code}, {response.reason}")
        observation_id = str(response.observation_id)
        if not observation_id.strip():
            raise CaptureError("CaptureObservation 成功响应缺少 observation_id")
        key = (arguments.scene, observation_id)
        topic_deadline = time.monotonic() + 2.0
        while key not in node.observations and time.monotonic() < topic_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if key not in node.observations:
            raise CaptureError("服务成功，但未收到匹配的 canonical Observation")
        observation = decode_observation(
            node.observations.pop(key),
            expected_scene=arguments.scene,
            expected_observation_id=observation_id,
        )
        if not np.array_equal(observation.stamp, stamp_array(response.stamp)):
            raise CaptureError("服务响应 stamp 与 Observation 不一致")

        observation_ns = int(observation.stamp[0]) * 1_000_000_000 + int(observation.stamp[1])
        if observation_ns < not_before_ns:
            raise CaptureError("Observation stamp 早于 Capture 请求的 not_before")
        tf_mode = "observation_stamp"
        try:
            transform = node.tf_buffer.lookup_transform(
                BASE_FRAME,
                LINK_FRAME,
                Time(nanoseconds=observation_ns),
                timeout=Duration(seconds=arguments.tf_timeout_sec),
            )
        except TransformException:
            tf_mode = "latest_while_stationary"
            try:
                transform = node.tf_buffer.lookup_transform(
                    BASE_FRAME,
                    LINK_FRAME,
                    Time(),
                    timeout=Duration(seconds=arguments.tf_timeout_sec),
                )
            except TransformException as error:
                raise CaptureError(
                    "无法取得 base_link→link7 TF；请启动 README 中只读 robot_state_publisher"
                ) from error
        T_base_link7 = transform_message_to_matrix(transform.transform)
        tf_stamp_ns = stamp_to_ns(transform.header.stamp)
        if tf_mode == "observation_stamp" and abs(tf_stamp_ns - observation_ns) > 1_000_000:
            raise CaptureError("按曝光时间查询到的 TF stamp 相差超过 1 ms")

        # TF 已冻结后继续观察一个完整窗口；最终静止证据同时覆盖 Capture 和 TF。
        post_deadline = time.monotonic() + arguments.stationary_window_sec
        while time.monotonic() < post_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        capture_start = capture_begin_monotonic - arguments.stationary_window_sec
        capture_window = [
            sample for sample in node.joints if sample.received_monotonic >= capture_start
        ]
        stationary = stationary_evidence(
            capture_window,
            minimum_duration_sec=2.0 * arguments.stationary_window_sec,
            maximum_position_span_rad=arguments.max_joint_span_rad,
            maximum_absolute_velocity_rad_s=arguments.max_joint_velocity_rad_s,
            maximum_feedback_age_sec=arguments.max_feedback_age_sec,
            now_monotonic=time.monotonic(),
            maximum_feedback_gap_sec=arguments.max_feedback_gap_sec,
            minimum_sample_count=arguments.min_joint_samples,
        )

        detection = detect_checkerboard(
            observation.rgb,
            observation.K,
            observation.D,
            minimum_border_margin_px=arguments.min_board_margin_px,
            maximum_reprojection_rms_px=arguments.max_reprojection_rms_px,
        )

        # 无论 TF 来自曝光时刻查询还是 latest 回退，都必须与本次连续、静止的
        # JointState 窗口有新鲜的时间配对，拒绝 tf2 缓存中的旧姿态。
        joint_nearest_tf, joint_tf_delta_sec = nearest_joint_sample(
            capture_window,
            tf_stamp_ns,
            arguments.max_joint_tf_delta_sec,
        )
        try:
            selected_joint, joint_observation_delta_sec = nearest_joint_sample(
                capture_window,
                observation_ns,
                arguments.max_joint_observation_delta_sec,
            )
            joint_selection_mode = "nearest_observation_ros_stamp"
        except CaptureError:
            # 相机 global time 与机械臂反馈可能不在同一时钟域。机械臂已在完整
            # 采集窗内通过静止门，故改选生成该 TF 的最近 q，并明确记录回退。
            selected_joint = joint_nearest_tf
            joint_observation_delta_sec = abs(
                selected_joint.stamp_ns - observation_ns
            ) * 1.0e-9
            joint_selection_mode = "nearest_tf_stamp_while_stationary"

        positions = selected_joint.positions
        assert_unique_sample(
            arguments.output_directory,
            sample_id=arguments.sample_id,
            observation_id=observation.observation_id,
            observation_stamp_ns=observation_ns,
            positions=positions,
            K=observation.K,
            optical_frame=observation.optical_frame,
            source_name=observation.source_name,
            minimum_joint_separation_rad=arguments.min_joint_separation_rad,
        )
        arrays = sample_arrays(
            sample_id=arguments.sample_id,
            observation=observation,
            detection=detection,
            stationary=stationary,
            T_base_link7=T_base_link7,
            tf_lookup_mode=tf_mode,
            tf_stamp_ns=tf_stamp_ns,
            selected_joint=selected_joint,
            joint_observation_delta_sec=joint_observation_delta_sec,
            joint_tf_delta_sec=joint_tf_delta_sec,
            joint_selection_mode=joint_selection_mode,
        )
        output = Path(arguments.output_directory) / f"{arguments.sample_id}.npz"
        saved = save_npz_atomic(arrays, output)
        print(
            json.dumps(
                {
                    "status": "saved",
                    "path": str(saved),
                    "sample_id": arguments.sample_id,
                    "observation_id": observation.observation_id,
                    "corner_count": int(len(detection.corners_px)),
                    "reprojection_rms_px": detection.reprojection_rms_px,
                    "joint_maximum_position_span_rad": stationary.maximum_position_span_rad,
                    "joint_observation_delta_sec": joint_observation_delta_sec,
                    "joint_tf_delta_sec": joint_tf_delta_sec,
                    "joint_selection_mode": joint_selection_mode,
                    "tf_lookup_mode": tf_mode,
                    "motion_commands_sent": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return saved
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读采集一个当前静止姿态的手眼标定样本（不会发运动命令）"
    )
    parser.add_argument("--sample-id", required=True, help="本姿态唯一 ID，例如 pose_001")
    parser.add_argument("--scene", default="handeye_calibration", help="Capture scene_id")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--observation-topic", default=OBSERVATION_TOPIC)
    parser.add_argument("--capture-service", default=CAPTURE_SERVICE)
    parser.add_argument("--joint-topic", default=JOINT_TOPIC)
    parser.add_argument("--discard-frames", type=int, default=3)
    parser.add_argument("--capture-timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=15.0)
    parser.add_argument("--tf-timeout-sec", type=float, default=1.0)
    parser.add_argument("--stationary-window-sec", type=float, default=1.0)
    parser.add_argument("--min-joint-samples", type=int, default=5)
    parser.add_argument("--max-feedback-age-sec", type=float, default=0.25)
    parser.add_argument("--max-feedback-gap-sec", type=float, default=0.25)
    parser.add_argument("--max-joint-span-rad", type=float, default=0.002)
    parser.add_argument("--max-joint-velocity-rad-s", type=float, default=0.01)
    parser.add_argument("--max-joint-observation-delta-sec", type=float, default=0.20)
    parser.add_argument("--max-joint-tf-delta-sec", type=float, default=0.20)
    parser.add_argument("--min-joint-separation-rad", type=float, default=0.03)
    parser.add_argument("--min-board-margin-px", type=float, default=8.0)
    parser.add_argument("--max-reprojection-rms-px", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    text_fields = (
        arguments.sample_id,
        arguments.scene,
        arguments.observation_topic,
        arguments.capture_service,
        arguments.joint_topic,
    )
    if any(not value.strip() for value in text_fields):
        parser.error("ID、scene、topic 和 frame 均不能为空")
    if "/" in arguments.sample_id or arguments.sample_id in (".", ".."):
        parser.error("--sample-id 不得包含路径分隔符")
    if arguments.discard_frames < 0:
        parser.error("--discard-frames 不能为负")
    if arguments.min_joint_samples < 2:
        parser.error("--min-joint-samples 必须至少为 2")
    positive = (
        arguments.capture_timeout_sec,
        arguments.wait_timeout_sec,
        arguments.tf_timeout_sec,
        arguments.stationary_window_sec,
        arguments.max_feedback_age_sec,
        arguments.max_feedback_gap_sec,
        arguments.max_joint_span_rad,
        arguments.max_joint_velocity_rad_s,
        arguments.max_joint_observation_delta_sec,
        arguments.max_joint_tf_delta_sec,
        arguments.min_joint_separation_rad,
        arguments.min_board_margin_px,
        arguments.max_reprojection_rms_px,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive):
        parser.error("所有时间和门限参数必须是有限正数")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    print("安全模式：只订阅 joint/TF/Observation 并调用 Capture；不会发布任何运动命令。")
    _run_ros_capture(arguments)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CaptureError as error:
        print(f"采集失败：{error}", file=__import__("sys").stderr)
        raise SystemExit(1)
