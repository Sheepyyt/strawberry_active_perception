#!/usr/bin/env python3
"""Capture a fresh strawberry and inspect the current robot without motion.

Only CaptureObservation is called. Robot state comes from subscriptions, and
Placo runs offline on a copied joint sample. This tool never enables a driver,
creates a motion action client, or publishes a robot command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for package in ("strawberry_learned_mask", "strawberry_active_perception_bridge"):
    sys.path.insert(0, str(ROOT / "perception_ws" / "src" / package))

from strawberry_active_perception_bridge.real_nbv_contract import (  # noqa: E402
    NERO_JOINT_NAMES,
    NERO_RESEARCH_SAFE_JOINT_LIMITS_RAD,
    decode_mask_mono8,
    estimate_target_center,
    ordered_joint_positions,
    wire_uint8,
)
from strawberry_learned_mask.provider import decode_depth_32fc1, decode_rgb8  # noqa: E402


def observation_matches(message: Any, scene: str, observation_id: str, not_before_ns: int) -> bool:
    """Reject a latched or mismatched image, even when the scene happens to match."""
    stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
    return bool(
        message.scene_id == scene
        and message.observation_id == observation_id
        and stamp >= not_before_ns
        and "mask=yolo11" in message.source_name
    )


def analyze_target(depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray) -> dict[str, Any]:
    """Use the supervisor's robust target-depth estimator; empty masks are valid failures."""
    selected = np.asarray(mask) == 255
    depth = np.asarray(depth, dtype=float)
    if depth.shape != selected.shape:
        raise ValueError("mask and depth dimensions differ")
    count = int(np.count_nonzero(selected))
    valid = selected & np.isfinite(depth) & (depth >= 0.20) & (depth <= 2.50)
    rows, columns = np.nonzero(selected)
    report: dict[str, Any] = {
        "passed": False,
        "mask_pixels": count,
        "valid_depth_pixels_in_mask": int(np.count_nonzero(valid)),
        "valid_depth_ratio_in_mask": float(np.count_nonzero(valid) / count) if count else 0.0,
        "minimum_mask_pixels": 200,
        "minimum_retained_target_pixels": 100,
        "depth_range_m": [0.20, 2.50],
        "bbox_xyxy": [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())] if count else None,
        "median_target_depth_m": None,
        "target_center_camera_m": None,
    }
    try:
        if count < 200:
            raise ValueError(f"YOLO 草莓轮廓只有 {count} 像素，需要至少 200")
        estimate = estimate_target_center(depth, mask, intrinsic, minimum_pixels=100)
        center = estimate.camera_xyz_m
        report.update({
            "passed": True,
            "reason": "已找到单颗草莓，近层有效深度足够；可生成本轮 NBV 预览",
            "median_target_depth_m": float(estimate.median_depth_m),
            "target_center_camera_m": center.tolist(),
            "retained_target_pixels": estimate.retained_pixels,
            "depth_layer_count": estimate.depth_layer_count,
            "target_off_axis_deg": math.degrees(math.atan2(float(np.linalg.norm(center[:2])), float(center[2]))),
            "depth_mad_m": float(estimate.depth_mad_m),
        })
    except ValueError as error:
        report["reason"] = str(error)
        report["adjustment"] = "先暂停流程；如需手动摆臂，先可靠承托并失能。让单颗草莓位于画面中部，建议距离 45–55 cm，再重新检查。"
    return report


def analyze_joints(names: list[str], positions: list[float]) -> dict[str, Any]:
    joints = ordered_joint_positions(names, positions, label="fresh feedback")
    limits = np.asarray(NERO_RESEARCH_SAFE_JOINT_LIMITS_RAD)
    margins = np.minimum(joints - limits[:, 0], limits[:, 1] - joints)
    limiting = int(np.argmin(margins))
    return {
        "passed": bool(np.all(margins >= 0.0)),
        "joint_names": list(NERO_JOINT_NAMES),
        "positions_rad": joints.tolist(),
        "positions_deg": np.degrees(joints).tolist(),
        "safe_limits_rad": limits.tolist(),
        "margin_to_safe_limits_rad": margins.tolist(),
        "minimum_margin_rad": float(margins[limiting]),
        "limiting_joint": NERO_JOINT_NAMES[limiting],
        "reason": "关节在现有保守限位内" if np.all(margins >= 0.0) else "当前关节超出保守限位；先在失能并扶稳时调整，不能直接开始 NBV",
    }


def offline_kinematics(joints: list[float]) -> dict[str, Any]:
    """Use the established Placo environment without mixing its NumPy with ROS/OpenCV."""
    script = """
import json, sys
from pathlib import Path
from strawberry_nero_control.ik_core import PlacoIKSolver
root = Path(sys.argv[1])
path = root / 'nero_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf'
solver = PlacoIKSolver(path)
joints = json.load(sys.stdin)
metrics = solver.configuration_metrics(joints, 'link7')
print('PREFLIGHT_JSON=' + json.dumps({'T_base_link7': solver.forward_kinematics(joints, 'link7').tolist(), 'sigma_min': metrics.sigma_min, 'condition_number': metrics.condition_number}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "nero_ws/src/strawberry_nero_control") + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-c", script, str(ROOT)],
        input=json.dumps(joints), text=True, capture_output=True, timeout=30,
        env=environment, check=False,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("PREFLIGHT_JSON=") and result.returncode == 0:
            output = json.loads(line.split("=", 1)[1])
            output["source"] = "offline Placo FK/Jacobian on fresh measured joints"
            output["nbv_candidate_singularity_reference"] = {"minimum_sigma": 0.10, "maximum_condition": 20.0}
            output["near_singular_for_nbv"] = output["sigma_min"] < 0.10 or output["condition_number"] > 20.0
            if not math.isfinite(output["condition_number"]):
                output["condition_number"] = None
            return output
    raise RuntimeError("Placo 当前姿态计算失败: " + (result.stderr or result.stdout)[-1600:])


def save_images(message: Any, output: Path) -> dict[str, Any]:
    import cv2

    rgb = decode_rgb8(message.color)
    depth = decode_depth_32fc1(message.depth)
    mask = decode_mask_mono8(message.target_mask)
    intrinsic = np.asarray(message.camera_info.k, dtype=float).reshape(3, 3)
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth image dimensions differ")
    target = analyze_target(depth, mask, intrinsic)
    overlay = rgb.copy()
    selected = mask == 255
    overlay[selected] = (0.4 * overlay[selected] + 0.6 * np.asarray([0, 255, 80])).astype(np.uint8)
    for name, image in (("rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), ("overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)), ("mask.png", mask)):
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError(f"cannot save {name}")
    np.savez_compressed(output / "observation.npz", rgb=rgb, depth_m=depth, mask=mask, K=intrinsic)
    target["artifacts"] = {name: str(output / name) for name in ("rgb.png", "overlay.png", "mask.png", "observation.npz")}
    target["image_size"] = [int(message.color.width), int(message.color.height)]
    return target


def run_check(output: Path, camera_only: bool, timeout: float) -> dict[str, Any]:
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from strawberry_perception_interfaces.msg import Observation
    from strawberry_perception_interfaces.srv import CaptureObservation

    report: dict[str, Any] = {
        "schema": "strawberry_demo_initialization/v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "camera_only": camera_only,
        "passed": False,
        "control_capability": "capture_service_and_state_subscribers_only_no_motion_or_enable",
    }
    rclpy.init()
    node = Node("strawberry_demo_initialization_" + uuid.uuid4().hex[:8])
    scene = "demo_check_" + uuid.uuid4().hex
    observations: dict[str, Any] = {}
    state: dict[str, Any] = {}
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(Observation, "/strawberry/perception/learned_observation", lambda message: observations.update({message.observation_id: message}) if message.scene_id == scene else None, qos)
    if not camera_only:
        node.create_subscription(JointState, "/feedback/joint_states", lambda message: state.update(joints=message, joint_received=time.monotonic()), 10)
        node.create_subscription(DiagnosticArray, "/strawberry_nero/diagnostics", lambda message: state.update(diagnostics=message, diagnostic_received=time.monotonic()), 10)

    def spin_until(predicate: Any, seconds: float, description: str) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(node, timeout_sec=0.1)
        raise TimeoutError(description)

    try:
        client = node.create_client(CaptureObservation, "/strawberry/perception/capture_observation")
        if not client.wait_for_service(timeout_sec=min(timeout, 10.0)):
            raise TimeoutError("相机采集服务未就绪，请先启动相机")
        request = CaptureObservation.Request()
        request.scene_id = scene
        request.not_before = node.get_clock().now().to_msg()
        not_before = int(request.not_before.sec) * 1_000_000_000 + int(request.not_before.nanosec)
        request.timeout.sec = 5
        request.discard_frames = 3
        request.require_color = True
        request.require_mask = False
        request.require_pose = False
        future = client.call_async(request)
        spin_until(future.done, timeout, "相机采集响应超时")
        response = future.result()
        report["capture"] = {"scene_id": scene, "observation_id": response.observation_id, "success": bool(response.success), "code": int(response.code), "reason": response.reason}
        if not response.success:
            raise RuntimeError("相机采集失败: " + response.reason)
        spin_until(lambda: response.observation_id in observations, timeout, "未收到本次新图像的 YOLO 结果，请检查 mask 进程")
        observation = observations[response.observation_id]
        if not observation_matches(observation, scene, response.observation_id, not_before):
            raise RuntimeError("YOLO 图像不是本次新采集，拒绝使用缓存")
        report["camera"] = save_images(observation, output)
        report["passed"] = report["camera"]["passed"]
        if not camera_only:
            after = time.monotonic()
            spin_until(lambda: state.get("joint_received", 0) >= after and state.get("diagnostic_received", 0) >= after, min(timeout, 6.0), "未收到新的机械臂关节/控制器状态；失能时可能没有反馈，请先完成抱持步骤")
            joints = state["joints"]
            report["robot"] = analyze_joints(list(joints.name), list(joints.position))
            report["robot"]["kinematics"] = offline_kinematics(report["robot"]["positions_rad"])
            diagnostics = state["diagnostics"]
            entries = [item for item in diagnostics.status if item.hardware_id == "nero"]
            if not entries:
                raise RuntimeError("控制器诊断缺少 NERO 状态")
            status = entries[-1]
            values = {entry.key: entry.value for entry in status.values}
            level = wire_uint8(status.level, "diagnostic level")
            report["controller"] = {"level": level, "message": status.message, "values": values}
            closed = values.get("execution_enabled", "").lower() == "false"
            fresh = values.get("feedback_fresh", "").lower() == "true"
            report["controller"]["passed"] = closed and fresh and values.get("mode") == "real" and level < 2
            report["passed"] = bool(report["passed"] and report["robot"]["passed"] and report["controller"]["passed"])
        report["reason"] = "初始观测通过，可生成新的本轮预览" if report["passed"] else "初始条件未通过；查看 camera/robot/controller 中的具体原因"
    except Exception as error:
        report["passed"] = False
        report["reason"] = str(error)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output-directory", required=True, type=Path)
    parser.add_argument("--camera-only", action="store_true", help="Camera/YOLO only; works with disabled robot and no CAN")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = run_check(output, args.camera_only, args.timeout)
    except Exception as error:
        report = {"schema": "strawberry_demo_initialization/v1", "passed": False, "reason": str(error), "camera_only": args.camera_only}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (output / "report.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
