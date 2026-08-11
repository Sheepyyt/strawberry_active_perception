#!/usr/bin/env python3
"""Small, isolated Placo-versus-NERO-IK comparison experiment.

The Placo half is read from the completed week-one dataset.  The NERO half
sends the exact same absolute link7 poses through the vendor ``move_p`` API and
records encoder/model feedback.  Nothing in the production controller is
modified or imported for motion control; its URDF FK is used only as a common
measurement ruler.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
WEEK1_DIR = THIS_DIR.parent
PROJECT_DIR = WEEK1_DIR.parent.parent
PLACO_CSV = WEEK1_DIR / "real" / "real_30x3.csv"
PLACO_JSON = WEEK1_DIR / "real" / "real_30x3.json"
DEFAULT_NATIVE_CSV = THIS_DIR / "native_results.csv"
COMPARISON_CSV = THIS_DIR / "comparison_results.csv"
SUMMARY_JSON = THIS_DIR / "comparison_summary.json"
URDF_PATH = (
    PROJECT_DIR
    / "nero_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf"
)

# Five progressively different targets from the already completed 30-target
# Placo experiment.  Their requested changes span 10.3--42.0 mm and
# 3.4--12.5 degrees without inventing a second target dataset.
DEFAULT_TARGET_IDS = ("P28", "P20", "P01", "P21", "P03")
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
NATIVE_CONTINUITY_LIMIT_RAD = 0.20
# Recovery reverses from a state already reached under the continuity guard.
# The small extra margin covers controller stopping distance without relaxing
# the actual IK comparison criterion above.
ANCHOR_RESET_LIMIT_RAD = 0.25
NATIVE_FIELDS = (
    "solver",
    "target_id",
    "repetition",
    "phase",
    "success",
    "reason",
    "commanded_position_offset_mm",
    "commanded_orientation_offset_deg",
    "ik_solve_time_ms",
    "command_to_motion_ms",
    "total_duration_s",
    "model_position_error_mm",
    "model_orientation_error_deg",
    "sdk_position_error_mm",
    "sdk_orientation_error_deg",
    "max_joint_delta_rad",
    "return_anchor_joint_error_rad",
    "final_joint_positions_rad",
    "same_start_placo_joint_positions_rad",
    "max_joint_solution_difference_rad",
    "same_start_placo_solve_time_ms",
    "reset_move_j_used",
    "reset_move_j_success",
    "reset_move_j_reason",
    "reset_move_j_duration_s",
    "recorded_at",
)


class ExperimentError(RuntimeError):
    """A checked refusal or runtime failure with a user-readable reason."""


@dataclass(frozen=True)
class Dataset:
    anchor_q: np.ndarray
    anchor_T: np.ndarray
    targets: dict[str, dict[str, Any]]


def _load_dataset() -> Dataset:
    if not PLACO_CSV.is_file() or not PLACO_JSON.is_file():
        raise ExperimentError("缺少 week1/real 下的 Placo 原始结果")
    raw = json.loads(PLACO_JSON.read_text(encoding="utf-8"))
    targets = {item["target_id"]: item for item in raw["target_definitions"]}
    return Dataset(
        anchor_q=np.asarray(raw["anchor_joints_rad"], dtype=float),
        anchor_T=np.asarray(raw["anchor_transform"], dtype=float),
        targets=targets,
    )


def _selected_targets(dataset: Dataset, target_ids: Iterable[str]) -> list[dict[str, Any]]:
    selected = []
    for target_id in target_ids:
        if target_id not in dataset.targets:
            raise ExperimentError(f"目标 {target_id} 不在现有 30 目标数据集中")
        selected.append(dataset.targets[target_id])
    return selected


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite), percentile))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() in ("true", "1", "yes")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    from scipy.spatial.transform import Rotation

    position = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    relative = target[:3, :3] @ actual[:3, :3].T
    angle = float(Rotation.from_matrix(relative).magnitude())
    return position, angle


def _pose_to_matrix(pose: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quaternion = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=float,
    )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise ExperimentError("收到零长度四元数")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def _print_targets(dataset: Dataset, targets: list[dict[str, Any]]) -> None:
    print("\n=== 共用目标（来自现有 Placo 30 目标实验）===")
    print("目标  位置变化(mm)  姿态变化(°)  参考最大Δq(rad)")
    for target in targets:
        reference_q = np.asarray(target["reference_joints_rad"], dtype=float)
        print(
            f"{target['target_id']:>3}  "
            f"{1000.0 * float(target['position_offset_m']):12.1f}  "
            f"{math.degrees(float(target['orientation_offset_rad'])):11.1f}  "
            f"{np.max(np.abs(reference_q - dataset.anchor_q)):15.6f}"
        )


def _read_placo_rows(target_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(target_ids)
    with PLACO_CSV.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["target_id"] in wanted]


def preview(target_ids: Iterable[str]) -> None:
    dataset = _load_dataset()
    targets = _selected_targets(dataset, target_ids)
    rows = _read_placo_rows(target_ids)
    _print_targets(dataset, targets)
    print("\n=== 已有 Placo 子集结果（5 目标 × 3 次）===")
    print(f"去程成功：{sum(_bool(r['target_success']) for r in rows)}/{len(rows)}")
    print(f"回程成功：{sum(_bool(r['return_success']) for r in rows)}/{len(rows)}")
    print(
        "去程位置误差 p95："
        f"{1000.0 * (_percentile((float(r['target_position_error_m']) for r in rows), 95) or 0.0):.3f} mm"
    )
    print(
        "去程姿态误差 p95："
        f"{math.degrees(_percentile((float(r['target_orientation_error_rad']) for r in rows), 95) or 0.0):.4f}°"
    )
    print(
        "Placo IK 时间 p50 / p95："
        f"{_percentile((float(r['target_solve_time_ms']) for r in rows), 50):.3f} / "
        f"{_percentile((float(r['target_solve_time_ms']) for r in rows), 95):.3f} ms"
    )
    print("\n本命令只读取旧数据，没有连接或控制机械臂。")
    print("NERO 固件 1.11 不提供原生纯 IK 计时/关节解读取；原生侧将测量整条 move_p 链路。")


class NativeRunner:
    """ROS 2 data collection around NERO's vendor ``move_p`` interface."""

    def __init__(self, dataset: Dataset) -> None:
        try:
            import rclpy
            from agx_arm_msgs.msg import AgxArmStatus
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import JointState
            from std_srvs.srv import SetBool
        except ImportError as exc:
            raise ExperimentError(
                "找不到 ROS 2/工作空间模块；请先 source ROS、虚拟环境和 install/setup.bash"
            ) from exc

        try:
            from strawberry_nero_control.ik_core import PlacoIKSolver
        except ImportError as exc:
            raise ExperimentError("找不到 strawberry_nero_control；请先构建并 source 工作空间") from exc

        self.rclpy = rclpy
        self.PoseStamped = PoseStamped
        self.JointState = JointState
        self.SetBool = SetBool
        rclpy.init(args=None)
        self.node = rclpy.create_node("nero_native_ik_comparison")
        self.dataset = dataset
        self.solver = PlacoIKSolver(URDF_PATH)
        self.safe_limits = self.solver.safe_joint_limits
        self.q: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.tcp_T: np.ndarray | None = None
        self.arm_status: Any | None = None
        self.q_time = 0.0
        self.tcp_time = 0.0
        self.status_time = 0.0

        self.move_p_pub = self.node.create_publisher(PoseStamped, "/control/move_p", 1)
        self.hold_pub = self.node.create_publisher(JointState, "/control/move_j", 1)
        self.gate_client = self.node.create_client(SetBool, "/control_enable")
        self.node.create_subscription(
            JointState, "/feedback/joint_states", self._joint_callback, 10
        )
        self.node.create_subscription(
            PoseStamped, "/feedback/tcp_pose", self._tcp_callback, 10
        )
        self.node.create_subscription(
            AgxArmStatus, "/feedback/arm_status", self._status_callback, 10
        )

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()

    def _joint_callback(self, message: Any) -> None:
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in JOINT_NAMES):
            return
        q = np.asarray([positions[name] for name in JOINT_NAMES], dtype=float)
        velocities = dict(zip(message.name, message.velocity))
        velocity = np.asarray([velocities.get(name, 0.0) for name in JOINT_NAMES])
        if np.all(np.isfinite(q)):
            self.q = q
            self.velocity = velocity if np.all(np.isfinite(velocity)) else np.zeros(7)
            self.q_time = time.monotonic()

    def _tcp_callback(self, message: Any) -> None:
        try:
            self.tcp_T = _pose_to_matrix(message.pose)
            self.tcp_time = time.monotonic()
        except ExperimentError:
            return

    def _status_callback(self, message: Any) -> None:
        self.arm_status = message
        self.status_time = time.monotonic()

    def _spin(self, timeout: float = 0.01) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout)

    def wait_feedback(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._spin(0.05)
            now = time.monotonic()
            if (
                self.q is not None
                and self.tcp_T is not None
                and self.arm_status is not None
                and now - self.q_time < 0.25
                and now - self.tcp_time < 0.25
                and now - self.status_time < 0.25
            ):
                return
        raise ExperimentError("5 秒内没有收到完整且新鲜的关节、TCP 和状态反馈")

    def _assert_fresh_and_healthy(self, *, allow_no_solution: bool = False) -> None:
        now = time.monotonic()
        if self.q is None or now - self.q_time > 0.25:
            raise ExperimentError("7 关节反馈缺失或超过 0.25 秒未更新")
        if self.tcp_T is None or now - self.tcp_time > 0.25:
            raise ExperimentError("TCP 位姿反馈缺失或超过 0.25 秒未更新")
        if self.arm_status is None or now - self.status_time > 0.25:
            raise ExperimentError("机械臂状态反馈缺失或超过 0.25 秒未更新")
        status = self.arm_status
        faults = []
        if int(status.ctrl_mode) != 1:
            faults.append(f"ctrl_mode={status.ctrl_mode}")
        # arm_status=2 means that the vendor Cartesian IK reported "no
        # solution".  It is a valid experimental outcome rather than a
        # physical fault, and firmware 1.11 can leave it latched after a
        # rejected move_p.  All other non-zero states remain hard faults.
        allowed_arm_states = (0, 2) if allow_no_solution else (0,)
        if int(status.arm_status) not in allowed_arm_states:
            faults.append(f"arm_status={status.arm_status}")
        if int(status.err_status) != 0:
            faults.append(f"err_status={status.err_status}")
        if any(status.joint_angle_limit):
            faults.append("关节限位报警")
        if any(status.communication_status_joint):
            faults.append("关节通信报警")
        if faults:
            raise ExperimentError("驱动状态异常：" + "，".join(faults))
        if np.any(self.q < self.safe_limits[:, 0]) or np.any(self.q > self.safe_limits[:, 1]):
            raise ExperimentError("真实关节角不在 Placo/URDF 保守安全限位内")

    def preflight(self) -> None:
        self.wait_feedback()
        self._assert_fresh_and_healthy(allow_no_solution=True)
        if self.move_p_pub.get_subscription_count() < 1:
            raise ExperimentError("/control/move_p 没有订阅者；NERO 驱动可能未启动")
        assert self.q is not None and self.tcp_T is not None
        anchor_error = float(np.max(np.abs(self.q - self.dataset.anchor_q)))
        if anchor_error > 0.020:
            raise ExperimentError(
                f"当前姿态离实验锚点 {anchor_error:.6f} rad（上限 0.020）；请先运行 center-ready"
            )
        model_T = self.solver.forward_kinematics(self.q, "link7")
        position_error, orientation_error = _pose_error(model_T, self.tcp_T)
        if position_error > 0.010 or orientation_error > math.radians(3.0):
            raise ExperimentError(
                "URDF link7 与 NERO TCP 坐标未对齐："
                f"{1000.0 * position_error:.2f} mm / {math.degrees(orientation_error):.2f}°"
            )
        print("\n=== 真机前置检查 ===")
        print(f"当前到实验锚点最大关节差：{anchor_error:.6f} rad")
        print(
            "URDF link7 与 NERO TCP 差："
            f"{1000.0 * position_error:.3f} mm / {math.degrees(orientation_error):.4f}°"
        )
        if int(self.arm_status.arm_status) == 2:
            print("提示：arm_status=2 是上一次 NERO 原生 IK 的“无解”记录；其他状态均正常。")
        print("反馈、限位、驱动状态和坐标对齐检查均通过。")

    def set_gate(self, enabled: bool) -> None:
        if not self.gate_client.wait_for_service(timeout_sec=3.0):
            raise ExperimentError("/control_enable 服务不可用")
        request = self.SetBool.Request()
        request.data = enabled
        future = self.gate_client.call_async(request)
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            self._spin(0.05)
        if not future.done() or future.result() is None or not future.result().success:
            raise ExperimentError(f"无法{'打开' if enabled else '关闭'} NERO 软件控制门")

    def soft_hold(self) -> None:
        if self.q is None:
            return
        message = self.JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = self.q.tolist()
        for _ in range(3):
            self.hold_pub.publish(message)
            self._spin(0.03)

    def _pose_message(self, transform: np.ndarray) -> Any:
        from scipy.spatial.transform import Rotation

        message = self.PoseStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            float(v) for v in transform[:3, 3]
        )
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = (float(v) for v in quaternion)
        return message

    def execute_segment(
        self,
        *,
        target_id: str,
        repetition: int,
        phase: str,
        target_T: np.ndarray,
        position_offset_mm: float,
        orientation_offset_deg: float,
    ) -> dict[str, Any]:
        self.wait_feedback()
        self._assert_fresh_and_healthy(allow_no_solution=True)
        assert self.q is not None
        start_q = self.q.copy()
        placo_result = self.solver.solve(target_T, start_q, "link7")
        if not placo_result.success:
            raise ExperimentError(
                "同一起点 Placo 对照解未通过，禁止发送 NERO 命令："
                f"{placo_result.message}"
            )
        placo_q = np.asarray(placo_result.joint_positions, dtype=float)
        message = self._pose_message(target_T)
        sent = time.monotonic()
        self.move_p_pub.publish(message)

        motion_started_at: float | None = None
        motion_confirmed = False
        settled_since: float | None = None
        max_joint_delta = 0.0
        final_model_error = (math.inf, math.inf)
        final_sdk_error = (math.inf, math.inf)
        reason = "15 秒运动超时"
        success = False
        deadline = sent + 15.0
        try:
            while time.monotonic() < deadline:
                self._spin(0.01)
                self._assert_fresh_and_healthy(allow_no_solution=True)
                assert self.q is not None and self.tcp_T is not None
                now = time.monotonic()
                joint_delta = float(np.max(np.abs(self.q - start_q)))
                max_joint_delta = max(max_joint_delta, joint_delta)
                if joint_delta > NATIVE_CONTINUITY_LIMIT_RAD:
                    reason = (
                        f"NERO 关节解跨度 {joint_delta:.6f} rad 超过 "
                        f"{NATIVE_CONTINUITY_LIMIT_RAD:.2f} rad"
                    )
                    raise ExperimentError(reason)
                if motion_started_at is None and joint_delta > 0.00035:
                    motion_started_at = now
                # 0.00035 rad is deliberately sensitive enough for latency
                # measurement, but encoder noise can cross it while the arm
                # is stationary.  Require 0.002 rad before treating the
                # command as real motion or starting the settle logic.
                if joint_delta > 0.002:
                    motion_confirmed = True
                if (
                    not motion_confirmed
                    and int(self.arm_status.arm_status) == 2
                    and now - sent > 0.75
                ):
                    reason = "NERO 原生 IK 报告无解（arm_status=2）"
                    motion_started_at = None
                    break
                if not motion_confirmed and now - sent > 2.0:
                    reason = "命令发送 2 秒后编码器仍未开始变化"
                    motion_started_at = None
                    break

                model_T = self.solver.forward_kinematics(self.q, "link7")
                final_model_error = _pose_error(target_T, model_T)
                final_sdk_error = _pose_error(target_T, self.tcp_T)
                velocity = self.velocity if self.velocity is not None else np.full(7, math.inf)
                is_still = float(np.max(np.abs(velocity))) < 0.02
                is_accurate = (
                    final_model_error[0] <= 0.010
                    and final_model_error[1] <= math.radians(5.0)
                )
                if motion_confirmed and is_still and is_accurate:
                    settled_since = settled_since or now
                    if now - settled_since >= 0.35:
                        success = True
                        reason = "目标已稳定到达"
                        break
                else:
                    settled_since = None
        except ExperimentError as exc:
            reason = str(exc)
            self.soft_hold()

        finished = time.monotonic()
        if not success:
            self.soft_hold()
        return {
            "solver": "nero_native_v111",
            "target_id": target_id,
            "repetition": repetition,
            "phase": phase,
            "success": success,
            "reason": reason,
            "commanded_position_offset_mm": position_offset_mm,
            "commanded_orientation_offset_deg": orientation_offset_deg,
            # Firmware 1.11 has no supported get_ik_joint_angles/timing API.
            "ik_solve_time_ms": "",
            "command_to_motion_ms": (
                "" if motion_started_at is None else 1000.0 * (motion_started_at - sent)
            ),
            "total_duration_s": finished - sent,
            "model_position_error_mm": 1000.0 * final_model_error[0],
            "model_orientation_error_deg": math.degrees(final_model_error[1]),
            "sdk_position_error_mm": 1000.0 * final_sdk_error[0],
            "sdk_orientation_error_deg": math.degrees(final_sdk_error[1]),
            "max_joint_delta_rad": max_joint_delta,
            "return_anchor_joint_error_rad": (
                float(np.max(np.abs(self.q - self.dataset.anchor_q)))
                if phase == "return" and self.q is not None
                else ""
            ),
            "final_joint_positions_rad": (
                json.dumps(self.q.tolist(), separators=(",", ":"))
                if self.q is not None
                else ""
            ),
            "same_start_placo_joint_positions_rad": json.dumps(
                placo_q.tolist(), separators=(",", ":")
            ),
            "max_joint_solution_difference_rad": (
                float(np.max(np.abs(self.q - placo_q))) if self.q is not None else ""
            ),
            "same_start_placo_solve_time_ms": placo_result.solve_time_ms,
            "reset_move_j_used": False,
            "reset_move_j_success": "",
            "reset_move_j_reason": "",
            "reset_move_j_duration_s": "",
            "recorded_at": _iso_now(),
        }

    def reset_to_anchor(self) -> dict[str, Any]:
        """Return to the known anchor with move_j; this is not an IK sample."""
        self.wait_feedback()
        self._assert_fresh_and_healthy(allow_no_solution=True)
        assert self.q is not None
        start_q = self.q.copy()
        target_q = self.dataset.anchor_q
        requested_delta = float(np.max(np.abs(target_q - start_q)))
        if requested_delta > ANCHOR_RESET_LIMIT_RAD:
            return {
                "success": False,
                "reason": (
                    f"锚点复位跨度 {requested_delta:.6f} rad 超过 "
                    f"{ANCHOR_RESET_LIMIT_RAD:.2f} rad"
                ),
                "duration_s": 0.0,
                "final_error_rad": requested_delta,
            }
        if requested_delta <= 0.003:
            return {
                "success": True,
                "reason": "已经位于锚点附近，无需发送命令",
                "duration_s": 0.0,
                "final_error_rad": requested_delta,
            }

        message = self.JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = target_q.tolist()
        sent = time.monotonic()
        self.hold_pub.publish(message)
        settled_since: float | None = None
        final_error = requested_delta
        reason = "锚点 move_j 复位超过 10 秒"
        success = False
        try:
            while time.monotonic() - sent < 10.0:
                self._spin(0.01)
                self._assert_fresh_and_healthy(allow_no_solution=True)
                assert self.q is not None
                if float(np.max(np.abs(self.q - start_q))) > ANCHOR_RESET_LIMIT_RAD:
                    raise ExperimentError(
                        f"锚点复位的真实关节跨度超过 {ANCHOR_RESET_LIMIT_RAD:.2f} rad"
                    )
                final_error = float(np.max(np.abs(self.q - target_q)))
                velocity = self.velocity if self.velocity is not None else np.full(7, math.inf)
                if final_error <= 0.003 and float(np.max(np.abs(velocity))) < 0.02:
                    settled_since = settled_since or time.monotonic()
                    if time.monotonic() - settled_since >= 0.35:
                        success = True
                        reason = "已用已知锚点关节角安全复位"
                        break
                else:
                    settled_since = None
        except ExperimentError as exc:
            reason = str(exc)
        if not success:
            self.soft_hold()
        return {
            "success": success,
            "reason": reason,
            "duration_s": time.monotonic() - sent,
            "final_error_rad": final_error,
        }


def _write_native_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=NATIVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_resume_rows(
    path: Path,
    target_ids: Iterable[str],
    repeats: int,
) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(NATIVE_FIELDS).issubset(set(reader.fieldnames or ())):
            raise ExperimentError("现有断点 CSV 的字段版本不兼容，不能安全续跑")
        rows = list(reader)
    allowed_targets = set(target_ids)
    phases_by_attempt: dict[tuple[str, int], set[str]] = {}
    seen_segments: set[tuple[str, int, str]] = set()
    for row in rows:
        target_id = row.get("target_id", "")
        repetition = int(row.get("repetition", 0))
        phase = row.get("phase", "")
        if target_id not in allowed_targets or not 1 <= repetition <= repeats:
            raise ExperimentError("断点 CSV 包含本次目标范围之外的数据")
        if phase not in ("outbound", "return"):
            raise ExperimentError("断点 CSV 包含未知阶段")
        segment = (target_id, repetition, phase)
        if segment in seen_segments:
            raise ExperimentError(f"断点 CSV 存在重复记录：{segment}")
        seen_segments.add(segment)
        phases_by_attempt.setdefault((target_id, repetition), set()).add(phase)
    # A safety-rejected outbound has no meaningful return command.  Add an
    # explicit skipped-return record so a later run can resume at the next
    # repetition without deleting the original rejection.
    for key, phases in list(phases_by_attempt.items()):
        if phases != {"outbound"}:
            continue
        outbound = next(
            row
            for row in rows
            if row["target_id"] == key[0]
            and int(row["repetition"]) == key[1]
            and row["phase"] == "outbound"
        )
        if not _bool(outbound["success"]):
            rows.append(_skipped_return(outbound))
            phases_by_attempt[key] = {"outbound", "return"}
    incomplete = [
        key
        for key, phases in phases_by_attempt.items()
        if phases != {"outbound", "return"}
    ]
    if incomplete:
        raise ExperimentError(f"断点停在一次往返的中间，不能自动续跑：{incomplete[0]}")
    return rows, set(phases_by_attempt)


def _record_reset(row: dict[str, Any], reset: dict[str, Any]) -> None:
    row["reset_move_j_used"] = True
    row["reset_move_j_success"] = bool(reset["success"])
    row["reset_move_j_reason"] = str(reset["reason"])
    row["reset_move_j_duration_s"] = float(reset["duration_s"])


def _is_native_no_solution(row: dict[str, Any]) -> bool:
    return "arm_status=2" in str(row["reason"])


def _is_recoverable_experimental_rejection(row: dict[str, Any]) -> bool:
    return _is_native_no_solution(row) or "NERO 关节解跨度" in str(row["reason"])


def _skipped_return(outbound: dict[str, Any]) -> dict[str, Any]:
    row = dict(outbound)
    row.update(
        {
            "phase": "return",
            "success": False,
            "reason": f"去程未成功（{outbound['reason']}），因此没有执行回程",
            "command_to_motion_ms": "",
            "total_duration_s": 0.0,
            "model_position_error_mm": "",
            "model_orientation_error_deg": "",
            "sdk_position_error_mm": "",
            "sdk_orientation_error_deg": "",
            "max_joint_delta_rad": 0.0,
            "return_anchor_joint_error_rad": 0.0,
            "final_joint_positions_rad": "",
            "max_joint_solution_difference_rad": "",
            "reset_move_j_used": False,
            "reset_move_j_success": "",
            "reset_move_j_reason": "",
            "reset_move_j_duration_s": "",
            "recorded_at": _iso_now(),
        }
    )
    return row


def run_native(args: argparse.Namespace) -> None:
    dataset = _load_dataset()
    targets = _selected_targets(dataset, args.targets)
    _print_targets(dataset, targets)
    runner = NativeRunner(dataset)
    rows: list[dict[str, Any]] = []
    gate_open = False
    try:
        runner.preflight()
        if not args.execute:
            print("\n预览结束：没有打开软件控制门，也没有发送 move_p。")
            return
        if args.resume and args.overwrite:
            raise ExperimentError("--resume 与 --overwrite 不能同时使用")
        completed_attempts: set[tuple[str, int]] = set()
        if args.output.exists():
            if args.resume:
                rows, completed_attempts = _load_resume_rows(
                    args.output, args.targets, args.repeats
                )
            elif not args.overwrite:
                raise ExperimentError(
                    f"结果文件已存在：{args.output}；续跑用 --resume，完整重做用 --overwrite"
                )
        elif args.resume:
            raise ExperimentError(f"找不到可续跑的结果文件：{args.output}")
        remaining_attempts = len(targets) * args.repeats - len(completed_attempts)
        print(
            "\n即将执行 NERO 原生 move_p："
            f"剩余 {remaining_attempts} 次往返（{2 * remaining_attempts} 段）。"
        )
        if completed_attempts:
            print(f"已从断点读取 {len(completed_attempts)} 次完整往返，将跳过这些记录。")
        print("过程中不会自动失能；异常时会发送当前关节保持并关闭软件控制门。")
        answer = input("确认空间清空、急停可触及；输入 NERO_COMPARE 继续：").strip()
        if answer != "NERO_COMPARE":
            raise ExperimentError("确认文字不匹配，未执行")
        runner.set_gate(True)
        gate_open = True
        for target in targets:
            target_id = target["target_id"]
            target_T = np.asarray(target["target_transform"], dtype=float)
            position_mm = 1000.0 * float(target["position_offset_m"])
            orientation_deg = math.degrees(float(target["orientation_offset_rad"]))
            for repetition in range(1, args.repeats + 1):
                if (target_id, repetition) in completed_attempts:
                    print(f"跳过已有断点：{target_id} 第 {repetition}/{args.repeats} 次")
                    continue
                print(f"\n=== NERO {target_id} 第 {repetition}/{args.repeats} 次：去程 ===")
                outbound = runner.execute_segment(
                    target_id=target_id,
                    repetition=repetition,
                    phase="outbound",
                    target_T=target_T,
                    position_offset_mm=position_mm,
                    orientation_offset_deg=orientation_deg,
                )
                rows.append(outbound)
                _print_native_result(outbound)
                if not outbound["success"]:
                    if not _is_recoverable_experimental_rejection(outbound):
                        _write_native_rows(args.output, rows)
                        raise ExperimentError(f"{target_id} 去程失败：{outbound['reason']}")
                    reset = runner.reset_to_anchor()
                    _record_reset(outbound, reset)
                    rows.append(_skipped_return(outbound))
                    _write_native_rows(args.output, rows)
                    print(
                        "去程拒绝已作为实验结果保存；"
                        f"锚点状态：{reset['reason']}"
                    )
                    if not reset["success"]:
                        raise ExperimentError(f"无解后的锚点复位失败：{reset['reason']}")
                    continue
                _write_native_rows(args.output, rows)

                print(f"\n=== NERO {target_id} 第 {repetition}/{args.repeats} 次：回程 ===")
                returned = runner.execute_segment(
                    target_id=target_id,
                    repetition=repetition,
                    phase="return",
                    target_T=dataset.anchor_T,
                    position_offset_mm=position_mm,
                    orientation_offset_deg=orientation_deg,
                )
                rows.append(returned)
                _print_native_result(returned)
                if not returned["success"]:
                    if not _is_recoverable_experimental_rejection(returned):
                        _write_native_rows(args.output, rows)
                        raise ExperimentError(f"{target_id} 回程失败：{returned['reason']}")
                    reset = runner.reset_to_anchor()
                    _record_reset(returned, reset)
                    print("NERO 回程拒绝已记录；改用已知锚点关节角复位（不计入 IK 对比）。")
                    print(
                        f"复位：{'成功' if reset['success'] else '失败'}，{reset['reason']}，"
                        f"最终关节误差={float(reset['final_error_rad']):.6f} rad"
                    )
                    if not reset["success"]:
                        _write_native_rows(args.output, rows)
                        raise ExperimentError(f"回程无解后的复位失败：{reset['reason']}")
                else:
                    return_error = float(returned["return_anchor_joint_error_rad"])
                    if return_error > 0.05:
                        reset = runner.reset_to_anchor()
                        _record_reset(returned, reset)
                        print(
                            f"回程到了相同末端位姿，但关节分支相差 {return_error:.6f} rad；"
                            "已用锚点关节角复位。"
                        )
                        if not reset["success"]:
                            _write_native_rows(args.output, rows)
                            raise ExperimentError(f"不同分支后的复位失败：{reset['reason']}")
                _write_native_rows(args.output, rows)
        print(f"\nNERO 原生实验完成：{args.output}")
    finally:
        if gate_open:
            try:
                runner.set_gate(False)
                print("NERO 软件控制门已关闭（电机仍保持使能，没有自动失能）。")
            except Exception as exc:  # Preserve the original result but warn loudly.
                print(f"警告：关闭软件控制门失败：{exc}", file=sys.stderr)
        runner.close()

    if len(rows) == len(targets) * args.repeats * 2:
        summarize(args.targets, args.output)


def reset_anchor(args: argparse.Namespace) -> None:
    """Preview or execute a monitored move_j reset after native IK no-solution."""
    dataset = _load_dataset()
    runner = NativeRunner(dataset)
    gate_open = False
    try:
        runner.wait_feedback()
        runner._assert_fresh_and_healthy(allow_no_solution=True)
        assert runner.q is not None
        delta = float(np.max(np.abs(runner.q - dataset.anchor_q)))
        print("\n=== 实验锚点复位预览 ===")
        print(f"当前 arm_status={int(runner.arm_status.arm_status)}（2 表示上次原生 IK 无解）")
        print(f"当前到锚点最大关节变化={delta:.6f} rad")
        if delta > ANCHOR_RESET_LIMIT_RAD:
            raise ExperimentError(
                f"锚点复位跨度超过 {ANCHOR_RESET_LIMIT_RAD:.2f} rad，拒绝执行"
            )
        if not args.execute:
            print("预览结束：没有打开软件控制门，没有发送 move_j。")
            return
        answer = input("确认空间清空、急停可触及；输入 RESET_ANCHOR 继续：").strip()
        if answer != "RESET_ANCHOR":
            raise ExperimentError("确认文字不匹配，未执行")
        runner.set_gate(True)
        gate_open = True
        result = runner.reset_to_anchor()
        print("\n=== 实验锚点复位结果 ===")
        print(f"结果：{'成功' if result['success'] else '失败'}，{result['reason']}")
        print(f"最终锚点关节误差={float(result['final_error_rad']):.6f} rad")
        if not result["success"]:
            raise ExperimentError(result["reason"])
    finally:
        if gate_open:
            try:
                runner.set_gate(False)
                print("NERO 软件控制门已关闭（电机仍保持使能，没有自动失能）。")
            except Exception as exc:
                print(f"警告：关闭软件控制门失败：{exc}", file=sys.stderr)
        runner.close()


def _print_native_result(row: dict[str, Any]) -> None:
    print(f"结果：{'成功' if row['success'] else '失败'}，{row['reason']}")
    print(
        "模型误差："
        f"{float(row['model_position_error_mm']):.3f} mm / "
        f"{float(row['model_orientation_error_deg']):.4f}°"
    )
    latency = row["command_to_motion_ms"]
    latency_text = "未检测到" if latency == "" else f"{float(latency):.1f} ms"
    print(
        f"命令到开始运动：{latency_text}，总时长：{float(row['total_duration_s']):.3f} s，"
        f"最大关节变化：{float(row['max_joint_delta_rad']):.6f} rad"
    )
    print(
        "NERO 最终关节解与同一起点 Placo 对照解最大差："
        f"{float(row['max_joint_solution_difference_rad']):.6f} rad"
    )


def _placo_comparison_rows(target_ids: Iterable[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in _read_placo_rows(target_ids):
        common = {
            "solver": "placo",
            "target_id": source["target_id"],
            "repetition": int(source["repetition"]),
            "commanded_position_offset_mm": 1000.0 * float(source["commanded_position_offset_m"]),
            "commanded_orientation_offset_deg": math.degrees(
                float(source["commanded_orientation_offset_rad"])
            ),
            "sdk_position_error_mm": "",
            "sdk_orientation_error_deg": "",
            "final_joint_positions_rad": "",
            "same_start_placo_joint_positions_rad": "",
            "max_joint_solution_difference_rad": "",
            "same_start_placo_solve_time_ms": "",
            "reset_move_j_used": False,
            "reset_move_j_success": "",
            "reset_move_j_reason": "",
            "reset_move_j_duration_s": "",
            "recorded_at": source["recorded_at"],
        }
        output.append(
            {
                **common,
                "phase": "outbound",
                "success": _bool(source["target_success"]),
                "reason": source["target_reason"],
                "ik_solve_time_ms": float(source["target_solve_time_ms"]),
                "command_to_motion_ms": 1000.0 * float(source["target_response_latency_s"]),
                "total_duration_s": float(source["target_total_duration_s"]),
                "model_position_error_mm": 1000.0 * float(source["target_position_error_m"]),
                "model_orientation_error_deg": math.degrees(
                    float(source["target_orientation_error_rad"])
                ),
                "max_joint_delta_rad": float(source["target_max_joint_delta_rad"]),
                "return_anchor_joint_error_rad": "",
            }
        )
        output.append(
            {
                **common,
                "phase": "return",
                "success": _bool(source["return_success"]),
                "reason": source["return_reason"],
                "ik_solve_time_ms": float(source["return_solve_time_ms"]),
                "command_to_motion_ms": 1000.0 * float(source["return_response_latency_s"]),
                "total_duration_s": float(source["return_total_duration_s"]),
                "model_position_error_mm": 1000.0 * float(source["return_position_error_m"]),
                "model_orientation_error_deg": math.degrees(
                    float(source["return_orientation_error_rad"])
                ),
                "max_joint_delta_rad": float(source["return_max_joint_delta_rad"]),
                "return_anchor_joint_error_rad": float(source["return_joint_error_rad"]),
            }
        )
    return output


def _metric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    data = []
    for row in rows:
        value = _float_or_none(row.get(field))
        if value is not None:
            data.append(value)
    return {
        "count": len(data),
        "p50": _percentile(data, 50),
        "p95": _percentile(data, 95),
        "max": max(data) if data else None,
    }


def _failure_category(row: dict[str, Any]) -> str:
    reason = str(row.get("reason", ""))
    if "arm_status=2" in reason:
        return "native_ik_no_solution"
    if "15 秒运动超时" in reason:
        return "no_meaningful_motion_timeout"
    if "去程未成功" in reason:
        return "return_skipped_after_outbound_rejection"
    if "NERO 关节解跨度" in reason:
        return "continuity_guard_rejection"
    return reason or "unknown"


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if _bool(row["success"])]
    failures = [row for row in rows if not _bool(row["success"])]
    success_metrics = (
        "ik_solve_time_ms",
        "command_to_motion_ms",
        "total_duration_s",
        "model_position_error_mm",
        "model_orientation_error_deg",
        "max_joint_delta_rad",
        "return_anchor_joint_error_rad",
        "max_joint_solution_difference_rad",
    )
    attempts = len(rows)
    return {
        "attempts": attempts,
        "successes": len(successful),
        "success_rate": len(successful) / attempts if attempts else None,
        "failure_categories": dict(Counter(_failure_category(row) for row in failures)),
        # Accuracy and timing are meaningful only for motions accepted as
        # successful.  Aborted P21 samples must not inflate final pose error.
        "successful_only": {
            field: _metric_summary(successful, field) for field in success_metrics
        },
        # These two fields describe branch choice even for a rejected motion.
        "all_attempts_branch_metrics": {
            "max_joint_delta_rad": _metric_summary(rows, "max_joint_delta_rad"),
            "max_joint_solution_difference_rad": _metric_summary(
                rows, "max_joint_solution_difference_rad"
            ),
        },
    }


def summarize(target_ids: Iterable[str], native_path: Path) -> None:
    if not native_path.is_file():
        raise ExperimentError(f"找不到 NERO 原生结果：{native_path}")
    placo_rows = _placo_comparison_rows(target_ids)
    with native_path.open(newline="", encoding="utf-8") as stream:
        native_rows = list(csv.DictReader(stream))
    expected = len(tuple(target_ids)) * 3 * 2
    if len(native_rows) != expected:
        raise ExperimentError(
            f"NERO 数据只有 {len(native_rows)} 段，完整默认实验应为 {expected} 段；"
            "保留了断点文件，但暂不生成最终对比"
        )
    combined = placo_rows + native_rows
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=NATIVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)

    native_return_rows = [
        row
        for row in native_rows
        if row["phase"] == "return"
    ]
    return_failure_categories = Counter(
        _failure_category(row)
        for row in native_return_rows
        if not _bool(row["success"])
    )
    summary: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": _iso_now(),
        "purpose": "Placo 与 NERO 原生 IK 的一次性性能对比；后续开发仍使用 Placo",
        "selected_target_ids": list(target_ids),
        "repetitions_per_target": 3,
        "segments_per_solver": expected,
        "measurement_note": (
            "两侧末端误差都用真实关节编码器与同一 URDF FK 计算。"
            "Placo 行是 Placo IK + 五次关节轨迹；NERO 行是原生 move_p（原生 IK + 原生轨迹）。"
        ),
        "native_ik_timing_note": (
            "NERO 固件 1.11 没有受支持的原生 IK 关节解/纯计算计时接口，"
            "因此 NERO 的 ik_solve_time_ms 为 null；不能把 move_p 总时长冒充 IK 时间。"
        ),
        "native_move_j_reset_note": (
            "NERO 原生 move_p 回程无解时，失败会保留；随后仅用已知锚点关节角 move_j 复位，"
            "该复位不属于 IK 样本，也不改变无解统计。reset_count 只统计 CSV 内自动记录的复位。"
        ),
        "native_move_j_reset_count": sum(
            _bool(row.get("reset_move_j_used")) for row in native_rows
        ),
        "native_return_breakdown": {
            "planned_returns": len(native_return_rows),
            "move_p_commands_sent": sum(
                _failure_category(row) != "return_skipped_after_outbound_rejection"
                for row in native_return_rows
            ),
            "successful_returns": sum(_bool(row["success"]) for row in native_return_rows),
            "failure_categories": dict(return_failure_categories),
        },
        "results": {},
        "artifacts": {
            "comparison_csv": COMPARISON_CSV.name,
            "native_csv": native_path.name,
            "placo_source_csv": str(PLACO_CSV.relative_to(WEEK1_DIR)),
        },
    }
    for solver in ("placo", "nero_native_v111"):
        solver_rows = [row for row in combined if row["solver"] == solver]
        summary["results"][solver] = {
            phase: _summarize_group([row for row in solver_rows if row["phase"] == phase])
            for phase in ("outbound", "return")
        }
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== 对比结果已生成 ===")
    print(f"逐段数据：{COMPARISON_CSV}")
    print(f"汇总数据：{SUMMARY_JSON}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="复用 week1 数据，对比 Placo 与 NERO 固件 1.11 原生 move_p"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview_parser = subparsers.add_parser("preview", help="只看选定目标和已有 Placo 数据")
    preview_parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_IDS))

    native_parser = subparsers.add_parser("native", help="NERO 原生真机预检/采集")
    native_parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_IDS))
    native_parser.add_argument("--execute", action="store_true", help="确认后实际发送 move_p")
    native_parser.add_argument("--output", type=Path, default=DEFAULT_NATIVE_CSV)
    native_parser.add_argument("--overwrite", action="store_true")
    native_parser.add_argument("--resume", action="store_true", help="从完整往返断点继续")
    native_parser.set_defaults(repeats=3)

    reset_parser = subparsers.add_parser(
        "reset-anchor", help="原生 IK 无解后，预览/执行已知关节锚点复位"
    )
    reset_parser.add_argument("--execute", action="store_true")

    report_parser = subparsers.add_parser("summarize", help="由两侧 CSV 生成最终对比")
    report_parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_IDS))
    report_parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE_CSV)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "preview":
            preview(args.targets)
        elif args.command == "native":
            run_native(args)
        elif args.command == "reset-anchor":
            reset_anchor(args)
        else:
            summarize(args.targets, args.native)
        return 0
    except (ExperimentError, KeyboardInterrupt) as exc:
        message = "用户取消" if isinstance(exc, KeyboardInterrupt) else str(exc)
        print(f"\n实验已停止或拒绝：{message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
