#!/usr/bin/env python3
"""Run a large, guarded NERO visitor demonstration."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional, Sequence

import numpy as np
from action_msgs.msg import GoalStatus
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args

from strawberry_nero_control.models import (
    NERO_JOINT_NAMES,
    READY_JOINT_POSITIONS,
)
from strawberry_nero_control.pose_demo import PoseDemoNode
from strawberry_nero_control.real_smoke_test import (
    NeroRealSmokeTest,
    POST_MOTION_SAFETY_NOTICE,
    SmokeTestError,
    nero_urdf_path,
    recovery_result_allows_placo,
)
from strawberry_nero_control.ros_utils import matrix_to_pose_stamped
from strawberry_nero_interfaces.action import MoveToPose
from strawberry_nero_interfaces.msg import IKResult as IKResultMsg

from demo_profiles import (
    DemoPlan,
    MAX_CONDITION_NUMBER,
    MAX_LEG_JOINT_DELTA_RAD,
    MAX_POSTURE_ERROR_RAD,
    MAX_PREVIEW_ORIENTATION_ERROR_RAD,
    MAX_PREVIEW_POSITION_ERROR_M,
    MIN_JOINT_LIMIT_MARGIN_RAD,
    MIN_SIGMA,
    PlannedTarget,
    joint_limit_margin,
    make_profile_solver,
    plan_profile,
)


MAX_CYCLES = 10
MAX_PAUSE_S = 10.0
READY_START_TOLERANCE_RAD = 0.05
MAX_FINAL_POSITION_ERROR_M = 0.020
MAX_FINAL_ORIENTATION_ERROR_RAD = math.radians(5.0)
MAX_FINAL_JOINT_ERROR_RAD = 0.050


class ExhibitionDemoNode(PoseDemoNode):
    """PoseDemo safety clients under a distinct exhibition node name."""

    def __init__(self) -> None:
        NeroRealSmokeTest.__init__(self, "nero_exhibition_demo")


def _cycles(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_CYCLES:
        raise argparse.ArgumentTypeError(
            f"cycles 必须在 1 到 {MAX_CYCLES} 之间"
        )
    return parsed


def _pause_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= MAX_PAUSE_S:
        raise argparse.ArgumentTypeError(
            f"pause 必须在 0 到 {MAX_PAUSE_S:.0f} 秒之间"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NERO 参观展示：poses 依次到达大幅固定位姿；trajectory "
            "沿多个安全路点跟踪分段五次平滑大弧线。"
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("poses", "trajectory"),
        default="poses",
        help="展示模式，默认 poses",
    )
    parser.add_argument(
        "--cycles",
        type=_cycles,
        default=1,
        help=f"重复轮数，1-{MAX_CYCLES}，默认 1",
    )
    parser.add_argument(
        "--pause",
        type=_pause_seconds,
        default=None,
        help="展示姿态停留秒数；poses 默认 1，trajectory 默认 0",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="只做完整离线 IK/轨迹检查，不连接 ROS，不运动",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过真机 START 确认；仅用于已经完成现场核查的受监督展示",
    )
    return parser


def _print_plan(plan: DemoPlan, cycles: int) -> None:
    print("\n=== 展示路线离线检查通过 ===")
    print(f"模式：{plan.mode}")
    print(f"每轮路点：{len(plan.targets)}，轮数：{cycles}")
    print(
        "link7 目标位置跨度 XYZ："
        f"{plan.xyz_span_m[0] * 1000.0:.0f} / "
        f"{plan.xyz_span_m[1] * 1000.0:.0f} / "
        f"{plan.xyz_span_m[2] * 1000.0:.0f} mm"
    )
    print(
        "主要展示弧线 link7.x："
        f"{plan.main_x_range_m[0]:.3f} ～ "
        f"{plan.main_x_range_m[1]:.3f} m（全部位于 -X）"
    )
    print(
        "每轮预计纯运动时间："
        f"{plan.predicted_motion_duration_s:.1f} s"
    )
    print("逐段预测：")
    for index, leg in enumerate(plan.targets, start=1):
        hold_text = "，展示停留" if leg.target.hold_at_target else ""
        print(
            f"  {index:02d}. {leg.target.label}: "
            f"Δq={leg.max_joint_delta_rad:.3f} rad, "
            f"T={leg.trajectory_duration_s:.2f} s, "
            f"sigma={leg.sigma_min:.3f}, "
            f"limit_margin={leg.joint_limit_margin_rad:.3f} rad"
            f"{hold_text}"
        )


def _confirm_real_show(plan: DemoPlan) -> None:
    if not sys.stdin.isatty():
        raise SmokeTestError(
            "真机展示确认需要交互式终端；自动化运行请在完成现场核查后显式添加 --yes"
        )
    print("\n=== 真机展示前确认 ===")
    print("1. 底座已可靠固定；相机和线缆不会被拉扯。")
    print("2. 底座周围半径至少 0.6 m、高度 0-0.7 m 已清空。")
    print("3. 观众距机械臂至少 1 m，机械臂下方无人。")
    print("4. 一名观察员在工作区外守住控制箱，且未启动其它控制发布者。")
    print(
        f"本次将执行 {len(plan.targets)} 个路点/轮，"
        "开始前会自动 recover 并 center-ready。"
    )
    answer = input("全部确认后输入 START：").strip()
    if answer != "START":
        raise SmokeTestError("确认词不匹配；没有发送展示运动请求")


def _prepare_real_arm(node: ExhibitionDemoNode) -> None:
    """Recover if needed, then use the existing ten-step ready procedure."""
    print("\n=== 自动准备：安全区恢复 -> ready ===")
    recovery = node.preview_recovery()
    if recovery_result_allows_placo(recovery):
        print("机械臂已经位于 Placo 安全范围。")
    else:
        node.execute_recovery()

    center_plan = node.preview_center_ready()
    if center_plan.target_poses:
        node.execute_center_ready(center_plan)
    else:
        print("机械臂已经位于 ready 邻域。")

    state = node.measured_state()
    ready_error = float(np.max(np.abs(
        state.positions - np.asarray(READY_JOINT_POSITIONS, dtype=float)
    )))
    print(f"ready 实测最大关节误差：{ready_error:.5f} rad")
    if ready_error > READY_START_TOLERANCE_RAD:
        raise SmokeTestError(
            f"ready 实测误差 {ready_error:.4f} rad 超过 "
            f"{READY_START_TOLERANCE_RAD:.3f} rad"
        )


def _require_sim_ready(node: ExhibitionDemoNode) -> None:
    """Keep simulation rehearsals deterministic and anchored at ready."""
    state = node.measured_state()
    error = float(np.max(np.abs(
        state.positions - np.asarray(READY_JOINT_POSITIONS, dtype=float)
    )))
    if error > READY_START_TOLERANCE_RAD:
        raise SmokeTestError(
            "仿真当前不在 ready；请重启 start_control.sh sim 后再运行"
        )


def _validate_live_ik(
    label: str,
    result: IKResultMsg,
    expected_posture: Sequence[float],
    safe_limits: np.ndarray,
) -> None:
    """Apply exhibition gates to the live SolveIK/Action result."""
    if not result.success or result.code != IKResultMsg.SUCCESS:
        raise SmokeTestError(
            f"{label} IK 被拒绝：code={result.code}, {result.reason}"
        )
    diagnostics = np.asarray(
        (
            result.position_error_m,
            result.orientation_error_rad,
            result.solve_time_ms,
            result.sigma_min,
            result.condition_number,
            result.max_joint_delta_rad,
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
        raise SmokeTestError(f"{label} IK 诊断无效")
    if tuple(result.solution_joint_state.name) != tuple(NERO_JOINT_NAMES):
        raise SmokeTestError(f"{label} IK 关节名称或顺序无效")
    solution = np.asarray(result.solution_joint_state.position, dtype=float)
    expected = np.asarray(expected_posture, dtype=float)
    if solution.shape != (7,) or not np.all(np.isfinite(solution)):
        raise SmokeTestError(f"{label} IK 没有返回完整有限的 7 关节解")

    checks = (
        (
            result.max_joint_delta_rad <= MAX_LEG_JOINT_DELTA_RAD,
            f"单段关节变化 {result.max_joint_delta_rad:.4f} rad 过大",
        ),
        (
            result.position_error_m <= MAX_PREVIEW_POSITION_ERROR_M,
            f"位置残差 {result.position_error_m * 1000.0:.2f} mm 过大",
        ),
        (
            result.orientation_error_rad
            <= MAX_PREVIEW_ORIENTATION_ERROR_RAD,
            "姿态残差超过 2°",
        ),
        (
            result.sigma_min >= MIN_SIGMA,
            f"sigma_min={result.sigma_min:.4f} 过低",
        ),
        (
            result.condition_number <= MAX_CONDITION_NUMBER,
            f"condition={result.condition_number:.2f} 过高",
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            raise SmokeTestError(f"{label} 展示附加门拒绝：{reason}")

    posture_error = float(np.max(np.abs(solution - expected)))
    if posture_error > MAX_POSTURE_ERROR_RAD:
        raise SmokeTestError(
            f"{label} 与预设姿态相差 {posture_error:.4f} rad"
        )
    margin = joint_limit_margin(solution, safe_limits)
    if margin < MIN_JOINT_LIMIT_MARGIN_RAD:
        raise SmokeTestError(
            f"{label} 关节限位余量 {margin:.4f} rad 不足"
        )


def _print_live_preview(label: str, result: IKResultMsg) -> None:
    print(
        f"[{label}] 现场预览通过："
        f"Δq={result.max_joint_delta_rad:.3f} rad, "
        f"残差={result.position_error_m * 1000.0:.2f} mm / "
        f"{math.degrees(result.orientation_error_rad):.2f}°, "
        f"sigma={result.sigma_min:.3f}, "
        f"condition={result.condition_number:.1f}"
    )


def _execute_simulated(
    node: ExhibitionDemoNode,
    target_pose,
    expected_posture: Sequence[float],
    label: str,
):
    """Execute one simulation goal with the same posture reference as real."""
    goal = MoveToPose.Goal()
    goal.target_pose = target_pose
    goal.controlled_frame = "link7"
    goal.posture_reference.name = list(NERO_JOINT_NAMES)
    goal.posture_reference.position = [
        float(value) for value in expected_posture
    ]
    goal.timeout.sec = 15
    wrapped = node._send_action_goal(
        node._move_client,
        goal,
        label,
        20.0,
    )
    if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
        detail = wrapped.result.ik_result
        raise SmokeTestError(
            f"{label} 仿真 Action 状态异常：status={wrapped.status}, "
            f"code={detail.code}, {detail.reason}"
        )
    return wrapped.result


def _validate_execution(
    node: ExhibitionDemoNode,
    label: str,
    result,
    expected_posture: Sequence[float],
) -> None:
    """Require a strong Action result and a fresh measured target match."""
    _validate_live_ik(
        f"{label} Action",
        result.ik_result,
        expected_posture,
        node._solver.safe_joint_limits,
    )
    final_errors = np.asarray(
        (result.final_position_error_m, result.final_orientation_error_rad),
        dtype=float,
    )
    if not np.all(np.isfinite(final_errors)) or np.any(final_errors < 0.0):
        raise SmokeTestError(f"{label} 最终实测误差无效")
    if result.final_position_error_m > MAX_FINAL_POSITION_ERROR_M:
        raise SmokeTestError(
            f"{label} 最终位置误差 "
            f"{result.final_position_error_m * 1000.0:.2f} mm 过大"
        )
    if result.final_orientation_error_rad > MAX_FINAL_ORIENTATION_ERROR_RAD:
        raise SmokeTestError(f"{label} 最终姿态误差超过 5°")

    measured = node.measured_state()
    joint_error = float(np.max(np.abs(
        measured.positions - np.asarray(expected_posture, dtype=float)
    )))
    print(
        f"[{label}] 到位：实测关节误差={joint_error:.4f} rad，"
        f"末端误差={result.final_position_error_m * 1000.0:.2f} mm / "
        f"{math.degrees(result.final_orientation_error_rad):.2f}°"
    )
    if joint_error > MAX_FINAL_JOINT_ERROR_RAD:
        raise SmokeTestError(
            f"{label} 实测关节误差 {joint_error:.4f} rad 过大"
        )


def _interruptible_pause(node: ExhibitionDemoNode, duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    while rclpy.ok() and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        rclpy.spin_once(node, timeout_sec=min(0.10, max(0.0, remaining)))


def _execute_leg(
    node: ExhibitionDemoNode,
    planned: PlannedTarget,
    simulation: bool,
    cycle: int,
    index: int,
    count: int,
) -> None:
    label = f"第 {cycle} 轮 {index}/{count}：{planned.target.label}"
    state, _ = node.current_pose()
    target_pose = matrix_to_pose_stamped(
        planned.transform,
        "base_link",
        node.get_clock().now().to_msg(),
    )
    preview = node._solve_pose_preview(
        target_pose,
        f"{label} 现场 SolveIK",
        posture_reference=planned.target.joints,
    )
    _validate_live_ik(
        label,
        preview,
        planned.target.joints,
        node._solver.safe_joint_limits,
    )
    _print_live_preview(label, preview)

    if simulation:
        result = _execute_simulated(
            node,
            target_pose,
            planned.target.joints,
            label,
        )
    else:
        result = node.execute_placo(
            target_pose,
            state,
            precision_test=False,
            label=label,
            posture_reference=planned.target.joints,
        )
    _validate_execution(
        node,
        label,
        result,
        planned.target.joints,
    )


def run_show(
    node: ExhibitionDemoNode,
    plan: DemoPlan,
    cycles: int,
    pause_s: float,
    assume_yes: bool,
) -> None:
    """Prepare once, then execute every target from fresh measured feedback."""
    simulation = node.controller_is_simulation()
    if simulation:
        print("\n检测到 sim 模式：只驱动 MeshCat 虚拟模型，不访问 CAN。")
        _require_sim_ready(node)
    else:
        if not assume_yes:
            _confirm_real_show(plan)
        _prepare_real_arm(node)

    for cycle in range(1, cycles + 1):
        print(f"\n========== 开始第 {cycle}/{cycles} 轮 ==========")
        for index, target in enumerate(plan.targets, start=1):
            _execute_leg(
                node,
                target,
                simulation,
                cycle,
                index,
                len(plan.targets),
            )
            should_pause = (
                plan.mode == "trajectory"
                or target.target.hold_at_target
            )
            if (
                pause_s > 0.0
                and should_pause
                and index < len(plan.targets)
            ):
                _interruptible_pause(node, pause_s)
        print(f"========== 第 {cycle}/{cycles} 轮完成并回到 ready ==========")


def main(args: Optional[Sequence[str]] = None) -> int:
    """Plan first, then run with deterministic cancellation and cleanup."""
    raw_args = list(sys.argv if args is None else [sys.argv[0], *args])
    cli_args = remove_ros_args(args=raw_args)[1:]
    arguments = _parser().parse_args(cli_args)
    pause_s = (
        (1.0 if arguments.mode == "poses" else 0.0)
        if arguments.pause is None
        else arguments.pause
    )

    try:
        solver = make_profile_solver(nero_urdf_path())
        plan = plan_profile(arguments.mode, solver)
        _print_plan(plan, arguments.cycles)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n离线规划失败：{error}", file=sys.stderr)
        return 2

    if arguments.preview_only:
        print("\n仅完成离线检查：未连接 ROS，未发送任何运动命令。")
        return 0

    rclpy.init(
        args=raw_args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    try:
        node = ExhibitionDemoNode()
        run_show(
            node,
            plan,
            arguments.cycles,
            pause_s,
            arguments.yes,
        )
        print("\n展示成功完成，机械臂已回到 ready，并保持使能。")
        print(POST_MOTION_SAFETY_NOTICE)
        return 0
    except (SmokeTestError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n展示已停止或被拒绝：{error}", file=sys.stderr)
        print(
            "没有自动跳到下一路点，也没有自动失能；请观察机械臂保持状态。",
            file=sys.stderr,
        )
        return 2
    except (KeyboardInterrupt, EOFError):
        print(
            "\n展示已取消；正在取消当前 Action。不要在机械臂悬空时直接失能。",
            file=sys.stderr,
        )
        return 130
    finally:
        if node is not None:
            node.cancel_active_goal()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
