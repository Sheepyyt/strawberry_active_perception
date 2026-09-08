"""Preview and execute one operator-supplied NERO Cartesian pose."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Callable, Optional, Sequence

import numpy as np
from action_msgs.msg import GoalStatus
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation

from strawberry_nero_interfaces.action import MoveToPose
from strawberry_nero_interfaces.msg import IKResult as IKResultMsg

from .real_smoke_test import (
    MeasuredState,
    NeroRealSmokeTest,
    POST_MOTION_SAFETY_NOTICE,
    SmokeTestError,
    require_confirmation,
    validate_smoke_ik,
)
from .ros_utils import matrix_to_pose_stamped, pose_error


BASE_FRAME = "base_link"
CONTROLLED_FRAME = "link7"
DEMO_MAX_POSITION_DELTA_M = 0.080
DEMO_MAX_ORIENTATION_DELTA_RAD = math.radians(30.0)
DEMO_MAX_JOINT_DELTA_RAD = 0.120
DEMO_MIN_JOINT_LIMIT_MARGIN_RAD = 0.010


def rotation_from_rpy_degrees(values: Sequence[float]) -> np.ndarray:
    """Convert roll, pitch and yaw in degrees to a rotation matrix."""
    rpy = np.asarray(values, dtype=float)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError("RPY 必须是 3 个有限角度")
    return Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()


def absolute_target_transform(
    position_m: Sequence[float],
    *,
    quaternion_xyzw: Optional[Sequence[float]] = None,
    rpy_degrees: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Construct a base-frame target from an absolute position and orientation."""
    position = np.asarray(position_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("绝对位置必须是 3 个有限米制数值")
    if (quaternion_xyzw is None) == (rpy_degrees is None):
        raise ValueError("四元数和 RPY 必须且只能选择一种")

    if quaternion_xyzw is not None:
        quaternion = np.asarray(quaternion_xyzw, dtype=float)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("四元数必须按 x y z w 提供 4 个有限数值")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-9:
            raise ValueError("四元数不能全为零")
        rotation = Rotation.from_quat(quaternion / norm).as_matrix()
    else:
        assert rpy_degrees is not None
        rotation = rotation_from_rpy_degrees(rpy_degrees)

    target = np.eye(4, dtype=float)
    target[:3, :3] = rotation
    target[:3, 3] = position
    return target


def relative_target_transform(
    current_transform: np.ndarray,
    xyz_mm: Sequence[float],
    rpy_degrees: Sequence[float],
    reference_frame: str,
) -> np.ndarray:
    """Apply a relative translation and rotation in base or link7 coordinates."""
    current = np.asarray(current_transform, dtype=float)
    translation = np.asarray(xyz_mm, dtype=float)
    if current.shape != (4, 4) or not np.all(np.isfinite(current)):
        raise ValueError("当前位姿矩阵无效")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("相对位移必须是 3 个有限毫米数值")
    if reference_frame not in ("base", "tool"):
        raise ValueError("相对坐标系只能是 base 或 tool")

    delta_position = translation / 1000.0
    delta_rotation = rotation_from_rpy_degrees(rpy_degrees)
    target = current.copy()
    if reference_frame == "base":
        target[:3, 3] += delta_position
        target[:3, :3] = delta_rotation @ current[:3, :3]
    else:
        target[:3, 3] += current[:3, :3] @ delta_position
        target[:3, :3] = current[:3, :3] @ delta_rotation
    return target


def validate_demo_delta(
    current_transform: np.ndarray,
    target_transform: np.ndarray,
) -> tuple[float, float]:
    """Reject a zero or overly large one-step demonstration target."""
    position_delta, orientation_delta = pose_error(
        target_transform,
        current_transform,
    )
    if (
        position_delta < 0.001
        and orientation_delta < math.radians(0.5)
    ):
        raise ValueError("目标位于 1 mm / 0.5° 死区内，无需运动")
    if position_delta > DEMO_MAX_POSITION_DELTA_M:
        raise ValueError("单次 Demo 的位置变化不能超过 80 mm")
    if orientation_delta > DEMO_MAX_ORIENTATION_DELTA_RAD:
        raise ValueError("单次 Demo 的姿态变化不能超过 30°")
    return float(position_delta), float(orientation_delta)


def minimum_joint_limit_margin(
    joints: Sequence[float],
    joint_limits: np.ndarray,
) -> float:
    """Return the smallest distance from a joint solution to either limit."""
    values = np.asarray(joints, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if values.shape != (7,) or limits.shape != (7, 2):
        raise ValueError("关节解或关节限位维度无效")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(limits)):
        raise ValueError("关节解或关节限位包含无效数值")
    lower_clearance = values - limits[:, 0]
    upper_clearance = limits[:, 1] - values
    return float(np.min(np.minimum(lower_clearance, upper_clearance)))


def pose_text(transform: np.ndarray) -> str:
    """Format one transform as position, quaternion and human-readable RPY."""
    position = transform[:3, 3]
    rotation = Rotation.from_matrix(transform[:3, :3])
    quaternion = rotation.as_quat()
    rpy = rotation.as_euler("xyz", degrees=True)
    return (
        f"位置 m      : [{position[0]:.6f}, {position[1]:.6f}, "
        f"{position[2]:.6f}]\n"
        f"四元数 xyzw : [{quaternion[0]:.7f}, {quaternion[1]:.7f}, "
        f"{quaternion[2]:.7f}, {quaternion[3]:.7f}]\n"
        f"RPY 度      : [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
    )


class PoseDemoNode(NeroRealSmokeTest):
    """Use the validated ROS clients and real-arm command gates for one pose."""

    def __init__(self) -> None:
        super().__init__("nero_pose_demo")

    def current_pose(self) -> tuple[MeasuredState, np.ndarray]:
        """Read a fresh safe joint state and calculate the current link7 pose."""
        state = self.measured_state()
        limits = self._solver.safe_joint_limits
        if np.any(state.positions < limits[:, 0]) or np.any(
            state.positions > limits[:, 1]
        ):
            raise SmokeTestError(
                "真实关节不在保守安全范围；请先使用 recover 工具恢复"
            )
        transform = self._solver.forward_kinematics(
            state.positions,
            CONTROLLED_FRAME,
        )
        return state, transform

    def controller_is_simulation(self) -> bool:
        """Read the controller mode instead of trusting a command-line flag."""
        if not self._parameter_client.wait_for_service(timeout_sec=5.0):
            raise SmokeTestError("无法读取 nero_control 模式")
        request = GetParameters.Request()
        request.names = ["simulation_mode"]
        response = self._wait_future(
            self._parameter_client.call_async(request),
            5.0,
            "读取 simulation_mode",
        )
        if (
            len(response.values) != 1
            or response.values[0].type != ParameterType.PARAMETER_BOOL
        ):
            raise SmokeTestError("simulation_mode 参数返回无效")
        return bool(response.values[0].bool_value)

    def preview_target(self, target_pose) -> IKResultMsg:
        """Call Placo SolveIK and apply the already-verified demo envelope."""
        result = self._solve_pose_preview(target_pose, "Placo 目标位姿预览")
        print("\n=== Placo IK 预览（没有发送运动命令）===")
        print(f"code={result.code}, success={result.success}")
        print(f"说明：{result.reason}")
        print(f"IK 耗时={result.solve_time_ms:.3f} ms")
        print(
            f"残差={result.position_error_m * 1000.0:.3f} mm / "
            f"{math.degrees(result.orientation_error_rad):.4f}°"
        )
        print(
            f"最大关节变化={result.max_joint_delta_rad:.6f} rad, "
            f"sigma_min={result.sigma_min:.4f}, "
            f"condition={result.condition_number:.2f}"
        )
        print(
            "目标关节角="
            + np.array2string(
                np.asarray(result.solution_joint_state.position),
                precision=6,
            )
        )
        validate_smoke_ik(result, DEMO_MAX_JOINT_DELTA_RAD)
        limit_margin = minimum_joint_limit_margin(
            result.solution_joint_state.position,
            self._solver.safe_joint_limits,
        )
        print(f"最小关节限位余量={limit_margin:.6f} rad")
        if limit_margin < DEMO_MIN_JOINT_LIMIT_MARGIN_RAD:
            raise SmokeTestError(
                "目标关节解距离保守限位不足 0.010 rad；"
                "请换方向或先回到 ready 邻域"
            )
        return result

    def execute_simulated(self, target_pose):
        """Send one MoveToPose goal in sim mode without touching hardware gates."""
        goal = MoveToPose.Goal()
        goal.target_pose = target_pose
        goal.controlled_frame = CONTROLLED_FRAME
        goal.timeout.sec = 15
        wrapped = self._send_action_goal(
            self._move_client,
            goal,
            "仿真目标位姿",
            20.0,
        )
        result = wrapped.result
        print("\n=== 仿真执行结果 ===")
        print(
            f"Action status={wrapped.status}, "
            f"code={result.ik_result.code}, "
            f"success={result.ik_result.success}"
        )
        print(f"说明：{result.ik_result.reason}")
        print(
            f"最终误差={result.final_position_error_m * 1000.0:.3f} mm / "
            f"{math.degrees(result.final_orientation_error_rad):.4f}°"
        )
        if (
            wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or not result.ik_result.success
        ):
            raise SmokeTestError("仿真 MoveToPose 没有成功完成")
        return result


def _parser() -> argparse.ArgumentParser:
    """Create a compact CLI for current, relative and absolute poses."""
    parser = argparse.ArgumentParser(
        description=(
            "给 NERO 的 link7 输入一个目标位姿。默认只做 Placo IK 预览；"
            "添加 --execute 才会执行。"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("current", help="只打印当前 link7 位姿")

    relative = subparsers.add_parser(
        "relative",
        help="相对当前位姿输入毫米位移和角度变化",
    )
    relative.add_argument(
        "--xyz-mm",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    relative.add_argument(
        "--rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("R", "P", "Y"),
    )
    relative.add_argument(
        "--frame",
        choices=("base", "tool"),
        default="base",
        help="变化量使用 base_link 或当前 link7 轴；默认 base",
    )
    relative.add_argument("--execute", action="store_true")

    absolute = subparsers.add_parser(
        "absolute",
        help="输入 base_link 中的绝对位置和姿态",
    )
    absolute.add_argument(
        "--position-m",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    orientation = absolute.add_mutually_exclusive_group(required=True)
    orientation.add_argument(
        "--quat-xyzw",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
    )
    orientation.add_argument(
        "--rpy-deg",
        nargs=3,
        type=float,
        metavar=("R", "P", "Y"),
    )
    absolute.add_argument("--execute", action="store_true")
    return parser


def run_demo(
    node: PoseDemoNode,
    arguments,
    input_function: Callable[[str], str] = input,
) -> int:
    """Print the current pose, preview the request, then optionally execute."""
    planned_state, current = node.current_pose()
    print("\n=== 当前 link7 位姿（base_link 坐标系）===")
    print(pose_text(current))
    if arguments.command == "current":
        print("\n只读取和计算了当前位姿，没有发送运动命令。")
        return 0

    if arguments.command == "relative":
        target = relative_target_transform(
            current,
            arguments.xyz_mm,
            arguments.rpy_deg,
            arguments.frame,
        )
    else:
        target = absolute_target_transform(
            arguments.position_m,
            quaternion_xyzw=arguments.quat_xyzw,
            rpy_degrees=arguments.rpy_deg,
        )

    position_delta, orientation_delta = validate_demo_delta(current, target)
    print("\n=== 请求的 link7 目标位姿（base_link 坐标系）===")
    print(pose_text(target))
    print(
        f"相对当前变化：{position_delta * 1000.0:.1f} mm / "
        f"{math.degrees(orientation_delta):.2f}°"
    )
    target_pose = matrix_to_pose_stamped(
        target,
        BASE_FRAME,
        node.get_clock().now().to_msg(),
    )
    node.preview_target(target_pose)
    if not arguments.execute:
        print("\n预览通过：Placo 已求解，但机械臂没有运动。")
        return 0

    simulation = node.controller_is_simulation()
    if simulation:
        print("\n检测到 sim 模式：只驱动虚拟模型，不访问 CAN。")
        node.execute_simulated(target_pose)
        return 0

    require_confirmation("MOVE_POSE", input_function)
    node.execute_placo(
        target_pose,
        planned_state,
        precision_test=True,
        label="现场目标位姿 Demo",
    )
    print("\n目标位姿 Demo 成功。")
    print(POST_MOTION_SAFETY_NOTICE)
    return 0


def main(args: Optional[Sequence[str]] = None) -> int:
    """Run the demo with deterministic cleanup and refusal exit codes."""
    raw_args = list(sys.argv if args is None else [sys.argv[0], *args])
    cli_args = remove_ros_args(args=raw_args)[1:]
    arguments = _parser().parse_args(cli_args)
    rclpy.init(
        args=raw_args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    try:
        node = PoseDemoNode()
        return run_demo(node, arguments)
    except (SmokeTestError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n操作已停止或被拒绝：{error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print(
            "\n操作已取消；异常时由观察人员在工作区外切断控制箱电源"
            "（机械臂可能下落）。",
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
