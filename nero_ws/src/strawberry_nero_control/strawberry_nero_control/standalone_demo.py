"""Run a CAN-free Placo and MeshCat motion demo without ROS 2 messages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

import numpy as np

from .ik_core import PlacoIKSolver
from .models import (
    IKErrorCode,
    IKResult,
    NERO_JOINT_NAMES,
    READY_JOINT_POSITIONS,
    TrajectoryResult,
)
from .trajectory import TrajectoryGenerator


@dataclass(frozen=True)
class StandalonePlan:
    """One offline target, its IK result and its optional safe trajectory."""

    urdf_path: Path
    start_joints: tuple[float, ...]
    target_transform: np.ndarray
    ik_result: IKResult
    trajectory_result: Optional[TrajectoryResult]


def default_urdf_path() -> Path:
    """Locate the NERO source URDF without using the ROS package index."""
    source_directory = Path(__file__).resolve().parents[2]
    return (
        source_directory
        / "agx_arm_ros"
        / "src"
        / "agx_arm_description"
        / "agx_arm_urdf"
        / "nero"
        / "urdf"
        / "nero_description.urdf"
    )


def plan_standalone_motion(
    urdf_path: str | Path,
    offset_xyz_m: Sequence[float] = (0.0, 0.0, -0.020),
) -> StandalonePlan:
    """Solve a small Cartesian offset and generate a safe quintic motion."""
    path = Path(urdf_path).expanduser().resolve()
    offset = np.asarray(offset_xyz_m, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("offset_xyz_m must contain three finite values")

    start = np.asarray(READY_JOINT_POSITIONS, dtype=float)
    solver = PlacoIKSolver(path)
    target = solver.forward_kinematics(start, "link7")
    target[:3, 3] += offset
    ik_result = solver.solve(target, start, "link7")

    trajectory_result = None
    if ik_result.success and ik_result.error_code != IKErrorCode.ALREADY_AT_TARGET:
        generator = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
        trajectory_result = generator.generate(
            start,
            ik_result.joint_positions,
        )

    return StandalonePlan(
        urdf_path=path,
        start_joints=tuple(float(value) for value in start),
        target_transform=target,
        ik_result=ik_result,
        trajectory_result=trajectory_result,
    )


def _resolved_urdf_content(urdf_path: Path) -> str:
    """Replace the one ROS package URI so Placo can load meshes directly."""
    content = urdf_path.read_text(encoding="utf-8")
    package_token = "package://agx_arm_description/"
    package_root = next(
        (
            parent
            for parent in urdf_path.parents
            if parent.name == "agx_arm_description"
        ),
        None,
    )
    if package_token in content:
        if package_root is None:
            raise ValueError(
                "cannot resolve package://agx_arm_description in URDF"
            )
        content = content.replace(
            package_token,
            f"file://{package_root.as_posix()}/",
        )
    return content


def _set_visual_joints(robot, positions: Sequence[float]) -> None:
    """Place all seven visual joints in the driver-defined order."""
    for name, value in zip(NERO_JOINT_NAMES, positions):
        robot.set_joint(name, float(value))
    robot.update_kinematics()


def _play_positions(robot, visualizer, positions, period_s: float) -> None:
    """Display a list of joint samples at the configured trajectory rate."""
    next_frame_time = time.monotonic()
    for joint_positions in positions:
        _set_visual_joints(robot, joint_positions)
        visualizer.display(robot.state.q)
        next_frame_time += period_s
        time.sleep(max(0.0, next_frame_time - time.monotonic()))


def animate_plan(
    plan: StandalonePlan,
    cycles: int = 2,
    wait_for_user: bool = True,
) -> None:
    """Show the planned motion in MeshCat; never send hardware commands."""
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    trajectory = plan.trajectory_result
    if trajectory is None or not trajectory.success or not trajectory.points:
        raise ValueError("the standalone plan has no executable trajectory")

    # These imports stay local so ``--dry-run`` needs only the IK stack.
    import meshcat
    import placo
    import placo_utils.visualization as visualization

    visualization.viewer = meshcat.Visualizer()
    viewer_url = visualization.viewer.url()
    flags = int(placo.Flags.ignore_collisions)
    flags |= int(placo.Flags.collision_as_visual)
    robot = placo.RobotWrapper(
        str(plan.urdf_path),
        flags,
        _resolved_urdf_content(plan.urdf_path),
    )
    visualizer = visualization.robot_viz(robot, "nero_standalone")

    forward_positions = [point.positions for point in trajectory.points]
    reverse_positions = list(reversed(forward_positions))
    _set_visual_joints(robot, plan.start_joints)
    visualizer.display(robot.state.q)
    visualization.frame_viz(
        "standalone_target",
        plan.target_transform,
        opacity=0.9,
        scale=0.8,
    )

    path_points = []
    for positions in forward_positions:
        _set_visual_joints(robot, positions)
        path_points.append(
            np.asarray(robot.get_T_world_frame("link7")[:3, 3], dtype=float)
        )
    _set_visual_joints(robot, plan.start_joints)
    visualizer.display(robot.state.q)
    visualization.path_viz(
        "standalone_link7_path",
        np.asarray(path_points),
        0x00A0FF,
    )

    print(f"\nMeshCat 地址：{viewer_url}")
    print("本程序只更新虚拟模型，绝不会连接 CAN 或发送电机命令。")
    if wait_for_user:
        input("请打开上面的网页；准备好后按 Enter 开始运动……")

    period_s = 1.0 / 50.0
    for cycle_index in range(cycles):
        print(f"播放第 {cycle_index + 1}/{cycles} 次：ready → target")
        _play_positions(robot, visualizer, forward_positions, period_s)
        time.sleep(0.25)
        print(f"播放第 {cycle_index + 1}/{cycles} 次：target → ready")
        _play_positions(robot, visualizer, reverse_positions, period_s)
        time.sleep(0.25)

    if wait_for_user:
        input("演示完成。按 Enter 关闭网页服务并退出……")


def _print_plan(plan: StandalonePlan, offset: Sequence[float]) -> None:
    """Print the safety-relevant IK and trajectory measurements."""
    result = plan.ik_result
    print("\n=== NERO 纯 Python Placo 演示 ===")
    print("模式：离线虚拟模型（无 ROS 2 消息、无驱动、无 CAN）")
    print(
        "link7 目标偏移 [m]："
        + np.array2string(np.asarray(offset, dtype=float), precision=4)
    )
    print(f"IK 状态：{result.error_code.name} - {result.message}")
    print(f"IK 耗时：{result.solve_time_ms:.3f} ms")
    print(f"位置残差：{result.position_error_m * 1000.0:.3f} mm")
    print(f"姿态残差：{np.degrees(result.orientation_error_rad):.4f} deg")
    print(
        f"奇异性：sigma_min={result.sigma_min:.4f}, "
        f"condition={result.condition_number:.2f}"
    )
    print(f"最大关节变化：{result.max_joint_delta_rad:.4f} rad")
    if result.joint_positions:
        joints = np.asarray(result.joint_positions, dtype=float)
        print("Placo 目标关节角 [rad]：" + np.array2string(joints, precision=6))

    trajectory = plan.trajectory_result
    if trajectory is not None:
        print(
            f"五次轨迹：{len(trajectory.points)} 点 / "
            f"{trajectory.duration_s:.3f} s"
        )
        print(
            f"轨迹峰值：{trajectory.peak_velocity_rad_s:.3f} rad/s, "
            f"{trajectory.peak_acceleration_rad_s2:.3f} rad/s^2"
        )


def _positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _argument_parser() -> argparse.ArgumentParser:
    """Create the standalone demonstration command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Use Placo to move a virtual NERO model without ROS 2 or CAN."
        )
    )
    parser.add_argument("--urdf", type=Path, default=default_urdf_path())
    parser.add_argument("--dx", type=float, default=0.0, help="x offset in metres")
    parser.add_argument("--dy", type=float, default=0.0, help="y offset in metres")
    parser.add_argument(
        "--dz",
        type=float,
        default=-0.020,
        help="z offset in metres (default: -0.020)",
    )
    parser.add_argument("--cycles", type=_positive_integer, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="solve and print results without starting MeshCat",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="start animation immediately and exit after playback",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Plan and optionally animate one small, hardware-free NERO motion."""
    arguments = _argument_parser().parse_args(argv)
    offset = (arguments.dx, arguments.dy, arguments.dz)
    try:
        plan = plan_standalone_motion(arguments.urdf, offset)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"无法创建离线演示：{error}", file=sys.stderr)
        return 1

    _print_plan(plan, offset)
    if not plan.ik_result.success:
        print("IK 已拒绝该目标；没有生成运动，也没有发送任何命令。")
        return 2
    if plan.ik_result.error_code == IKErrorCode.ALREADY_AT_TARGET:
        print("目标位于死区内，不需要也不会播放运动。")
        return 0
    if plan.trajectory_result is None or not plan.trajectory_result.success:
        message = (
            "missing trajectory"
            if plan.trajectory_result is None
            else plan.trajectory_result.message
        )
        print(f"轨迹已拒绝：{message}", file=sys.stderr)
        return 3
    if arguments.dry_run:
        print("dry-run 完成：只计算，未启动 MeshCat。")
        return 0

    try:
        animate_plan(
            plan,
            cycles=arguments.cycles,
            wait_for_user=not arguments.no_prompt,
        )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，离线演示已停止。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
