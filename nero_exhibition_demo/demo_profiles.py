"""Large, conservative NERO exhibition profiles and offline validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from strawberry_nero_control.ik_core import PlacoIKSolver
from strawberry_nero_control.models import (
    IKConfig,
    IKErrorCode,
    NERO_JOINT_NAMES,
    READY_JOINT_POSITIONS,
    TrajectoryConfig,
)
from strawberry_nero_control.trajectory import TrajectoryGenerator


# The arm first extends into this posture, then joint1 sweeps the whole shape
# around the base.  Only joint1 changes during the wide part of the show, so
# the arm keeps the same well-conditioned shape throughout the sweep.
SHOW_CENTER_JOINTS = (0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0)
SHOW_LEFT_JOINTS = (1.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0)
SHOW_RIGHT_JOINTS = (-1.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0)
EXPAND_TRANSITION_JOINTS = tuple(
    tuple(
        float(ready + fraction * (show - ready))
        for ready, show in zip(READY_JOINT_POSITIONS, SHOW_CENTER_JOINTS)
    )
    for fraction in (0.25, 0.50, 0.75)
)

# These client-side gates are intentionally tighter than the ordinary live
# controller gates.  A failure stops the show before the next Action goal.
MAX_LEG_JOINT_DELTA_RAD = 0.55
MIN_JOINT_LIMIT_MARGIN_RAD = 0.20
MAX_PREVIEW_POSITION_ERROR_M = 0.005
MAX_PREVIEW_ORIENTATION_ERROR_RAD = math.radians(2.0)
MIN_SIGMA = 0.10
MAX_CONDITION_NUMBER = 20.0
MAX_POSTURE_ERROR_RAD = 0.010
MAX_LEG_DURATION_S = 6.50


@dataclass(frozen=True)
class DemoTarget:
    """One named, model-derived exhibition target."""

    label: str
    joints: tuple[float, ...]
    hold_at_target: bool = False


@dataclass(frozen=True)
class PlannedTarget:
    """Offline-validated transform and diagnostics for one target."""

    target: DemoTarget
    transform: np.ndarray
    predicted_joints: tuple[float, ...]
    max_joint_delta_rad: float
    position_error_m: float
    orientation_error_rad: float
    sigma_min: float
    condition_number: float
    joint_limit_margin_rad: float
    trajectory_duration_s: float


@dataclass(frozen=True)
class DemoPlan:
    """A complete show profile that starts and finishes at ready."""

    mode: str
    targets: tuple[PlannedTarget, ...]
    xyz_span_m: tuple[float, float, float]
    main_x_range_m: tuple[float, float]
    predicted_motion_duration_s: float


def profile_targets(mode: str) -> tuple[DemoTarget, ...]:
    """Return fixed-pose or wide-arc targets, always ending at ready."""
    expand = tuple(
        DemoTarget(
            f"向 -X 展开过渡 {index + 1}/{len(EXPAND_TRANSITION_JOINTS)}",
            joints,
        )
        for index, joints in enumerate(EXPAND_TRANSITION_JOINTS)
    )
    retract_joints = tuple(reversed(EXPAND_TRANSITION_JOINTS))
    retract = tuple(
        DemoTarget(
            f"从 -X 收回过渡 {index + 1}/{len(retract_joints)}",
            joints,
        )
        for index, joints in enumerate(retract_joints)
    )
    if mode == "poses":
        return (
            *expand,
            DemoTarget("展开到 -X 展示中心", SHOW_CENTER_JOINTS, True),
            DemoTarget("前往左侧的中间路点", (0.5, *SHOW_CENTER_JOINTS[1:])),
            DemoTarget("左侧大幅展示", SHOW_LEFT_JOINTS, True),
            DemoTarget("离开左侧的中间路点", (0.5, *SHOW_CENTER_JOINTS[1:])),
            DemoTarget("回到展示中心", SHOW_CENTER_JOINTS, True),
            DemoTarget("前往右侧的中间路点", (-0.5, *SHOW_CENTER_JOINTS[1:])),
            DemoTarget("右侧大幅展示", SHOW_RIGHT_JOINTS, True),
            DemoTarget("离开右侧的中间路点", (-0.5, *SHOW_CENTER_JOINTS[1:])),
            DemoTarget("再次回到展示中心", SHOW_CENTER_JOINTS, True),
            *retract,
            DemoTarget("回到 ready", tuple(READY_JOINT_POSITIONS)),
        )
    if mode == "trajectory":
        # The controller accepts one safe pose Action at a time.  These points
        # therefore form a piecewise minimum-jerk arc, with measured settling
        # at every waypoint, instead of an unmonitored direct command stream.
        joint1_waypoints = (0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0)
        arc = tuple(
            DemoTarget(
                f"大弧线路点 {index + 1}/{len(joint1_waypoints)}",
                (
                    value,
                    SHOW_CENTER_JOINTS[1],
                    SHOW_CENTER_JOINTS[2],
                    SHOW_CENTER_JOINTS[3],
                    SHOW_CENTER_JOINTS[4],
                    SHOW_CENTER_JOINTS[5],
                    SHOW_CENTER_JOINTS[6],
                ),
            )
            for index, value in enumerate(joint1_waypoints)
        )
        return (
            *expand,
            DemoTarget("展开到 -X 轨迹起点", SHOW_CENTER_JOINTS),
            *arc,
            *retract,
            DemoTarget("轨迹结束并回到 ready", tuple(READY_JOINT_POSITIONS)),
        )
    raise ValueError("mode 必须是 poses 或 trajectory")


def make_profile_solver(
    urdf_path: str | Path,
    *,
    timeout_s: float = 0.10,
) -> PlacoIKSolver:
    """Build a planner matching normal Demo limits with a stable check timeout."""
    config = IKConfig(
        position_tolerance_m=0.050,
        position_convergence_target_m=0.001,
        orientation_tolerance_rad=math.radians(5.0),
        max_joint_delta_rad=1.50,
        timeout_s=float(timeout_s),
    )
    return PlacoIKSolver(urdf_path, config=config)


def joint_limit_margin(
    joints: Sequence[float],
    safe_limits: np.ndarray,
) -> float:
    """Return the nearest conservative joint-limit clearance."""
    values = np.asarray(joints, dtype=float)
    limits = np.asarray(safe_limits, dtype=float)
    if values.shape != (7,) or limits.shape != (7, 2):
        raise ValueError("关节或限位维度无效")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(limits)):
        raise ValueError("关节或限位包含无效数值")
    return float(np.min(np.minimum(
        values - limits[:, 0],
        limits[:, 1] - values,
    )))


def _validate_definition(
    targets: Sequence[DemoTarget],
    safe_limits: np.ndarray,
) -> None:
    """Reject accidental edits that shrink or weaken the exhibition profile."""
    if not targets:
        raise ValueError("展示 profile 不能为空")
    ready = np.asarray(READY_JOINT_POSITIONS, dtype=float)
    if not np.allclose(targets[-1].joints, ready, rtol=0.0, atol=1.0e-12):
        raise ValueError("展示 profile 必须以 ready 结束")

    values = np.asarray([target.joints for target in targets], dtype=float)
    if values.shape != (len(targets), 7) or not np.all(np.isfinite(values)):
        raise ValueError("展示 profile 必须包含完整有限的 7 关节目标")
    if tuple(NERO_JOINT_NAMES) != tuple(f"joint{i}" for i in range(1, 8)):
        raise ValueError("当前项目的 NERO 关节顺序已改变")

    for target in targets:
        margin = joint_limit_margin(target.joints, safe_limits)
        if margin < MIN_JOINT_LIMIT_MARGIN_RAD:
            raise ValueError(
                f"{target.label} 的关节限位余量只有 {margin:.4f} rad"
            )

    joint1_span = float(np.ptp(values[:, 0]))
    if joint1_span < 2.0 - 1.0e-12:
        raise ValueError("展示 profile 的 joint1 总摆幅小于 2.0 rad")


def _validate_ik_result(
    label: str,
    expected_posture: Sequence[float],
    result,
    safe_limits: np.ndarray,
) -> float:
    """Apply the stricter exhibition gates to one Placo result."""
    if not result.success or result.error_code != IKErrorCode.SUCCESS:
        raise ValueError(
            f"{label} 离线 IK 被拒绝：code={int(result.error_code)}, "
            f"{result.message}"
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
        raise ValueError(f"{label} 离线 IK 诊断无效")
    if result.max_joint_delta_rad > MAX_LEG_JOINT_DELTA_RAD:
        raise ValueError(
            f"{label} 单段变化 {result.max_joint_delta_rad:.4f} rad 过大"
        )
    if result.position_error_m > MAX_PREVIEW_POSITION_ERROR_M:
        raise ValueError(f"{label} 位置残差超过 5 mm")
    if result.orientation_error_rad > MAX_PREVIEW_ORIENTATION_ERROR_RAD:
        raise ValueError(f"{label} 姿态残差超过 2°")
    if result.sigma_min < MIN_SIGMA:
        raise ValueError(f"{label} sigma_min={result.sigma_min:.4f} 过低")
    if result.condition_number > MAX_CONDITION_NUMBER:
        raise ValueError(
            f"{label} condition={result.condition_number:.2f} 过高"
        )

    solution = np.asarray(result.joint_positions, dtype=float)
    expected = np.asarray(expected_posture, dtype=float)
    posture_error = float(np.max(np.abs(solution - expected)))
    if posture_error > MAX_POSTURE_ERROR_RAD:
        raise ValueError(
            f"{label} 与预设姿态相差 {posture_error:.4f} rad"
        )
    margin = joint_limit_margin(solution, safe_limits)
    if margin < MIN_JOINT_LIMIT_MARGIN_RAD:
        raise ValueError(f"{label} IK 解的关节限位余量不足")
    return margin


def plan_profile(
    mode: str,
    solver: PlacoIKSolver,
) -> DemoPlan:
    """Solve and trajectory-check every leg without ROS or CAN commands."""
    targets = profile_targets(mode)
    safe_limits = solver.safe_joint_limits
    _validate_definition(targets, safe_limits)
    trajectory = TrajectoryGenerator(
        TrajectoryConfig(
            frequency_hz=50.0,
            max_velocity_rad_s=0.30,
            max_acceleration_rad_s2=0.50,
        ),
        joint_limits=safe_limits,
    )

    previous = np.asarray(READY_JOINT_POSITIONS, dtype=float)
    planned = []
    target_positions = []
    main_target_x = []
    for target in targets:
        target_transform = solver.forward_kinematics(target.joints, "link7")
        target_positions.append(target_transform[:3, 3].copy())
        if np.allclose(
            np.asarray(target.joints[1:], dtype=float),
            np.asarray(SHOW_CENTER_JOINTS[1:], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            main_target_x.append(float(target_transform[0, 3]))
        result = solver.solve(
            target_transform,
            previous,
            "link7",
            posture_reference_joints=target.joints,
        )
        margin = _validate_ik_result(
            target.label,
            target.joints,
            result,
            safe_limits,
        )
        segment = trajectory.generate(previous, result.joint_positions)
        if not segment.success:
            raise ValueError(
                f"{target.label} 五次轨迹被拒绝：{segment.message}"
            )
        if segment.duration_s > MAX_LEG_DURATION_S:
            raise ValueError(
                f"{target.label} 安全轨迹耗时 {segment.duration_s:.2f}s 过长"
            )
        planned.append(PlannedTarget(
            target=target,
            transform=target_transform,
            predicted_joints=tuple(float(value) for value in result.joint_positions),
            max_joint_delta_rad=float(result.max_joint_delta_rad),
            position_error_m=float(result.position_error_m),
            orientation_error_rad=float(result.orientation_error_rad),
            sigma_min=float(result.sigma_min),
            condition_number=float(result.condition_number),
            joint_limit_margin_rad=margin,
            trajectory_duration_s=float(segment.duration_s),
        ))
        previous = np.asarray(result.joint_positions, dtype=float)

    points = np.asarray(target_positions, dtype=float)
    xyz_span = tuple(float(value) for value in np.ptp(points, axis=0))
    if xyz_span[1] < 0.55:
        raise ValueError("末端左右展示跨度小于 0.55 m")
    if not main_target_x:
        raise ValueError("展示 profile 没有主要 -X 弧线路点")
    main_x_range = (min(main_target_x), max(main_target_x))
    if main_x_range[1] > -0.18:
        raise ValueError("主要展示弧线没有完全位于 base_link -X")
    return DemoPlan(
        mode=mode,
        targets=tuple(planned),
        xyz_span_m=xyz_span,
        main_x_range_m=main_x_range,
        predicted_motion_duration_s=float(sum(
            target.trajectory_duration_s for target in planned
        )),
    )
