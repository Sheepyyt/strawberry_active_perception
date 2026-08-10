"""Deterministic 30-pose dataset for the Week-1 NERO acceptance test."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional, Sequence

import numpy as np

from .ik_core import IKResult, PlacoIKSolver
from .models import IKErrorCode, NERO_JOINT_NAMES, READY_JOINT_POSITIONS
from .ros_utils import pose_error
from .trajectory import TrajectoryGenerator, TrajectoryResult


ACCEPTANCE_DATASET_VERSION = "week1-30-v2"
ACCEPTANCE_TARGET_COUNT = 30
ACCEPTANCE_REPEATS = 3
ACCEPTANCE_TOTAL_TARGET_ATTEMPTS = (
    ACCEPTANCE_TARGET_COUNT * ACCEPTANCE_REPEATS
)
ACCEPTANCE_MAX_JOINT_DELTA_RAD = 0.12
ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD = 0.02
ACCEPTANCE_MIN_SIGMA = 0.10
ACCEPTANCE_MAX_CONDITION = 20.0

# These small offsets are applied to the central anchor joint vector only to
# construct known-reachable Cartesian targets through FK. Placo still receives
# only the resulting target pose; it does not receive these reference joints.
ACCEPTANCE_JOINT_AMPLITUDES_RAD = np.asarray(
    [0.110, 0.100, 0.100, 0.090, 0.090, 0.080, 0.080],
    dtype=float,
)
_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17)


@dataclass(frozen=True)
class AcceptanceTargetPlan:
    """One known-reachable target and its outbound/return safety evidence."""

    index: int
    target_id: str
    reference_joints: np.ndarray
    target_transform: np.ndarray
    position_offset_m: float
    orientation_offset_rad: float
    outbound_ik: IKResult
    return_ik: Optional[IKResult]
    outbound_trajectory: Optional[TrajectoryResult]
    return_trajectory: Optional[TrajectoryResult]
    predicted_return_joint_error_rad: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class AcceptanceSuitePlan:
    """A frozen anchor and all 30 deterministic acceptance targets."""

    dataset_id: str
    anchor_joints: np.ndarray
    anchor_transform: np.ndarray
    targets: tuple[AcceptanceTargetPlan, ...]

    @property
    def passed(self) -> bool:
        """Return true only when every target and return path is accepted."""
        return len(self.targets) == ACCEPTANCE_TARGET_COUNT and all(
            target.accepted for target in self.targets
        )


def _radical_inverse(index: int, base: int) -> float:
    """Return one deterministic Halton coordinate in the open unit interval."""
    if index <= 0 or base <= 1:
        raise ValueError("Halton index/base are invalid")
    value = 0.0
    fraction = 1.0 / base
    current = index
    while current:
        current, digit = divmod(current, base)
        value += digit * fraction
        fraction /= base
    return value


def deterministic_joint_offsets() -> tuple[np.ndarray, ...]:
    """Generate 30 fixed, balanced seven-joint perturbations near ready."""
    offsets = []
    # Skip the earliest Halton points, whose cross-axis distribution is less
    # even. The fixed skip is part of the versioned dataset definition.
    for sample_index in range(19, 19 + ACCEPTANCE_TARGET_COUNT):
        unit = np.asarray([
            _radical_inverse(sample_index, base)
            for base in _HALTON_BASES
        ])
        offset = (2.0 * unit - 1.0) * ACCEPTANCE_JOINT_AMPLITUDES_RAD
        offsets.append(offset)
    return tuple(offsets)


def _ik_refusal(result: IKResult, phase: str) -> Optional[str]:
    if not result.success or result.error_code != IKErrorCode.SUCCESS:
        return f"{phase} IK 被拒绝：{result.message}"
    values = np.asarray([
        result.position_error_m,
        result.orientation_error_rad,
        result.solve_time_ms,
        result.max_joint_delta_rad,
        result.sigma_min,
        result.condition_number,
    ])
    if not np.all(np.isfinite(values)):
        return f"{phase} IK 诊断包含无效数值"
    if result.max_joint_delta_rad > ACCEPTANCE_MAX_JOINT_DELTA_RAD:
        return (
            f"{phase}关节变化超过 "
            f"{ACCEPTANCE_MAX_JOINT_DELTA_RAD:.2f} rad"
        )
    if (
        result.sigma_min < ACCEPTANCE_MIN_SIGMA
        or result.condition_number > ACCEPTANCE_MAX_CONDITION
    ):
        return f"{phase}过于接近奇异点"
    return None


def _dataset_id(
    anchor_joints: np.ndarray,
    target_transforms: Sequence[np.ndarray],
) -> str:
    payload = {
        "version": ACCEPTANCE_DATASET_VERSION,
        "anchor_joints": np.round(anchor_joints, 12).tolist(),
        "target_transforms": [
            np.round(transform, 12).tolist()
            for transform in target_transforms
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{ACCEPTANCE_DATASET_VERSION}-{digest[:12]}"


def plan_acceptance_suite(
    solver: PlacoIKSolver,
    trajectory_generator: TrajectoryGenerator,
    anchor_joints: Sequence[float] = READY_JOINT_POSITIONS,
) -> AcceptanceSuitePlan:
    """Build and precheck all 30 known-reachable pose/return pairs."""
    anchor = np.asarray(anchor_joints, dtype=float)
    if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
        raise ValueError("acceptance anchor must contain seven finite joints")
    limits = solver.safe_joint_limits
    if np.any(anchor < limits[:, 0]) or np.any(anchor > limits[:, 1]):
        raise ValueError("acceptance anchor violates conservative joint limits")

    anchor_transform = solver.forward_kinematics(anchor, "link7")
    targets = []
    for index, offset in enumerate(deterministic_joint_offsets()):
        reference = anchor + offset
        if np.any(reference < limits[:, 0]) or np.any(
            reference > limits[:, 1]
        ):
            raise ValueError(
                f"acceptance target {index + 1:02d} reference is outside limits"
            )
        target_transform = solver.forward_kinematics(reference, "link7")
        position_offset, orientation_offset = pose_error(
            anchor_transform,
            target_transform,
        )
        outbound = solver.solve(target_transform, anchor, "link7")
        refusal = _ik_refusal(outbound, "去程")
        outbound_trajectory = None
        return_result = None
        return_trajectory = None
        return_joint_error = math.inf

        if refusal is None:
            outbound_trajectory = trajectory_generator.generate(
                anchor,
                outbound.joint_positions,
            )
            if not outbound_trajectory.success:
                refusal = f"去程轨迹被拒绝：{outbound_trajectory.message}"
        if refusal is None:
            return_result = solver.solve(
                anchor_transform,
                outbound.joint_positions,
                "link7",
                posture_reference_joints=anchor,
            )
            refusal = _ik_refusal(return_result, "回程")
        if refusal is None and return_result is not None:
            return_trajectory = trajectory_generator.generate(
                outbound.joint_positions,
                return_result.joint_positions,
            )
            if not return_trajectory.success:
                refusal = f"回程轨迹被拒绝：{return_trajectory.message}"
        if refusal is None and return_result is not None:
            return_joint_error = float(np.max(np.abs(
                np.asarray(return_result.joint_positions) - anchor
            )))
            if return_joint_error > ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD:
                refusal = "预计回程关节误差超过 0.02 rad"

        targets.append(AcceptanceTargetPlan(
            index=index,
            target_id=f"P{index + 1:02d}",
            reference_joints=reference.copy(),
            target_transform=target_transform.copy(),
            position_offset_m=float(position_offset),
            orientation_offset_rad=float(orientation_offset),
            outbound_ik=outbound,
            return_ik=return_result,
            outbound_trajectory=outbound_trajectory,
            return_trajectory=return_trajectory,
            predicted_return_joint_error_rad=return_joint_error,
            accepted=refusal is None,
            reason="通过" if refusal is None else refusal,
        ))

    transforms = [target.target_transform for target in targets]
    return AcceptanceSuitePlan(
        dataset_id=_dataset_id(anchor, transforms),
        anchor_joints=anchor.copy(),
        anchor_transform=anchor_transform.copy(),
        targets=tuple(targets),
    )


TARGET_REPORT_FIELDS = (
    "index",
    "target_id",
    "accepted",
    "reason",
    "position_offset_m",
    "orientation_offset_rad",
    "outbound_solve_time_ms",
    "outbound_position_error_m",
    "outbound_orientation_error_rad",
    "outbound_max_joint_delta_rad",
    "outbound_sigma_min",
    "outbound_condition_number",
    "outbound_trajectory_duration_s",
    "outbound_peak_velocity_rad_s",
    "outbound_peak_acceleration_rad_s2",
    "return_solve_time_ms",
    "return_position_error_m",
    "return_orientation_error_rad",
    "return_max_joint_delta_rad",
    "return_sigma_min",
    "return_condition_number",
    "return_trajectory_duration_s",
    "return_peak_velocity_rad_s",
    "return_peak_acceleration_rad_s2",
    "predicted_return_joint_error_rad",
    "reference_joints_rad",
    "target_transform_row_major",
)


def target_to_row(target: AcceptanceTargetPlan) -> dict:
    """Convert one target plan into the stable CSV report schema."""
    outbound_trajectory = target.outbound_trajectory
    return_result = target.return_ik
    return_trajectory = target.return_trajectory
    return {
        "index": target.index,
        "target_id": target.target_id,
        "accepted": target.accepted,
        "reason": target.reason,
        "position_offset_m": target.position_offset_m,
        "orientation_offset_rad": target.orientation_offset_rad,
        "outbound_solve_time_ms": target.outbound_ik.solve_time_ms,
        "outbound_position_error_m": target.outbound_ik.position_error_m,
        "outbound_orientation_error_rad": (
            target.outbound_ik.orientation_error_rad
        ),
        "outbound_max_joint_delta_rad": (
            target.outbound_ik.max_joint_delta_rad
        ),
        "outbound_sigma_min": target.outbound_ik.sigma_min,
        "outbound_condition_number": target.outbound_ik.condition_number,
        "outbound_trajectory_duration_s": (
            "" if outbound_trajectory is None
            else outbound_trajectory.duration_s
        ),
        "outbound_peak_velocity_rad_s": (
            "" if outbound_trajectory is None
            else outbound_trajectory.peak_velocity_rad_s
        ),
        "outbound_peak_acceleration_rad_s2": (
            "" if outbound_trajectory is None
            else outbound_trajectory.peak_acceleration_rad_s2
        ),
        "return_solve_time_ms": (
            "" if return_result is None else return_result.solve_time_ms
        ),
        "return_position_error_m": (
            "" if return_result is None else return_result.position_error_m
        ),
        "return_orientation_error_rad": (
            "" if return_result is None
            else return_result.orientation_error_rad
        ),
        "return_max_joint_delta_rad": (
            "" if return_result is None
            else return_result.max_joint_delta_rad
        ),
        "return_sigma_min": (
            "" if return_result is None else return_result.sigma_min
        ),
        "return_condition_number": (
            "" if return_result is None
            else return_result.condition_number
        ),
        "return_trajectory_duration_s": (
            "" if return_trajectory is None
            else return_trajectory.duration_s
        ),
        "return_peak_velocity_rad_s": (
            "" if return_trajectory is None
            else return_trajectory.peak_velocity_rad_s
        ),
        "return_peak_acceleration_rad_s2": (
            "" if return_trajectory is None
            else return_trajectory.peak_acceleration_rad_s2
        ),
        "predicted_return_joint_error_rad": (
            target.predicted_return_joint_error_rad
        ),
        "reference_joints_rad": json.dumps(
            target.reference_joints.tolist(),
            separators=(",", ":"),
        ),
        "target_transform_row_major": json.dumps(
            target.target_transform.reshape(-1).tolist(),
            separators=(",", ":"),
        ),
    }


def write_acceptance_target_set(
    plan: AcceptanceSuitePlan,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write the stable, visible 30-target CSV/JSON definition."""
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "target_set_30.csv"
    json_path = directory / "target_set_30.json"
    rows = [target_to_row(target) for target in plan.targets]
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TARGET_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    payload = {
        "schema_version": 1,
        "dataset_version": ACCEPTANCE_DATASET_VERSION,
        "dataset_id": plan.dataset_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_count": len(plan.targets),
        "repeats_per_target": ACCEPTANCE_REPEATS,
        "target_attempts_required": ACCEPTANCE_TOTAL_TARGET_ATTEMPTS,
        "passed_offline_precheck": plan.passed,
        "anchor_joints_rad": plan.anchor_joints.tolist(),
        "anchor_transform": plan.anchor_transform.tolist(),
        "joint_names": list(NERO_JOINT_NAMES),
        "joint_amplitudes_rad": ACCEPTANCE_JOINT_AMPLITUDES_RAD.tolist(),
        "thresholds": {
            "max_joint_delta_rad": ACCEPTANCE_MAX_JOINT_DELTA_RAD,
            "max_return_joint_error_rad": (
                ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD
            ),
            "min_sigma": ACCEPTANCE_MIN_SIGMA,
            "max_condition": ACCEPTANCE_MAX_CONDITION,
        },
        "targets": rows,
        "important_note": (
            "Targets are known-reachable URDF/FK poses near ready. This does "
            "not prove environment collision clearance or absolute physical "
            "camera accuracy."
        ),
    }
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(json_path)
    return csv_path, json_path


def print_acceptance_suite(plan: AcceptanceSuitePlan) -> None:
    """Print a compact 30-target precheck table."""
    print("\n=== 第一周 30 个安全目标离线预检 ===")
    print(f"数据集：{plan.dataset_id}")
    print(
        "目标  结果  位移(mm)  转角(deg)  去程Δq(rad)  "
        "回点Δq(rad)  sigma_min"
    )
    for target in plan.targets:
        print(
            f"{target.target_id:>4}  "
            f"{'通过' if target.accepted else '拒绝':<4}  "
            f"{target.position_offset_m * 1000.0:>8.2f}  "
            f"{math.degrees(target.orientation_offset_rad):>9.3f}  "
            f"{target.outbound_ik.max_joint_delta_rad:>11.6f}  "
            f"{target.predicted_return_joint_error_rad:>11.6f}  "
            f"{target.outbound_ik.sigma_min:>9.4f}"
        )
        if not target.accepted:
            print(f"      原因：{target.reason}")


def animate_acceptance_suite(
    plan: AcceptanceSuitePlan,
    urdf_path: Path,
    playback_rate: float = 2.0,
) -> None:
    """Play all 30 prechecked out-and-return paths in MeshCat only."""
    if not plan.passed:
        raise ValueError("cannot animate a failed acceptance suite")
    if not math.isfinite(playback_rate) or playback_rate <= 0.0:
        raise ValueError("playback_rate must be finite and positive")

    # Keep visualization optional: offline calculations do not import MeshCat.
    import meshcat
    import placo
    import placo_utils.visualization as visualization

    from .standalone_demo import _resolved_urdf_content

    path = Path(urdf_path).expanduser().resolve()
    visualization.viewer = meshcat.Visualizer()
    flags = int(placo.Flags.ignore_collisions)
    flags |= int(placo.Flags.collision_as_visual)
    robot = placo.RobotWrapper(
        str(path),
        flags,
        _resolved_urdf_content(path),
    )
    visualizer = visualization.robot_viz(robot, "nero_week1_targets")

    def set_positions(positions) -> None:
        for joint_name, value in zip(NERO_JOINT_NAMES, positions):
            robot.set_joint(joint_name, float(value))
        robot.update_kinematics()

    def display_positions(positions) -> None:
        set_positions(positions)
        visualizer.display(robot.state.q)

    for target in plan.targets:
        visualization.frame_viz(
            f"week1_targets/{target.target_id}",
            target.target_transform,
            opacity=0.45,
            scale=0.25,
        )
        path_points = []
        assert target.outbound_trajectory is not None
        for point in target.outbound_trajectory.points:
            set_positions(point.positions)
            path_points.append(np.asarray(
                robot.get_T_world_frame("link7")[:3, 3],
                dtype=float,
            ))
        visualization.path_viz(
            f"week1_paths/{target.target_id}",
            np.asarray(path_points),
            0x00A0FF,
        )

    display_positions(plan.anchor_joints)
    print(f"\nMeshCat 地址：{visualization.viewer.url()}")
    print(
        "这里只播放虚拟模型和 Placo 五次轨迹，不加载驱动、不连接 CAN。"
    )
    input("打开网页检查 30 条路径；准备好后按 Enter 开始依次播放……")
    period = 1.0 / (50.0 * playback_rate)
    for target in plan.targets:
        assert target.outbound_trajectory is not None
        assert target.return_trajectory is not None
        print(f"播放 {target.target_id}/P30")
        for point in target.outbound_trajectory.points:
            display_positions(point.positions)
            time.sleep(period)
        for point in target.return_trajectory.points:
            display_positions(point.positions)
            time.sleep(period)
    display_positions(plan.anchor_joints)
    input("30 个目标播放完成。按 Enter 关闭 MeshCat 并退出……")
