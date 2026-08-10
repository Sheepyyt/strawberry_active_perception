"""Fixed six-direction Placo validation shared by offline and real tests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Optional, Sequence

import numpy as np
from ament_index_python.packages import get_package_share_directory

from .ik_core import IKResult, PlacoIKSolver
from .models import IKErrorCode, READY_JOINT_POSITIONS
from .trajectory import TrajectoryGenerator, TrajectoryResult


AXIS_TEST_DISTANCE_M = 0.015
AXIS_MAX_JOINT_DELTA_RAD = 0.08
AXIS_MAX_RETURN_JOINT_ERROR_RAD = 0.02
AXIS_MIN_SIGMA = 0.10
AXIS_MAX_CONDITION = 20.0


@dataclass(frozen=True)
class AxisDirection:
    """One signed local axis in the frozen link7 frame."""

    label: str
    axis_index: int
    sign: float
    distance_m: float


AXIS_DIRECTIONS = (
    AxisDirection("+X", 0, 1.0, AXIS_TEST_DISTANCE_M),
    AxisDirection("-X", 0, -1.0, AXIS_TEST_DISTANCE_M),
    AxisDirection("+Y", 1, 1.0, AXIS_TEST_DISTANCE_M),
    AxisDirection("-Y", 1, -1.0, AXIS_TEST_DISTANCE_M),
    AxisDirection("+Z", 2, 1.0, AXIS_TEST_DISTANCE_M),
    AxisDirection("-Z", 2, -1.0, AXIS_TEST_DISTANCE_M),
)


@dataclass(frozen=True)
class AxisCasePlan:
    """Offline evidence for one outbound and return pair."""

    direction: AxisDirection
    target_transform: np.ndarray
    outbound_ik: IKResult
    return_ik: Optional[IKResult]
    outbound_trajectory: Optional[TrajectoryResult]
    return_trajectory: Optional[TrajectoryResult]
    predicted_return_joint_error_rad: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class AxisSuitePlan:
    """Frozen anchor and all six deterministic direction plans."""

    anchor_joints: np.ndarray
    anchor_transform: np.ndarray
    cases: tuple[AxisCasePlan, ...]

    @property
    def passed(self) -> bool:
        """Return true only when every outbound and return pair is accepted."""
        return len(self.cases) == len(AXIS_DIRECTIONS) and all(
            case.accepted for case in self.cases
        )


REPORT_FIELDS = (
    "index",
    "direction",
    "distance_m",
    "anchor_joint1_rad",
    "anchor_joint2_rad",
    "anchor_joint3_rad",
    "anchor_joint4_rad",
    "anchor_joint5_rad",
    "anchor_joint6_rad",
    "anchor_joint7_rad",
    "accepted",
    "reason",
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
    "return_joint_error_rad",
    "outbound_final_position_error_m",
    "outbound_final_orientation_error_rad",
    "return_final_position_error_m",
    "return_final_orientation_error_rad",
)


class AxisReportWriter:
    """Rewrite one durable CSV/JSON pair as validation progresses."""

    def __init__(self, output_directory: Path, mode: str) -> None:
        if mode not in ("offline", "real"):
            raise ValueError("axis report mode must be offline or real")
        self.output_directory = Path(output_directory).expanduser()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{time.time_ns() % 1_000_000_000:09d}"
        stem = f"axis_suite_{mode}_{stamp}_{suffix}"
        self.csv_path = self.output_directory / f"{stem}.csv"
        self.json_path = self.output_directory / f"{stem}.json"
        self.mode = mode

    def write(
        self,
        rows: Sequence[dict],
        *,
        anchor_joints: Sequence[float],
        completed: bool,
        passed: bool,
        message: str,
    ) -> None:
        """Persist current rows atomically enough for supervised interruption."""
        complete_rows = [
            {field: row.get(field, "") for field in REPORT_FIELDS}
            for row in rows
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(complete_rows)
        summary = {
            "mode": self.mode,
            "distances_m": {
                direction.label: direction.distance_m
                for direction in AXIS_DIRECTIONS
            },
            "directions_total": len(AXIS_DIRECTIONS),
            "directions_recorded": len(complete_rows),
            "completed": bool(completed),
            "passed": bool(passed),
            "message": str(message),
            "anchor_joints_rad": [float(value) for value in anchor_joints],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "thresholds": {
                "max_joint_delta_rad": AXIS_MAX_JOINT_DELTA_RAD,
                "max_return_joint_error_rad": (
                    AXIS_MAX_RETURN_JOINT_ERROR_RAD
                ),
                "min_sigma": AXIS_MIN_SIGMA,
                "max_condition": AXIS_MAX_CONDITION,
            },
            "rows": complete_rows,
            "csv_path": str(self.csv_path),
            "json_path": str(self.json_path),
        }
        temporary = self.json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.json_path)


def default_urdf_path() -> Path:
    """Locate the installed NERO URDF."""
    share = Path(get_package_share_directory("agx_arm_description"))
    return (
        share
        / "agx_arm_urdf"
        / "nero"
        / "urdf"
        / "nero_description.urdf"
    )


def axis_target(
    anchor_transform: np.ndarray,
    direction: AxisDirection,
    distance_m: Optional[float] = None,
) -> np.ndarray:
    """Offset position along one axis of the frozen link7 orientation."""
    distance = (
        direction.distance_m
        if distance_m is None
        else float(distance_m)
    )
    if not math.isfinite(distance) or not math.isclose(
        distance,
        direction.distance_m,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "axis-suite displacement must match its locked direction profile"
        )
    anchor = np.asarray(anchor_transform, dtype=float)
    if anchor.shape != (4, 4) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor transform must be a finite 4 by 4 matrix")
    if direction not in AXIS_DIRECTIONS:
        raise ValueError("direction is not part of the fixed axis suite")
    target = anchor.copy()
    target[:3, 3] += (
        anchor[:3, direction.axis_index] * direction.sign * distance
    )
    return target


def _ik_refusal(result: IKResult, phase: str) -> Optional[str]:
    if not result.success or int(result.error_code) != IKErrorCode.SUCCESS:
        return f"{phase} IK 被拒绝：{result.message}"
    values = (
        result.position_error_m,
        result.orientation_error_rad,
        result.max_joint_delta_rad,
        result.sigma_min,
        result.condition_number,
    )
    if not all(math.isfinite(value) for value in values):
        return f"{phase} IK 诊断包含无效数值"
    if result.max_joint_delta_rad > AXIS_MAX_JOINT_DELTA_RAD:
        return f"{phase}关节变化超过 0.08 rad"
    if (
        result.sigma_min < AXIS_MIN_SIGMA
        or result.condition_number > AXIS_MAX_CONDITION
    ):
        return f"{phase}过于接近奇异点"
    return None


def plan_axis_case(
    solver: PlacoIKSolver,
    trajectory_generator: TrajectoryGenerator,
    anchor_joints: Sequence[float],
    direction: AxisDirection,
    anchor_transform: Optional[np.ndarray] = None,
) -> AxisCasePlan:
    """Plan one locked-direction pair from its own measured anchor."""
    anchor = np.asarray(anchor_joints, dtype=float)
    if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor joints must contain seven finite values")
    if direction not in AXIS_DIRECTIONS:
        raise ValueError("direction is not part of the fixed axis suite")
    if anchor_transform is None:
        anchor_transform = solver.forward_kinematics(anchor, "link7")
    anchor_transform = np.asarray(anchor_transform, dtype=float)
    target = axis_target(anchor_transform, direction)
    outbound = solver.solve(target, anchor, "link7")
    refusal = _ik_refusal(outbound, "去程")
    outbound_trajectory = None
    return_result = None
    return_trajectory = None
    return_joint_error = float("inf")
    if refusal is None:
        outbound_trajectory = trajectory_generator.generate(
            anchor, outbound.joint_positions
        )
        if not outbound_trajectory.success:
            refusal = f"去程轨迹被拒绝：{outbound_trajectory.message}"
    if refusal is None:
        return_result = solver.solve(
            anchor_transform,
            outbound.joint_positions,
            "link7",
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
        if return_joint_error > AXIS_MAX_RETURN_JOINT_ERROR_RAD:
            refusal = "预计回程关节误差超过 0.02 rad"
    return AxisCasePlan(
        direction=direction,
        target_transform=target,
        outbound_ik=outbound,
        return_ik=return_result,
        outbound_trajectory=outbound_trajectory,
        return_trajectory=return_trajectory,
        predicted_return_joint_error_rad=return_joint_error,
        accepted=refusal is None,
        reason="通过" if refusal is None else refusal,
    )


def plan_axis_suite(
    solver: PlacoIKSolver,
    trajectory_generator: TrajectoryGenerator,
    anchor_joints: Sequence[float],
) -> AxisSuitePlan:
    """Plan all six outbound/return pairs without sending robot commands."""
    anchor = np.asarray(anchor_joints, dtype=float)
    if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor joints must contain seven finite values")
    anchor_transform = solver.forward_kinematics(anchor, "link7")
    cases = tuple(
        plan_axis_case(
            solver,
            trajectory_generator,
            anchor,
            direction,
            anchor_transform,
        )
        for direction in AXIS_DIRECTIONS
    )
    return AxisSuitePlan(
        anchor_joints=anchor.copy(),
        anchor_transform=anchor_transform.copy(),
        cases=cases,
    )


def case_to_row(
    index: int,
    case: AxisCasePlan,
    anchor_joints: Optional[Sequence[float]] = None,
) -> dict:
    """Convert one offline plan into the shared report schema."""
    outbound_trajectory = case.outbound_trajectory
    return_ik = case.return_ik
    return_trajectory = case.return_trajectory
    row = {
        "index": index,
        "direction": case.direction.label,
        "distance_m": case.direction.distance_m,
        "accepted": case.accepted,
        "reason": case.reason,
        "outbound_solve_time_ms": case.outbound_ik.solve_time_ms,
        "outbound_position_error_m": case.outbound_ik.position_error_m,
        "outbound_orientation_error_rad": (
            case.outbound_ik.orientation_error_rad
        ),
        "outbound_max_joint_delta_rad": case.outbound_ik.max_joint_delta_rad,
        "outbound_sigma_min": case.outbound_ik.sigma_min,
        "outbound_condition_number": case.outbound_ik.condition_number,
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
            "" if return_ik is None else return_ik.solve_time_ms
        ),
        "return_position_error_m": (
            "" if return_ik is None else return_ik.position_error_m
        ),
        "return_orientation_error_rad": (
            "" if return_ik is None else return_ik.orientation_error_rad
        ),
        "return_max_joint_delta_rad": (
            "" if return_ik is None else return_ik.max_joint_delta_rad
        ),
        "return_sigma_min": (
            "" if return_ik is None else return_ik.sigma_min
        ),
        "return_condition_number": (
            "" if return_ik is None else return_ik.condition_number
        ),
        "return_trajectory_duration_s": (
            "" if return_trajectory is None else return_trajectory.duration_s
        ),
        "return_peak_velocity_rad_s": (
            "" if return_trajectory is None
            else return_trajectory.peak_velocity_rad_s
        ),
        "return_peak_acceleration_rad_s2": (
            "" if return_trajectory is None
            else return_trajectory.peak_acceleration_rad_s2
        ),
        "return_joint_error_rad": case.predicted_return_joint_error_rad,
    }
    if anchor_joints is not None:
        anchor = np.asarray(anchor_joints, dtype=float)
        if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
            raise ValueError("report anchor must contain seven finite values")
        for joint_index, value in enumerate(anchor, start=1):
            row[f"anchor_joint{joint_index}_rad"] = float(value)
    return row


def print_axis_suite(plan: AxisSuitePlan) -> None:
    """Print a compact Chinese table suitable for operator review."""
    print("\n=== Placo 六方向分级位移离线预检 ===")
    print(
        "方向  距离(mm)  结果  去程残差(mm)  去程Δq(rad)  "
        "回程残差(mm)  回点Δq(rad)"
    )
    for case in plan.cases:
        return_error = (
            float("nan")
            if case.return_ik is None
            else case.return_ik.position_error_m * 1000.0
        )
        print(
            f"{case.direction.label:>3}  "
            f"{case.direction.distance_m * 1000.0:>8.0f}  "
            f"{'通过' if case.accepted else '拒绝':<4}  "
            f"{case.outbound_ik.position_error_m * 1000.0:>11.3f}  "
            f"{case.outbound_ik.max_joint_delta_rad:>11.6f}  "
            f"{return_error:>11.3f}  "
            f"{case.predicted_return_joint_error_rad:>11.6f}"
        )
        if not case.accepted:
            print(f"     原因：{case.reason}")


def run_offline_axis_suite(
    urdf_path: Path,
    output_directory: Path,
    anchor_joints: Sequence[float],
) -> tuple[AxisSuitePlan, AxisReportWriter]:
    """Run six-direction planning and save CSV/JSON evidence."""
    solver = PlacoIKSolver(urdf_path)
    trajectory = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
    plan = plan_axis_suite(solver, trajectory, anchor_joints)
    rows = [
        case_to_row(index, case, plan.anchor_joints)
        for index, case in enumerate(plan.cases)
    ]
    writer = AxisReportWriter(output_directory, "offline")
    writer.write(
        rows,
        anchor_joints=plan.anchor_joints,
        completed=True,
        passed=plan.passed,
        message="六方向离线预检通过" if plan.passed else "六方向离线预检失败",
    )
    return plan, writer


def main(argv=None) -> None:
    """Run the pure-Python six-direction Placo and trajectory precheck."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path.home()
            / ".ros"
            / "strawberry_nero_control"
            / "axis_suite"
        ),
    )
    parser.add_argument(
        "--anchor-joints",
        type=float,
        nargs=7,
        default=READY_JOINT_POSITIONS,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    arguments = parser.parse_args(argv)
    plan, writer = run_offline_axis_suite(
        arguments.urdf or default_urdf_path(),
        arguments.output_dir,
        arguments.anchor_joints,
    )
    print_axis_suite(plan)
    print(f"\nCSV：{writer.csv_path}")
    print(f"JSON：{writer.json_path}")
    if not plan.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
