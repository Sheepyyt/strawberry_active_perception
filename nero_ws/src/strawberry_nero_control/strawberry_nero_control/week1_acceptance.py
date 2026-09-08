"""Formal Week-1 offline and supervised real-arm acceptance tools."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable, Optional, Sequence

import numpy as np
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
import rclpy

from strawberry_nero_interfaces.action import RecoverToSafe
from strawberry_nero_interfaces.msg import IKResult as IKResultMsg

from .acceptance_dataset import (
    ACCEPTANCE_DATASET_VERSION,
    ACCEPTANCE_MAX_JOINT_DELTA_RAD,
    ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD,
    ACCEPTANCE_REPEATS,
    ACCEPTANCE_TARGET_COUNT,
    ACCEPTANCE_TOTAL_TARGET_ATTEMPTS,
    AcceptanceSuitePlan,
    animate_acceptance_suite,
    plan_acceptance_suite,
    print_acceptance_suite,
    write_acceptance_target_set,
)
from .axis_suite import REPORT_FIELDS as AXIS_REPORT_FIELDS
from .offline_benchmark import (
    FORMAL_OFFLINE_SAMPLES,
    run_benchmark,
)
from .real_smoke_test import (
    ExecutionObservation,
    MeasuredState,
    NeroRealSmokeTest,
    POST_MOTION_SAFETY_NOTICE,
    SMOKE_MAX_FINAL_ORIENTATION_ERROR_RAD,
    SMOKE_MAX_FINAL_POSITION_ERROR_M,
    SmokeTestError,
    nero_urdf_path,
    require_confirmation,
    validate_smoke_ik,
)
from .ros_utils import matrix_to_pose_stamped, pose_error
from .trajectory import TrajectoryGenerator
from .validation_paths import (
    week1_runtime_directory,
    week1_validation_directory,
)


SESSION_SCHEMA_VERSION = 1
DEFAULT_BATCH_TARGETS = 5
MAX_BATCH_TARGETS = 5
ANCHOR_JOINT_TOLERANCE_RAD = 0.02
FORMAL_REAL_SUCCESS_RATE = 0.95
FORMAL_POSITION_P95_M = 0.010
FORMAL_ORIENTATION_P95_RAD = math.radians(5.0)
FORMAL_RESPONSE_LATENCY_P95_S = 0.200
FORMAL_MIN_TARGET_ORIENTATION_RAD = math.radians(3.0)
FORMAL_MIN_DATASET_MAX_POSITION_M = 0.030
FORMAL_MIN_DATASET_MAX_ORIENTATION_RAD = math.radians(10.0)


REAL_REPORT_FIELDS = (
    "attempt_index",
    "target_index",
    "target_id",
    "repetition",
    "commanded_position_offset_m",
    "commanded_orientation_offset_rad",
    "target_success",
    "target_code",
    "target_reason",
    "target_solve_time_ms",
    "target_position_error_m",
    "target_orientation_error_rad",
    "target_sigma_min",
    "target_condition_number",
    "target_max_joint_delta_rad",
    "target_response_latency_s",
    "target_motion_start_detected",
    "target_total_duration_s",
    "return_success",
    "return_code",
    "return_reason",
    "return_solve_time_ms",
    "return_position_error_m",
    "return_orientation_error_rad",
    "return_sigma_min",
    "return_condition_number",
    "return_max_joint_delta_rad",
    "return_response_latency_s",
    "return_motion_start_detected",
    "return_total_duration_s",
    "return_joint_error_rad",
    "return_recovered_after_stop",
    "return_recovered_at",
    "attempt_accepted",
    "stop_stage",
    "recorded_at",
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite), percentile))


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def create_real_session(
    plan: AcceptanceSuitePlan,
    runtime_directory: Path,
) -> Path:
    """Create one hidden, resumable session after a complete no-motion preview."""
    if not plan.passed:
        raise ValueError("cannot create a real session from a failed target set")
    directory = Path(runtime_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{time.time_ns() % 1_000_000_000:09d}"
    path = directory / f"week1_real_{stamp}_{suffix}.json"
    payload = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "dataset_version": ACCEPTANCE_DATASET_VERSION,
        "dataset_id": plan.dataset_id,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "status": "previewed",
        "anchor_joints_rad": plan.anchor_joints.tolist(),
        "anchor_transform": plan.anchor_transform.tolist(),
        "targets": [
            {
                "index": target.index,
                "target_id": target.target_id,
                "target_transform": target.target_transform.tolist(),
                "reference_joints_rad": target.reference_joints.tolist(),
                "position_offset_m": target.position_offset_m,
                "orientation_offset_rad": target.orientation_offset_rad,
            }
            for target in plan.targets
        ],
        "target_count": ACCEPTANCE_TARGET_COUNT,
        "repeats_per_target": ACCEPTANCE_REPEATS,
        "required_target_attempts": ACCEPTANCE_TOTAL_TARGET_ATTEMPTS,
        "next_attempt_index": 0,
        "blocked": False,
        "blocked_reason": "",
        "rows": [],
        "operator_safe_batches": [],
    }
    _atomic_write_json(path, payload)
    return path


def load_real_session(path: Path) -> dict:
    """Load and structurally validate one resumable session file."""
    session_path = Path(path).expanduser().resolve()
    if not session_path.is_file():
        raise SmokeTestError(f"验收会话不存在：{session_path}")
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeTestError(f"无法读取验收会话：{error}") from error
    if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise SmokeTestError("验收会话版本不受支持")
    if payload.get("dataset_version") != ACCEPTANCE_DATASET_VERSION:
        raise SmokeTestError("验收会话的数据集版本不匹配")
    if payload.get("target_count") != ACCEPTANCE_TARGET_COUNT:
        raise SmokeTestError("验收会话不是完整 30 目标数据集")
    if payload.get("repeats_per_target") != ACCEPTANCE_REPEATS:
        raise SmokeTestError("验收会话不是每目标 3 次")
    targets = payload.get("targets")
    rows = payload.get("rows")
    if not isinstance(targets, list) or len(targets) != ACCEPTANCE_TARGET_COUNT:
        raise SmokeTestError("验收会话目标列表损坏")
    if not isinstance(rows, list):
        raise SmokeTestError("验收会话结果列表损坏")
    next_attempt = payload.get("next_attempt_index")
    if (
        not isinstance(next_attempt, int)
        or next_attempt < 0
        or next_attempt > ACCEPTANCE_TOTAL_TARGET_ATTEMPTS
    ):
        raise SmokeTestError("验收会话进度值无效")
    if len(rows) != next_attempt:
        raise SmokeTestError("验收会话进度与结果行数不一致")
    anchor = np.asarray(payload.get("anchor_joints_rad"), dtype=float)
    transform = np.asarray(payload.get("anchor_transform"), dtype=float)
    if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
        raise SmokeTestError("验收会话锚点关节无效")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SmokeTestError("验收会话锚点位姿无效")
    payload["_path"] = str(session_path)
    return payload


def save_real_session(session: dict) -> None:
    """Checkpoint progress without copying intermediate data into Git."""
    path = Path(session["_path"])
    durable = {
        key: value for key, value in session.items()
        if not key.startswith("_")
    }
    durable["updated_at"] = _timestamp()
    _atomic_write_json(path, durable)


def _target_transform(session: dict, target_index: int) -> np.ndarray:
    transform = np.asarray(
        session["targets"][target_index]["target_transform"],
        dtype=float,
    )
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SmokeTestError(f"P{target_index + 1:02d} 位姿数据无效")
    return transform


def _result_fields(
    prefix: str,
    result,
    observation: Optional[ExecutionObservation],
) -> dict:
    ik = result.ik_result
    return {
        f"{prefix}_success": bool(
            ik.success and ik.code == IKResultMsg.SUCCESS
        ),
        f"{prefix}_code": int(ik.code),
        f"{prefix}_reason": str(ik.reason),
        f"{prefix}_solve_time_ms": float(ik.solve_time_ms),
        f"{prefix}_position_error_m": float(
            result.final_position_error_m
        ),
        f"{prefix}_orientation_error_rad": float(
            result.final_orientation_error_rad
        ),
        f"{prefix}_sigma_min": float(ik.sigma_min),
        f"{prefix}_condition_number": float(ik.condition_number),
        f"{prefix}_max_joint_delta_rad": float(ik.max_joint_delta_rad),
        f"{prefix}_response_latency_s": (
            "" if observation is None
            or observation.response_latency_s is None
            else float(observation.response_latency_s)
        ),
        f"{prefix}_motion_start_detected": bool(
            observation is not None and observation.motion_start_detected
        ),
        f"{prefix}_total_duration_s": (
            "" if observation is None
            else float(observation.total_duration_s)
        ),
    }


def _empty_attempt_row(attempt_index: int) -> dict:
    target_index, repetition_index = divmod(
        attempt_index,
        ACCEPTANCE_REPEATS,
    )
    row = {field: "" for field in REAL_REPORT_FIELDS}
    row.update({
        "attempt_index": attempt_index,
        "target_index": target_index,
        "target_id": f"P{target_index + 1:02d}",
        "repetition": repetition_index + 1,
        "target_success": False,
        "return_success": False,
        "return_recovered_after_stop": False,
        "attempt_accepted": False,
        "recorded_at": _timestamp(),
    })
    return row


def _pending_return_row(session: dict) -> Optional[dict]:
    """Return the last partially completed target that still needs a return."""
    rows = session.get("rows", [])
    if not session.get("blocked") or not rows:
        return None
    row = rows[-1]
    if (
        bool(row.get("target_success"))
        and not bool(row.get("return_success"))
        and row.get("stop_stage") in ("return_precheck", "return_execution")
    ):
        return row
    return None


def _record_recovered_return(
    session: dict,
    return_fields: dict,
    final_positions: Sequence[float],
) -> bool:
    """Complete a stopped target row after an explicit return-to-anchor."""
    row = _pending_return_row(session)
    if row is None:
        return False
    anchor = np.asarray(session["anchor_joints_rad"], dtype=float)
    final = np.asarray(final_positions, dtype=float)
    return_error = float(np.max(np.abs(final - anchor)))
    if return_error > ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD:
        raise SmokeTestError(
            "补做回程后的关节误差超过 0.02 rad："
            f"{return_error:.6f} rad"
        )
    row.update(return_fields)
    row["return_joint_error_rad"] = return_error
    row["return_recovered_after_stop"] = True
    row["return_recovered_at"] = _timestamp()
    row["attempt_accepted"] = bool(
        row.get("target_success") and row.get("return_success")
    )
    row["stop_stage"] = "return_recovered_after_stop"
    session.setdefault("recovery_events", []).append({
        "attempt_index": int(row["attempt_index"]),
        "target_id": str(row["target_id"]),
        "repetition": int(row["repetition"]),
        "event": "explicit_return_to_anchor_completed",
        "at": row["return_recovered_at"],
        "return_joint_error_rad": return_error,
    })
    session["blocked_reason"] = (
        "回程已显式补做并返回锚点；继续前仍需使用 "
        "--resume-after-stop 确认已检查停止原因"
    )
    save_real_session(session)
    return True


def _numeric_rows(rows: Sequence[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value in (None, ""):
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def write_final_real_report(
    session: dict,
    output_directory: Path,
) -> tuple[Path, Path, dict]:
    """Promote only one completed session into the visible project results."""
    rows = session["rows"]
    if len(rows) != ACCEPTANCE_TOTAL_TARGET_ATTEMPTS:
        raise SmokeTestError("只有完成 90 个目标尝试后才能生成正式报告")
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "real_30x3.csv"
    json_path = directory / "real_30x3.json"

    complete_rows = [
        {field: row.get(field, "") for field in REAL_REPORT_FIELDS}
        for row in rows
    ]
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REAL_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(complete_rows)
    temporary_csv.replace(csv_path)

    target_successes = sum(bool(row.get("target_success")) for row in rows)
    return_successes = sum(bool(row.get("return_success")) for row in rows)
    success_rate = target_successes / ACCEPTANCE_TOTAL_TARGET_ATTEMPTS
    position_p95 = _percentile(
        _numeric_rows(rows, "target_position_error_m"), 95.0
    )
    orientation_p95 = _percentile(
        _numeric_rows(rows, "target_orientation_error_rad"), 95.0
    )
    latency_values = _numeric_rows(rows, "target_response_latency_s")
    latency_p95 = _percentile(latency_values, 95.0)
    solve_p95 = _percentile(
        _numeric_rows(rows, "target_solve_time_ms"), 95.0
    )
    safe_batches = session.get("operator_safe_batches", [])
    all_batches_safe = bool(safe_batches) and all(
        bool(batch.get("no_collision_confirmed"))
        for batch in safe_batches
    )
    reported_codes = [
        int(value)
        for field in ("target_code", "return_code")
        for value in (row.get(field) for row in rows)
        if value not in (None, "")
    ]
    joint_limit_failures = sum(
        code == IKResultMsg.JOINT_LIMIT_VIOLATION
        for code in reported_codes
    )
    position_offsets = [
        float(target["position_offset_m"])
        for target in session["targets"]
    ]
    orientation_offsets = [
        float(target["orientation_offset_rad"])
        for target in session["targets"]
    ]
    acceptance = {
        "exactly_30_targets_x_3_attempts": (
            len(rows) == ACCEPTANCE_TOTAL_TARGET_ATTEMPTS
        ),
        "target_success_rate_at_least_95_percent": (
            success_rate >= FORMAL_REAL_SUCCESS_RATE
        ),
        "position_error_p95_at_most_10_mm": (
            position_p95 is not None
            and position_p95 <= FORMAL_POSITION_P95_M
        ),
        "orientation_error_p95_at_most_5_deg": (
            orientation_p95 is not None
            and orientation_p95 <= FORMAL_ORIENTATION_P95_RAD
        ),
        "response_latency_p95_at_most_200_ms": (
            len(latency_values) == target_successes
            and latency_p95 is not None
            and latency_p95 <= FORMAL_RESPONSE_LATENCY_P95_S
        ),
        "all_anchor_returns_succeeded": (
            return_successes == ACCEPTANCE_TOTAL_TARGET_ATTEMPTS
        ),
        "operator_confirmed_zero_collisions_for_every_batch": (
            all_batches_safe
        ),
        "zero_reported_joint_limit_failures": joint_limit_failures == 0,
        "session_not_blocked": not bool(session.get("blocked")),
        "all_targets_change_orientation_by_at_least_3_deg": (
            len(orientation_offsets) == ACCEPTANCE_TARGET_COUNT
            and min(orientation_offsets) >= FORMAL_MIN_TARGET_ORIENTATION_RAD
        ),
        "dataset_reaches_at_least_30_mm_translation": (
            max(position_offsets) >= FORMAL_MIN_DATASET_MAX_POSITION_M
        ),
        "dataset_reaches_at_least_10_deg_orientation": (
            max(orientation_offsets)
            >= FORMAL_MIN_DATASET_MAX_ORIENTATION_RAD
        ),
    }
    summary = {
        "schema_version": 1,
        "benchmark": "week1_real_30_targets_x_3",
        "generated_at": _timestamp(),
        "dataset_version": session["dataset_version"],
        "dataset_id": session["dataset_id"],
        "anchor_joints_rad": session["anchor_joints_rad"],
        "anchor_transform": session["anchor_transform"],
        "target_definitions": session["targets"],
        "targets": ACCEPTANCE_TARGET_COUNT,
        "repeats_per_target": ACCEPTANCE_REPEATS,
        "target_attempts": len(rows),
        "target_successes": target_successes,
        "target_success_rate": success_rate,
        "return_successes": return_successes,
        "ik_time_p95_ms": solve_p95,
        "model_position_error_p95_m": position_p95,
        "model_orientation_error_p95_rad": orientation_p95,
        "response_latency_p95_s": latency_p95,
        "response_latency_measurements": len(latency_values),
        "commanded_position_offset_range_m": [
            min(position_offsets),
            max(position_offsets),
        ],
        "commanded_orientation_offset_range_rad": [
            min(orientation_offsets),
            max(orientation_offsets),
        ],
        "joint_limit_failure_count": joint_limit_failures,
        "operator_safe_batches": safe_batches,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "artifacts": {
            "details_csv": csv_path.name,
            "summary_json": json_path.name,
        },
        "important_note": (
            "Cartesian errors use encoder feedback plus the same URDF/FK. "
            "They do not validate absolute physical camera accuracy. "
            "Collision status is an explicit operator observation."
        ),
    }
    _atomic_write_json(json_path, summary)
    return csv_path, json_path, summary


def promote_axis_result(source_json: Path, output_directory: Path) -> tuple[Path, Path]:
    """Copy only one completed/passed six-axis result into visible evidence."""
    source = Path(source_json).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read axis result: {error}") from error
    if not payload.get("completed") or not payload.get("passed"):
        raise ValueError("axis result must be completed and passed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("axis result must contain six completed directions")
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "axis_15mm.csv"
    json_path = directory / "axis_15mm.json"
    complete_rows = [
        {field: row.get(field, "") for field in AXIS_REPORT_FIELDS}
        for row in rows
    ]
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AXIS_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(complete_rows)
    temporary_csv.replace(csv_path)
    durable = {
        key: value for key, value in payload.items()
        if key not in ("csv_path", "json_path")
    }
    durable["promoted_at"] = _timestamp()
    durable["artifacts"] = {
        "details_csv": csv_path.name,
        "summary_json": json_path.name,
    }
    _atomic_write_json(json_path, durable)
    return csv_path, json_path


class Week1AcceptanceNode(NeroRealSmokeTest):
    """ROS client for previewed, batched and resumable formal validation."""

    def __init__(self) -> None:
        super().__init__("nero_week1_acceptance")

    @staticmethod
    def _require_already_safe(result: RecoverToSafe.Result) -> None:
        if not (
            result.success
            and result.code == RecoverToSafe.Result.ALREADY_SAFE
            and not result.executed
            and result.robot_is_safe
        ):
            raise SmokeTestError(
                "恢复预览没有确认 ALREADY_SAFE；请先完成 recover"
            )

    def _require_ready_neighborhood(self, state: MeasuredState) -> None:
        from .models import READY_JOINT_POSITIONS

        error = float(np.max(np.abs(
            state.positions - np.asarray(READY_JOINT_POSITIONS, dtype=float)
        )))
        if error > 0.05:
            raise SmokeTestError(
                "正式验收必须从 ready 中心邻域开始；"
                f"当前最大关节差={error:.6f} rad，请先运行 center-ready"
            )

    def _plan_for_session(self, session: dict) -> AcceptanceSuitePlan:
        anchor = np.asarray(session["anchor_joints_rad"], dtype=float)
        trajectory = TrajectoryGenerator(
            joint_limits=self._solver.safe_joint_limits
        )
        plan = plan_acceptance_suite(self._solver, trajectory, anchor)
        if plan.dataset_id != session["dataset_id"]:
            raise SmokeTestError("会话目标数据哈希不匹配，拒绝执行")
        return plan

    def preview_real_session(self, runtime_directory: Path) -> Path:
        """Check all 30 targets through direct Placo and ROS SolveIK only."""
        safe = self.preview_recovery()
        self._require_already_safe(safe)
        state = self.measured_state()
        self._require_ready_neighborhood(state)
        trajectory = TrajectoryGenerator(
            joint_limits=self._solver.safe_joint_limits
        )
        plan = plan_acceptance_suite(
            self._solver,
            trajectory,
            state.positions,
        )
        print_acceptance_suite(plan)
        if not plan.passed:
            raise SmokeTestError("30 目标直接 Placo/轨迹预检未全部通过")

        stamp = self.get_clock().now().to_msg()
        print("\n=== 30 目标 ROS 2 SolveIK 复核（仍不运动） ===")
        for target in plan.targets:
            target_pose = matrix_to_pose_stamped(
                target.target_transform,
                "base_link",
                stamp,
            )
            result = self._solve_pose_preview(
                target_pose,
                f"{target.target_id} ROS SolveIK 预览",
            )
            validate_smoke_ik(result, ACCEPTANCE_MAX_JOINT_DELTA_RAD)
            if result.max_joint_delta_rad > ACCEPTANCE_MAX_JOINT_DELTA_RAD:
                raise SmokeTestError(
                    f"{target.target_id} ROS 解关节变化超过 0.12 rad"
                )
            direct = np.asarray(target.outbound_ik.joint_positions)
            service = np.asarray(result.solution_joint_state.position)
            difference = float(np.max(np.abs(direct - service)))
            if difference > ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD:
                raise SmokeTestError(
                    f"{target.target_id} 直接解与 ROS 解相差 "
                    f"{difference:.6f} rad"
                )
            print(
                f"{target.target_id}: 通过，"
                f"残差={result.position_error_m * 1000.0:.3f} mm / "
                f"{math.degrees(result.orientation_error_rad):.4f}°，"
                f"IK={result.solve_time_ms:.3f} ms，"
                f"Δq={result.max_joint_delta_rad:.6f} rad"
            )
        path = create_real_session(plan, runtime_directory)
        print(f"\n预检会话（中间断点，不纳入 Git）：{path}")
        print("30 个目标全部通过，未打开软件门，未发送真机运动命令。")
        return path

    def _require_session_anchor(
        self,
        session: dict,
        state: MeasuredState,
    ) -> None:
        anchor_joints = np.asarray(session["anchor_joints_rad"], dtype=float)
        joint_error = float(np.max(np.abs(
            state.positions - anchor_joints
        )))
        current_pose = self._solver.forward_kinematics(
            state.positions,
            "link7",
        )
        anchor_pose = np.asarray(session["anchor_transform"], dtype=float)
        position_error, orientation_error = pose_error(
            anchor_pose,
            current_pose,
        )
        if (
            joint_error > ANCHOR_JOINT_TOLERANCE_RAD
            or position_error > SMOKE_MAX_FINAL_POSITION_ERROR_M
            or orientation_error > SMOKE_MAX_FINAL_ORIENTATION_ERROR_RAD
        ):
            raise SmokeTestError(
                "机械臂不在该会话锚点："
                f"关节差={joint_error:.6f} rad，"
                f"位姿差={position_error * 1000.0:.3f} mm / "
                f"{math.degrees(orientation_error):.3f}°。"
                "请先用 return-anchor 预览/返回，或重新创建会话。"
            )

    def preview_batch(
        self,
        session: dict,
        batch_targets: int,
    ) -> tuple[int, int]:
        """Preview only the next bounded group and return attempt bounds."""
        safe = self.preview_recovery()
        self._require_already_safe(safe)
        state = self.measured_state()
        self._require_session_anchor(session, state)
        self._plan_for_session(session)
        start_attempt = int(session["next_attempt_index"])
        if start_attempt >= ACCEPTANCE_TOTAL_TARGET_ATTEMPTS:
            print("会话的 90 个目标尝试已经全部完成。")
            return start_attempt, start_attempt
        start_target = start_attempt // ACCEPTANCE_REPEATS
        end_target = min(
            ACCEPTANCE_TARGET_COUNT,
            start_target + batch_targets,
        )
        end_attempt = end_target * ACCEPTANCE_REPEATS
        stamp = self.get_clock().now().to_msg()
        print(
            "\n=== 下一批正式验收预览（不运动） ===\n"
            f"尝试 {start_attempt + 1}–{end_attempt}/"
            f"{ACCEPTANCE_TOTAL_TARGET_ATTEMPTS}，"
            f"目标 P{start_target + 1:02d}–P{end_target:02d}"
        )
        for target_index in range(start_target, end_target):
            definition = session["targets"][target_index]
            transform = _target_transform(session, target_index)
            target_pose = matrix_to_pose_stamped(
                transform,
                "base_link",
                stamp,
            )
            result = self._solve_pose_preview(
                target_pose,
                f"P{target_index + 1:02d} 批次预览",
            )
            validate_smoke_ik(result, ACCEPTANCE_MAX_JOINT_DELTA_RAD)
            if result.max_joint_delta_rad > ACCEPTANCE_MAX_JOINT_DELTA_RAD:
                raise SmokeTestError(
                    f"P{target_index + 1:02d} 关节变化超过 0.12 rad"
                )
            print(
                f"P{target_index + 1:02d}: "
                f"请求变化={float(definition['position_offset_m']) * 1000.0:.1f} mm / "
                f"{math.degrees(float(definition['orientation_offset_rad'])):.1f}°，"
                f"{result.position_error_m * 1000.0:.3f} mm / "
                f"{math.degrees(result.orientation_error_rad):.4f}°，"
                f"Δq={result.max_joint_delta_rad:.6f} rad，"
                f"sigma={result.sigma_min:.4f}"
            )
        print("本次只是预览，没有打开软件门或发送运动命令。")
        return start_attempt, end_attempt

    def execute_batch(
        self,
        session: dict,
        start_attempt: int,
        end_attempt: int,
        input_function: Callable[[str], str],
        allow_resume_after_stop: bool,
        visible_output_directory: Path,
    ) -> Optional[dict]:
        """Execute one prechecked batch, checkpointing every target/return pair."""
        if start_attempt == end_attempt:
            if len(session["rows"]) == ACCEPTANCE_TOTAL_TARGET_ATTEMPTS:
                _, _, summary = write_final_real_report(
                    session,
                    visible_output_directory,
                )
                if session.get("blocked"):
                    raise SmokeTestError(
                        "90 次尝试已经结束，但最后状态被标记为中止；"
                        "已保存未通过的最终报告，不能补写安全确认。"
                    )
                return summary
            return None
        if session.get("blocked") and not allow_resume_after_stop:
            raise SmokeTestError(
                "上次批次中途停止。检查原因并回到会话锚点后，"
                "必须显式添加 --resume-after-stop 才能跳过失败尝试继续。"
            )
        if session.get("blocked"):
            session["blocked"] = False
            session["blocked_reason"] = ""
            session.setdefault("resume_acknowledgements", []).append({
                "at": _timestamp(),
                "next_attempt_index": start_attempt,
            })
            save_real_session(session)

        require_confirmation("RUN_BATCH", input_function)
        anchor_transform = np.asarray(
            session["anchor_transform"], dtype=float
        )
        anchor_joints = np.asarray(
            session["anchor_joints_rad"], dtype=float
        )
        for attempt_index in range(start_attempt, end_attempt):
            row = _empty_attempt_row(attempt_index)
            target_index = int(row["target_index"])
            target_id = str(row["target_id"])
            repetition = int(row["repetition"])
            definition = session["targets"][target_index]
            row["commanded_position_offset_m"] = float(
                definition["position_offset_m"]
            )
            row["commanded_orientation_offset_rad"] = float(
                definition["orientation_offset_rad"]
            )
            stage = "target_precheck"
            try:
                start_state = self.measured_state()
                self._require_session_anchor(session, start_state)
                target_transform = _target_transform(session, target_index)
                target_pose = matrix_to_pose_stamped(
                    target_transform,
                    "base_link",
                    self.get_clock().now().to_msg(),
                )
                target_preview = self._solve_pose_preview(
                    target_pose,
                    f"{target_id} 第 {repetition}/3 次实时预览",
                )
                validate_smoke_ik(
                    target_preview,
                    ACCEPTANCE_MAX_JOINT_DELTA_RAD,
                )
                if (
                    target_preview.max_joint_delta_rad
                    > ACCEPTANCE_MAX_JOINT_DELTA_RAD
                ):
                    raise SmokeTestError("实时目标关节变化超过 0.12 rad")
                print(
                    f"\n=== {target_id} 第 {repetition}/3 次：去程 ==="
                )
                stage = "target_execution"
                target_result = self.execute_placo(
                    target_pose,
                    start_state,
                    precision_test=True,
                    label=f"正式验收 {target_id} 第 {repetition}/3 次去程",
                )
                row.update(_result_fields(
                    "target",
                    target_result,
                    self.last_execution_observation,
                ))

                stage = "return_precheck"
                return_state = self.measured_state()
                anchor_pose = matrix_to_pose_stamped(
                    anchor_transform,
                    "base_link",
                    self.get_clock().now().to_msg(),
                )
                return_preview = self._solve_pose_preview(
                    anchor_pose,
                    f"{target_id} 第 {repetition}/3 次回程预览",
                    posture_reference=anchor_joints,
                )
                validate_smoke_ik(
                    return_preview,
                    ACCEPTANCE_MAX_JOINT_DELTA_RAD,
                )
                if (
                    return_preview.max_joint_delta_rad
                    > ACCEPTANCE_MAX_JOINT_DELTA_RAD
                ):
                    raise SmokeTestError("实时回程关节变化超过 0.12 rad")
                print(
                    f"\n=== {target_id} 第 {repetition}/3 次：回程 ==="
                )
                stage = "return_execution"
                return_result = self.execute_placo(
                    anchor_pose,
                    return_state,
                    precision_test=True,
                    label=f"正式验收 {target_id} 第 {repetition}/3 次回程",
                    posture_reference=anchor_joints,
                )
                row.update(_result_fields(
                    "return",
                    return_result,
                    self.last_execution_observation,
                ))
                final_state = self.measured_state()
                return_joint_error = float(np.max(np.abs(
                    final_state.positions
                    - np.asarray(session["anchor_joints_rad"], dtype=float)
                )))
                row["return_joint_error_rad"] = return_joint_error
                if return_joint_error > ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD:
                    raise SmokeTestError(
                        "回到锚点后的关节误差超过 0.02 rad："
                        f"{return_joint_error:.6f} rad"
                    )
                row["attempt_accepted"] = bool(
                    row["target_success"] and row["return_success"]
                )
                row["stop_stage"] = ""
                print(
                    f"{target_id} 第 {repetition}/3 次完成，"
                    f"回点关节误差={return_joint_error:.6f} rad"
                )
            except BaseException as error:
                row["attempt_accepted"] = False
                row["stop_stage"] = stage
                detail = str(error).strip() or type(error).__name__
                if not row.get("target_reason"):
                    row["target_reason"] = detail
                session["rows"].append(row)
                session["next_attempt_index"] = attempt_index + 1
                session["blocked"] = True
                session["blocked_reason"] = (
                    f"{target_id} repetition {repetition} stopped at "
                    f"{stage}: {detail}"
                )
                session["status"] = "blocked"
                save_real_session(session)
                print(
                    f"\n中途断点已保存：{session['_path']}",
                    file=sys.stderr,
                )
                raise

            session["rows"].append(row)
            session["next_attempt_index"] = attempt_index + 1
            session["status"] = "running"
            save_real_session(session)

        try:
            require_confirmation("BATCH_SAFE", input_function)
        except BaseException as error:
            detail = str(error).strip() or type(error).__name__
            session["blocked"] = True
            session["blocked_reason"] = (
                "operator did not confirm the completed batch was collision free: "
                f"{detail}"
            )
            session["status"] = "blocked"
            save_real_session(session)
            raise
        session["operator_safe_batches"].append({
            "start_attempt": start_attempt,
            "end_attempt_exclusive": end_attempt,
            "no_collision_confirmed": True,
            "confirmed_at": _timestamp(),
        })
        completed = end_attempt == ACCEPTANCE_TOTAL_TARGET_ATTEMPTS
        session["status"] = "completed" if completed else "running"
        save_real_session(session)
        if not completed:
            print(
                f"\n本批完成并已确认无碰撞。总进度：{end_attempt}/"
                f"{ACCEPTANCE_TOTAL_TARGET_ATTEMPTS}。"
            )
            return None

        csv_path, json_path, summary = write_final_real_report(
            session,
            visible_output_directory,
        )
        print(f"\n第一周真机最终 CSV：{csv_path}")
        print(f"第一周真机最终 JSON：{json_path}")
        print(f"正式真机验收：{'通过' if summary['passed'] else '未通过'}")
        return summary

    def return_to_session_anchor(
        self,
        session: dict,
        execute: bool,
        input_function: Callable[[str], str],
    ) -> None:
        """Preview or explicitly execute a small return to a session anchor."""
        safe = self.preview_recovery()
        self._require_already_safe(safe)
        state = self.measured_state()
        anchor_transform = np.asarray(
            session["anchor_transform"], dtype=float
        )
        anchor_joints = np.asarray(
            session["anchor_joints_rad"], dtype=float
        )
        anchor_pose = matrix_to_pose_stamped(
            anchor_transform,
            "base_link",
            self.get_clock().now().to_msg(),
        )
        result = self._solve_pose_preview(
            anchor_pose,
            "返回会话锚点预览",
            posture_reference=anchor_joints,
        )
        if result.code == IKResultMsg.ALREADY_AT_TARGET and result.success:
            final_state = self.measured_state()
            self._require_session_anchor(session, final_state)
            repaired = _record_recovered_return(
                session,
                {
                    "return_success": True,
                    "return_code": int(result.code),
                    "return_reason": "补做回程时已经位于会话锚点",
                    "return_solve_time_ms": float(result.solve_time_ms),
                    "return_position_error_m": float(
                        result.position_error_m
                    ),
                    "return_orientation_error_rad": float(
                        result.orientation_error_rad
                    ),
                    "return_sigma_min": float(result.sigma_min),
                    "return_condition_number": float(
                        result.condition_number
                    ),
                    "return_max_joint_delta_rad": float(
                        result.max_joint_delta_rad
                    ),
                    "return_response_latency_s": "",
                    "return_motion_start_detected": False,
                    "return_total_duration_s": 0.0,
                },
                final_state.positions,
            )
            print("已经位于会话锚点死区内，没有发送运动命令。")
            if repaired:
                print("上次中断的回程记录已补全。")
            return
        validate_smoke_ik(result, ACCEPTANCE_MAX_JOINT_DELTA_RAD)
        if result.max_joint_delta_rad > ACCEPTANCE_MAX_JOINT_DELTA_RAD:
            raise SmokeTestError("返回会话锚点需要超过 0.12 rad，拒绝执行")
        print(
            "\n=== 返回会话锚点预览 ===\n"
            f"残差={result.position_error_m * 1000.0:.3f} mm / "
            f"{math.degrees(result.orientation_error_rad):.4f}°，"
            f"最大Δq={result.max_joint_delta_rad:.6f} rad"
        )
        if not execute:
            print("预览完成：没有打开软件门或发送运动命令。")
            return
        require_confirmation("RETURN_ANCHOR", input_function)
        execution_result = self.execute_placo(
            anchor_pose,
            state,
            precision_test=True,
            label="正式验收返回会话锚点",
            posture_reference=anchor_joints,
        )
        final_state = self.measured_state()
        self._require_session_anchor(session, final_state)
        repaired = _record_recovered_return(
            session,
            _result_fields(
                "return",
                execution_result,
                self.last_execution_observation,
            ),
            final_state.positions,
        )
        print("已稳定返回会话锚点；现在可以显式恢复下一批。")
        if repaired:
            print("上次中断的回程记录已补全，并保留了恢复标记。")


def _bounded_batch_targets(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed > MAX_BATCH_TARGETS:
        raise argparse.ArgumentTypeError(
            f"batch-targets must be between 1 and {MAX_BATCH_TARGETS}"
        )
    return parsed


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser(
        "offline-100",
        help="run the deterministic 100-target IK and trajectory acceptance",
    )
    offline.add_argument("--urdf", type=Path, default=None)
    offline.add_argument("--seed", type=int, default=20260806)
    offline.add_argument("--joint-range", type=float, default=0.15)
    offline.add_argument("--output-dir", type=Path, default=None)

    targets = subparsers.add_parser(
        "targets-30",
        help="generate and save the deterministic 30-pose real target set",
    )
    targets.add_argument("--urdf", type=Path, default=None)
    targets.add_argument("--output-dir", type=Path, default=None)
    targets.add_argument(
        "--meshcat",
        action="store_true",
        help="play all 30 paths in a CAN-free MeshCat model",
    )
    targets.add_argument("--playback-rate", type=float, default=2.0)

    archive = subparsers.add_parser(
        "promote-axis",
        help="promote one completed six-axis JSON into visible project evidence",
    )
    archive.add_argument("--source-json", type=Path, required=True)
    archive.add_argument("--output-dir", type=Path, default=None)

    preview = subparsers.add_parser(
        "real-preview",
        help="preview all 30 targets through live ROS SolveIK; never move",
    )
    preview.add_argument("--runtime-dir", type=Path, default=None)

    batch = subparsers.add_parser(
        "real-batch",
        help="preview or execute at most five targets x three repetitions",
    )
    batch.add_argument("--session", type=Path, required=True)
    batch.add_argument(
        "--batch-targets",
        type=_bounded_batch_targets,
        default=DEFAULT_BATCH_TARGETS,
    )
    batch.add_argument("--execute", action="store_true")
    batch.add_argument("--resume-after-stop", action="store_true")
    batch.add_argument("--output-dir", type=Path, default=None)

    return_anchor = subparsers.add_parser(
        "return-anchor",
        help="preview or execute a small return to one saved session anchor",
    )
    return_anchor.add_argument("--session", type=Path, required=True)
    return_anchor.add_argument("--execute", action="store_true")
    return parser


def _run_without_ros(arguments) -> Optional[int]:
    visible_root = week1_validation_directory()
    if arguments.command == "offline-100":
        output = arguments.output_dir or visible_root / "offline"
        summary = run_benchmark(
            arguments.urdf or nero_urdf_path(),
            output,
            FORMAL_OFFLINE_SAMPLES,
            arguments.seed,
            arguments.joint_range,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["passed"] else 1
    if arguments.command == "targets-30":
        solver_path = arguments.urdf or nero_urdf_path()
        from .ik_core import PlacoIKSolver

        solver = PlacoIKSolver(solver_path)
        trajectory = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
        plan = plan_acceptance_suite(solver, trajectory)
        print_acceptance_suite(plan)
        output = arguments.output_dir or visible_root / "targets"
        csv_path, json_path = write_acceptance_target_set(plan, output)
        print(f"\n30 目标 CSV：{csv_path}")
        print(f"30 目标 JSON：{json_path}")
        if arguments.meshcat:
            animate_acceptance_suite(
                plan,
                solver_path,
                arguments.playback_rate,
            )
        return 0 if plan.passed else 1
    if arguments.command == "promote-axis":
        output = arguments.output_dir or visible_root / "smoke"
        csv_path, json_path = promote_axis_result(
            arguments.source_json,
            output,
        )
        print(f"六方向最终 CSV：{csv_path}")
        print(f"六方向最终 JSON：{json_path}")
        return 0
    return None


def run_ros_command(
    node: Week1AcceptanceNode,
    arguments,
    input_function: Callable[[str], str] = input,
) -> int:
    """Run one real command with preview-first semantics."""
    if arguments.command == "real-preview":
        runtime = arguments.runtime_dir or week1_runtime_directory()
        node.preview_real_session(runtime)
        return 0

    session = load_real_session(arguments.session)
    if arguments.command == "return-anchor":
        node.return_to_session_anchor(
            session,
            arguments.execute,
            input_function,
        )
        if arguments.execute:
            print(POST_MOTION_SAFETY_NOTICE)
        return 0

    start_attempt, end_attempt = node.preview_batch(
        session,
        arguments.batch_targets,
    )
    if not arguments.execute:
        print("\n批次预览完成：没有发送真机运动请求。")
        return 0
    output = (
        arguments.output_dir
        if arguments.output_dir is not None
        else week1_validation_directory() / "real"
    )
    node.execute_batch(
        session,
        start_attempt,
        end_attempt,
        input_function,
        arguments.resume_after_stop,
        output,
    )
    print(POST_MOTION_SAFETY_NOTICE)
    return 0


def main(args: Optional[Sequence[str]] = None) -> int:
    """Dispatch pure-offline commands without ever initializing ROS/CAN."""
    raw_args = list(sys.argv if args is None else [sys.argv[0], *args])
    cli_args = remove_ros_args(args=raw_args)[1:]
    arguments = _argument_parser().parse_args(cli_args)
    try:
        offline_result = _run_without_ros(arguments)
        if offline_result is not None:
            return offline_result
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n验收工具已停止：{error}", file=sys.stderr)
        return 2

    rclpy.init(
        args=raw_args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    try:
        node = Week1AcceptanceNode()
        return run_ros_command(node, arguments)
    except (SmokeTestError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n验收工具已停止或拒绝：{error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print(
            "\n验收批次已取消；断点保留在隐藏运行目录。"
            "确认机械臂保持状态；异常时由观察人员在工作区外切断控制箱电源，"
            "机械臂可能下落。",
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
