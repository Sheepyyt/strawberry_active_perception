"""Deterministic 100-target offline acceptance benchmark for NERO Placo IK."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory

from .ik_core import PlacoIKSolver
from .models import IKErrorCode, READY_JOINT_POSITIONS
from .trajectory import TrajectoryGenerator
from .validation_paths import week1_validation_directory


FORMAL_OFFLINE_SAMPLES = 100
FORMAL_SUCCESS_RATE = 0.99
FORMAL_IK_P95_MS = 20.0


def _default_urdf() -> Path:
    share = Path(get_package_share_directory("agx_arm_description"))
    return (
        share
        / "agx_arm_urdf"
        / "nero"
        / "urdf"
        / "nero_description.urdf"
    )


def _percentile(values, percentile):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=float), percentile))


def _negative_checks(solver, ready):
    checks = {}

    malformed = np.eye(4)
    malformed[0, 0] = 2.0
    result = solver.solve(malformed, ready)
    checks["invalid_target"] = (
        int(result.error_code) == IKErrorCode.INVALID_TARGET
    )

    unreachable = np.eye(4)
    unreachable[:3, 3] = (5.0, 5.0, 5.0)
    result = solver.solve(unreachable, ready)
    checks["unreachable"] = (
        int(result.error_code) == IKErrorCode.UNREACHABLE
    )

    unsafe = ready.copy()
    unsafe[1] = solver.safe_joint_limits[1, 0] - 0.01
    result = solver.solve(solver.forward_kinematics(ready), unsafe)
    checks["joint_limit"] = (
        int(result.error_code) == IKErrorCode.JOINT_LIMIT_VIOLATION
    )

    zero = np.zeros(7)
    result = solver.solve(solver.forward_kinematics(zero), zero)
    checks["zero_pose_singularity"] = (
        int(result.error_code) == IKErrorCode.NEAR_SINGULARITY
    )
    return checks


def _write_reports(
    output_directory: Path,
    samples: int,
    rows: list[dict],
    summary: dict,
) -> tuple[Path, Path]:
    """Atomically replace the one important offline evidence pair."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = "offline_100" if samples == FORMAL_OFFLINE_SAMPLES else (
        f"offline_{samples}"
    )
    csv_path = output_directory / f"{stem}.csv"
    json_path = output_directory / f"{stem}.json"

    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    durable_summary = dict(summary)
    durable_summary["artifacts"] = {
        "details_csv": csv_path.name,
        "summary_json": json_path.name,
    }
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(durable_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(json_path)
    return csv_path, json_path


def run_benchmark(
    urdf_path: Path,
    output_directory: Path,
    samples: int,
    seed: int,
    joint_range_rad: float,
) -> dict:
    """Run known-reachable current-state-to-target IK and trajectory cases."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not math.isfinite(joint_range_rad) or not 0.0 < joint_range_rad <= 0.30:
        raise ValueError("joint-range must be in (0, 0.30] rad")

    solver = PlacoIKSolver(urdf_path)
    trajectory_generator = TrajectoryGenerator(
        joint_limits=solver.safe_joint_limits
    )
    ready = np.asarray(READY_JOINT_POSITIONS, dtype=float)
    rng = np.random.default_rng(seed)
    safe_lower = solver.safe_joint_limits[:, 0] + 0.01
    safe_upper = solver.safe_joint_limits[:, 1] - 0.01
    rows = []
    for index in range(samples):
        current_joints = np.clip(
            ready + rng.uniform(
                -0.5 * joint_range_rad,
                0.5 * joint_range_rad,
                7,
            ),
            safe_lower,
            safe_upper,
        )
        known_reachable_joints = np.clip(
            current_joints + rng.uniform(
                -joint_range_rad,
                joint_range_rad,
                7,
            ),
            safe_lower,
            safe_upper,
        )
        target = solver.forward_kinematics(known_reachable_joints)
        result = solver.solve(target, current_joints)
        trajectory = None
        if result.success and result.error_code == IKErrorCode.SUCCESS:
            trajectory = trajectory_generator.generate(
                current_joints,
                result.joint_positions,
            )
        trajectory_success = bool(
            trajectory is not None and trajectory.success
        )
        accepted = bool(
            result.success
            and result.error_code == IKErrorCode.SUCCESS
            and trajectory_success
        )
        reason = result.message
        if result.success and not trajectory_success:
            reason = (
                "trajectory was not generated"
                if trajectory is None
                else trajectory.message
            )
        rows.append({
            "index": index,
            "success": accepted,
            "code": int(result.error_code),
            "reason": reason,
            "solve_time_ms": result.solve_time_ms,
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
            "sigma_min": result.sigma_min,
            "condition_number": result.condition_number,
            "max_joint_delta_rad": result.max_joint_delta_rad,
            "trajectory_success": trajectory_success,
            "trajectory_duration_s": (
                "" if trajectory is None else trajectory.duration_s
            ),
            "trajectory_peak_velocity_rad_s": (
                "" if trajectory is None
                else trajectory.peak_velocity_rad_s
            ),
            "trajectory_peak_acceleration_rad_s2": (
                "" if trajectory is None
                else trajectory.peak_acceleration_rad_s2
            ),
            "current_joints_rad": json.dumps(
                current_joints.tolist(), separators=(",", ":")
            ),
            "known_reachable_joints_rad": json.dumps(
                known_reachable_joints.tolist(), separators=(",", ":")
            ),
            "solution_joints_rad": json.dumps(
                list(result.joint_positions), separators=(",", ":")
            ),
        })

    solve_times = [row["solve_time_ms"] for row in rows]
    successes = sum(bool(row["success"]) for row in rows)
    success_rate = successes / samples
    ik_time_p95_ms = _percentile(solve_times, 95.0)
    negative_checks = _negative_checks(solver, ready)
    acceptance = {
        "at_least_100_targets": samples >= FORMAL_OFFLINE_SAMPLES,
        "success_rate_at_least_99_percent": (
            success_rate >= FORMAL_SUCCESS_RATE
        ),
        "ik_time_p95_at_most_20_ms": (
            ik_time_p95_ms is not None
            and ik_time_p95_ms <= FORMAL_IK_P95_MS
        ),
        "all_accepted_trajectories_safe": all(
            bool(row["trajectory_success"])
            for row in rows
            if row["success"]
        ),
        "all_negative_checks_pass": all(negative_checks.values()),
    }
    summary = {
        "schema_version": 2,
        "benchmark": "week1_offline_known_reachable_fk_to_placo_ik",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "samples": samples,
        "successes": successes,
        "success_rate": success_rate,
        "ik_time_p50_ms": _percentile(solve_times, 50.0),
        "ik_time_p95_ms": ik_time_p95_ms,
        "ik_time_max_ms": max(solve_times),
        "position_error_p95_m": _percentile(
            [row["position_error_m"] for row in rows], 95.0
        ),
        "orientation_error_p95_rad": _percentile(
            [row["orientation_error_rad"] for row in rows], 95.0
        ),
        "max_joint_delta_rad": max(
            float(row["max_joint_delta_rad"]) for row in rows
        ),
        "min_sigma": min(float(row["sigma_min"]) for row in rows),
        "max_condition": max(
            float(row["condition_number"]) for row in rows
        ),
        "negative_checks": negative_checks,
        "acceptance": acceptance,
        "seed": seed,
        "joint_range_rad": joint_range_rad,
        "urdf_path_at_run": str(Path(urdf_path).resolve()),
        "passed": all(acceptance.values()),
        "important_note": (
            "Targets are known reachable because they are generated by FK "
            "from safe joints. Errors are model based, not external physical "
            "camera measurements."
        ),
    }
    csv_path, json_path = _write_reports(
        Path(output_directory).expanduser().resolve(),
        samples,
        rows,
        summary,
    )
    result = dict(summary)
    result["csv_path"] = str(csv_path)
    result["json_path"] = str(json_path)
    return result


def main(argv=None) -> None:
    """Run the formal offline benchmark and exit nonzero on refusal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=FORMAL_OFFLINE_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--joint-range", type=float, default=0.15)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "visible final-results directory; defaults to "
            "<project>/validation/week1/offline"
        ),
    )
    arguments = parser.parse_args(argv)
    output_directory = (
        arguments.output_dir
        if arguments.output_dir is not None
        else week1_validation_directory() / "offline"
    )
    summary = run_benchmark(
        arguments.urdf or _default_urdf(),
        output_directory,
        arguments.samples,
        arguments.seed,
        arguments.joint_range,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
