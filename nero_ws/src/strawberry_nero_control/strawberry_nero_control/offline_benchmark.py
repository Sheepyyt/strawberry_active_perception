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


def _default_urdf() -> Path:
    share = Path(get_package_share_directory('agx_arm_description'))
    return (
        share
        / 'agx_arm_urdf'
        / 'nero'
        / 'urdf'
        / 'nero_description.urdf'
    )


def _percentile(values, percentile):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _negative_checks(solver, ready):
    checks = {}

    malformed = np.eye(4)
    malformed[0, 0] = 2.0
    result = solver.solve(malformed, ready)
    checks['invalid_target'] = int(result.error_code) == IKErrorCode.INVALID_TARGET

    unreachable = np.eye(4)
    unreachable[:3, 3] = (5.0, 5.0, 5.0)
    result = solver.solve(unreachable, ready)
    checks['unreachable'] = int(result.error_code) == IKErrorCode.UNREACHABLE

    unsafe = ready.copy()
    unsafe[1] = solver.safe_joint_limits[1, 0] - 0.01
    result = solver.solve(solver.forward_kinematics(ready), unsafe)
    checks['joint_limit'] = (
        int(result.error_code) == IKErrorCode.JOINT_LIMIT_VIOLATION
    )

    zero = np.zeros(7)
    result = solver.solve(solver.forward_kinematics(zero), zero)
    checks['zero_pose_singularity'] = (
        int(result.error_code) == IKErrorCode.NEAR_SINGULARITY
    )
    return checks


def run_benchmark(
    urdf_path: Path,
    output_directory: Path,
    samples: int,
    seed: int,
    joint_range_rad: float,
) -> dict:
    """Run known-reachable FK-to-IK cases and write CSV/JSON evidence."""
    if samples <= 0:
        raise ValueError('samples must be positive')
    if not math.isfinite(joint_range_rad) or not 0.0 < joint_range_rad <= 0.30:
        raise ValueError('joint-range must be in (0, 0.30] rad')

    solver = PlacoIKSolver(urdf_path)
    ready = np.asarray(READY_JOINT_POSITIONS, dtype=float)
    rng = np.random.default_rng(seed)
    safe_lower = solver.safe_joint_limits[:, 0] + 0.01
    safe_upper = solver.safe_joint_limits[:, 1] - 0.01
    rows = []
    for index in range(samples):
        target_joints = np.clip(
            ready + rng.uniform(-joint_range_rad, joint_range_rad, 7),
            safe_lower,
            safe_upper,
        )
        target = solver.forward_kinematics(target_joints)
        result = solver.solve(target, ready)
        row = {
            'index': index,
            'success': result.success,
            'code': int(result.error_code),
            'reason': result.message,
            'solve_time_ms': result.solve_time_ms,
            'position_error_m': result.position_error_m,
            'orientation_error_rad': result.orientation_error_rad,
            'sigma_min': result.sigma_min,
            'condition_number': result.condition_number,
            'max_joint_delta_rad': result.max_joint_delta_rad,
            'target_joints_rad': json.dumps(target_joints.tolist()),
            'solution_joints_rad': json.dumps(list(result.joint_positions)),
        }
        rows.append(row)

    solve_times = [row['solve_time_ms'] for row in rows]
    successes = sum(bool(row['success']) for row in rows)
    negative_checks = _negative_checks(solver, ready)
    summary = {
        'samples': samples,
        'successes': successes,
        'success_rate': successes / samples,
        'ik_time_p50_ms': _percentile(solve_times, 50.0),
        'ik_time_p95_ms': _percentile(solve_times, 95.0),
        'ik_time_max_ms': max(solve_times),
        'position_error_p95_m': _percentile(
            [row['position_error_m'] for row in rows], 95.0
        ),
        'orientation_error_p95_rad': _percentile(
            [row['orientation_error_rad'] for row in rows], 95.0
        ),
        'negative_checks': negative_checks,
        'acceptance': {
            'success_rate_at_least_99_percent': successes / samples >= 0.99,
            'ik_time_p95_at_most_20_ms': _percentile(solve_times, 95.0) <= 20.0,
            'all_negative_checks_pass': all(negative_checks.values()),
        },
        'seed': seed,
        'joint_range_rad': joint_range_rad,
        'urdf_path': str(urdf_path),
    }
    summary['passed'] = all(summary['acceptance'].values())

    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = output_directory / f'offline_ik_{stamp}.csv'
    json_path = output_directory / f'offline_ik_{stamp}.json'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    summary['csv_path'] = str(csv_path)
    summary['json_path'] = str(json_path)
    return summary


def main(argv=None) -> None:
    """Command-line entry point; exits nonzero when acceptance fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260806)
    parser.add_argument('--joint-range', type=float, default=0.15)
    parser.add_argument('--urdf', type=Path, default=None)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path.home() / '.ros' / 'strawberry_nero_control' / 'offline',
    )
    arguments = parser.parse_args(argv)
    summary = run_benchmark(
        arguments.urdf or _default_urdf(),
        arguments.output_dir,
        arguments.samples,
        arguments.seed,
        arguments.joint_range,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
