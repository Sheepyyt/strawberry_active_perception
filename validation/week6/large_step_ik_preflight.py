#!/usr/bin/env python3
"""Deterministic, hardware-free 50/100 mm camera-motion IK preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from strawberry_nero_control.ik_core import PlacoIKSolver
from strawberry_nero_control.models import READY_JOINT_POSITIONS


ROOT = Path(__file__).resolve().parents[2]
URDF = (
    ROOT
    / "nero_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf"
    / "nero_description.urdf"
)
HANDEYE_REPORT = (
    ROOT
    / "validation/week3/artifacts/stability_pose001_030_factory_raw_D.json"
)
HANDEYE_SHA256 = (
    "31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f"
)
DIRECTIONS = {
    "camera_right": (1.0, 0.0, 0.0),
    "camera_left": (-1.0, 0.0, 0.0),
    "camera_down": (0.0, 1.0, 0.0),
    "camera_up": (0.0, -1.0, 0.0),
    "camera_forward": (0.0, 0.0, 1.0),
    "camera_backward": (0.0, 0.0, -1.0),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_preflight() -> dict[str, object]:
    if sha256_file(HANDEYE_REPORT) != HANDEYE_SHA256:
        raise RuntimeError("formal hand-eye report SHA-256 mismatch")
    handeye_document = json.loads(HANDEYE_REPORT.read_text(encoding="utf-8"))
    handeye = np.asarray(
        handeye_document["T_link7_camera_optical"], dtype=np.float64
    )
    solver = PlacoIKSolver(URDF)
    ready = np.asarray(READY_JOINT_POSITIONS, dtype=np.float64)
    base_link7 = solver.forward_kinematics(ready)
    base_camera = base_link7 @ handeye
    records: list[dict[str, object]] = []
    for distance_m in (0.05, 0.10):
        for name, unit_vector in DIRECTIONS.items():
            camera_target = base_camera.copy()
            camera_target[:3, 3] += (
                base_camera[:3, :3]
                @ (distance_m * np.asarray(unit_vector, dtype=np.float64))
            )
            link7_target = camera_target @ np.linalg.inv(handeye)
            result = solver.solve(link7_target, ready)
            accepted = bool(
                result.success
                and result.position_error_m <= 0.003
                and result.orientation_error_rad <= np.deg2rad(2.0)
                and result.max_joint_delta_rad <= 0.35
                and result.sigma_min >= 0.10
                and result.condition_number <= 20.0
            )
            records.append(
                {
                    "distance_m": distance_m,
                    "direction": name,
                    "accepted_by_large_profile": accepted,
                    "solver_success": bool(result.success),
                    "error_code": int(result.error_code),
                    "message": result.message,
                    "position_error_m": float(result.position_error_m),
                    "orientation_error_deg": float(
                        np.rad2deg(result.orientation_error_rad)
                    ),
                    "max_joint_delta_rad": float(result.max_joint_delta_rad),
                    "sigma_min": float(result.sigma_min),
                    "condition_number": float(result.condition_number),
                }
            )
    five_cm = [item for item in records if item["distance_m"] == 0.05]
    ten_cm = [item for item in records if item["distance_m"] == 0.10]
    return {
        "schema": "strawberry_large_step_ik_preflight/v1",
        "hardware_or_ros_used": False,
        "start_pose": "NERO READY_JOINT_POSITIONS",
        "handeye_report": str(HANDEYE_REPORT.relative_to(ROOT)),
        "handeye_sha256": HANDEYE_SHA256,
        "limits": {
            "translation_range_m": [0.05, 0.10],
            "max_joint_delta_rad": 0.35,
            "max_position_error_m": 0.003,
            "max_orientation_error_deg": 2.0,
            "min_sigma": 0.10,
            "max_condition_number": 20.0,
        },
        "five_cm_accepted": sum(
            bool(item["accepted_by_large_profile"]) for item in five_cm
        ),
        "five_cm_total": len(five_cm),
        "ten_cm_accepted": sum(
            bool(item["accepted_by_large_profile"]) for item in ten_cm
        ),
        "ten_cm_total": len(ten_cm),
        "records": records,
        "interpretation": (
            "This proves only offline reachability around READY. Live preview "
            "must solve again from measured joints before any real motion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_preflight()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
