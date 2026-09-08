#!/usr/bin/env python3
"""Run deterministic G2 Gradient-NBV acceptance and write strict JSON."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NBV_SOURCE_ROOT = (
    REPOSITORY_ROOT / "perception_ws" / "src" / "strawberry_gradient_nbv"
)
os.sys.path.insert(0, str(NBV_SOURCE_ROOT))

from strawberry_gradient_nbv.core import GradientNBVCore, NBVInputError  # noqa: E402
from strawberry_gradient_nbv.fixtures import make_multiview_fixture  # noqa: E402


UPSTREAM_COMMIT = "81b501defc117732f66478ed01cb528b15140208"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _write_coverage_csv_atomic(path: Path, report: dict[str, Any]) -> None:
    """Write the five-view coverage curve and planner diagnostics."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "view_index",
                    "observation_id",
                    "coverage",
                    "coverage_increment",
                    "current_gain",
                    "planned_gain",
                    "gain_improvement",
                    "translation_step_m",
                ),
            )
            writer.writeheader()
            previous = None
            for index, record in enumerate(report["multiview"]["records"]):
                coverage = float(record["coverage"])
                writer.writerow(
                    {
                        "view_index": index,
                        "observation_id": record["observation_id"],
                        "coverage": coverage,
                        "coverage_increment": (
                            0.0 if previous is None else coverage - previous
                        ),
                        "current_gain": record["current_gain"],
                        "planned_gain": record["planned_gain"],
                        "gain_improvement": record["gain_improvement"],
                        "translation_step_m": record["step_m"],
                    }
                )
                previous = coverage
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _result_geometry(result, observation, config) -> dict[str, Any]:
    position = result.pose[:3, 3]
    target = np.asarray(config["target_center"], dtype=float)
    target_direction = target - position
    target_direction /= np.linalg.norm(target_direction)
    cosine = float(np.clip(np.dot(result.pose[:3, 2], target_direction), -1.0, 1.0))
    return {
        "step_m": float(np.linalg.norm(position - observation.pose[:3, 3])),
        "look_at_error_deg": float(np.degrees(np.arccos(cosine))),
        "within_observation_bounds": bool(
            np.all(position >= np.asarray(config["observation_min"]) - 1.0e-6)
            and np.all(position <= np.asarray(config["observation_max"]) + 1.0e-6)
        ),
        "pose": result.pose.tolist(),
    }


def _multiview_acceptance(device: str) -> tuple[dict[str, Any], list[float]]:
    fixture = make_multiview_fixture(160, 100)
    config = dict(fixture.config)
    config["scene_id"] = fixture.scene_id
    core = GradientNBVCore(device)
    core.configure(config)
    coverage: list[float] = []
    records = []
    for observation in fixture.observations():
        started = time.perf_counter()
        result = core.update_and_plan(
            observation.depth,
            observation.mask,
            observation.K,
            observation.pose,
        )
        _synchronize(device)
        elapsed = time.perf_counter() - started
        coverage.append(result.coverage)
        records.append(
            {
                "observation_id": observation.observation_id,
                "coverage": result.coverage,
                "gain": result.gain,
                "current_gain": result.current_gain,
                "planned_gain": result.planned_gain,
                "gain_improvement": result.gain_improvement,
                "wall_time_sec": elapsed,
                "core_time_ms": result.compute_time_ms,
                "observed_voxel_count": result.observed_voxel_count,
                "occupied_voxel_count": result.occupied_voxel_count,
                "loss_history": list(result.loss_history),
                **_result_geometry(result, observation, config),
            }
        )
    growth = np.diff(coverage)
    geometry_passed = all(
        record["step_m"] <= config["max_step"] + 1.0e-6
        and record["step_m"] >= 1.0e-4
        and record["look_at_error_deg"] <= 1.0
        and record["within_observation_bounds"]
        and record["gain_improvement"] > 1.0e-6
        and record["planned_gain"] > record["current_gain"]
        for record in records
    )
    return (
        {
            "fixture": fixture.scene_id,
            "coverage_definition": (
                "target-ROI voxels ever touched by a valid ray / total target-ROI voxels"
            ),
            "records": records,
            "coverage": coverage,
            "coverage_growth": growth.tolist(),
            "positive_growth_count_last_four": int(np.count_nonzero(growth > 0.0)),
            "final_minus_first": coverage[-1] - coverage[0],
            "coverage_gate_passed": bool(
                np.all(growth >= 0.0)
                and np.count_nonzero(growth > 0.0) >= 3
                and coverage[-1] - coverage[0] >= 0.20
            ),
            "geometry_gate_passed": geometry_passed,
        },
        coverage,
    )


def _performance_acceptance(device: str) -> dict[str, Any]:
    fixture = make_multiview_fixture(640, 400)
    observation = next(fixture.observations())
    config = dict(fixture.config)
    config["scene_id"] = fixture.scene_id
    core = GradientNBVCore(device)
    core.configure(config)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = core.update_and_plan(
        observation.depth,
        observation.mask,
        observation.K,
        observation.pose,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated_mib = (
        torch.cuda.max_memory_allocated() / (1024.0**2) if device == "cuda" else None
    )
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved() / (1024.0**2) if device == "cuda" else None
    )
    passed = elapsed <= 2.0 and (
        peak_allocated_mib is None or peak_allocated_mib <= 2048.0
    )
    return {
        "input_resolution": [640, 400],
        "algorithm_resolution": [320, 200],
        "samples_per_ray": config["samples_per_ray"],
        "optimization_steps": config["optimization_steps"],
        "wall_time_sec": elapsed,
        "core_time_ms": result.compute_time_ms,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "gain": result.gain,
        "current_gain": result.current_gain,
        "planned_gain": result.planned_gain,
        "gain_improvement": result.gain_improvement,
        "coverage": result.coverage,
        "gate_passed": passed,
    }


def _reproducibility_acceptance(device: str) -> dict[str, Any]:
    fixture = make_multiview_fixture(96, 64)
    observation = next(fixture.observations())
    config = dict(fixture.config)
    config.update(scene_id=fixture.scene_id, voxel_size=0.006, samples_per_ray=32)
    results = []
    for _ in range(2):
        core = GradientNBVCore(device)
        core.configure(config)
        results.append(
            core.update_and_plan(
                observation.depth,
                observation.mask,
                observation.K,
                observation.pose,
            )
        )
    pose_difference = float(np.max(np.abs(results[0].pose - results[1].pose)))
    gain_difference = abs(results[0].gain - results[1].gain)
    current_gain_difference = abs(results[0].current_gain - results[1].current_gain)
    return {
        "pose_max_absolute_difference": pose_difference,
        "gain_absolute_difference": gain_difference,
        "current_gain_absolute_difference": current_gain_difference,
        "gate_passed": (
            pose_difference == 0.0
            and gain_difference == 0.0
            and current_gain_difference == 0.0
        ),
    }


def _mask_semantics_acceptance(device: str) -> dict[str, Any]:
    """Prove that mask location, not only ray geometry, changes the gain map."""
    fixture = make_multiview_fixture(160, 100)
    observation = next(fixture.observations())
    config = dict(fixture.config)
    config["scene_id"] = fixture.scene_id
    shifted_mask = np.roll(
        observation.mask, observation.mask.shape[1] // 5, axis=1
    )
    results = []
    for mask in (observation.mask, shifted_mask):
        core = GradientNBVCore(device)
        core.configure(config)
        results.append(
            core.update_and_plan(
                observation.depth, mask, observation.K, observation.pose
            )
        )
    gain_delta = abs(results[0].current_gain - results[1].current_gain)
    pose_delta = float(np.linalg.norm(results[0].pose[:3, 3] - results[1].pose[:3, 3]))
    same_area = int(np.count_nonzero(observation.mask)) == int(
        np.count_nonzero(shifted_mask)
    )
    passed = bool(
        same_area
        and gain_delta >= 0.05
        and results[0].current_gain >= results[1].current_gain * 1.15
        and results[0].gain_improvement > 1.0e-6
        and results[1].gain_improvement > 1.0e-6
    )
    return {
        "ablation": "same-area truth mask versus horizontally shifted mask",
        "masked_pixel_count": int(np.count_nonzero(observation.mask)),
        "same_mask_area": same_area,
        "truth_current_gain": results[0].current_gain,
        "shifted_current_gain": results[1].current_gain,
        "current_gain_absolute_difference": gain_delta,
        "planned_pose_translation_difference_m": pose_delta,
        "truth_gain_improvement": results[0].gain_improvement,
        "shifted_gain_improvement": results[1].gain_improvement,
        "gate_passed": passed,
    }


def _invalid_input_acceptance() -> dict[str, Any]:
    fixture = make_multiview_fixture(64, 48)
    observation = next(fixture.observations())
    config = dict(fixture.config)
    config.update(
        scene_id=fixture.scene_id,
        voxel_size=0.006,
        samples_per_ray=16,
        optimization_steps=1,
        downsample=1,
    )
    cases = {}
    for name in (
        "empty_mask",
        "all_bad_depth",
        "bad_K",
        "size_mismatch",
        "bad_pose",
        "target_depth_invalid",
    ):
        core = GradientNBVCore("cpu")
        core.configure(config)
        depth = observation.depth.copy()
        mask = observation.mask.copy()
        intrinsics = observation.K.copy()
        pose = observation.pose.copy()
        if name == "empty_mask":
            mask.fill(0)
        elif name == "all_bad_depth":
            depth.fill(np.nan)
        elif name == "bad_K":
            intrinsics[0, 0] = 0.0
        elif name == "size_mismatch":
            mask = mask[:-1]
        elif name == "bad_pose":
            pose[0, 0] = 2.0
        else:
            depth[mask == 255] = np.nan
        state_before = (
            core._log_odds.clone(),
            core._semantic_log_odds.clone(),
            core._ever_observed.clone(),
        )
        try:
            core.update_and_plan(depth, mask, intrinsics, pose)
            rejected = False
            reason = "unexpected success"
        except NBVInputError as error:
            rejected = True
            reason = str(error)
        unchanged = all(
            torch.equal(before, after)
            for before, after in zip(
                state_before,
                (core._log_odds, core._semantic_log_odds, core._ever_observed),
            )
        )
        cases[name] = {
            "rejected": rejected,
            "map_unchanged": unchanged,
            "reason": reason,
        }
    return {
        "cases": cases,
        "gate_passed": all(
            case["rejected"] and case["map_unchanged"] for case in cases.values()
        ),
    }


def build_report() -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multiview, _ = _multiview_acceptance(device)
    performance = _performance_acceptance(device)
    reproducibility = _reproducibility_acceptance(device)
    mask_semantics = _mask_semantics_acceptance(device)
    invalid_inputs = _invalid_input_acceptance()
    core_source = (NBV_SOURCE_ROOT / "strawberry_gradient_nbv" / "core.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = (
        "import rclpy",
        "import rospy",
        "import moveit",
        "from moveit",
        "import abb_control",
        "open3d",
        "matplotlib",
        "cv_bridge",
    )
    independence_passed = not any(value in core_source for value in forbidden)
    checks = {
        "multiview_coverage": multiview["coverage_gate_passed"],
        "next_view_geometry": multiview["geometry_gate_passed"],
        "performance": performance["gate_passed"],
        "fixed_seed_reproducibility": reproducibility["gate_passed"],
        "mask_semantics": mask_semantics["gate_passed"],
        "invalid_inputs_are_atomic": invalid_inputs["gate_passed"],
        "core_import_independence": independence_passed,
    }
    return {
        "schema_version": 1,
        "artifact": "g2_nbv_acceptance",
        "generated_utc": _utc_now(),
        "gate": "G2",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "upstream_reference_commit": UPSTREAM_COMMIT,
        "multiview": multiview,
        "performance": performance,
        "reproducibility": reproducibility,
        "mask_semantics": mask_semantics,
        "invalid_inputs": invalid_inputs,
        "independence": {
            "forbidden_imports": list(forbidden),
            "gate_passed": independence_passed,
            "note": "Core check is performed without sourcing or launching ROS/MoveIt/ABB.",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        help=(
            "coverage-curve CSV path; defaults to <output stem>_coverage.csv"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build_report()
    _write_json_atomic(arguments.output, report)
    coverage_path = arguments.coverage_csv or arguments.output.with_name(
        f"{arguments.output.stem}_coverage.csv"
    )
    _write_coverage_csv_atomic(coverage_path, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
