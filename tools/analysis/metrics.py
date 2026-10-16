"""Read-only, ROS-free metrics from an immutable recorded NBV run.

Candidate sensitivity is conditional on each recorded planning state. It does
not simulate observations or the outcome of an alternative motion policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RADII_MM = (100.0, 75.0, 50.0, 25.0, 10.0, 5.0)
MINIMUM_DEPTHS_M = (0.42, 0.43, 0.45)


class AnalysisError(ValueError):
    """Recorded evidence is missing, ambiguous, or inconsistent."""


def _required(value: dict, key: str, context: str) -> Any:
    if key not in value:
        raise AnalysisError(f"{context}: missing {key}")
    return value[key]


def _number(value: Any, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise AnalysisError(f"{context}: expected a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{context}: expected a finite number") from error
    if not math.isfinite(result):
        raise AnalysisError(f"{context}: expected a finite number")
    return result


def _read_json(path: Path, hashes: dict) -> dict:
    try:
        data = path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError) as error:
        raise AnalysisError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(document, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return document


def _index_snapshots(directory: Path, scene_id: str, ids: list[str], schema: str) -> dict:
    found = {}
    for path in sorted(directory.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                scene = str(archive["scene_id"].item())
                observation = str(archive["observation_id"].item())
                if scene != scene_id or observation not in ids:
                    continue
                if str(archive["schema"].item()) != schema:
                    raise AnalysisError(f"{path}: unsupported snapshot schema")
                if observation in found:
                    raise AnalysisError(f"duplicate {schema} snapshot for {scene_id}/{observation}")
                found[observation] = path
        except (OSError, KeyError, ValueError) as error:
            if isinstance(error, AnalysisError):
                raise
            raise AnalysisError(f"invalid snapshot {path}: {error}") from error
    missing = [observation for observation in ids if observation not in found]
    if missing:
        raise AnalysisError(f"missing {schema} snapshots: {missing}")
    return found


def _load_npz(path: Path, hashes: dict) -> dict:
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _grid(snapshot: dict, context: str) -> dict:
    fields = ("dimensions", "origin_m", "voxel_size_m", "target_center_m", "target_roi_size_m")
    grid = {key: np.asarray(_required(snapshot, key, context)) for key in fields}
    for key in fields:
        expected = () if key == "voxel_size_m" else (3,)
        if grid[key].shape != expected or not np.all(np.isfinite(grid[key])):
            raise AnalysisError(f"{context}: invalid grid {key}")
    dimensions = grid["dimensions"]
    if np.any(dimensions <= 0) or not np.array_equal(dimensions, dimensions.astype(int)):
        raise AnalysisError(f"{context}: invalid grid dimensions")
    if float(grid["voxel_size_m"]) <= 0 or np.any(grid["target_roi_size_m"] <= 0):
        raise AnalysisError(f"{context}: invalid grid size")
    shape = tuple(int(x) for x in dimensions)
    for key in ("target", "occupied", "observed"):
        array = np.asarray(_required(snapshot, key, context))
        if array.shape != shape or array.dtype != np.bool_:
            raise AnalysisError(f"{context}: {key} must be a boolean array on the declared grid")
    grid["world_frame"] = np.asarray(_required(snapshot, "world_frame", context))
    return grid


def _candidate_round(index: int, label: str, source: str, records: list,
                     selected: dict | None, directions: int) -> dict:
    if not isinstance(records, list) or not records:
        raise AnalysisError(f"{source}: missing candidate audit")
    rows, identities, indices = [], set(), set()
    for record in records:
        context = f"{source} candidate"
        radius = _number(_required(record, "radius_m", context), context) * 1000
        matches = [r for r in RADII_MM if math.isclose(r, radius, abs_tol=1e-6)]
        if len(matches) != 1:
            raise AnalysisError(f"{source}: unsupported radius {radius} mm")
        radius = matches[0]
        candidate_index = int(_required(record, "candidate_index", context))
        direction = int(_required(record, "direction_index", context))
        if candidate_index in indices or (radius, direction) in identities:
            raise AnalysisError(f"{source}: duplicate candidate index or radius/direction")
        indices.add(candidate_index)
        identities.add((radius, direction))
        target_depth = _number(_required(record, "target_depth_m", context), context)
        minimum = _number(_required(record, "minimum_target_view_distance_m", context), context)
        if minimum > min(MINIMUM_DEPTHS_M) + 1e-9:
            raise AnalysisError(f"{source}: sensitivity thresholds would relax the recorded minimum")
        reachable = _required(record, "ik_gate_passed", context)
        scored = _required(record, "gain_scored", context)
        if not isinstance(reachable, bool) or not isinstance(scored, bool):
            raise AnalysisError(f"{source}: candidate gate flags must be boolean")
        current = gain = improvement = tolerance = positive = None
        if scored:
            if not reachable:
                raise AnalysisError(f"{source}: scored candidate did not pass IK")
            current = _number(_required(record, "current_gain", context), context)
            gain = _number(_required(record, "candidate_gain", context), context)
            improvement = gain - current
            recorded = _number(_required(record, "gain_improvement", context), context)
            if not math.isclose(recorded, improvement, rel_tol=1e-9, abs_tol=1e-12):
                raise AnalysisError(f"{source}: inconsistent candidate gain improvement")
            tolerance = max(1e-9, abs(current) * 1e-6)
            positive = improvement > tolerance
        rows.append({"round_index": index, "source": source,
                     "candidate_index": candidate_index, "direction_index": direction,
                     "radius_mm": radius, "target_depth_m": target_depth,
                     "ik_gate_passed": reachable, "gain_scored": scored,
                     "current_gain": current, "candidate_gain": gain,
                     "gain_improvement": improvement, "gain_tolerance": tolerance,
                     "positive": positive, "rejection": record.get("rejection", "")})
    expected = {(r, direction) for r in RADII_MM for direction in range(directions)}
    if identities != expected:
        raise AnalysisError(f"{source}: missing or unexpected native lattice candidates")

    def counts(group: list[dict]) -> dict:
        scored_gains = [row["gain_improvement"] for row in group if row["gain_scored"]]
        return {"total": len(group), "reachable": sum(row["ik_gate_passed"] for row in group),
                "scored": len(scored_gains), "positive": sum(row["positive"] is True for row in group),
                "best_gain_improvement": max(scored_gains) if scored_gains else None}

    selected_radius = None
    if selected is not None:
        selected_id = _required(selected, "candidate_index", source)
        selected_rows = [row for row in rows if row["candidate_index"] == selected_id]
        if len(selected_rows) != 1 or selected_rows[0]["positive"] is not True:
            raise AnalysisError(f"{source}: selected candidate is missing or has no positive gain")
        selected_radius = selected_rows[0]["radius_mm"]
        if not math.isclose(_number(_required(selected, "radius_m", source), source) * 1000,
                            selected_radius, abs_tol=1e-6):
            raise AnalysisError(f"{source}: selected radius disagrees with candidate audit")
    radii, thresholds, cross = [], [], []
    for radius in RADII_MM:
        group = [row for row in rows if row["radius_mm"] == radius]
        radii.append({"radius_mm": radius, **counts(group)})
        for minimum in MINIMUM_DEPTHS_M:
            passed = [row for row in group if row["target_depth_m"] + 1e-9 >= minimum]
            cross.append({"radius_mm": radius, "minimum_depth_m": minimum,
                          "depth_pass": len(passed), **counts(passed)})
    for minimum in MINIMUM_DEPTHS_M:
        passed = [row for row in rows if row["target_depth_m"] + 1e-9 >= minimum]
        thresholds.append({"minimum_depth_m": minimum, "depth_pass": len(passed), **counts(passed)})
    return {"round_index": index, "label": label, "source": source,
            "selected_radius_mm": selected_radius, "radii": radii, "thresholds": thresholds,
            "radius_thresholds": cross, "candidate_rows": rows}


def _roi_mask(grid: dict) -> np.ndarray:
    # Match core.configure's float32 centres and inclusive ROI tolerance.
    origin = grid["origin_m"].astype(np.float32)
    target = grid["target_center_m"].astype(np.float32)
    half = (grid["target_roi_size_m"] / 2).astype(np.float32)
    size = np.float32(grid["voxel_size_m"])
    axes = [np.abs(origin[i] + (np.arange(int(n), dtype=np.float32) + np.float32(.5))
                   * size - target[i]) <= half[i] + np.float32(1e-7)
            for i, n in enumerate(grid["dimensions"])]
    return axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]


def _analyze_run(run: Path) -> dict:
    """Analyze original artifacts without writing files or importing ROS.

    Raises AnalysisError rather than silently omitting ambiguous/missing views.
    Near-layer depth statistics retain the estimator's recorded definition;
    mask-validity is independently recomputed over every masked depth pixel.
    """
    run = Path(run).expanduser().resolve()
    hashes: dict[str, str] = {}
    execution_path = run / "execution.json"
    document = _read_json(execution_path, hashes)
    scene = str(_required(document, "scene_id", str(execution_path)))
    corrected = _required(document, "corrected_observation", "execution")
    if corrected.get("scene_id") != scene:
        raise AnalysisError("initial observation scene does not match execution")
    steps = _required(document, "motion_steps", "execution")
    if not isinstance(steps, list):
        raise AnalysisError("motion_steps must be a list")
    for index, step in enumerate(steps, 1):
        if step.get("step_index") != index:
            raise AnalysisError("motion steps are missing, duplicated, or out of order")
        if step.get("execution", {}).get("action_status") != 4:
            raise AnalysisError(f"step {index}: action did not complete successfully")
        update = _required(step, "post_map_update", f"step {index}")
        if update.get("scene_id") != scene or update.get("success") is not True:
            raise AnalysisError(f"step {index}: missing successful same-scene map update")
    ids = [_required(corrected, "observation_id", "initial observation")]
    ids += [_required(step["post_map_update"], "observation_id", "post map update") for step in steps]
    if len(set(ids)) != len(ids):
        raise AnalysisError("duplicate execution observation_id")
    maps = _index_snapshots(run / "maps", scene, ids, "strawberry_gradient_nbv_map/v1")
    observations = _index_snapshots(run / "observations", scene, ids,
                                    "strawberry_canonical_observation_snapshot/v1")
    config = _required(document["configuration"], "request_semantics", "configuration")
    depth_min = _number(_required(config, "depth_min_m", "configuration"), "depth_min_m")
    depth_max = _number(_required(config, "depth_max_m", "configuration"), "depth_max_m")
    if not 0 <= depth_min < depth_max:
        raise AnalysisError("invalid configured depth range")
    views, baseline_grid, previous_target, previous_coverage = [], None, None, None
    previous_observed = None
    roi = None
    for index, observation_id in enumerate(ids):
        snapshot = _load_npz(maps[observation_id], hashes)
        observation = _load_npz(observations[observation_id], hashes)
        context = str(maps[observation_id])
        grid = _grid(snapshot, context)
        if baseline_grid is None:
            baseline_grid = grid
            roi = _roi_mask(grid)
            if not np.any(roi):
                raise AnalysisError(f"{context}: target ROI contains no voxel centres")
        elif any(not np.array_equal(grid[key], baseline_grid[key]) for key in grid):
            raise AnalysisError(f"{context}: grid or world frame differs from initial map")
        if str(observation["world_frame"].item()) != str(grid["world_frame"].item()):
            raise AnalysisError(f"{context}: observation/map world frames differ")
        depth, mask = observation["depth_m"], observation["mask"] > 0
        if depth.ndim != 2 or depth.shape != mask.shape:
            raise AnalysisError(f"{context}: depth and mask pixel grids differ")
        valid = mask & np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
        mask_pixels, valid_pixels = int(mask.sum()), int(valid.sum())
        if not mask_pixels or not valid_pixels:
            raise AnalysisError(f"{context}: no valid masked depth")
        target = document["final_target"] if index == 0 else steps[index - 1]["post_target"]
        if target.get("mask_pixels") != mask_pixels or target.get("valid_mask_pixels") != valid_pixels:
            raise AnalysisError(f"{context}: target audit pixel counts disagree with observation")
        coverage = _number(snapshot["coverage"].item(), context)
        if not 0 <= coverage <= 1:
            raise AnalysisError(f"{context}: coverage is outside [0,1]")
        roi_observed = int(np.count_nonzero(snapshot["observed"] & roi))
        reconstructed_coverage = roi_observed / int(np.count_nonzero(roi))
        if not math.isclose(coverage, reconstructed_coverage, rel_tol=0, abs_tol=1e-12):
            raise AnalysisError(f"{context}: coverage disagrees with observed target-ROI voxels")
        if previous_observed is not None and np.any(previous_observed & ~snapshot["observed"]):
            raise AnalysisError(f"{context}: ever-observed map lost voxels")
        update = document["raw_next_view"] if index == 0 else steps[index - 1]["post_map_update"]
        if update.get("observation_id") != observation_id or not math.isclose(
                coverage, _number(update.get("coverage"), context), abs_tol=1e-6):
            raise AnalysisError(f"{context}: map snapshot disagrees with execution update")
        current_target = snapshot["target"]
        step = None if index == 0 else steps[index - 1]
        views.append({"index": index, "observation_id": observation_id,
                      "observation_path": str(observations[observation_id]),
                      "map_path": str(maps[observation_id]), "coverage": coverage,
                      "roi_observed_voxels": roi_observed,
                      "coverage_gain_pp": 0.0 if index == 0 else (coverage - previous_coverage) * 100,
                      "target_voxels": int(current_target.sum()),
                      "target_occupied_voxels": int((current_target & snapshot["occupied"]).sum()),
                      "target_added": 0 if index == 0 else int((current_target & ~previous_target).sum()),
                      "target_removed": 0 if index == 0 else int((previous_target & ~current_target).sum()),
                      "mask_pixels": mask_pixels, "valid_depth_pixels": valid_pixels,
                      "valid_depth_fraction": valid_pixels / mask_pixels,
                      "median_depth_mm": _number(target["median_depth_m"], context) * 1000,
                      "depth_mad_mm": _number(target["depth_mad_m"], context) * 1000,
                      "depth_layer_count": target.get("depth_layer_count"),
                      "depth_statistics_source": "recorded nearest supported masked-depth layer",
                      "actual_translation_mm": 0.0 if step is None else _number(
                          step["cumulative_camera_motion"]["actual_step_translation_m"], context) * 1000,
                      "position_error_mm": None if step is None else _number(
                          step["post_pose_error"]["translation_m"], context) * 1000})
        previous_target, previous_coverage = current_target, coverage
        previous_observed = snapshot["observed"]

    rounds = []
    directions = int(document["session_policy"].get("reachable_candidate_direction_count", 18)) + 1
    if steps:
        preview_path = run / "preview.json"
        preview = _read_json(preview_path, hashes)
        actual_hash = hashes[str(preview_path)]
        authorization = _required(document, "execution_authorization", "execution")
        if (authorization.get("plan_sha256") != actual_hash
                or steps[0].get("plan_sha256_binding") != actual_hash):
            raise AnalysisError("preview SHA does not match execution authorization / first motion")
        if preview.get("scene_id") != scene:
            raise AnalysisError("SHA-bound preview belongs to a different scene")
        frozen = _required(steps[0], "selected_candidate", "first motion")
        selected = _required(preview, "selected_candidate", "preview")
        for key in ("T_base_camera", "T_base_link7"):
            frozen_pose = np.asarray(_required(frozen, key, "first motion"), dtype=float)
            selected_pose = np.asarray(_required(selected, key, "preview"), dtype=float)
            if (frozen_pose.shape != (4, 4) or not np.all(np.isfinite(frozen_pose))
                    or not np.array_equal(frozen_pose, selected_pose)):
                raise AnalysisError(f"first motion {key} differs from SHA-bound preview")
        rounds.append(_candidate_round(1, "第 1 次选点（SHA 绑定预演）", str(preview_path) + "#ik_candidates",
                                       _required(preview, "ik_candidates", "preview"), selected, directions))
        for index, step in enumerate(steps[1:], 2):
            source = f"{execution_path}#motion_steps[{index - 1}].ik_candidates"
            if step.get("source_map_observation_id") != ids[index - 1]:
                raise AnalysisError(f"step {index}: candidate source is not the preceding mapped view")
            rounds.append(_candidate_round(index, f"第 {index} 次选点", source,
                                           _required(step, "ik_candidates", source),
                                           _required(step, "selected_candidate", source), directions))
        terminal = steps[-1].get("next_view_convergence")
        if document.get("session_progress", {}).get("converged_early") and terminal is None:
            raise AnalysisError("converged run is missing terminal candidate audit")
        if terminal is not None:
            source = f"{execution_path}#motion_steps[{len(steps) - 1}].next_view_convergence.ik_candidates"
            if terminal.get("source_observation_id") != ids[-1] or terminal.get("motion_goal_sent") is not False:
                raise AnalysisError("terminal candidate audit has inconsistent observation or action provenance")
            rounds.append(_candidate_round(len(steps) + 1, "终止选点（未执行运动）", source,
                                           _required(terminal, "ik_candidates", source), None, directions))
            if any(row["positive"] is True for row in rounds[-1]["candidate_rows"]):
                raise AnalysisError("terminal no-positive-gain audit contains a positive candidate")
    progress = document.get("session_progress", {})
    initial, final = views[0], views[-1]
    summary = {"completed_actions": len(steps), "map_updates": len(steps),
               "roi_voxel_count": int(np.count_nonzero(roi)),
               "mapped_observations": len(views), "initial_coverage": initial["coverage"],
               "final_coverage": final["coverage"],
               "coverage_gain_pp": (final["coverage"] - initial["coverage"]) * 100,
               "initial_target_voxels": initial["target_voxels"], "final_target_voxels": final["target_voxels"],
               "coverage_target": document["session_policy"].get("coverage_target"),
               "scientific_acceptance_passed": progress.get("scientific_acceptance_passed"),
               "stop_reason": progress.get("convergence_reason") or progress.get("termination_reason") or document.get("reason"),
               "started_utc": document.get("started_utc"), "finished_utc": document.get("finished_utc")}
    return {"run_id": run.name, "scene_id": scene, "run_directory": str(run),
            "depth_range_m": [depth_min, depth_max],
            "source_hashes": hashes, "views": views, "rounds": rounds, "summary": summary,
            "candidate_rows": [row for round_data in rounds for row in round_data["candidate_rows"]],
            "definitions": {"coverage": "fraction of target-ROI voxels touched by valid rays; not fruit completeness",
                            "target_voxels": "voxels with positive target semantic evidence; not verified new surface",
                            "sensitivity": "recorded-state candidate filtering only; not alternative-policy rollout",
                            "positive_gain_tolerance": "max(1e-9, abs(current_gain) * 1e-6)"}}


def analyze_run(run: Path) -> dict:
    """Return validated metrics without modifying any original run artifact."""
    try:
        return _analyze_run(run)
    except AnalysisError:
        raise
    except (KeyError, IndexError, TypeError, OSError) as error:
        raise AnalysisError(f"missing or malformed run evidence: {error}") from error
