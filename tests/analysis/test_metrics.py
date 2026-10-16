"""Evidence provenance, voxel arithmetic, and frozen-state gain semantics."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("analysis_metrics", ROOT / "tools/analysis/metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def _write_json(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def _bind_preview(run):
    document = json.loads((run / "execution.json").read_text())
    digest = hashlib.sha256((run / "preview.json").read_bytes()).hexdigest()
    document["execution_authorization"]["plan_sha256"] = digest
    document["motion_steps"][0]["plan_sha256_binding"] = digest
    _write_json(run / "execution.json", document)


def _candidates(terminal=False):
    records = []
    for radius in metrics.RADII_MM:
        for direction in range(19):
            scored = direction < 3
            improvement = (-.001 if terminal else .01) if direction == 0 else (.5e-6 if direction == 1 else -.001)
            record = {"candidate_index": len(records), "direction_index": direction,
                      "radius_m": radius / 1000, "target_depth_m": .425 if scored else .46,
                      "minimum_target_view_distance_m": .42,
                      "ik_gate_passed": scored, "gain_scored": scored,
                      "T_base_camera": np.eye(4).tolist(), "T_base_link7": np.eye(4).tolist()}
            if scored:
                record.update(current_gain=1., candidate_gain=1. + improvement,
                              gain_improvement=improvement)
            records.append(record)
    return records


@pytest.fixture
def run(tmp_path):
    run = tmp_path / "recorded_run"
    (run / "maps").mkdir(parents=True)
    (run / "observations").mkdir()
    target_audit = {"mask_pixels": 16, "valid_mask_pixels": 12,
                    "median_depth_m": .5, "depth_mad_m": .002, "depth_layer_count": 1}
    first_candidates = _candidates()
    preview = {"scene_id": "test_scene", "ik_candidates": first_candidates,
               "selected_candidate": copy.deepcopy(first_candidates[0])}
    _write_json(run / "preview.json", preview)
    steps = []
    roi_indices = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1)]
    targets = [{0, 1}, {1, 2}, {1, 2, 3}, {1, 3, 4}, {1, 3, 4, 5}]
    for index in range(5):
        observation_id = f"view_{index}"
        mask = np.full((4, 4), 255, np.uint8)
        depth = np.full((4, 4), .5, np.float32)
        depth[0] = np.nan
        np.savez(run / "observations" / f"{observation_id}.npz",
                 schema="strawberry_canonical_observation_snapshot/v1",
                 scene_id="test_scene", observation_id=observation_id,
                 world_frame="base_link", mask=mask, depth_m=depth)
        observed = np.zeros((4, 4, 4), dtype=bool)
        for xyz in roi_indices[:index + 1]:
            observed[xyz] = True
        target = np.zeros_like(observed)
        target.flat[list(targets[index])] = True
        np.savez(run / "maps" / f"{observation_id}.npz",
                 schema="strawberry_gradient_nbv_map/v1", scene_id="test_scene",
                 observation_id=observation_id, world_frame="base_link",
                 dimensions=[4, 4, 4], origin_m=[-.02, -.02, -.02], voxel_size_m=.01,
                 target_center_m=[0., 0., 0.], target_roi_size_m=[.02, .02, .02],
                 target=target, occupied=target, observed=observed, coverage=(index + 1) / 8)
        if index:
            candidates = _candidates()
            selected = candidates[[0, 19, 76, 95][index - 1]]
            steps.append({"step_index": index, "execution": {"action_status": 4},
                          "post_map_update": {"success": True, "scene_id": "test_scene",
                                              "observation_id": observation_id, "coverage": (index + 1) / 8},
                          "post_target": target_audit,
                          "post_pose_error": {"translation_m": .001},
                          "cumulative_camera_motion": {"actual_step_translation_m": .01 * index},
                          "source_map_observation_id": f"view_{index - 1}",
                          "selected_candidate": copy.deepcopy(selected), "ik_candidates": candidates})
    steps[-1]["next_view_convergence"] = {"source_observation_id": "view_4", "motion_goal_sent": False,
                                         "ik_candidates": _candidates(terminal=True)}
    document = {"scene_id": "test_scene", "motion_steps": steps,
                "corrected_observation": {"scene_id": "test_scene", "observation_id": "view_0"},
                "final_target": target_audit, "raw_next_view": {"observation_id": "view_0", "coverage": .125},
                "configuration": {"request_semantics": {"depth_min_m": .2, "depth_max_m": 2.5}},
                "session_policy": {"reachable_candidate_direction_count": 18, "coverage_target": .8},
                "session_progress": {"converged_early": True, "convergence_reason": "no positive gain",
                                     "scientific_acceptance_passed": False},
                "execution_authorization": {},
                # This diagnostic must never replace the SHA-bound first round.
                "ik_candidates": [{"diagnostic_only": True}]}
    _write_json(run / "execution.json", document)
    _bind_preview(run)
    return run


def _rewrite_npz(path, **changes):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values.update(changes)
    np.savez(path, **values)


def test_pairing_roi_set_differences_and_no_source_mutation(run):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}
    result = metrics.analyze_run(run)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}
    assert before == after
    assert result["summary"]["roi_voxel_count"] == 8  # Not the 64-voxel full map.
    assert [v["roi_observed_voxels"] for v in result["views"]] == [1, 2, 3, 4, 5]
    assert [(v["target_added"], v["target_removed"]) for v in result["views"]] == [(0, 0), (1, 1), (1, 0), (1, 1), (1, 0)]
    assert [v["valid_depth_fraction"] for v in result["views"]] == [.75] * 5
    assert result["views"][0]["depth_mad_mm"] == 2  # Recorded near-layer MAD, not raw-mask MAD.
    assert result["views"][0]["position_error_mm"] is None
    assert result["summary"]["completed_actions"] == result["summary"]["map_updates"] == 4
    assert result["summary"]["coverage_gain_pp"] == 50
    assert result["source_hashes"] == {str(p): digest for p, digest in before.items()}
    json.dumps(result, allow_nan=False)


def test_frozen_preview_source_tolerance_and_unscored_nulls(run):
    result = metrics.analyze_run(run)
    assert [r["selected_radius_mm"] for r in result["rounds"]] == [100, 75, 10, 5, None]
    first = result["rounds"][0]
    assert first["source"].endswith("preview.json#ik_candidates")
    assert first["radii"][0]["positive"] == 1  # Tiny >0 improvement is below source tolerance.
    unscored = first["candidate_rows"][3]
    assert all(unscored[k] is None for k in ("current_gain", "candidate_gain", "gain_improvement", "positive"))
    assert first["thresholds"][2]["best_gain_improvement"] is None
    assert first["thresholds"][2]["scored"] == 0
    assert len(first["radius_thresholds"]) == 18
    assert result["rounds"][-1]["thresholds"][0]["positive"] == 0


@pytest.mark.parametrize("kind", ["maps", "observations"])
def test_missing_or_duplicate_snapshot_is_an_error(run, kind):
    path = run / kind / "view_2.npz"
    duplicate = path.with_name("duplicate.npz")
    shutil.copyfile(path, duplicate)
    with pytest.raises(metrics.AnalysisError, match="duplicate"):
        metrics.analyze_run(run)
    duplicate.unlink()
    path.unlink()
    with pytest.raises(metrics.AnalysisError, match="missing"):
        metrics.analyze_run(run)


@pytest.mark.parametrize("field,value", [("origin_m", [.0, -.02, -.02]),
                                        ("voxel_size_m", .011),
                                        ("world_frame", "another_frame")])
def test_grid_mismatch_rejected_before_voxel_differences(run, field, value):
    _rewrite_npz(run / "maps/view_2.npz", **{field: value})
    with pytest.raises(metrics.AnalysisError, match="grid or world frame differs"):
        metrics.analyze_run(run)


def test_roi_coverage_is_recomputed(run):
    _rewrite_npz(run / "maps/view_1.npz", coverage=.5)
    with pytest.raises(metrics.AnalysisError, match="observed target-ROI"):
        metrics.analyze_run(run)


def test_first_round_requires_matching_frozen_preview_sha(run):
    with (run / "preview.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(metrics.AnalysisError, match="preview SHA"):
        metrics.analyze_run(run)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "relax"])
def test_candidate_audit_must_be_complete_unique_and_only_tightened(run, mutation):
    path = run / "preview.json"
    preview = json.loads(path.read_text())
    if mutation == "missing":
        preview["ik_candidates"].pop()
    elif mutation == "duplicate":
        preview["ik_candidates"][1] = copy.deepcopy(preview["ik_candidates"][0])
    else:
        preview["ik_candidates"][0]["minimum_target_view_distance_m"] = .43
    _write_json(path, preview)
    _bind_preview(run)
    with pytest.raises(metrics.AnalysisError, match="missing|duplicate|relax"):
        metrics.analyze_run(run)


def test_nonexecution_preview_snapshots_are_excluded(run):
    for directory in ("maps", "observations"):
        source = run / directory / "view_0.npz"
        other = source.with_name("preview.npz")
        shutil.copyfile(source, other)
        _rewrite_npz(other, observation_id="preview_only")
    result = metrics.analyze_run(run)
    assert len(result["views"]) == 5
    assert not any(path.endswith("preview.npz") for path in result["source_hashes"])
