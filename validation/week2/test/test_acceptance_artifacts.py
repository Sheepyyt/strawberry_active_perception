"""Regression checks for the small, committed Week 2 evidence documents."""

import json
from pathlib import Path


ARTIFACTS = Path(__file__).parents[1] / "artifacts"


def _load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def test_g2_artifact_meets_the_approved_numerical_gates() -> None:
    report = _load("g2_nbv_acceptance.json")
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    coverage = report["multiview"]["coverage"]
    assert all(after >= before for before, after in zip(coverage, coverage[1:]))
    assert sum(after > before for before, after in zip(coverage, coverage[1:])) >= 3
    assert coverage[-1] - coverage[0] >= 0.20
    assert report["performance"]["wall_time_sec"] <= 2.0
    assert report["performance"]["peak_allocated_mib"] <= 2048.0
    assert report["mask_semantics"]["gate_passed"] is True


def test_g3_artifact_combines_scale_and_later_multiframe_edge_evidence() -> None:
    report = _load("g3_camera_runtime_summary.json")
    assert report["status"] == "passed_for_stationary_camera_stage"
    assert report["hardware_transport"]["negotiated_speed_mbps"] == 480.0
    assert report["hardware_transport"]["superspeed"] is False
    run = report["corrected_long_run"]
    stamps = run["sensor_stamp_metrics"]
    assert run["gate_result"] == "passed_for_exact_profile"
    assert stamps["color_rate_hz"] >= 9.0
    assert stamps["depth_rate_hz"] >= 9.0
    assert stamps["color_estimated_drop_fraction"] <= 0.01
    assert stamps["depth_estimated_drop_fraction"] <= 0.01
    assert run["rgb_depth_pairing_within_5ms"]["absolute_skew_p99_ms"] <= 5.0
    assert report["final_gate_decision"]["full_G3"] == (
        "passed_for_stationary_camera_stage"
    )
    passed = set(report["gate_items"]["passed"])
    assert "checkerboard_metric_scale_accuracy" in passed
    assert "rgb_pnp_registered_depth_plane_consistency" in passed
    assert "twenty_frame_four_side_physical_color_depth_edge_alignment" in passed
    assert "color_depth_edge_alignment" not in report["gate_items"]["not_tested"]
    assert report["checkerboard_geometry"]["physical_edge_result"] == "not_testable"
    assert report["checkerboard_multiframe_edge_alignment"]["status"] == "passed"
    smoke = report["real_single_view_nbv_smoke"]
    assert smoke["status"] == "passed"
    assert smoke["mask_pixels"] >= 200
    assert smoke["next_view_finite"] is True
    assert smoke["translation_step_m"] <= 0.10
    assert smoke["look_at_error_deg"] <= 1.0
    assert smoke["duplicate_observation_idempotent"] is True
    assert smoke["motion_executed"] is False


def test_checkerboard_metric_passes_while_edge_remains_not_testable() -> None:
    report = _load("g3_checkerboard_geometry.json")
    assert report["status"] == "partial_pass_edge_not_testable"
    assert report["measured_reference"]["square_size_mm"] == 30.0
    assert report["detection"]["detected_corner_count"] == 88
    assert report["depth_plane"]["residual_p90_mm"] <= 2.0
    assert report["depth_plane"]["roi_valid_depth_fraction"] >= 0.8
    assert report["metric_scale"]["median_absolute_error_percent"] <= 2.0
    assert report["metric_scale"]["horizontal_median_absolute_error_percent"] <= 2.0
    assert report["metric_scale"]["vertical_median_absolute_error_percent"] <= 2.0
    comparison = report["rgb_pnp_vs_registered_depth"]
    assert comparison["centre_3d_difference_mm"] <= comparison[
        "metric_gate_limit_mm"
    ]
    assert comparison["plane_normal_difference_deg"] <= 1.0
    edge = report["physical_edge_alignment"]
    assert edge["gate_result"] == "not_testable"
    assert edge["gate_passed"] is None
    assert edge["measurement_prerequisites"][
        "enough_physical_depth_separation"
    ] is False
    assert edge["measurement_prerequisites"]["enough_synchronized_frames"] is False
    assert len(edge["sides"]) == 4
    assert all(side["accepted_profiles"] > 0 for side in edge["sides"])
    assert report["motion_executed"] is False


def test_multiframe_checkerboard_passes_all_four_physical_edges() -> None:
    report = _load("g3_checkerboard_multiframe_geometry.json")
    assert report["status"] == "passed"
    assert report["input"]["frame_count"] == 20
    assert report["input"]["unique_observation_ids"] == 20
    assert all(report["checks"].values())
    edge = report["physical_edge_alignment"]
    assert edge["gate_passed"] is True
    assert edge["p95_absolute_error_px"] <= edge["global_p95_limit_px"]
    assert len(edge["sides"]) == 4
    for side in edge["sides"]:
        assert side["gate_passed"] is True
        assert side["accepted_profile_fraction"] >= edge[
            "minimum_profile_fraction_per_side"
        ]
        assert side["accepted_along_edge_span_fraction"] >= edge[
            "minimum_along_edge_span_fraction_per_side"
        ]
        assert side["median_absolute_error_px"] <= edge["side_median_limit_px"]
        assert side["median_background_separation_m"] >= edge[
            "minimum_background_separation_m"
        ]
    assert report["motion_executed"] is False


def test_real_single_view_nbv_artifact_is_safe_and_finite() -> None:
    report = _load("g3_real_single_view_nbv.json")
    assert report["status"] == "passed"
    observation = report["canonical_observation"]
    assert observation["mask_pixels"] >= 200
    assert observation["mask_valid_depth_fraction"] >= 0.8
    assert observation["skew_matches_header_difference_within_1ns"] is True
    next_view = report["next_view"]
    assert next_view["success"] is True
    assert next_view["translation_step_m"] <= 0.1
    assert next_view["look_at_error_deg"] <= 1.0
    assert next_view["compute_time_ms"] <= 2000.0
    assert report["idempotency"]["map_updated_once"] is True
    assert not any(report["runtime_safety"].values())


def test_g4_artifact_contains_live_zero_motion_evidence() -> None:
    report = _load("g4_gradient_to_placo_pipeline.json")
    assert report["status"] == "passed"
    assert report["idempotency"]["duplicate_action_exact_cached_result"] is True
    assert report["next_view"]["translation_step_m"] <= 0.10
    assert report["next_view"]["look_at_error_deg"] <= 1.0
    safety = report["safety"]
    assert safety["passed"] is True
    assert safety["controller_simulation_mode"] is True
    assert safety["controller_execution_enabled"] is False
    assert safety["motion_command_count_observed"] == 0
    assert safety["known_vendor_driver_nodes_detected"] == []
    assert safety["move_action_used_by_fixture"] is False
    result = report["ik_preview"]["values"]
    assert result["success"] is True
    assert result["controlled_frame"] == "link7"


def test_final_report_records_resolved_g0_dependency() -> None:
    report = _load("final_test_report.json")
    assert report["gate_summary"]["G0"] == "passed"
    assert report["software_regression"]["camera_driver_packages"][
        "rosdep_status"
    ] == "all_system_dependencies_satisfied"
