"""Pure safety-contract tests for the one-shot real NBV supervisor."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from strawberry_active_perception_bridge.real_nbv_contract import (
    CONVERGENCE_COVERAGE_DELTA,
    DEFAULT_ALPHAS,
    LARGE_MOTION_LIMITS,
    LARGE_MOTION_PROFILE,
    MotionSessionLedger,
    NEAR_BEST_GAIN_RATIO,
    REACHABLE_VIEW_DIRECTION_COUNT,
    REACHABLE_VIEW_RADII_M,
    aggregate_depth_mask,
    camera_matrix,
    compare_gradient_candidates,
    decode_depth_32fc1,
    decode_mask_mono8,
    estimate_target_center,
    five_frame_identity_sha256,
    independently_segmented_camera_candidates,
    motion_limits_for_profile,
    normalize_nbv_configuration,
    project_numerical_step_overshoot,
    reachable_view_lattice,
    require_unit_quaternion,
    segmented_camera_candidates,
    select_near_best_reachable_candidate,
    validate_ik_solution,
    validate_raw_next_view,
    wire_bool,
    wire_uint8,
)
from strawberry_active_perception_bridge.real_nbv_supervisor import (
    DEFAULT_PLAN_PATH,
    DEFAULT_PLAN_SHA256,
    DEFAULT_REPORT_SHA256,
    GateClosureError,
    MAX_FROZEN_TARGET_CENTER_DRIFT_M,
    RealNBVSupervisor,
    SupervisorError,
    _authorization_token,
    _failure_gates_closed,
    _load_execution_plan,
    _require_v3_execution_plan,
    _session_policy,
    _validate_bound_session_policy,
    _validate_controller_parameter_values,
    _validate_upstream_mask_source,
    _verify_new_aggregate_batch,
    _verify_new_batch_pair,
)
from strawberry_active_perception_bridge.transforms import (
    pose_components_to_matrix,
)


def test_upstream_mask_can_be_disabled_only_for_learned_observation() -> None:
    assert _validate_upstream_mask_source(
        "/strawberry/perception/observation", True
    )
    assert not _validate_upstream_mask_source(
        "/strawberry/perception/learned_observation", False
    )
    with pytest.raises(SupervisorError, match="learned Observation"):
        _validate_upstream_mask_source(
            "/strawberry/perception/observation", False
        )
    with pytest.raises(SupervisorError, match="boolean"):
        _validate_upstream_mask_source(
            "/strawberry/perception/learned_observation", 0
        )


def test_frozen_target_identity_gate_matches_measured_camera_repeatability() -> None:
    """The identity gate covers measured jitter but remains tightly bounded."""
    assert MAX_FROZEN_TARGET_CENTER_DRIFT_M == pytest.approx(0.005)
    assert MAX_FROZEN_TARGET_CENTER_DRIFT_M < 0.020


def _image(array: np.ndarray, encoding: str, padding: int = 0):
    if encoding == "32FC1":
        rows = array.astype("<f4").view(np.uint8).reshape(array.shape[0], -1)
        unit = 4
    else:
        rows = array.astype(np.uint8)
        unit = 1
    padded = np.zeros((array.shape[0], rows.shape[1] + padding), dtype=np.uint8)
    padded[:, : rows.shape[1]] = rows
    return SimpleNamespace(
        encoding=encoding,
        height=array.shape[0],
        width=array.shape[1],
        step=array.shape[1] * unit + padding,
        is_bigendian=False,
        data=padded.tobytes(),
    )


def test_padded_depth_mask_and_robust_target_centre() -> None:
    depth = np.full((20, 30), 1.0, dtype=np.float32)
    depth[5, 5] = 2.4
    mask = np.full((20, 30), 255, dtype=np.uint8)
    decoded_depth = decode_depth_32fc1(_image(depth, "32FC1", padding=8))
    decoded_mask = decode_mask_mono8(_image(mask, "mono8", padding=3))
    info = SimpleNamespace(
        height=20,
        width=30,
        k=[100.0, 0.0, 14.5, 0.0, 100.0, 9.5, 0.0, 0.0, 1.0],
        d=[0.0] * 5,
    )
    intrinsic = camera_matrix(info, decoded_depth.shape)
    estimate = estimate_target_center(
        decoded_depth,
        decoded_mask,
        intrinsic,
        minimum_pixels=200,
    )
    np.testing.assert_allclose(estimate.camera_xyz_m, (0.005, 0.005, 1.0))
    assert estimate.mask_pixels == 600
    assert estimate.valid_mask_pixels == 600
    assert estimate.depth_layer_count == 2
    assert estimate.selected_layer_pixels == 599
    assert estimate.retained_pixels == 599


def test_target_centre_selects_nearest_supported_depth_layer() -> None:
    depth = np.full((20, 30), 2.0, dtype=np.float32)
    depth.reshape(-1)[:220] = 0.36
    mask = np.full(depth.shape, 255, dtype=np.uint8)
    intrinsic = np.asarray(
        [[100.0, 0.0, 14.5], [0.0, 100.0, 9.5], [0.0, 0.0, 1.0]]
    )

    estimate = estimate_target_center(
        depth, mask, intrinsic, minimum_pixels=200
    )

    assert estimate.valid_mask_pixels == 600
    assert estimate.depth_layer_count == 2
    assert estimate.selected_layer_pixels == 220
    assert estimate.retained_pixels == 220
    assert estimate.median_depth_m == pytest.approx(0.36)
    assert estimate.camera_xyz_m[2] == pytest.approx(0.36)


def test_target_centre_does_not_let_background_satisfy_foreground_gate() -> None:
    depth = np.full((20, 30), 2.0, dtype=np.float32)
    depth.reshape(-1)[:180] = 0.36
    mask = np.full(depth.shape, 255, dtype=np.uint8)
    intrinsic = np.asarray(
        [[100.0, 0.0, 14.5], [0.0, 100.0, 9.5], [0.0, 0.0, 1.0]]
    )

    with pytest.raises(ValueError, match="nearest target-depth layer has 180"):
        estimate_target_center(depth, mask, intrinsic, minimum_pixels=200)


def test_target_centre_ignores_tiny_nearer_outlier_layer() -> None:
    depth = np.full((20, 30), 2.0, dtype=np.float32)
    depth.reshape(-1)[:250] = 0.36
    depth.reshape(-1)[:3] = 0.21
    mask = np.full(depth.shape, 255, dtype=np.uint8)
    intrinsic = np.asarray(
        [[100.0, 0.0, 14.5], [0.0, 100.0, 9.5], [0.0, 0.0, 1.0]]
    )

    estimate = estimate_target_center(
        depth, mask, intrinsic, minimum_pixels=200
    )

    assert estimate.retained_pixels == 247
    assert estimate.median_depth_m == pytest.approx(0.36)


def test_mask_and_quaternion_are_fail_closed() -> None:
    invalid_mask = np.zeros((20, 20), dtype=np.uint8)
    invalid_mask[0, 0] = 1
    with pytest.raises(ValueError, match="0 or 255"):
        decode_mask_mono8(_image(invalid_mask, "mono8"))
    with pytest.raises(ValueError, match="norm must be 1"):
        require_unit_quaternion((0.0, 0.0, 0.0, 2.0))


def test_jazzy_one_byte_uint8_fields_are_decoded_numerically() -> None:
    assert wire_uint8(b"\x00") == 0
    assert wire_uint8(b"\x01") == 1
    assert wire_bool(b"\x00") is False
    assert wire_bool(b"\x01") is True
    depth = np.asarray([[1.0]], dtype=np.float32)
    message = _image(depth, "32FC1")
    message.is_bigendian = b"\x00"
    assert decode_depth_32fc1(message)[0, 0] == pytest.approx(1.0)


def test_five_frame_depth_mask_aggregation_uses_all_pixel_evidence() -> None:
    depth_values = np.asarray(
        [
            [[1.0, 2.0, np.nan], [1.0, np.nan, 5.0]],
            [[9.0, 4.0, 10.0], [np.nan, np.nan, 7.0]],
            [[3.0, np.nan, 30.0], [3.0, np.nan, 9.0]],
            [[7.0, 8.0, np.nan], [np.nan, np.nan, np.inf]],
            [[5.0, 6.0, 20.0], [np.nan, np.nan, -np.inf]],
        ]
    )
    vote_counts = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    masks = np.zeros_like(depth_values, dtype=np.uint8)
    for frame_index in range(5):
        masks[frame_index][frame_index < vote_counts] = 255

    result = aggregate_depth_mask(depth_values, masks)

    np.testing.assert_allclose(
        result.depth_m,
        np.asarray([[5.0, 5.0, 20.0], [np.nan, np.nan, 7.0]]),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        result.finite_support,
        np.asarray([[5, 4, 3], [2, 0, 3]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(result.mask_votes, vote_counts)
    np.testing.assert_array_equal(
        result.mask,
        np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8),
    )
    assert result.audit_counts == {
        "input_frame_count": 5,
        "pixel_count": 6,
        "finite_depth_sample_count": 17,
        "output_finite_depth_pixel_count": 4,
        "output_nan_depth_pixel_count": 2,
        "foreground_mask_sample_count": 15,
        "output_foreground_mask_pixel_count": 3,
    }
    assert all(type(value) is int for value in result.audit_counts.values())


def test_five_frame_aggregation_is_order_independent_not_frame_selection() -> None:
    second_pixels = (10.0, 6.0, 9.0, 7.0, 8.0)
    depths = [
        np.asarray([[float(index), second_pixels[index - 1]]], dtype=np.float64)
        for index in range(1, 6)
    ]
    masks = [
        np.asarray([[255 if index in (0, 2, 4) else 0, 255 if index < 2 else 0]])
        for index in range(5)
    ]
    forward = aggregate_depth_mask(depths, masks)
    reverse = aggregate_depth_mask(depths[::-1], masks[::-1])

    np.testing.assert_array_equal(forward.depth_m, np.asarray([[3.0, 8.0]]))
    assert not any(np.array_equal(forward.depth_m, frame) for frame in depths)
    np.testing.assert_array_equal(forward.depth_m, reverse.depth_m)
    np.testing.assert_array_equal(forward.mask, reverse.mask)
    np.testing.assert_array_equal(forward.finite_support, reverse.finite_support)
    assert forward.audit_counts == reverse.audit_counts


def test_five_frame_aggregation_rejects_count_shape_and_mask_errors() -> None:
    depth = np.ones((2, 3), dtype=np.float64)
    mask = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly 5"):
        aggregate_depth_mask([depth] * 4, [mask] * 5)
    with pytest.raises(ValueError, match="does not match"):
        aggregate_depth_mask([depth] * 4 + [np.ones((3, 2))], [mask] * 5)
    bad_masks = [mask.copy() for _ in range(5)]
    bad_masks[2][0, 0] = 1
    with pytest.raises(ValueError, match="exactly 0 or 255"):
        aggregate_depth_mask([depth] * 5, bad_masks)


def test_five_frame_identity_is_canonical_and_recomputed() -> None:
    scenes = [f"scene_{index}" for index in range(5)]
    observations = [f"obs_{index}" for index in range(5)]
    stamps = [100 + index for index in range(5)]
    first = five_frame_identity_sha256(scenes, observations, stamps)
    second = five_frame_identity_sha256(
        tuple(scenes), tuple(observations), tuple(stamps)
    )
    assert first == second
    assert len(first) == 64
    changed = observations.copy()
    changed[4] = "obs_changed"
    assert five_frame_identity_sha256(scenes, changed, stamps) != first
    with pytest.raises(ValueError, match="increasing"):
        five_frame_identity_sha256(scenes, observations, stamps[::-1])


def _nbv_configuration(scene_id: str = "frozen_scene") -> dict:
    return {
        "scene_id": scene_id,
        "world_frame": "base_link",
        "target_center_m": [0.2, -0.5, 0.25],
        "map_size_m": [0.3, 0.3, 0.3],
        "target_roi_size_m": [0.15, 0.15, 0.15],
        "observation_min_m": [0.25, 0.03, 0.33],
        "observation_max_m": [0.28, 0.06, 0.36],
        "voxel_size_m": 0.003,
        "depth_min_m": 0.2,
        "depth_max_m": 2.5,
        "samples_per_ray": 128,
        "optimization_steps": 10,
        "max_step_m": 0.005,
        "random_seed": 0,
    }


def test_configure_nbv_semantics_freeze_wire_values_and_map_origin() -> None:
    evidence = normalize_nbv_configuration(_nbv_configuration())
    assert evidence.request["voxel_size_m"] == float(np.float32(0.003))
    assert evidence.request["max_step_m"] == float(np.float32(0.005))
    assert evidence.voxel_dimensions == (100, 100, 100)
    expected_origin = np.asarray([0.2, -0.5, 0.25]) - (
        np.asarray([100, 100, 100]) * float(np.float32(0.003)) / 2.0
    )
    np.testing.assert_allclose(evidence.map_origin_m, expected_origin)
    assert len(evidence.sha256) == 64
    changed = _nbv_configuration()
    changed["target_center_m"][0] += 0.000001
    changed_evidence = normalize_nbv_configuration(changed)
    assert changed_evidence.sha256 != evidence.sha256
    assert changed_evidence.map_origin_m[0] != evidence.map_origin_m[0]
    with pytest.raises(ValueError, match="max_step"):
        invalid = _nbv_configuration()
        invalid["max_step_m"] = 0.006
        normalize_nbv_configuration(invalid)


def test_large_motion_profile_allows_only_a_separate_ten_centimetre_config() -> None:
    values = _nbv_configuration()
    values["max_step_m"] = 0.10
    with pytest.raises(ValueError, match="selected motion profile"):
        normalize_nbv_configuration(values)
    evidence = normalize_nbv_configuration(
        values,
        max_step_ceiling_m=LARGE_MOTION_LIMITS.maximum_camera_step_m,
    )
    assert evidence.request["max_step_m"] == float(np.float32(0.10))
    assert motion_limits_for_profile(LARGE_MOTION_PROFILE) == LARGE_MOTION_LIMITS
    assert LARGE_MOTION_LIMITS.minimum_camera_step_m == pytest.approx(0.001)
    assert not LARGE_MOTION_LIMITS.minimum_step_inclusive
    assert REACHABLE_VIEW_RADII_M == (0.10, 0.075, 0.05, 0.025, 0.01, 0.005)
    assert REACHABLE_VIEW_DIRECTION_COUNT == 18
    assert NEAR_BEST_GAIN_RATIO == pytest.approx(0.90)


def test_reachable_view_lattice_is_deterministic_large_first_and_looks_at_target() -> None:
    current = np.eye(4)
    current[:3, 3] = (0.0, 0.0, -0.6)
    raw = current.copy()
    raw[0, 3] += 0.10
    target = np.zeros(3)
    first = reachable_view_lattice(
        current,
        raw,
        target,
        (-0.31, -0.31, -0.91),
        (0.31, 0.31, -0.29),
    )
    second = reachable_view_lattice(
        current,
        raw,
        target,
        (-0.31, -0.31, -0.91),
        (0.31, 0.31, -0.29),
    )
    assert 1 < len(first) <= 256
    assert [item.radius_m for item in first] == [item.radius_m for item in second]
    assert first[0].radius_m == pytest.approx(0.10)
    assert first[-1].radius_m == pytest.approx(0.005)
    for lhs, rhs in zip(first, second):
        np.testing.assert_allclose(lhs.pose, rhs.pose)
        optical_forward = lhs.pose[:3, 2]
        target_direction = target - lhs.pose[:3, 3]
        target_direction /= np.linalg.norm(target_direction)
        np.testing.assert_allclose(optical_forward, target_direction, atol=1e-12)
        assert lhs.motion.translation_m == pytest.approx(lhs.radius_m)


def test_reachable_view_lattice_filters_bounds_and_invalid_inputs() -> None:
    current = np.eye(4)
    current[2, 3] = -0.6
    raw = current.copy()
    raw[0, 3] = 0.1
    candidates = reachable_view_lattice(
        current,
        raw,
        (0.0, 0.0, 0.0),
        (-0.011, -0.011, -0.611),
        (0.011, 0.011, -0.589),
    )
    assert all(candidate.radius_m <= 0.01 + 1e-12 for candidate in candidates)
    with pytest.raises(ValueError, match="strictly descending"):
        reachable_view_lattice(
            current,
            raw,
            (0.0, 0.0, 0.0),
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
            radii_m=(0.01, 0.02),
        )


def test_reachable_gain_selection_prefers_larger_near_best_not_unreachable() -> None:
    selection = select_near_best_reachable_candidate(
        current_gain=1.0,
        candidate_gains=(1.80, 1.75, 1.20),
        candidate_translations_m=(0.025, 0.10, 0.075),
        joint_limit_clearances_rad=(0.3, 0.2, 0.4),
    )
    assert selection is not None
    # The 100 mm candidate retains >90% of the best improvement, so the
    # demonstrably larger move wins even though the 25 mm point scores highest.
    assert selection.selected_index == 1
    assert selection.useful_indices == (0, 1, 2)
    assert selection.near_best_indices == (0, 1)

    no_improvement = select_near_best_reachable_candidate(
        current_gain=1.0,
        candidate_gains=(1.0, 0.9),
        candidate_translations_m=(0.10, 0.05),
        joint_limit_clearances_rad=(0.2, 0.3),
    )
    assert no_improvement is None


def test_supervisor_large_profile_filters_ik_before_one_read_only_gain_call() -> None:
    class _SolveClient:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def wait_for_service(timeout_sec):
            return timeout_sec == 5.0

        def call_async(self, _request):
            self.calls += 1
            result = SimpleNamespace(
                success=True,
                code=0,
                reason="ok",
                position_error_m=0.0,
                orientation_error_rad=0.0,
                sigma_min=0.2,
                condition_number=5.0,
                max_joint_delta_rad=0.0,
                solution_joint_state=SimpleNamespace(
                    name=[f"joint{i}" for i in range(1, 8)],
                    position=[0.0] * 7,
                ),
            )
            return SimpleNamespace(result=result)

    class _GainClient:
        def __init__(self, current):
            self.current = current
            self.calls = 0
            self.last_request = None

        @staticmethod
        def wait_for_service(timeout_sec):
            return timeout_sec == 5.0

        def call_async(self, request):
            self.calls += 1
            self.last_request = request
            gains = []
            for pose in request.candidate_poses:
                position = np.array(
                    (
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                    )
                )
                distance = np.linalg.norm(position - self.current[:3, 3])
                improvement = 0.80 if abs(distance - 0.025) < 1e-6 else 0.20
                if abs(distance - 0.10) < 1e-6:
                    improvement = 0.75
                gains.append(1.0 + improvement)
            return SimpleNamespace(
                success=True,
                code=0,
                reason="ok",
                current_gain=1.0,
                candidate_gains=gains,
            )

    current = np.eye(4)
    current[2, 3] = -0.6
    raw = current.copy()
    raw[0, 3] = 0.1
    node = SimpleNamespace(
        _active_nbv_configuration={
            "target_center_m": [0.0, 0.0, 0.0],
            "observation_min_m": [-0.31, -0.31, -0.91],
            "observation_max_m": [0.31, 0.31, -0.29],
        },
        _solve_client=_SolveClient(),
        _evaluate_candidates_client=_GainClient(current),
        report=SimpleNamespace(transform_link7_camera_optical=np.eye(4)),
        base_frame="base_link",
        scene_id="scene",
        operation_timeout=20.0,
        motion_limits=LARGE_MOTION_LIMITS,
        get_parameter=lambda name: SimpleNamespace(
            value={
                "max_selected_camera_rotation_deg": 15.0,
                "max_ik_joint_delta_rad": 0.35,
            }[name]
        ),
        _fresh_joints=lambda: np.zeros(7),
        _wait_future=lambda future, _timeout, _label: future,
    )
    selected, records = RealNBVSupervisor._solve_reachable_view_lattice(
        node,
        current,
        raw,
        SimpleNamespace(sec=10, nanosec=0),
        "obs",
    )
    assert selected is not None
    assert selected["camera_translation_m"] == pytest.approx(0.10)
    assert selected["gain_improvement"] == pytest.approx(0.75)
    assert selected["selection_contract"].startswith("IK-safe candidates")
    assert node._solve_client.calls == len(records)
    assert node._evaluate_candidates_client.calls == 1
    assert len(node._evaluate_candidates_client.last_request.candidate_poses) <= 256


def test_raw_step_and_segmented_rotation_use_true_so3_distance() -> None:
    current = np.eye(4)
    angle = np.radians(120.0)
    raw = pose_components_to_matrix(
        (0.005, 0.0, 0.0),
        (0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)),
    )
    motion = validate_raw_next_view(current, raw, gain=1.0)
    assert motion.translation_m == pytest.approx(0.005)
    assert np.degrees(motion.rotation_rad) == pytest.approx(120.0)
    segments = segmented_camera_candidates(current, raw)
    assert tuple(item[0] for item in segments) == DEFAULT_ALPHAS
    assert segments[-1][2].translation_m == pytest.approx(0.005 * 0.03125)
    assert np.degrees(segments[-1][2].rotation_rad) == pytest.approx(3.75)


def test_translation_and_rotation_can_be_segmented_independently() -> None:
    """A large rotation can be shortened without losing useful translation."""
    current = np.eye(4)
    angle = np.radians(120.0)
    raw = pose_components_to_matrix(
        (0.005, 0.0, 0.0),
        (0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)),
    )
    candidates = independently_segmented_camera_candidates(current, raw)
    assert len(candidates) == len(DEFAULT_ALPHAS) * (len(DEFAULT_ALPHAS) + 1)
    translation_alpha, rotation_alpha, _, motion = candidates[4]
    assert translation_alpha == 1.0
    assert rotation_alpha == 0.0625
    assert motion.translation_m == pytest.approx(0.005)
    assert np.degrees(motion.rotation_rad) == pytest.approx(7.5)
    translation_alpha, rotation_alpha, candidate, motion = candidates[6]
    assert translation_alpha == 1.0
    assert rotation_alpha == 0.0
    np.testing.assert_allclose(candidate[:3, :3], current[:3, :3])
    assert motion.translation_m == pytest.approx(0.005)
    assert motion.rotation_rad == pytest.approx(0.0)


def test_gradient_direction_disagreement_is_separate_from_endpoint_safety() -> None:
    artifact_current = np.eye(4)
    artifact_target = np.eye(4)
    artifact_target[0, 3] = 0.0025
    live_current = np.eye(4)
    live_current[0, 3] = 0.00002
    live_target = live_current.copy()
    live_target[0, 3] += 0.0025 * np.cos(np.radians(108.0))
    live_target[1, 3] += 0.0025 * np.sin(np.radians(108.0))
    evidence = compare_gradient_candidates(
        artifact_current,
        artifact_target,
        live_current,
        live_target,
    )
    assert np.degrees(
        evidence.translation_direction_disagreement_rad
    ) == pytest.approx(108.0)
    assert evidence.target_motion.translation_m > 0.004
    assert evidence.target_motion.translation_m < evidence.triangle_bound_m
    # This metric describes optimizer repeatability.  Each endpoint retains
    # its own independently validated 2.5 mm motion.
    assert evidence.artifact_step_m == pytest.approx(0.0025)
    assert evidence.live_step_m == pytest.approx(0.0025)


@pytest.mark.parametrize("distance", [0.001, 0.005002])
def test_raw_step_rejects_deadband_and_over_limit(distance: float) -> None:
    target = np.eye(4)
    target[0, 3] = distance
    with pytest.raises(ValueError):
        validate_raw_next_view(np.eye(4), target, gain=1.0)


def test_raw_step_accepts_sub_micrometre_frozen_start_drift() -> None:
    target = np.eye(4)
    target[0, 3] = 0.0050008
    motion = validate_raw_next_view(np.eye(4), target, gain=1.0)
    assert motion.translation_m == pytest.approx(0.0050008)

    with pytest.raises(ValueError, match="overshoot_tolerance_m"):
        validate_raw_next_view(
            np.eye(4), target, gain=1.0, overshoot_tolerance_m=float("nan")
        )


def test_only_sub_micrometre_step_overshoot_is_projected() -> None:
    current = np.eye(4)
    target = np.eye(4)
    target[0, 3] = 0.0050002
    projected, correction = project_numerical_step_overshoot(current, target)
    assert correction == pytest.approx(0.0000002)
    assert np.linalg.norm(projected[:3, 3]) == pytest.approx(0.005)
    assert np.array_equal(projected[:3, :3], target[:3, :3])

    target[0, 3] = 0.005002
    with pytest.raises(ValueError, match="beyond"):
        project_numerical_step_overshoot(current, target)

    # A dynamic target is not frozen until after its fresh preflight pose.  A
    # bounded stationary start drift can therefore be removed before freezing.
    target[0, 3] = 0.005011
    projected, correction = project_numerical_step_overshoot(
        current, target, overshoot_tolerance_m=0.001
    )
    assert correction == pytest.approx(0.000011)
    assert np.linalg.norm(projected[:3, 3]) == pytest.approx(0.005)


def test_ik_validation_recomputes_joint_delta_and_all_precision_gates() -> None:
    current = np.zeros(7)
    solution = np.linspace(0.0, 0.07, 7)
    accepted = validate_ik_solution(
        current_joints=current,
        solution_names=[f"joint{index}" for index in range(1, 8)],
        solution_positions=solution,
        reported_max_joint_delta_rad=0.07,
        position_error_m=0.001,
        orientation_error_rad=0.01,
        sigma_min=0.2,
        condition_number=5.0,
    )
    assert accepted.max_joint_delta_rad == pytest.approx(0.07)
    assert accepted.minimum_joint_limit_clearance_rad > 0.001
    with pytest.raises(ValueError, match="exceeds limit"):
        validate_ik_solution(
            current_joints=current,
            solution_names=[f"joint{index}" for index in range(1, 8)],
            solution_positions=np.linspace(0.0, 0.09, 7),
            reported_max_joint_delta_rad=0.09,
            position_error_m=0.001,
            orientation_error_rad=0.01,
            sigma_min=0.2,
            condition_number=5.0,
        )


def test_ik_validation_rejects_solution_on_research_joint_limit() -> None:
    current = np.zeros(7)
    names = [f"joint{index}" for index in range(1, 8)]
    solution = np.zeros(7)
    solution[1] = -1.7050934149601134
    with pytest.raises(ValueError, match="joint2 clearance"):
        validate_ik_solution(
            current_joints=current,
            solution_names=names,
            solution_positions=solution,
            reported_max_joint_delta_rad=abs(solution[1]),
            position_error_m=0.001,
            orientation_error_rad=0.01,
            sigma_min=0.2,
            condition_number=5.0,
            max_joint_delta_rad=2.0,
        )

    solution[1] += 0.0015
    accepted = validate_ik_solution(
        current_joints=current,
        solution_names=names,
        solution_positions=solution,
        reported_max_joint_delta_rad=abs(solution[1]),
        position_error_m=0.001,
        orientation_error_rad=0.01,
        sigma_min=0.2,
        condition_number=5.0,
        max_joint_delta_rad=2.0,
    )
    assert accepted.minimum_joint_limit_clearance_rad == pytest.approx(0.0015)
    assert accepted.limiting_joint_name == "joint2"


def test_supervisor_motion_surface_is_one_shot_and_explicitly_guarded() -> None:
    source = (
        __import__(
            "strawberry_active_perception_bridge.real_nbv_supervisor",
            fromlist=["__file__"],
        )
        .__file__
    )
    text = open(source, encoding="utf-8").read()
    assert "create_publisher(\n            JointState" not in text
    assert text.count("move_client.send_goal_async(goal)") == 1
    assert "operator_workspace_clearance_confirmed" in text
    assert "EXECUTE_REAL_NBV_ONCE" in text
    assert "execution_plan_sha256" in text
    assert "require_aggregate_execution_plan is a fixed safety invariant" in text
    assert "audited_upper_bounds" in text
    assert "v2 remains evidence-only" in text
    assert "max_frozen_target_center_drift_m" in text
    assert "fresh optimizer output is diagnostic only" in text
    assert "minimum_pixels=1" in text
    assert "resulting aggregate is\n                # still required to pass" in text
    assert "if selected is None and not self.execute_requested" in text
    assert "if not translation_agrees or not rotation_agrees" not in text
    assert "exact_sha_frozen_v3" in text
    assert "for _attempt in range(2)" in text
    assert "isinstance(error, GateClosureError)" in text


def test_execution_plan_is_bound_to_exact_reviewed_bytes_and_semantics() -> None:
    document, digest, path = _load_execution_plan(
        DEFAULT_PLAN_PATH,
        DEFAULT_PLAN_SHA256,
        DEFAULT_REPORT_SHA256,
    )
    assert digest == DEFAULT_PLAN_SHA256
    assert path.is_file()
    assert document["scene_id"] == "real_nbv_once_20260814T032716Z"
    assert document["observation"]["observation_id"] == (
        "agg5_73c6d0cd9429f8b7f2cc478d"
    )
    assert document["selected_candidate"]["alpha"] == 0.5
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _load_execution_plan(
            DEFAULT_PLAN_PATH,
            "0" * 64,
            DEFAULT_REPORT_SHA256,
        )


def _minimal_aggregate_v2_outer_preview() -> dict:
    source_outer = json.loads(
        Path(DEFAULT_PLAN_PATH).read_text(encoding="utf-8")
    )
    source = source_outer.get("execution_plan_candidate", source_outer)
    scene_id = "aggregate_v2_contract_scene"
    selected = copy.deepcopy(source["selected_candidate"])
    exact_tf = copy.deepcopy(source["exact_tf"])
    observation_id = "agg5_contract_observation"
    pose_span = {
        "max_translation_m": 0.0001,
        "max_rotation_deg": 0.05,
        "max_translation_pair_zero_based": [0, 4],
        "max_rotation_pair_zero_based": [1, 3],
        "translation_limit_m": 0.00025,
        "rotation_limit_deg": 0.1,
    }

    def batch(label: str, start_stamp: int, aggregate_id: str) -> dict:
        ids = [f"{label}_{index}" for index in range(5)]
        scenes = [f"{label}_scene_{index}" for index in range(5)]
        stamps = [start_stamp + index for index in range(5)]
        return {
            "selection_performed": False,
            "capture_count": 5,
            "member_observation_ids": ids,
            "raw_scenes": scenes,
            "member_stamps_ns": stamps,
            "pose_span": copy.deepcopy(pose_span),
            "aggregate_observation_id": aggregate_id,
            "aggregate_identity_sha256": five_frame_identity_sha256(
                scenes, ids, stamps
            ),
            "reference_member_index_zero_based": 2,
            "reference_observation_id": ids[2],
            "camera_info_exactly_equal": True,
            "member_exact_tfs": [
                {
                    "requested_stamp_ns": stamp,
                    "returned_stamp_ns": stamp,
                    "stamp_difference_ns": 0,
                }
                for stamp in stamps
            ],
        }

    bootstrap_batch = batch("bootstrap", 100, "agg5_bootstrap")
    planning_batch = batch("planning", 200, observation_id)
    candidate = {
        "schema_version": "strawberry_real_nbv_aggregate_plan/v2",
        "status": "passed",
        "scene_id": scene_id,
        "safety": {
            "passed": True,
            "controller_execution_enabled": False,
            "motion_command_count_observed": 0,
            "gate_clients_created": False,
            "motion_action_clients_created": False,
            "command_publishers_created": False,
            "controller_diagnostic_fresh_after_solve": True,
        },
        "handeye_report": {"sha256": DEFAULT_REPORT_SHA256},
        "observation": {
            "scene_id": scene_id,
            "observation_id": observation_id,
        },
        "nbv": {
            "planned_gain": source["nbv"]["planned_gain"],
            "strict_gain_evidence": "core_invariant_nonzero_translation",
        },
        "aggregation": {
            "capture_count_per_batch": 5,
            "selection_performed": False,
            "bootstrap_member_ids": bootstrap_batch["member_observation_ids"],
            "bootstrap_raw_scenes": bootstrap_batch["raw_scenes"],
            "bootstrap_member_stamps_ns": bootstrap_batch["member_stamps_ns"],
            "bootstrap_aggregate_observation_id": bootstrap_batch[
                "aggregate_observation_id"
            ],
            "bootstrap_aggregate_identity_sha256": bootstrap_batch[
                "aggregate_identity_sha256"
            ],
            "bootstrap_pose_span": bootstrap_batch["pose_span"],
            "planning_member_ids": planning_batch["member_observation_ids"],
            "planning_raw_scenes": planning_batch["raw_scenes"],
            "planning_member_stamps_ns": planning_batch["member_stamps_ns"],
            "planning_aggregate_observation_id": planning_batch[
                "aggregate_observation_id"
            ],
            "planning_aggregate_identity_sha256": planning_batch[
                "aggregate_identity_sha256"
            ],
            "planning_pose_span": planning_batch["pose_span"],
        },
        "selected_candidate": selected,
        "exact_tf": exact_tf,
        "ik": copy.deepcopy(source["ik"]),
    }
    return {
        "schema": "strawberry_real_nbv_once_supervisor/v1",
        "status": "passed_preview_only",
        "execute_requested": False,
        "preview_motion_commands_observed": 0,
        "execution_plan_candidate": candidate,
        "selected_candidate": copy.deepcopy(selected),
        "final_tf": {
            "T_base_camera_optical": copy.deepcopy(
                exact_tf["T_base_camera_optical"]
            )
        },
        "corrected_observation": {"observation_id": observation_id},
        "bootstrap_batch": bootstrap_batch,
        "planning_batch": planning_batch,
    }


def _minimal_frozen_config_v3_outer_preview() -> dict:
    document = _minimal_aggregate_v2_outer_preview()
    source_outer = json.loads(
        Path(DEFAULT_PLAN_PATH).read_text(encoding="utf-8")
    )
    source = source_outer.get("execution_plan_candidate", source_outer)
    candidate = document["execution_plan_candidate"]
    candidate["schema_version"] = (
        "strawberry_real_nbv_frozen_config_plan/v3"
    )
    scene_id = candidate["scene_id"]
    values = copy.deepcopy(source["nbv_configuration"])
    values["scene_id"] = scene_id
    evidence = normalize_nbv_configuration(values)
    voxel_grid = {
        "dimensions": list(evidence.voxel_dimensions),
        "origin_m": evidence.map_origin_m.tolist(),
    }
    candidate["nbv_configuration"] = evidence.request
    candidate["nbv_configuration_sha256"] = evidence.sha256
    candidate["derived_voxel_grid"] = voxel_grid
    candidate["target"] = {
        "center_base_link_m": evidence.request["target_center_m"],
        "planning_batch_center_base_link_m": evidence.request[
            "target_center_m"
        ],
    }
    document["configuration"] = {
        "request_semantics": copy.deepcopy(evidence.request),
        "request_sha256": evidence.sha256,
        "derived_voxel_grid": copy.deepcopy(voxel_grid),
        "response_code": 0,
    }
    document["bootstrap_target"] = {
        "base_xyz_m": copy.deepcopy(evidence.request["target_center_m"])
    }
    document["final_target"] = {
        "base_xyz_m": copy.deepcopy(evidence.request["target_center_m"])
    }
    return document


def _write_exact_json(path: Path, document: dict) -> str:
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_outer_preview_loads_exact_embedded_aggregate_v2_candidate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate_v2_outer_preview.json"
    expected_sha256 = _write_exact_json(
        path, _minimal_aggregate_v2_outer_preview()
    )

    candidate, actual_sha256, resolved_path = _load_execution_plan(
        path,
        expected_sha256,
        DEFAULT_REPORT_SHA256,
    )

    assert actual_sha256 == expected_sha256
    assert resolved_path == path.resolve()
    assert candidate["schema_version"] == (
        "strawberry_real_nbv_aggregate_plan/v2"
    )
    assert candidate["_verified_outer_schema"] == (
        "strawberry_real_nbv_once_supervisor/v1"
    )
    assert candidate["aggregation"]["capture_count_per_batch"] == 5


def test_v3_plan_loads_only_when_full_configure_semantics_recompute(
    tmp_path: Path,
) -> None:
    document = _minimal_frozen_config_v3_outer_preview()
    path = tmp_path / "frozen_config_v3.json"
    digest = _write_exact_json(path, document)
    candidate, actual, _resolved = _load_execution_plan(
        path, digest, DEFAULT_REPORT_SHA256
    )
    assert actual == digest
    assert candidate["schema_version"].endswith("/v3")
    assert candidate["nbv_configuration"] == document["configuration"][
        "request_semantics"
    ]
    expected = normalize_nbv_configuration(candidate["nbv_configuration"])
    assert candidate["nbv_configuration_sha256"] == expected.sha256
    assert candidate["derived_voxel_grid"]["origin_m"] == pytest.approx(
        expected.map_origin_m
    )
    _require_v3_execution_plan(candidate)
    with pytest.raises(RuntimeError, match="v2 remains evidence-only"):
        _require_v3_execution_plan(
            _minimal_aggregate_v2_outer_preview()[
                "execution_plan_candidate"
            ]
        )


def test_v3_plan_rejects_changed_config_or_derived_origin(tmp_path: Path) -> None:
    document = _minimal_frozen_config_v3_outer_preview()
    document["execution_plan_candidate"]["nbv_configuration"][
        "target_center_m"
    ][0] += 0.000001
    document["configuration"]["request_semantics"]["target_center_m"][
        0
    ] += 0.000001
    path = tmp_path / "frozen_config_changed.json"
    digest = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="SHA256 does not recompute"):
        _load_execution_plan(path, digest, DEFAULT_REPORT_SHA256)

    document = _minimal_frozen_config_v3_outer_preview()
    document["execution_plan_candidate"]["derived_voxel_grid"]["origin_m"][
        0
    ] += 0.001
    path = tmp_path / "frozen_origin_changed.json"
    digest = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="origin does not recompute"):
        _load_execution_plan(path, digest, DEFAULT_REPORT_SHA256)


def test_bounded_session_policy_is_exactly_sha_bound_in_outer_preview(
    tmp_path: Path,
) -> None:
    document = _minimal_frozen_config_v3_outer_preview()
    policy = _session_policy(
        10,
        coverage_target=0.65,
        coverage_plateau_delta=0.004,
        coverage_plateau_patience=3,
    )
    document["session_policy"] = copy.deepcopy(policy)
    document["max_motion_steps"] = 10
    document["execution_plan_candidate"]["session_policy"] = copy.deepcopy(
        policy
    )
    path = tmp_path / "three_step_preview.json"
    digest = _write_exact_json(path, document)
    candidate, _actual, _resolved = _load_execution_plan(
        path, digest, DEFAULT_REPORT_SHA256
    )
    assert _validate_bound_session_policy(
        candidate,
        10,
        coverage_target=0.65,
        coverage_plateau_delta=0.004,
        coverage_plateau_patience=3,
    ) == policy
    assert _authorization_token(1) == "EXECUTE_REAL_NBV_ONCE"
    assert _authorization_token(3) == "EXECUTE_REAL_NBV_SESSION_3"
    assert _authorization_token(10) == "EXECUTE_REAL_NBV_SESSION_10"

    document["session_policy"]["cumulative_translation_max_m"] = 0.020
    document["execution_plan_candidate"]["session_policy"] = copy.deepcopy(
        document["session_policy"]
    )
    digest = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="SHA/semantics do not recompute"):
        _load_execution_plan(path, digest, DEFAULT_REPORT_SHA256)


def test_large_motion_policy_prefers_visible_steps_but_allows_safe_fallback() -> None:
    policy = _session_policy(
        8,
        motion_profile=LARGE_MOTION_PROFILE,
        minimum_target_pixels=100,
        coverage_target=0.20,
    )
    assert policy["schema"] == "strawberry_real_nbv_motion_session_policy/v4"
    assert policy["single_step_translation_min_exclusive_m"] == 0.001
    assert policy["single_step_translation_min_inclusive_m"] is None
    assert policy["single_step_translation_max_m"] == 0.10
    assert policy["preferred_visible_translation_m"] == 0.05
    assert policy["reachable_candidate_radii_m"] == list(REACHABLE_VIEW_RADII_M)
    assert policy["reachable_candidate_direction_count"] == 18
    assert policy["near_best_gain_ratio"] == pytest.approx(0.90)
    assert policy["cumulative_translation_max_m"] == 0.60
    assert policy["cumulative_rotation_max_deg"] == pytest.approx(90.0)
    assert policy["maximum_ik_joint_delta_rad"] == 0.35
    assert policy["maximum_ik_position_error_m"] == 0.005
    assert policy["maximum_final_position_error_m"] == 0.005
    assert policy["minimum_valid_mask_depth_pixels"] == 100
    assert _authorization_token(
        8, LARGE_MOTION_PROFILE
    ) == "EXECUTE_REAL_NBV_LARGE_SESSION_8"
    assert _validate_bound_session_policy(
        {"session_policy": policy},
        8,
        motion_profile=LARGE_MOTION_PROFILE,
        minimum_target_pixels=100,
        coverage_target=0.20,
    ) == policy

    ledger = MotionSessionLedger(
        max_motion_steps=8,
        initial_camera=np.eye(4),
        initial_coverage=0.01,
        motion_limits=LARGE_MOTION_LIMITS,
        minimum_target_pixels=100,
        coverage_target=0.20,
    )
    accepted = ledger.validate_next_planned_step(
        planned_start_camera=np.eye(4),
        planned_camera=_translated_camera(0.05),
    )
    assert accepted.translation_m == pytest.approx(0.05)
    smaller_fallback = ledger.validate_next_planned_step(
        planned_start_camera=np.eye(4),
        planned_camera=_translated_camera(0.005),
    )
    assert smaller_fallback.translation_m == pytest.approx(0.005)
    with pytest.raises(ValueError, match="selected motion profile"):
        ledger.validate_next_planned_step(
            planned_start_camera=np.eye(4),
            planned_camera=_translated_camera(0.001),
        )

    with pytest.raises(ValueError, match="at least 100"):
        MotionSessionLedger(
            max_motion_steps=8,
            initial_camera=np.eye(4),
            initial_coverage=0.01,
            motion_limits=LARGE_MOTION_LIMITS,
            minimum_target_pixels=99,
            coverage_target=0.20,
        )


def test_large_motion_profile_uses_eight_as_backstop_not_stop_rule() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=8,
        initial_camera=np.eye(4),
        initial_coverage=0.01,
        motion_limits=LARGE_MOTION_LIMITS,
        minimum_target_pixels=100,
        coverage_target=0.99,
    )
    start = np.eye(4)
    for index in range(1, 9):
        target = _translated_camera(index * 0.075)
        ledger.validate_next_planned_step(
            planned_start_camera=start,
            planned_camera=target,
        )
        ledger.record_closed_step(
            planned_start_camera=start,
            planned_camera=target,
            actual_camera=target,
            coverage_after=0.01 + index * 0.01,
            target_valid_mask_depth_pixels=150,
            reported_motion_goal_count=index,
            gates_closed=True,
        )
        start = target
    summary = ledger.summary()
    assert summary["motion_goal_count"] == 8
    assert summary["converged_early"] is False
    assert summary["final_coverage"] == pytest.approx(0.09)
    assert summary["motion_limits"]["cumulative_translation_max_m"] == 0.60
    assert summary["motion_limits"]["cumulative_rotation_max_deg"] == pytest.approx(
        90.0
    )


def test_small_motion_policy_keeps_two_hundred_pixel_floor() -> None:
    with pytest.raises(RuntimeError, match="at least 200"):
        _session_policy(1, minimum_target_pixels=199)


def _controller_snapshot(
    precision_joint_delta: float,
    *,
    solver_position_tolerance_m: float = 0.002,
    precision_position_error_m: float = 0.003,
    precision_final_position_tolerance_m: float = 0.003,
) -> dict[str, object]:
    return {
        "simulation_mode": False,
        "first_motion_test_mode": False,
        "precision_test_mode": True,
        "precision_max_joint_delta_rad": precision_joint_delta,
        "ik_position_tolerance_m": solver_position_tolerance_m,
        "precision_max_ik_position_error_m": precision_position_error_m,
        "precision_final_position_tolerance_m": (
            precision_final_position_tolerance_m
        ),
        "verified_driver_speed_percent": 10,
        "execution_enabled_on_start": False,
    }


def test_controller_profile_must_match_selected_motion_envelope() -> None:
    small = _controller_snapshot(0.12)
    small_limits = motion_limits_for_profile("small_verified")
    assert _validate_controller_parameter_values(small, small_limits) == small
    with pytest.raises(SupervisorError, match="incompatible"):
        _validate_controller_parameter_values(small, LARGE_MOTION_LIMITS)

    large = _controller_snapshot(
        0.35,
        solver_position_tolerance_m=0.005,
        precision_position_error_m=0.005,
        precision_final_position_tolerance_m=0.005,
    )
    assert _validate_controller_parameter_values(
        large, LARGE_MOTION_LIMITS
    ) == large
    with pytest.raises(SupervisorError, match="incompatible"):
        _validate_controller_parameter_values(
            _controller_snapshot(0.351), LARGE_MOTION_LIMITS
        )


def test_authorization_receipt_is_written_before_first_goal(tmp_path: Path) -> None:
    receipt_path = tmp_path / "preview.json.consumed.json"
    fake = SimpleNamespace(
        _authorization_plan_sha256="a" * 64,
        _authorization_receipt_path=receipt_path,
        max_motion_steps=3,
        _audit={},
    )
    RealNBVSupervisor._consume_session_authorization(fake)
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert saved["plan_sha256"] == "a" * 64
    assert saved["max_motion_steps"] == 3
    assert saved["reusable"] is False
    assert fake._audit["authorization_receipt"]["path"] == str(receipt_path)


def test_failure_audit_uses_session_wide_gate_closure_latch() -> None:
    """A not-yet-executed next step must not hide the prior gate closure."""
    node = SimpleNamespace(_gates_closed_proven=True)
    assert _failure_gates_closed(node, SupervisorError("step 2 IK failed"))
    assert not _failure_gates_closed(node, GateClosureError("close failed"))
    node._gates_closed_proven = False
    assert not _failure_gates_closed(
        node, SupervisorError("extra command publisher")
    )


def _new_batch(label: str, start: int) -> dict:
    scenes = [f"new_session_{label}_scene_{index}" for index in range(5)]
    ids = [f"new_{label}_obs_{index}" for index in range(5)]
    stamps = [start + index for index in range(5)]
    return {
        "raw_scenes": scenes,
        "member_observation_ids": ids,
        "member_stamps_ns": stamps,
        "aggregate_observation_id": "agg5_" + five_frame_identity_sha256(
            scenes, ids, stamps
        )[:24],
    }


def test_execution_capture_batches_are_new_and_mutually_disjoint() -> None:
    plan = _minimal_frozen_config_v3_outer_preview()[
        "execution_plan_candidate"
    ]
    bootstrap = _new_batch("bootstrap", 1000)
    planning = _new_batch("planning", 2000)
    assert _verify_new_aggregate_batch(
        bootstrap, plan, "bootstrap"
    )["all_identity_sets_disjoint"]
    assert _verify_new_aggregate_batch(
        planning, plan, "planning"
    )["all_identity_sets_disjoint"]
    assert _verify_new_batch_pair(
        bootstrap, planning
    )["all_identity_sets_disjoint"]

    reused = _new_batch("reused", 3000)
    reused["raw_scenes"][0] = plan["aggregation"][
        "planning_raw_scenes"
    ][0]
    with pytest.raises(RuntimeError, match="reuses frozen"):
        _verify_new_aggregate_batch(reused, plan, "planning")

    planning["member_observation_ids"][0] = bootstrap[
        "member_observation_ids"
    ][0]
    with pytest.raises(RuntimeError, match="identities overlap"):
        _verify_new_batch_pair(bootstrap, planning)


def test_outer_preview_rejects_selected_candidate_drift(tmp_path: Path) -> None:
    document = _minimal_aggregate_v2_outer_preview()
    document["selected_candidate"]["alpha"] = 0.25
    path = tmp_path / "aggregate_v2_candidate_drift.json"
    exact_sha256 = _write_exact_json(path, document)

    with pytest.raises(
        RuntimeError,
        match="embedded selected candidate differs from preview result",
    ):
        _load_execution_plan(path, exact_sha256, DEFAULT_REPORT_SHA256)


def test_outer_preview_rejects_non_exact_sha(tmp_path: Path) -> None:
    path = tmp_path / "aggregate_v2_wrong_sha.json"
    exact_sha256 = _write_exact_json(
        path, _minimal_aggregate_v2_outer_preview()
    )
    replacement = "0" if exact_sha256[0] != "0" else "1"
    wrong_sha256 = replacement + exact_sha256[1:]

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _load_execution_plan(path, wrong_sha256, DEFAULT_REPORT_SHA256)


def test_aggregate_preview_rejects_frame_selection_or_duplicate_evidence(
    tmp_path: Path,
) -> None:
    document = _minimal_aggregate_v2_outer_preview()
    document["execution_plan_candidate"]["aggregation"][
        "selection_performed"
    ] = True
    path = tmp_path / "aggregate_v2_selected_frame.json"
    exact_sha256 = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="no frame was selected"):
        _load_execution_plan(path, exact_sha256, DEFAULT_REPORT_SHA256)

    document = _minimal_aggregate_v2_outer_preview()
    duplicate = document["bootstrap_batch"]["member_observation_ids"][0]
    document["bootstrap_batch"]["member_observation_ids"][1] = duplicate
    document["execution_plan_candidate"]["aggregation"][
        "bootstrap_member_ids"
    ][1] = duplicate
    path = tmp_path / "aggregate_v2_duplicate_frame.json"
    exact_sha256 = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="identity evidence is invalid"):
        _load_execution_plan(path, exact_sha256, DEFAULT_REPORT_SHA256)

    document = _minimal_aggregate_v2_outer_preview()
    document["bootstrap_batch"]["aggregate_identity_sha256"] = "0" * 64
    document["execution_plan_candidate"]["aggregation"][
        "bootstrap_aggregate_identity_sha256"
    ] = "0" * 64
    path = tmp_path / "aggregate_v2_forged_identity_hash.json"
    exact_sha256 = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="does not recompute"):
        _load_execution_plan(path, exact_sha256, DEFAULT_REPORT_SHA256)


def test_aggregate_preview_rejects_pose_span_or_inexact_tf(tmp_path: Path) -> None:
    document = _minimal_aggregate_v2_outer_preview()
    document["bootstrap_batch"]["pose_span"]["max_translation_m"] = 0.0006
    document["execution_plan_candidate"]["aggregation"][
        "bootstrap_pose_span"
    ]["max_translation_m"] = 0.0006
    path = tmp_path / "aggregate_v2_pose_span.json"
    exact_sha256 = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="pose span exceeds"):
        _load_execution_plan(path, exact_sha256, DEFAULT_REPORT_SHA256)

    document = _minimal_aggregate_v2_outer_preview()
    document["planning_batch"]["member_exact_tfs"][3][
        "returned_stamp_ns"
    ] += 1
    path = tmp_path / "aggregate_v2_inexact_tf.json"
    exact_sha256 = _write_exact_json(path, document)
    with pytest.raises(RuntimeError, match="exact exposure TF"):
        _load_execution_plan(path, exact_sha256, DEFAULT_REPORT_SHA256)


def _translated_camera(x_m: float) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x_m
    return transform


def test_three_step_session_ledger_enforces_coverage_and_cumulative_bounds() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    for index, (position, actual_position, coverage) in enumerate(
        (
            (0.004, 0.0038, 0.38),
            (0.0088, 0.0086, 0.47),
            (0.0136, 0.0134, 0.52),
        ),
        start=1,
    ):
        evidence = ledger.record_closed_step(
            planned_camera=_translated_camera(position),
            actual_camera=_translated_camera(actual_position),
            coverage_after=coverage,
            target_valid_mask_depth_pixels=500,
            reported_motion_goal_count=index,
            gates_closed=True,
        )
        assert evidence.step_index == index
    summary = ledger.summary()
    assert summary["motion_goal_count"] == 3
    assert summary["scientific_acceptance_passed"] is True
    assert summary["total_coverage_delta"] == pytest.approx(0.22)


def test_session_planned_step_starts_from_measured_previous_pose() -> None:
    """Accepted terminal error must not make a clamped next step look too long."""
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    ledger.record_closed_step(
        planned_camera=_translated_camera(0.005),
        actual_camera=_translated_camera(0.0052),
        coverage_after=0.36,
        target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    evidence = ledger.record_closed_step(
        planned_start_camera=_translated_camera(0.00525),
        # The second target is exactly 5 mm from the measured 5.2 mm pose,
        # plus a 0.05 mm stationary preflight shift.
        planned_camera=_translated_camera(0.01025),
        actual_camera=_translated_camera(0.0101),
        coverage_after=0.43,
        target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=2,
        gates_closed=True,
    )
    assert evidence.planned_step_translation_m == pytest.approx(0.005)
    assert ledger.summary()["motion_goal_count"] == 2


def test_session_ledger_accepts_same_sub_micrometre_step_tolerance() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    evidence = ledger.record_closed_step(
        planned_camera=_translated_camera(0.0050008),
        actual_camera=_translated_camera(0.0049),
        coverage_after=0.36,
        target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    assert evidence.planned_step_translation_m == pytest.approx(0.0050008)

    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    with pytest.raises(ValueError, match="planned camera step"):
        ledger.record_closed_step(
            planned_camera=_translated_camera(0.0050012),
            actual_camera=_translated_camera(0.0049),
            coverage_after=0.36,
            target_valid_mask_depth_pixels=400,
            reported_motion_goal_count=1,
            gates_closed=True,
        )


def test_session_stops_after_two_consecutive_small_coverage_gains() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.40,
    )
    first = ledger.record_closed_step(
        planned_camera=_translated_camera(0.004),
        actual_camera=_translated_camera(0.0039),
        coverage_after=0.40 + CONVERGENCE_COVERAGE_DELTA - 0.0001,
        target_valid_mask_depth_pixels=300,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    assert first.coverage_plateau_count == 1
    assert first.converged is False
    second = ledger.record_closed_step(
        planned_start_camera=_translated_camera(0.0039),
        planned_camera=_translated_camera(0.0079),
        actual_camera=_translated_camera(0.0078),
        coverage_after=0.40 + 2 * (CONVERGENCE_COVERAGE_DELTA - 0.0001),
        target_valid_mask_depth_pixels=300,
        reported_motion_goal_count=2,
        gates_closed=True,
    )
    assert second.coverage_plateau_count == 2
    assert second.converged is True
    summary = ledger.summary()
    assert summary["converged_early"] is True
    assert "2 consecutive" in summary["convergence_reason"]


def test_large_gain_resets_plateau_and_absolute_target_stops_session() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=5,
        initial_camera=np.eye(4),
        initial_coverage=0.40,
        coverage_target=0.50,
    )
    first = ledger.record_closed_step(
        planned_camera=_translated_camera(0.003),
        actual_camera=_translated_camera(0.003),
        coverage_after=0.404,
        target_valid_mask_depth_pixels=300,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    assert first.coverage_plateau_count == 1
    second = ledger.record_closed_step(
        planned_start_camera=_translated_camera(0.003),
        planned_camera=_translated_camera(0.006),
        actual_camera=_translated_camera(0.006),
        coverage_after=0.46,
        target_valid_mask_depth_pixels=300,
        reported_motion_goal_count=2,
        gates_closed=True,
    )
    assert second.coverage_plateau_count == 0
    final = ledger.record_closed_step(
        planned_start_camera=_translated_camera(0.006),
        planned_camera=_translated_camera(0.009),
        actual_camera=_translated_camera(0.009),
        coverage_after=0.51,
        target_valid_mask_depth_pixels=300,
        reported_motion_goal_count=3,
        gates_closed=True,
    )
    assert final.coverage_target_reached is True
    assert final.converged is True
    assert ledger.summary()["coverage_target_reached"] is True
    with pytest.raises(ValueError, match="converged session"):
        ledger.record_closed_step(
            planned_start_camera=_translated_camera(0.009),
            planned_camera=_translated_camera(0.012),
            actual_camera=_translated_camera(0.012),
            coverage_after=0.52,
            target_valid_mask_depth_pixels=300,
            reported_motion_goal_count=4,
            gates_closed=True,
        )


def test_session_can_converge_on_next_view_motion_deadband() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.24,
    )
    ledger.record_closed_step(
        planned_camera=_translated_camera(0.004),
        actual_camera=_translated_camera(0.0039),
        coverage_after=0.35,
        target_valid_mask_depth_pixels=300,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    ledger.record_closed_step(
        planned_start_camera=_translated_camera(0.0039),
        planned_camera=_translated_camera(0.0079),
        actual_camera=_translated_camera(0.0078),
        coverage_after=0.416,
        target_valid_mask_depth_pixels=300,
        reported_motion_goal_count=2,
        gates_closed=True,
    )

    reason = "Gradient-NBV requested no motion above the 1 mm deadband"
    ledger.mark_converged(reason)
    summary = ledger.summary()

    assert summary["motion_goal_count"] == 2
    assert summary["converged_early"] is True
    assert summary["convergence_reason"] == reason
    assert summary["terminated"] is False
    assert summary["scientific_acceptance_passed"] is False


def test_session_cannot_converge_before_any_closed_motion() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.24,
    )
    with pytest.raises(ValueError, match="before a motion goal"):
        ledger.mark_converged("deadband reached")


@pytest.mark.parametrize(
    "reason,gates_closed",
    (
        ("step 2 SolveIK failed", True),
        ("step 2 lost the red target", True),
        ("step 2 reused an Observation identity", True),
        ("step 2 NBV map coverage decreased", True),
        ("step 2 gate closure failed", False),
        ("step 2 found an extra command publisher", True),
    ),
)
def test_fault_latch_prevents_every_later_goal(reason, gates_closed) -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    ledger.record_closed_step(
        planned_camera=_translated_camera(0.004),
        actual_camera=_translated_camera(0.0039),
        coverage_after=0.36,
        target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    ledger.abort(reason, gates_closed=gates_closed)
    with pytest.raises(ValueError, match="terminated session"):
        ledger.record_closed_step(
            planned_camera=_translated_camera(0.008),
            actual_camera=_translated_camera(0.008),
            coverage_after=0.42,
            target_valid_mask_depth_pixels=400,
            reported_motion_goal_count=2,
            gates_closed=True,
        )
    summary = ledger.summary()
    assert summary["motion_goal_count"] == 1
    assert summary["termination_reason"] == reason
    assert summary["gates_closed_at_termination"] is gates_closed


def test_session_rejects_second_step_without_closed_gates_and_bad_target() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    with pytest.raises(ValueError, match="gates must be proven closed"):
        ledger.record_closed_step(
            planned_camera=_translated_camera(0.004),
            actual_camera=_translated_camera(0.004),
            coverage_after=0.35,
            target_valid_mask_depth_pixels=400,
            reported_motion_goal_count=1,
            gates_closed=False,
        )
    with pytest.raises(ValueError, match="fewer than 200"):
        ledger.record_closed_step(
            planned_camera=_translated_camera(0.004),
            actual_camera=_translated_camera(0.004),
            coverage_after=0.35,
            target_valid_mask_depth_pixels=199,
            reported_motion_goal_count=1,
            gates_closed=True,
        )


def test_session_accepts_one_to_ten_steps_and_enforces_cumulative_motion() -> None:
    MotionSessionLedger(
        max_motion_steps=2,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    MotionSessionLedger(
        max_motion_steps=10,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        MotionSessionLedger(
            max_motion_steps=0,
            initial_camera=np.eye(4),
            initial_coverage=0.30,
        )
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        MotionSessionLedger(
            max_motion_steps=11,
            initial_camera=np.eye(4),
            initial_coverage=0.30,
        )
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    ledger.record_closed_step(
        planned_camera=_translated_camera(0.004),
        actual_camera=_translated_camera(0.007),
        coverage_after=0.35,
        target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    ledger.record_closed_step(
        planned_camera=_translated_camera(0.011),
        actual_camera=_translated_camera(0.002),
        coverage_after=0.40,
        target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=2,
        gates_closed=True,
    )
    with pytest.raises(ValueError, match="exceeds 15 mm"):
        ledger.record_closed_step(
            planned_camera=_translated_camera(0.006),
            actual_camera=_translated_camera(0.007),
            coverage_after=0.45,
            target_valid_mask_depth_pixels=400,
            reported_motion_goal_count=3,
            gates_closed=True,
        )


def test_ten_step_session_rejects_path_budget_before_fourth_motion() -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=10,
        initial_camera=np.eye(4),
        initial_coverage=0.20,
    )
    positions = (0.005, 0.010, 0.015)
    coverages = (0.25, 0.30, 0.35)
    previous = np.eye(4)
    for index, (position, coverage) in enumerate(
        zip(positions, coverages), start=1
    ):
        target = _translated_camera(position)
        ledger.validate_next_planned_step(
            planned_start_camera=previous,
            planned_camera=target,
        )
        ledger.record_closed_step(
            planned_start_camera=previous,
            planned_camera=target,
            actual_camera=target,
            coverage_after=coverage,
            target_valid_mask_depth_pixels=400,
            reported_motion_goal_count=index,
            gates_closed=True,
        )
        previous = target
    with pytest.raises(ValueError, match="exceeds 15 mm"):
        ledger.validate_next_planned_step(
            planned_start_camera=previous,
            planned_camera=_translated_camera(0.010),
        )
