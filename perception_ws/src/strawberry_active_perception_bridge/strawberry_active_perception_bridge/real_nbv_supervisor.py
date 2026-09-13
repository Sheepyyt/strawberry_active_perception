"""Bounded real Observation -> Gradient-NBV -> Placo supervisor.

The default ``execute:=false`` path is motion-free.  Physical execution is a
separate branch that must bind the exact accepted preview artifact
and SHA, re-check the current camera pose and SolveIK, pass the controller and
driver safety gates, and receive an explicit operator authorization token.
It defaults to one MoveToPose goal and permits a SHA-bound convergence session
of at most ten goals under an explicit session authorization contract.  Any
inability to prove both software gates closed is a fatal onsite-stop condition.

Gradient-NBV runs in its separate ``.venv-nbv`` ROS process.  This system-
Python node republishes a copy of the real Observation with the exposure-time
``T_base_camera_optical`` obtained from exact TF and the hash-bound formal
hand-eye report.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from action_msgs.msg import GoalStatus
from agx_arm_msgs.msg import AgxArmStatus
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from strawberry_nero_interfaces.action import MoveToPose
from strawberry_nero_interfaces.msg import IKResult
from strawberry_nero_interfaces.srv import SolveIK
from strawberry_perception_interfaces.action import ComputeNextView
from strawberry_perception_interfaces.msg import NextView, Observation
from strawberry_perception_interfaces.srv import (
    CaptureObservation,
    ConfigureNBV,
    EvaluateViewCandidates,
)
from tf2_ros import Buffer, TransformException, TransformListener

from .calibration_report import load_verified_handeye_report
from .real_nbv_contract import (
    ALLOWED_SESSION_MOTION_STEPS,
    CONVERGENCE_COVERAGE_DELTA,
    DEFAULT_ALPHAS,
    DEFAULT_COVERAGE_PLATEAU_PATIENCE,
    LARGE_MOTION_PROFILE,
    MAX_CAMERA_STEP_M,
    MAX_IK_JOINT_DELTA_RAD,
    MAX_SELECTED_CAMERA_ROTATION_RAD,
    MAX_SESSION_ROTATION_RAD,
    MAX_SESSION_TRANSLATION_M,
    MIN_CAMERA_STEP_M,
    MotionBudgetExhausted,
    MotionLimits,
    MotionSessionLedger,
    NEAR_BEST_GAIN_RATIO,
    REACHABLE_VIEW_DIRECTION_COUNT,
    REACHABLE_VIEW_RADII_M,
    SMALL_MOTION_LIMITS,
    SMALL_MOTION_PROFILE,
    aggregate_five_frame_depth_mask,
    array_list,
    camera_matrix,
    camera_motion,
    camera_step_above_minimum,
    compare_gradient_candidates,
    decode_depth_32fc1,
    decode_mask_mono8,
    estimate_target_center,
    five_frame_identity_sha256,
    independently_segmented_camera_candidates,
    motion_limits_for_profile,
    normalize_nbv_configuration,
    ordered_joint_positions,
    project_numerical_step_overshoot,
    reachable_view_lattice,
    require_unit_quaternion,
    select_near_best_reachable_candidate,
    stamp_nanoseconds,
    transform_point,
    validate_ik_solution,
    validate_raw_next_view,
    wire_bool,
    wire_uint8,
)
from .transforms import (
    camera_target_to_link7,
    matrix_to_pose_components,
    pose_components_to_matrix,
    validate_rigid_transform,
)


DEFAULT_REPORT_PATH = (
    "/home/yyt/strawberry_active_perception/validation/week3/artifacts/"
    "stability_pose001_030_factory_raw_D.json"
)
DEFAULT_REPORT_SHA256 = (
    "31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f"
)
DEFAULT_OUTPUT_PATH = (
    "/home/yyt/strawberry_active_perception/artifacts/week4/"
    "real_nbv_frozen_config_v3_preview.json"
)
DEFAULT_PLAN_PATH = (
    "/home/yyt/strawberry_active_perception/validation/week4/artifacts/"
    "real_nbv_frozen_config_v3_preview.json"
)
DEFAULT_PLAN_SHA256 = (
    "8e3d032a794a9a5a37dfed4e24964daffefde32fd17df11d97fba529f14e256f"
)
ONE_STEP_AUTHORIZATION_TOKEN = "EXECUTE_REAL_NBV_ONCE"
SESSION_AUTHORIZATION_TOKEN_PREFIX = "EXECUTE_REAL_NBV_SESSION_"
FEEDBACK_MAX_AGE_SEC = 0.20
STATIONARY_SPEED_RAD_SEC = 0.020
FORBIDDEN_COMMAND_TOPICS = (
    "/control/joint_states",
    "/control/move_p",
    "/control/move_l",
    "/control/move_c",
    "/control/move_js",
    "/control/move_mit",
)
AGGREGATE_CAPTURE_COUNT = 5
AGGREGATE_MIN_FINITE_COUNT = 3
AGGREGATE_MASK_MAJORITY_COUNT = 3
AGGREGATE_MAX_TRANSLATION_SPAN_M = 0.00050
AGGREGATE_MAX_ROTATION_SPAN_DEG = 0.10
MAX_PLAN_START_TRANSLATION_DRIFT_M = 0.001
MAX_PLAN_START_ROTATION_DRIFT_DEG = 0.5
MAX_PLAN_TARGET_TRANSLATION_DRIFT_M = 0.0015
MAX_PLAN_TARGET_ROTATION_DRIFT_DEG = 0.5
# This is an identity/corroboration gate, not an input to ConfigureNBV during
# execution.  Live Gemini measurements of the small red target showed
# 1.45--2.07 mm maximum within-batch centre variation even after fixed
# five-frame aggregation.  A later pre-gate single-frame check differed from
# its frozen five-frame centre by 3.58 mm while the target and camera were
# stationary.  Keep a fixed 5 mm ceiling: it covers that measured sensor
# repeatability while remaining far below the independent 20 mm
# post-motion "target lost or moved" gate.  The SHA-frozen configuration,
# voxel origin and first motion target remain unchanged by fresh captures.
MAX_FROZEN_TARGET_CENTER_DRIFT_M = 0.005
AGGREGATE_PLAN_V2 = "strawberry_real_nbv_aggregate_plan/v2"
AGGREGATE_PLAN_V3 = "strawberry_real_nbv_frozen_config_plan/v3"


class SupervisorError(RuntimeError):
    """One fail-closed refusal with an actionable audit reason."""


class GateClosureError(SupervisorError):
    """A fatal inability to prove both command gates closed."""


def _validate_controller_parameter_values(
    decoded: dict[str, Any], motion_limits: MotionLimits
) -> dict[str, Any]:
    """Require real precision mode and a profile-compatible joint gate."""
    expected_modes = {
        "simulation_mode": False,
        "first_motion_test_mode": False,
        "precision_test_mode": True,
        "execution_enabled_on_start": False,
    }
    actual_modes = {name: decoded.get(name) for name in expected_modes}
    if (
        actual_modes != expected_modes
        or decoded.get("verified_driver_speed_percent") != 10
    ):
        raise SupervisorError(
            f"nero_control is not in locked real precision mode: {decoded}"
        )
    configured_joint_delta = float(
        decoded.get("precision_max_joint_delta_rad", math.nan)
    )
    required_joint_delta = motion_limits.maximum_ik_joint_delta_rad
    if not (
        required_joint_delta - 1.0e-12
        <= configured_joint_delta
        <= 0.35 + 1.0e-12
    ):
        raise SupervisorError(
            "nero_control precision joint-delta gate is incompatible with "
            f"{motion_limits.profile}: controller={configured_joint_delta:.6f}rad, "
            f"required=[{required_joint_delta:.6f}, 0.350000]rad"
        )
    solver_position_tolerance = float(
        decoded.get("ik_position_tolerance_m", math.nan)
    )
    precision_position_error = float(
        decoded.get("precision_max_ik_position_error_m", math.nan)
    )
    precision_final_tolerance = float(
        decoded.get("precision_final_position_tolerance_m", math.nan)
    )
    required_position_error = motion_limits.maximum_ik_position_error_m
    required_final_tolerance = motion_limits.maximum_final_position_error_m
    if not (
        0.0 < solver_position_tolerance <= required_position_error + 1.0e-12
        and abs(precision_position_error - required_position_error) <= 1.0e-12
        and abs(precision_final_tolerance - required_final_tolerance) <= 1.0e-12
    ):
        raise SupervisorError(
            "nero_control Cartesian error gates are incompatible with "
            f"{motion_limits.profile}: {decoded}"
        )
    return decoded


def _failure_gates_closed(node: Any, error: Exception) -> bool:
    """Report closed gates only from the session-wide proven latch."""
    return (
        not isinstance(error, GateClosureError)
        and bool(node._gates_closed_proven)
    )


def _candidate_selection_evidence(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Return the optional large-workspace lattice decision evidence."""
    if candidate.get("segmentation_contract") != "reachable_view_lattice":
        return None
    keys = (
        "candidate_index",
        "direction_index",
        "direction_world",
        "radius_m",
        "current_gain",
        "candidate_gain",
        "gain_improvement",
        "best_gain_improvement",
        "near_best_gain_floor",
        "near_best_candidate_count",
        "reachable_candidate_count",
        "useful_candidate_count",
        "selection_contract",
    )
    if any(key not in candidate for key in keys):
        raise SupervisorError("reachable-view candidate audit is incomplete")
    return {key: copy.deepcopy(candidate[key]) for key in keys}


def _legacy_session_policy(max_motion_steps: int) -> dict[str, Any]:
    """Recompute the historical one-/three-step policy for old evidence."""
    if isinstance(max_motion_steps, bool) or max_motion_steps not in (1, 3):
        raise SupervisorError("legacy max_motion_steps must be exactly 1 or 3")
    policy = {
        "schema": "strawberry_real_nbv_motion_session_policy/v1",
        "max_motion_steps": int(max_motion_steps),
        "single_step_translation_min_exclusive_m": MIN_CAMERA_STEP_M,
        "single_step_translation_max_m": MAX_CAMERA_STEP_M,
        "single_step_rotation_max_deg": math.degrees(
            MAX_SELECTED_CAMERA_ROTATION_RAD
        ),
        "cumulative_translation_max_m": MAX_SESSION_TRANSLATION_M,
        "cumulative_rotation_max_deg": math.degrees(MAX_SESSION_ROTATION_RAD),
        "convergence_coverage_delta": CONVERGENCE_COVERAGE_DELTA,
        "minimum_valid_mask_depth_pixels": 200,
        "configure_once_and_keep_same_map": True,
        "five_frame_aggregate_after_every_motion": True,
        "automatic_return_or_disable": False,
    }
    payload = json.dumps(
        policy, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    policy["sha256"] = hashlib.sha256(payload).hexdigest()
    return policy


def _session_policy(
    max_motion_steps: int,
    *,
    motion_profile: str = SMALL_MOTION_PROFILE,
    minimum_target_pixels: int = 200,
    coverage_target: float | None = None,
    coverage_plateau_delta: float = CONVERGENCE_COVERAGE_DELTA,
    coverage_plateau_patience: int = DEFAULT_COVERAGE_PLATEAU_PATIENCE,
) -> dict[str, Any]:
    """Return the exact policy bound by a preview and one operator token."""
    try:
        limits = motion_limits_for_profile(motion_profile)
    except ValueError as error:
        raise SupervisorError(str(error)) from error
    if (
        isinstance(max_motion_steps, bool)
        or max_motion_steps not in ALLOWED_SESSION_MOTION_STEPS
        or max_motion_steps > limits.maximum_motion_steps
    ):
        raise SupervisorError(
            "max_motion_steps must be an integer in [1, "
            f"{limits.maximum_motion_steps}] for profile {limits.profile}"
        )
    if (
        isinstance(minimum_target_pixels, bool)
        or not isinstance(minimum_target_pixels, int)
    ):
        raise SupervisorError("minimum_target_pixels must be an integer")
    minimum_pixels_floor = (
        200 if limits.profile == SMALL_MOTION_PROFILE else 100
    )
    if minimum_target_pixels < minimum_pixels_floor:
        raise SupervisorError(
            "minimum_target_pixels must be at least "
            f"{minimum_pixels_floor} for profile {limits.profile}"
        )
    if isinstance(coverage_plateau_delta, bool):
        raise SupervisorError("coverage_plateau_delta must be numeric")
    plateau_delta = float(coverage_plateau_delta)
    if not math.isfinite(plateau_delta) or not 0.0 < plateau_delta <= 0.05:
        raise SupervisorError(
            "coverage_plateau_delta must be finite and in (0, 0.05]"
        )
    if (
        isinstance(coverage_plateau_patience, bool)
        or not isinstance(coverage_plateau_patience, int)
        or not 1 <= coverage_plateau_patience <= 10
    ):
        raise SupervisorError("coverage_plateau_patience must be in [1, 10]")
    normalized_target = None
    if isinstance(coverage_target, bool):
        raise SupervisorError("coverage_target must be numeric")
    if coverage_target is not None and float(coverage_target) != 0.0:
        normalized_target = float(coverage_target)
        if not math.isfinite(normalized_target) or not 0.0 < normalized_target <= 1.0:
            raise SupervisorError("coverage_target must be 0 (disabled) or in (0, 1]")
    policy = {
        "schema": (
            "strawberry_real_nbv_motion_session_policy/v2"
            if limits.profile == SMALL_MOTION_PROFILE
            else "strawberry_real_nbv_motion_session_policy/v4"
        ),
        "max_motion_steps": int(max_motion_steps),
        "single_step_translation_min_exclusive_m": (
            limits.minimum_camera_step_m
            if not limits.minimum_step_inclusive
            else None
        ),
        "single_step_translation_min_inclusive_m": (
            limits.minimum_camera_step_m
            if limits.minimum_step_inclusive
            else None
        ),
        "single_step_translation_max_m": limits.maximum_camera_step_m,
        "single_step_rotation_max_deg": math.degrees(
            limits.maximum_camera_rotation_rad
        ),
        "maximum_ik_joint_delta_rad": limits.maximum_ik_joint_delta_rad,
        "maximum_ik_position_error_m": limits.maximum_ik_position_error_m,
        "maximum_final_position_error_m": (
            limits.maximum_final_position_error_m
        ),
        "cumulative_translation_max_m": limits.maximum_session_translation_m,
        "cumulative_rotation_max_deg": math.degrees(
            limits.maximum_session_rotation_rad
        ),
        "coverage_target": normalized_target,
        "coverage_plateau_delta": plateau_delta,
        "coverage_plateau_patience": int(coverage_plateau_patience),
        "motion_deadband_m": limits.minimum_camera_step_m,
        "stop_at_first_reached_condition": True,
        "minimum_valid_mask_depth_pixels": int(minimum_target_pixels),
        "configure_once_and_keep_same_map": True,
        "five_frame_aggregate_after_every_motion": True,
        "automatic_return_or_disable": False,
    }
    if limits.profile == SMALL_MOTION_PROFILE:
        # Preserve byte-for-byte v2 policy compatibility with existing plans.
        policy.pop("single_step_translation_min_inclusive_m")
        policy.pop("maximum_ik_joint_delta_rad")
        policy.pop("maximum_ik_position_error_m")
        policy.pop("maximum_final_position_error_m")
    else:
        policy["motion_profile"] = limits.profile
        policy["preferred_visible_translation_m"] = 0.050
        policy["reachable_candidate_radii_m"] = list(REACHABLE_VIEW_RADII_M)
        policy["reachable_candidate_direction_count"] = (
            REACHABLE_VIEW_DIRECTION_COUNT
        )
        policy["near_best_gain_ratio"] = NEAR_BEST_GAIN_RATIO
        policy["candidate_selection_contract"] = (
            "finite_direction_distance_lattice_ik_first_then_gain; "
            "within_90_percent_of_best_gain_choose_largest_translation"
        )
    payload = json.dumps(
        policy, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    policy["sha256"] = hashlib.sha256(payload).hexdigest()
    return policy


def _validated_stored_session_policy(policy: Any) -> dict[str, Any]:
    """Recompute a policy SHA, retaining read-only support for v1 evidence."""
    if not isinstance(policy, dict):
        raise SupervisorError("execution plan session policy is invalid")
    schema = policy.get("schema")
    steps = policy.get("max_motion_steps")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise SupervisorError("execution plan session policy step count is invalid")
    if schema == "strawberry_real_nbv_motion_session_policy/v1":
        expected = _legacy_session_policy(steps)
    elif schema == "strawberry_real_nbv_motion_session_policy/v2":
        try:
            expected = _session_policy(
                steps,
                motion_profile=SMALL_MOTION_PROFILE,
                minimum_target_pixels=policy.get(
                    "minimum_valid_mask_depth_pixels"
                ),
                coverage_target=policy.get("coverage_target"),
                coverage_plateau_delta=policy.get("coverage_plateau_delta"),
                coverage_plateau_patience=policy.get(
                    "coverage_plateau_patience"
                ),
            )
        except (TypeError, ValueError) as error:
            raise SupervisorError(
                "execution plan session policy values are invalid"
            ) from error
    elif schema == "strawberry_real_nbv_motion_session_policy/v3":
        expected = {
            "schema": "strawberry_real_nbv_motion_session_policy/v3",
            "max_motion_steps": steps,
            "single_step_translation_min_exclusive_m": None,
            "single_step_translation_min_inclusive_m": 0.05,
            "single_step_translation_max_m": 0.1,
            "single_step_rotation_max_deg": math.degrees(math.radians(15.0)),
            "maximum_ik_joint_delta_rad": 0.35,
            "maximum_ik_position_error_m": 0.005,
            "maximum_final_position_error_m": 0.005,
            "cumulative_translation_max_m": 0.3,
            "cumulative_rotation_max_deg": 45.0,
            "coverage_target": policy.get("coverage_target"),
            "coverage_plateau_delta": policy.get("coverage_plateau_delta"),
            "coverage_plateau_patience": policy.get(
                "coverage_plateau_patience"
            ),
            "motion_deadband_m": 0.05,
            "stop_at_first_reached_condition": True,
            "minimum_valid_mask_depth_pixels": policy.get(
                "minimum_valid_mask_depth_pixels"
            ),
            "configure_once_and_keep_same_map": True,
            "five_frame_aggregate_after_every_motion": True,
            "automatic_return_or_disable": False,
            "motion_profile": LARGE_MOTION_PROFILE,
        }
        payload = json.dumps(
            expected, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        expected["sha256"] = hashlib.sha256(payload).hexdigest()
    elif schema == "strawberry_real_nbv_motion_session_policy/v4":
        try:
            expected = _session_policy(
                steps,
                motion_profile=policy.get("motion_profile", ""),
                minimum_target_pixels=policy.get(
                    "minimum_valid_mask_depth_pixels"
                ),
                coverage_target=policy.get("coverage_target"),
                coverage_plateau_delta=policy.get("coverage_plateau_delta"),
                coverage_plateau_patience=policy.get(
                    "coverage_plateau_patience"
                ),
            )
        except (TypeError, ValueError) as error:
            raise SupervisorError(
                "execution plan large-motion policy values are invalid"
            ) from error
    else:
        raise SupervisorError("unsupported execution plan session policy schema")
    if policy != expected:
        raise SupervisorError(
            "execution plan session policy SHA/semantics do not recompute"
        )
    return copy.deepcopy(expected)


def _authorization_token(
    max_motion_steps: int,
    motion_profile: str = SMALL_MOTION_PROFILE,
) -> str:
    """Return the explicit token spelling for one bounded session."""
    try:
        limits = motion_limits_for_profile(motion_profile)
    except ValueError as error:
        raise SupervisorError(str(error)) from error
    if (
        isinstance(max_motion_steps, bool)
        or max_motion_steps not in ALLOWED_SESSION_MOTION_STEPS
        or max_motion_steps > limits.maximum_motion_steps
    ):
        raise SupervisorError("max_motion_steps is invalid for the motion profile")
    if limits.profile == LARGE_MOTION_PROFILE:
        return f"EXECUTE_REAL_NBV_LARGE_SESSION_{max_motion_steps}"
    if max_motion_steps == 1:
        return ONE_STEP_AUTHORIZATION_TOKEN
    return f"{SESSION_AUTHORIZATION_TOKEN_PREFIX}{max_motion_steps}"


def _validate_bound_session_policy(
    execution_plan: dict[str, Any],
    max_motion_steps: int,
    *,
    motion_profile: str = SMALL_MOTION_PROFILE,
    minimum_target_pixels: int = 200,
    coverage_target: float | None = None,
    coverage_plateau_delta: float = CONVERGENCE_COVERAGE_DELTA,
    coverage_plateau_patience: int = DEFAULT_COVERAGE_PLATEAU_PATIENCE,
) -> dict[str, Any]:
    """Require the runtime convergence policy to match the frozen preview."""
    expected = _session_policy(
        max_motion_steps,
        motion_profile=motion_profile,
        minimum_target_pixels=minimum_target_pixels,
        coverage_target=coverage_target,
        coverage_plateau_delta=coverage_plateau_delta,
        coverage_plateau_patience=coverage_plateau_patience,
    )
    stored = execution_plan.get("session_policy")
    if stored is None and max_motion_steps == 1:
        if (
            coverage_target is not None
            or abs(
                float(coverage_plateau_delta) - CONVERGENCE_COVERAGE_DELTA
            ) > 1.0e-12
            or coverage_plateau_patience
            != DEFAULT_COVERAGE_PLATEAU_PATIENCE
        ):
            raise SupervisorError(
                "legacy one-step plan cannot bind custom convergence settings"
            )
        legacy = _legacy_session_policy(1)
        return {**legacy, "legacy_v3_policy_inferred": True}
    if stored != expected:
        raise SupervisorError(
            "execution plan session policy differs from requested runtime policy"
        )
    return copy.deepcopy(expected)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: str | Path, document: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _duration_message(seconds: float):
    result = Duration(seconds=seconds).to_msg()
    return result


def _strict_pose_matrix(pose: Any, label: str) -> np.ndarray:
    position = pose.pose.position
    orientation = pose.pose.orientation
    quaternion = require_unit_quaternion(
        (orientation.x, orientation.y, orientation.z, orientation.w),
        f"{label} quaternion",
    )
    return pose_components_to_matrix(
        (position.x, position.y, position.z), quaternion
    )


def _strict_transform_matrix(transform: Any, label: str) -> np.ndarray:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    quaternion = require_unit_quaternion(
        (rotation.x, rotation.y, rotation.z, rotation.w),
        f"{label} quaternion",
    )
    return pose_components_to_matrix(
        (translation.x, translation.y, translation.z), quaternion
    )


def _fill_pose_stamped(message: Any, transform: np.ndarray, frame_id: str, stamp: Any) -> None:
    position, quaternion = matrix_to_pose_components(transform)
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.pose.position.x = float(position[0])
    message.pose.position.y = float(position[1])
    message.pose.position.z = float(position[2])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])


def _target_audit(estimate: Any, centre_base: np.ndarray) -> dict[str, Any]:
    return {
        "camera_xyz_m": array_list(estimate.camera_xyz_m),
        "base_xyz_m": array_list(centre_base),
        "mask_pixels": estimate.mask_pixels,
        "valid_mask_pixels": estimate.valid_mask_pixels,
        "depth_layer_count": estimate.depth_layer_count,
        "selected_layer_pixels": estimate.selected_layer_pixels,
        "selected_layer_range_m": [
            estimate.selected_layer_min_depth_m,
            estimate.selected_layer_max_depth_m,
        ],
        "retained_pixels": estimate.retained_pixels,
        "median_depth_m": estimate.median_depth_m,
        "depth_mad_m": estimate.depth_mad_m,
        "method": (
            "nearest supported masked-depth layer, median/MAD trim, then "
            "component-wise 3-D median"
        ),
    }


def _camera_info_signature(message: Any) -> tuple[Any, ...]:
    """Return every calibration/grid field that must stay exact in a batch."""
    roi = message.roi
    return (
        int(message.width),
        int(message.height),
        str(message.distortion_model),
        tuple(float(value) for value in message.d),
        tuple(float(value) for value in message.k),
        tuple(float(value) for value in message.r),
        tuple(float(value) for value in message.p),
        int(message.binning_x),
        int(message.binning_y),
        int(roi.x_offset),
        int(roi.y_offset),
        int(roi.height),
        int(roi.width),
        bool(roi.do_rectify),
    )


def _verify_new_aggregate_batch(
    batch: dict[str, Any],
    execution_plan: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Prove one execution batch shares no raw identity with the artifact."""
    aggregation = execution_plan.get("aggregation", {})
    artifact_ids = set(
        aggregation.get("bootstrap_member_ids", ())
    ) | set(aggregation.get("planning_member_ids", ()))
    artifact_scenes = set(
        aggregation.get("bootstrap_raw_scenes", ())
    ) | set(aggregation.get("planning_raw_scenes", ()))
    artifact_stamps = set(
        aggregation.get("bootstrap_member_stamps_ns", ())
    ) | set(aggregation.get("planning_member_stamps_ns", ()))
    artifact_aggregates = {
        aggregation.get("bootstrap_aggregate_observation_id"),
        aggregation.get("planning_aggregate_observation_id"),
    }
    current_ids = set(batch.get("member_observation_ids", ()))
    current_scenes = set(batch.get("raw_scenes", ()))
    current_stamps = set(batch.get("member_stamps_ns", ()))
    current_aggregate = batch.get("aggregate_observation_id")
    if (
        len(artifact_ids) != 10
        or len(artifact_scenes) != 10
        or len(artifact_stamps) != 10
        or len(artifact_aggregates) != 2
        or None in artifact_aggregates
        or len(current_ids) != 5
        or len(current_scenes) != 5
        or len(current_stamps) != 5
        or not str(current_aggregate).strip()
    ):
        raise SupervisorError(
            f"execution {label} batch independence evidence is incomplete"
        )
    overlap = {
        "observation_ids": sorted(current_ids & artifact_ids),
        "raw_scenes": sorted(current_scenes & artifact_scenes),
        "stamps_ns": sorted(current_stamps & artifact_stamps),
        "aggregate_observation_id": (
            [current_aggregate]
            if current_aggregate in artifact_aggregates
            else []
        ),
    }
    if any(overlap.values()):
        raise SupervisorError(
            f"execution {label} batch reuses frozen artifact capture identity"
        )
    return {
        "label": label,
        "artifact_raw_capture_count": len(artifact_ids),
        "current_raw_capture_count": len(current_ids),
        "overlap": overlap,
        "all_identity_sets_disjoint": True,
        "new_aggregate_observation_id": current_aggregate,
    }


def _verify_new_batch_pair(
    bootstrap: dict[str, Any], planning: dict[str, Any]
) -> dict[str, Any]:
    """Require all ten execution captures and both aggregates to be distinct."""
    bootstrap_ids = set(bootstrap.get("member_observation_ids", ()))
    planning_ids = set(planning.get("member_observation_ids", ()))
    bootstrap_scenes = set(bootstrap.get("raw_scenes", ()))
    planning_scenes = set(planning.get("raw_scenes", ()))
    bootstrap_stamps = set(bootstrap.get("member_stamps_ns", ()))
    planning_stamps = set(planning.get("member_stamps_ns", ()))
    if any(
        len(values) != 5
        for values in (
            bootstrap_ids,
            planning_ids,
            bootstrap_scenes,
            planning_scenes,
            bootstrap_stamps,
            planning_stamps,
        )
    ):
        raise SupervisorError("new execution batch identity evidence is incomplete")
    overlap = {
        "observation_ids": sorted(
            bootstrap_ids & planning_ids
        ),
        "raw_scenes": sorted(
            bootstrap_scenes & planning_scenes
        ),
        "stamps_ns": sorted(
            bootstrap_stamps & planning_stamps
        ),
        "aggregate_observation_id": (
            [bootstrap.get("aggregate_observation_id")]
            if bootstrap.get("aggregate_observation_id")
            == planning.get("aggregate_observation_id")
            else []
        ),
    }
    if any(overlap.values()):
        raise SupervisorError(
            "new execution bootstrap/planning capture identities overlap"
        )
    return {
        "raw_capture_count": 10,
        "aggregate_count": 2,
        "overlap": overlap,
        "all_identity_sets_disjoint": True,
    }


def _require_v3_execution_plan(execution_plan: dict[str, Any]) -> None:
    """Keep v1/v2 artifacts readable as evidence but outside motion authority."""
    if execution_plan.get("schema_version") != AGGREGATE_PLAN_V3:
        raise SupervisorError(
            "execution requires a v3 five-frame plan with frozen ConfigureNBV "
            "semantics; v2 remains evidence-only"
        )


def _load_execution_plan(
    path: str | Path,
    expected_sha256: str,
    report_sha256: str,
) -> tuple[dict[str, Any], str, Path]:
    """Load the exact accepted read-only plan and reject semantic drift."""
    digest = str(expected_sha256).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SupervisorError("execution_plan_sha256 must be 64 lowercase hex digits")
    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file():
        raise SupervisorError(f"execution plan does not exist: {plan_path}")
    payload = plan_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, digest):
        raise SupervisorError(
            f"execution plan SHA256 mismatch: expected {digest}, got {actual}"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupervisorError(f"execution plan is invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise SupervisorError("execution plan root must be an object")
    outer_document = document
    aggregate_v2 = False
    aggregate_schema = ""
    configuration_evidence = None
    plan_motion_limits: MotionLimits = SMALL_MOTION_LIMITS
    if document.get("schema") == "strawberry_real_nbv_once_supervisor/v1":
        if document.get("status") != "passed_preview_only":
            raise SupervisorError("aggregate preview outer status must be passed_preview_only")
        if document.get("execute_requested") is not False:
            raise SupervisorError("aggregate preview must have execute_requested=false")
        if document.get("preview_motion_commands_observed") != 0:
            raise SupervisorError("aggregate preview observed motion commands")
        candidate = document.get("execution_plan_candidate")
        if not isinstance(candidate, dict):
            raise SupervisorError("aggregate preview has no execution_plan_candidate")
        stored_policy = candidate.get("session_policy")
        if stored_policy is not None:
            validated_policy = _validated_stored_session_policy(stored_policy)
            try:
                plan_motion_limits = motion_limits_for_profile(
                    validated_policy.get(
                        "motion_profile", SMALL_MOTION_PROFILE
                    )
                )
            except ValueError as error:
                raise SupervisorError(str(error)) from error
            if document.get("session_policy") != stored_policy:
                raise SupervisorError(
                    "outer and embedded session policies differ"
                )
        aggregate_schema = str(candidate.get("schema_version", ""))
        if aggregate_schema not in (AGGREGATE_PLAN_V2, AGGREGATE_PLAN_V3):
            raise SupervisorError("unsupported aggregate execution plan schema")
        embedded_selected = candidate.get("selected_candidate", {})
        outer_selected = document.get("selected_candidate", {})
        for key in ("T_base_camera", "T_base_link7", "alpha"):
            if embedded_selected.get(key) != outer_selected.get(key):
                raise SupervisorError(
                    "embedded selected candidate differs from preview result"
                )
        if (
            stored_policy is not None
            and stored_policy.get("schema")
            == "strawberry_real_nbv_motion_session_policy/v4"
        ):
            outer_selection = _candidate_selection_evidence(outer_selected)
            if (
                outer_selection is None
                or embedded_selected.get("reachable_view_selection")
                != outer_selection
            ):
                raise SupervisorError(
                    "embedded reachable-view selection differs from preview audit"
                )
            if not math.isclose(
                float(outer_selection["radius_m"]),
                float(outer_selected.get("camera_translation_m", math.nan)),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise SupervisorError(
                    "reachable-view radius differs from selected camera translation"
                )
        if candidate.get("exact_tf", {}).get("T_base_camera_optical") != document.get(
            "final_tf", {}
        ).get("T_base_camera_optical"):
            raise SupervisorError("embedded current camera differs from preview exact TF")
        if candidate.get("observation", {}).get("observation_id") != document.get(
            "corrected_observation", {}
        ).get("observation_id"):
            raise SupervisorError("embedded observation differs from preview result")
        document = candidate
        aggregate_v2 = True
    supported_schema = (
        aggregate_schema
        if aggregate_v2
        else "strawberry_real_nbv_readonly_plan/v1"
    )
    if document.get("schema_version") != supported_schema:
        raise SupervisorError("unsupported execution plan schema")
    if document.get("status") != "passed":
        raise SupervisorError("execution plan status must be passed")
    safety = document.get("safety")
    if not isinstance(safety, dict) or safety.get("passed") is not True:
        raise SupervisorError("execution plan safety evidence did not pass")
    if safety.get("controller_execution_enabled") is not False:
        raise SupervisorError("preview was not recorded with controller gate closed")
    if safety.get("motion_command_count_observed") != 0:
        raise SupervisorError("preview observed unexpected motion commands")
    false_safety_fields = ["gate_clients_created", "motion_action_clients_created"]
    false_safety_fields.append(
        "command_publishers_created" if aggregate_v2 else "ros_publishers_created"
    )
    for key in false_safety_fields:
        if safety.get(key) is not False:
            raise SupervisorError(f"preview safety field {key} must be false")
    if safety.get("controller_diagnostic_fresh_after_solve") is not True:
        raise SupervisorError("preview controller diagnostic was not fresh")
    report = document.get("handeye_report")
    if not isinstance(report, dict) or report.get("sha256") != report_sha256:
        raise SupervisorError("execution plan hand-eye SHA does not match runtime report")
    observation = document.get("observation")
    if not isinstance(observation, dict) or not str(
        observation.get("observation_id", "")
    ).strip():
        raise SupervisorError("execution plan observation binding is missing")
    scene_id = str(document.get("scene_id", "")).strip()
    configured_scene = (
        document.get("observation", {}).get("scene_id")
        if aggregate_v2
        else document.get("nbv_config", {}).get("scene_id")
    )
    if not scene_id or configured_scene != scene_id:
        raise SupervisorError("execution plan scene binding is inconsistent")
    nbv = document.get("nbv")
    if not isinstance(nbv, dict):
        raise SupervisorError("execution plan NBV evidence is missing")
    if aggregate_v2:
        planned_gain = float(nbv.get("planned_gain", math.nan))
        if (
            not math.isfinite(planned_gain)
            or planned_gain <= 0.0
            or nbv.get("strict_gain_evidence")
            != "core_invariant_nonzero_translation"
        ):
            raise SupervisorError("aggregate plan has no valid strict-gain evidence")
        aggregation = document.get("aggregation")
        if not isinstance(aggregation, dict) or aggregation.get(
            "capture_count_per_batch"
        ) != AGGREGATE_CAPTURE_COUNT:
            raise SupervisorError("aggregate plan does not bind two complete 5-frame batches")
        if aggregation.get("selection_performed") is not False:
            raise SupervisorError("aggregate plan must prove that no frame was selected")
        batch_ids: list[str] = []
        batch_scenes: list[str] = []
        batch_stamps: list[int] = []
        batch_aggregate_ids: list[str] = []
        for label in ("bootstrap", "planning"):
            ids = aggregation.get(f"{label}_member_ids")
            scenes = aggregation.get(f"{label}_raw_scenes")
            stamps = aggregation.get(f"{label}_member_stamps_ns")
            pose_span = aggregation.get(f"{label}_pose_span")
            aggregate_id = str(
                aggregation.get(f"{label}_aggregate_observation_id", "")
            )
            aggregate_digest = str(
                aggregation.get(f"{label}_aggregate_identity_sha256", "")
            )
            if (
                not isinstance(ids, list)
                or len(ids) != AGGREGATE_CAPTURE_COUNT
                or any(not str(value).strip() for value in ids)
                or len(set(ids)) != AGGREGATE_CAPTURE_COUNT
                or not isinstance(scenes, list)
                or len(scenes) != AGGREGATE_CAPTURE_COUNT
                or any(not str(value).strip() for value in scenes)
                or len(set(scenes)) != AGGREGATE_CAPTURE_COUNT
                or not isinstance(stamps, list)
                or len(stamps) != AGGREGATE_CAPTURE_COUNT
                or any(not isinstance(value, int) or value <= 0 for value in stamps)
                or len(set(stamps)) != AGGREGATE_CAPTURE_COUNT
                or stamps != sorted(stamps)
                or not aggregate_id
                or len(aggregate_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in aggregate_digest
                )
            ):
                raise SupervisorError(
                    f"aggregate plan {label} batch identity evidence is invalid"
                )
            try:
                recomputed_digest = five_frame_identity_sha256(
                    scenes, ids, stamps
                )
            except ValueError as error:
                raise SupervisorError(
                    f"aggregate plan {label} identity cannot be recomputed: {error}"
                ) from error
            if not hmac.compare_digest(recomputed_digest, aggregate_digest):
                raise SupervisorError(
                    f"aggregate plan {label} identity SHA256 does not recompute"
                )
            if not isinstance(pose_span, dict):
                raise SupervisorError(
                    f"aggregate plan {label} pose-span evidence is missing"
                )
            span_values = np.asarray(
                (
                    pose_span.get("max_translation_m", math.nan),
                    pose_span.get("max_rotation_deg", math.nan),
                ),
                dtype=float,
            )
            if (
                not np.all(np.isfinite(span_values))
                or np.any(span_values < 0.0)
                or span_values[0] > AGGREGATE_MAX_TRANSLATION_SPAN_M + 1.0e-12
                or span_values[1] > AGGREGATE_MAX_ROTATION_SPAN_DEG + 1.0e-9
            ):
                raise SupervisorError(
                    f"aggregate plan {label} camera pose span exceeds the fixed gate"
                )
            outer_batch = outer_document.get(f"{label}_batch")
            if (
                not isinstance(outer_batch, dict)
                or outer_batch.get("selection_performed") is not False
                or outer_batch.get("capture_count") != AGGREGATE_CAPTURE_COUNT
                or outer_batch.get("member_observation_ids") != ids
                or outer_batch.get("raw_scenes") != scenes
                or outer_batch.get("member_stamps_ns") != stamps
                or outer_batch.get("pose_span") != pose_span
                or outer_batch.get("aggregate_observation_id") != aggregate_id
                or outer_batch.get("aggregate_identity_sha256")
                != aggregate_digest
                or outer_batch.get("reference_member_index_zero_based") != 2
                or outer_batch.get("reference_observation_id") != ids[2]
                or outer_batch.get("camera_info_exactly_equal") is not True
            ):
                raise SupervisorError(
                    f"embedded {label} aggregation differs from outer preview audit"
                )
            exact_tfs = outer_batch.get("member_exact_tfs")
            if not isinstance(exact_tfs, list) or len(exact_tfs) != 5:
                raise SupervisorError(
                    f"aggregate plan {label} exact-TF evidence is incomplete"
                )
            for stamp, exact_tf in zip(stamps, exact_tfs):
                if (
                    not isinstance(exact_tf, dict)
                    or exact_tf.get("requested_stamp_ns") != stamp
                    or exact_tf.get("returned_stamp_ns") != stamp
                    or exact_tf.get("stamp_difference_ns") != 0
                ):
                    raise SupervisorError(
                        f"aggregate plan {label} does not bind exact exposure TF"
                    )
            batch_ids.extend(str(value) for value in ids)
            batch_scenes.extend(str(value) for value in scenes)
            batch_stamps.extend(int(value) for value in stamps)
            batch_aggregate_ids.append(aggregate_id)
        if (
            len(set(batch_ids)) != 10
            or len(set(batch_scenes)) != 10
            or len(set(batch_stamps)) != 10
            or len(set(batch_aggregate_ids)) != 2
        ):
            raise SupervisorError("aggregate bootstrap/planning batches are not independent")
        if aggregation.get("planning_aggregate_observation_id") != observation.get(
            "observation_id"
        ):
            raise SupervisorError("aggregate plan observation is not the planning aggregate")
        if aggregate_schema == AGGREGATE_PLAN_V3:
            frozen_configuration = document.get("nbv_configuration")
            try:
                configuration_evidence = normalize_nbv_configuration(
                    frozen_configuration,
                    max_step_ceiling_m=(
                        plan_motion_limits.maximum_camera_step_m
                    ),
                )
            except (TypeError, ValueError) as error:
                raise SupervisorError(
                    f"frozen ConfigureNBV semantics are invalid: {error}"
                ) from error
            if configuration_evidence.request != frozen_configuration:
                raise SupervisorError(
                    "frozen ConfigureNBV semantics are not wire-canonical"
                )
            if not hmac.compare_digest(
                configuration_evidence.sha256,
                str(document.get("nbv_configuration_sha256", "")),
            ):
                raise SupervisorError("frozen ConfigureNBV SHA256 does not recompute")
            voxel_grid = document.get("derived_voxel_grid")
            expected_voxel_grid = {
                "dimensions": list(configuration_evidence.voxel_dimensions),
                "origin_m": array_list(configuration_evidence.map_origin_m),
            }
            if voxel_grid != expected_voxel_grid:
                raise SupervisorError("frozen voxel-grid origin does not recompute")
            outer_configuration = outer_document.get("configuration")
            if (
                not isinstance(outer_configuration, dict)
                or outer_configuration.get("request_semantics")
                != configuration_evidence.request
                or outer_configuration.get("request_sha256")
                != configuration_evidence.sha256
                or outer_configuration.get("derived_voxel_grid")
                != expected_voxel_grid
                or outer_configuration.get("response_code") != 0
            ):
                raise SupervisorError(
                    "embedded ConfigureNBV semantics differ from outer preview audit"
                )
            if (
                configuration_evidence.request["scene_id"] != scene_id
                or configuration_evidence.request["world_frame"] != "base_link"
            ):
                raise SupervisorError(
                    "frozen ConfigureNBV scene/world binding is inconsistent"
                )
            target = document.get("target")
            if (
                not isinstance(target, dict)
                or target.get("center_base_link_m")
                != configuration_evidence.request["target_center_m"]
                or outer_document.get("bootstrap_target", {}).get("base_xyz_m")
                != configuration_evidence.request["target_center_m"]
                or outer_document.get("final_target", {}).get("base_xyz_m")
                != target.get("planning_batch_center_base_link_m")
            ):
                raise SupervisorError(
                    "frozen ConfigureNBV target differs from preview target evidence"
                )
    else:
        gain_improvement = float(nbv.get("gain_improvement", math.nan))
        if not math.isfinite(gain_improvement) or gain_improvement <= 0.0:
            raise SupervisorError(
                "execution plan has no positive numerical gain improvement"
            )
    selected = document.get("selected_candidate")
    if not isinstance(selected, dict):
        raise SupervisorError("execution plan selected_candidate is missing")
    try:
        artifact_current = validate_rigid_transform(
            np.asarray(document["exact_tf"]["T_base_camera_optical"], dtype=float),
            "artifact current camera",
        )
        artifact_target = validate_rigid_transform(
            np.asarray(selected["T_base_camera"], dtype=float),
            "artifact selected camera",
        )
        validate_rigid_transform(
            np.asarray(selected["T_base_link7"], dtype=float),
            "artifact selected link7",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SupervisorError(f"execution plan target matrices are invalid: {error}") from error
    motion = camera_motion(artifact_current, artifact_target)
    if configuration_evidence is not None:
        configuration = configuration_evidence.request
        observation_min = np.asarray(configuration["observation_min_m"], dtype=float)
        observation_max = np.asarray(configuration["observation_max_m"], dtype=float)
        for label, position in (
            ("artifact current camera", artifact_current[:3, 3]),
            ("artifact selected camera", artifact_target[:3, 3]),
        ):
            if np.any(position < observation_min - 1.0e-9) or np.any(
                position > observation_max + 1.0e-9
            ):
                raise SupervisorError(
                    f"{label} lies outside frozen observation bounds"
                )
    if not camera_step_above_minimum(
        motion.translation_m, plan_motion_limits
    ) or (
        motion.translation_m
        > plan_motion_limits.maximum_camera_step_m + 1.0e-9
    ):
        raise SupervisorError(
            "artifact camera translation is outside its bound motion profile"
        )
    if (
        motion.rotation_rad
        > plan_motion_limits.maximum_camera_rotation_rad + 1.0e-9
    ):
        raise SupervisorError(
            "artifact camera rotation exceeds its bound motion profile"
        )
    if float(selected.get("alpha", math.nan)) not in DEFAULT_ALPHAS:
        raise SupervisorError("artifact selected alpha is not in the audited sequence")
    ik = document.get("ik")
    if not isinstance(ik, dict) or ik.get("success") is not True or ik.get("code") != 0:
        raise SupervisorError("artifact IK did not return exact SUCCESS")
    ik_values = np.asarray(
        (
            ik.get("max_joint_delta_rad"),
            ik.get("position_error_m"),
            ik.get("orientation_error_rad"),
            ik.get("sigma_min"),
            ik.get("condition_number"),
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(ik_values)):
        raise SupervisorError("artifact IK diagnostics are non-finite")
    if (
        ik_values[0] > plan_motion_limits.maximum_ik_joint_delta_rad
        or ik_values[1] > 0.003
        or ik_values[2] > math.radians(2.0)
        or ik_values[3] < 0.10
        or ik_values[4] > 20.0
    ):
        raise SupervisorError("artifact IK diagnostics violate execution gates")
    # Preserve provenance of the exact outer bytes without letting callers
    # accidentally treat the embedded candidate as a separately hashed file.
    document["_verified_outer_schema"] = outer_document.get(
        "schema", outer_document.get("schema_version")
    )
    return document, actual, plan_path


class RealNBVSupervisor(Node):
    """Preview once, then optionally run one bounded convergence session."""

    def __init__(self) -> None:
        """Create observers and read-only clients; defer motion entities."""
        super().__init__("real_nbv_supervisor")
        self.declare_parameter("execute", False)
        self.declare_parameter("motion_profile", SMALL_MOTION_PROFILE)
        self.declare_parameter("max_motion_steps", 1)
        self.declare_parameter("coverage_target", 0.0)
        self.declare_parameter(
            "coverage_plateau_delta", CONVERGENCE_COVERAGE_DELTA
        )
        self.declare_parameter(
            "coverage_plateau_patience", DEFAULT_COVERAGE_PLATEAU_PATIENCE
        )
        self.declare_parameter("operator_workspace_clearance_confirmed", False)
        self.declare_parameter("execution_authorization_token", "")
        self.declare_parameter("execution_plan_path", DEFAULT_PLAN_PATH)
        self.declare_parameter("execution_plan_sha256", DEFAULT_PLAN_SHA256)
        self.declare_parameter("authorization_receipt_path", "")
        self.declare_parameter("require_aggregate_execution_plan", True)
        self.declare_parameter(
            "execution_output_path",
            "/home/yyt/strawberry_active_perception/artifacts/week4/"
            "real_nbv_motion_session_execution.json",
        )
        self.declare_parameter("scene_id", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("link_frame", "link7")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter(
            "raw_observation_topic", "/strawberry/perception/observation"
        )
        self.declare_parameter(
            "corrected_observation_topic",
            "/strawberry/perception/real_nbv_observation",
        )
        self.declare_parameter(
            "capture_service", "/strawberry/perception/capture_observation"
        )
        self.declare_parameter("configure_service", "/strawberry/nbv/configure")
        self.declare_parameter(
            "evaluate_candidates_service", "/strawberry/nbv/evaluate_candidates"
        )
        self.declare_parameter(
            "compute_action", "/strawberry/nbv/compute_next_view"
        )
        self.declare_parameter("solve_ik_service", "/strawberry_nero/solve_ik")
        self.declare_parameter("joint_state_topic", "/feedback/joint_states")
        self.declare_parameter("motion_command_topic", "/control/move_j")
        self.declare_parameter("arm_status_topic", "/feedback/arm_status")
        self.declare_parameter(
            "controller_diagnostics_topic", "/strawberry_nero/diagnostics"
        )
        self.declare_parameter(
            "controller_parameters_service", "/nero_control/get_parameters"
        )
        self.declare_parameter("driver_gate_service", "/control_enable")
        self.declare_parameter(
            "controller_gate_service", "/strawberry_nero/enable_execution"
        )
        self.declare_parameter(
            "move_action", "/strawberry_nero/move_to_pose"
        )
        self.declare_parameter("handeye_report_path", DEFAULT_REPORT_PATH)
        self.declare_parameter("handeye_report_sha256", DEFAULT_REPORT_SHA256)
        self.declare_parameter("minimum_calibration_samples", 30)
        self.declare_parameter("output_path", DEFAULT_OUTPUT_PATH)
        # The verified USB2 profile can show short color-stream scheduling
        # gaps. Five seconds keeps capture-on-demand reproducible while the
        # service still fails closed instead of publishing a stale frame.
        self.declare_parameter("capture_timeout_sec", 5.0)
        self.declare_parameter("capture_discard_frames", 3)
        self.declare_parameter("planning_capture_count", 5)
        self.declare_parameter("aggregate_min_finite_count", 3)
        self.declare_parameter("aggregate_mask_majority_count", 3)
        self.declare_parameter("max_batch_camera_translation_span_m", 0.00050)
        self.declare_parameter("max_batch_camera_rotation_span_deg", 0.10)
        self.declare_parameter("post_motion_camera_settle_delay_sec", 2.0)
        self.declare_parameter("operation_timeout_sec", 20.0)
        self.declare_parameter("tf_timeout_sec", 1.0)
        self.declare_parameter("minimum_target_pixels", 200)
        self.declare_parameter("depth_min_m", 0.20)
        self.declare_parameter("depth_max_m", 2.50)
        self.declare_parameter("voxel_size_m", 0.003)
        self.declare_parameter("map_size_m", 0.30)
        self.declare_parameter("target_roi_size_m", 0.15)
        self.declare_parameter("max_step_m", MAX_CAMERA_STEP_M)
        self.declare_parameter("observation_bound_half_extent_m", 0.016)
        self.declare_parameter("max_target_drift_m", 0.020)
        self.declare_parameter(
            "max_frozen_target_center_drift_m",
            MAX_FROZEN_TARGET_CENTER_DRIFT_M,
        )
        self.declare_parameter(
            "max_plan_start_translation_drift_m",
            MAX_PLAN_START_TRANSLATION_DRIFT_M,
        )
        self.declare_parameter(
            "max_plan_start_rotation_drift_deg",
            MAX_PLAN_START_ROTATION_DRIFT_DEG,
        )
        self.declare_parameter(
            "max_plan_target_translation_drift_m",
            MAX_PLAN_TARGET_TRANSLATION_DRIFT_M,
        )
        self.declare_parameter(
            "max_plan_target_rotation_drift_deg",
            MAX_PLAN_TARGET_ROTATION_DRIFT_DEG,
        )
        self.declare_parameter("max_plan_solution_drift_rad", 0.010)
        self.declare_parameter("post_target_position_tolerance_m", 0.005)
        self.declare_parameter("post_target_rotation_tolerance_deg", 2.0)
        self.declare_parameter(
            "max_selected_camera_rotation_deg",
            math.degrees(MAX_SELECTED_CAMERA_ROTATION_RAD),
        )
        self.declare_parameter("max_ik_joint_delta_rad", MAX_IK_JOINT_DELTA_RAD)
        self.declare_parameter("candidate_alphas", list(DEFAULT_ALPHAS))

        self.execute_requested = bool(self.get_parameter("execute").value)
        self.motion_profile = str(
            self.get_parameter("motion_profile").value
        ).strip()
        try:
            self.motion_limits = motion_limits_for_profile(self.motion_profile)
        except ValueError as error:
            raise SupervisorError(str(error)) from error
        max_motion_steps_value = self.get_parameter("max_motion_steps").value
        if (
            isinstance(max_motion_steps_value, bool)
            or not isinstance(max_motion_steps_value, int)
            or max_motion_steps_value not in ALLOWED_SESSION_MOTION_STEPS
            or max_motion_steps_value > self.motion_limits.maximum_motion_steps
        ):
            raise SupervisorError(
                "max_motion_steps must be an integer in [1, "
                f"{self.motion_limits.maximum_motion_steps}] for profile "
                f"{self.motion_profile}"
            )
        self.max_motion_steps = int(max_motion_steps_value)
        coverage_target_value = float(self.get_parameter("coverage_target").value)
        self.coverage_target = (
            None if coverage_target_value == 0.0 else coverage_target_value
        )
        self.coverage_plateau_delta = float(
            self.get_parameter("coverage_plateau_delta").value
        )
        patience_value = self.get_parameter("coverage_plateau_patience").value
        if isinstance(patience_value, bool) or not isinstance(patience_value, int):
            raise SupervisorError("coverage_plateau_patience must be an integer")
        self.coverage_plateau_patience = int(patience_value)
        minimum_target_pixels_value = self.get_parameter(
            "minimum_target_pixels"
        ).value
        if (
            isinstance(minimum_target_pixels_value, bool)
            or not isinstance(minimum_target_pixels_value, int)
        ):
            raise SupervisorError("minimum_target_pixels must be an integer")
        self.minimum_target_pixels = int(minimum_target_pixels_value)
        self.session_policy = _session_policy(
            self.max_motion_steps,
            motion_profile=self.motion_profile,
            minimum_target_pixels=self.minimum_target_pixels,
            coverage_target=self.coverage_target,
            coverage_plateau_delta=self.coverage_plateau_delta,
            coverage_plateau_patience=self.coverage_plateau_patience,
        )
        if self.get_parameter("require_aggregate_execution_plan").value is not True:
            raise SupervisorError(
                "require_aggregate_execution_plan is a fixed safety invariant"
            )
        self.base_frame = str(self.get_parameter("base_frame").value).strip()
        self.link_frame = str(self.get_parameter("link_frame").value).strip()
        self.camera_frame = str(self.get_parameter("camera_frame").value).strip()
        if (self.base_frame, self.link_frame, self.camera_frame) != (
            "base_link",
            "link7",
            "camera_color_optical_frame",
        ):
            raise SupervisorError(
                "real supervisor frames must be exactly base_link/link7/"
                "camera_color_optical_frame"
            )
        output_parameter = (
            "execution_output_path" if self.execute_requested else "output_path"
        )
        self.output_path = str(self.get_parameter(output_parameter).value).strip()
        if not self.output_path:
            raise SupervisorError("output_path must be non-empty")
        requested_scene = str(self.get_parameter("scene_id").value).strip()
        self.scene_id = requested_scene or (
            "real_nbv_once_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        self.capture_session_id = (
            f"{self.scene_id}_capture_{time.time_ns()}_{os.getpid()}"
        )
        self.raw_topic = str(
            self.get_parameter("raw_observation_topic").value
        ).strip()
        self.corrected_topic = str(
            self.get_parameter("corrected_observation_topic").value
        ).strip()
        if not self.raw_topic or not self.corrected_topic:
            raise SupervisorError("observation topics must be non-empty")
        if self.raw_topic == self.corrected_topic:
            raise SupervisorError("raw and corrected Observation topics must differ")

        self.report = load_verified_handeye_report(
            str(self.get_parameter("handeye_report_path").value),
            str(self.get_parameter("handeye_report_sha256").value),
            int(self.get_parameter("minimum_calibration_samples").value),
        )
        self.operation_timeout = float(
            self.get_parameter("operation_timeout_sec").value
        )
        self.tf_timeout = float(self.get_parameter("tf_timeout_sec").value)
        if self.operation_timeout <= 0.0 or self.tf_timeout <= 0.0:
            raise SupervisorError("operation and TF timeouts must be positive")
        if (
            int(self.get_parameter("planning_capture_count").value)
            != AGGREGATE_CAPTURE_COUNT
            or int(self.get_parameter("aggregate_min_finite_count").value)
            != AGGREGATE_MIN_FINITE_COUNT
            or int(self.get_parameter("aggregate_mask_majority_count").value)
            != AGGREGATE_MASK_MAJORITY_COUNT
        ):
            raise SupervisorError(
                "the audited aggregation contract is fixed at 5 captures, "
                "3 finite depth samples, and 3 mask votes"
            )
        batch_translation_limit = float(
            self.get_parameter("max_batch_camera_translation_span_m").value
        )
        batch_rotation_limit_deg = float(
            self.get_parameter("max_batch_camera_rotation_span_deg").value
        )
        if (
            not 0.0 < batch_translation_limit
            <= AGGREGATE_MAX_TRANSLATION_SPAN_M
            or not 0.0 < batch_rotation_limit_deg
            <= AGGREGATE_MAX_ROTATION_SPAN_DEG
        ):
            raise SupervisorError(
                "five-frame camera pose-span gates may only be made stricter than "
                "0.50 mm / 0.10 degrees"
            )
        settle_delay = float(
            self.get_parameter("post_motion_camera_settle_delay_sec").value
        )
        if not math.isfinite(settle_delay) or not 0.0 <= settle_delay <= 10.0:
            raise SupervisorError(
                "post_motion_camera_settle_delay_sec must be within [0, 10]"
            )
        audited_upper_bounds = {
            "max_step_m": self.motion_limits.maximum_camera_step_m,
            "max_target_drift_m": 0.020,
            "max_frozen_target_center_drift_m": (
                MAX_FROZEN_TARGET_CENTER_DRIFT_M
            ),
            "max_plan_start_translation_drift_m": (
                MAX_PLAN_START_TRANSLATION_DRIFT_M
            ),
            "max_plan_start_rotation_drift_deg": MAX_PLAN_START_ROTATION_DRIFT_DEG,
            "max_plan_target_translation_drift_m": (
                MAX_PLAN_TARGET_TRANSLATION_DRIFT_M
            ),
            "max_plan_target_rotation_drift_deg": (
                MAX_PLAN_TARGET_ROTATION_DRIFT_DEG
            ),
            "max_plan_solution_drift_rad": 0.010,
            "post_target_position_tolerance_m": 0.005,
            "post_target_rotation_tolerance_deg": 2.0,
            "max_selected_camera_rotation_deg": math.degrees(
                self.motion_limits.maximum_camera_rotation_rad
            ),
            "max_ik_joint_delta_rad": (
                self.motion_limits.maximum_ik_joint_delta_rad
            ),
        }
        for parameter_name, audited_ceiling in audited_upper_bounds.items():
            parameter_value = float(self.get_parameter(parameter_name).value)
            if (
                not math.isfinite(parameter_value)
                or parameter_value <= 0.0
                or parameter_value > audited_ceiling + 1.0e-12
            ):
                raise SupervisorError(
                    f"{parameter_name} may only be made stricter than its "
                    f"audited ceiling {audited_ceiling:g}"
                )
        observation_half_extent = float(
            self.get_parameter("observation_bound_half_extent_m").value
        )
        required_half_extent = (
            self.motion_limits.maximum_session_translation_m + 0.001
            if self.max_motion_steps > 1
            else self.motion_limits.maximum_camera_step_m + 0.001
        )
        maximum_half_extent = (
            0.016
            if self.motion_profile == SMALL_MOTION_PROFILE
            else self.motion_limits.maximum_session_translation_m + 0.001
        )
        if (
            not math.isfinite(observation_half_extent)
            or observation_half_extent < required_half_extent - 1.0e-12
            or observation_half_extent > maximum_half_extent + 1.0e-12
        ):
            raise SupervisorError(
                "observation_bound_half_extent_m must cover the selected session "
                "envelope and cannot exceed its audited bound "
                f"{maximum_half_extent:g} m (need {required_half_extent:g})"
            )

        state_qos = QoSProfile(
            depth=32,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        live_qos = QoSProfile(
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._observations: dict[tuple[str, str], tuple[Observation, float]] = {}
        self._joint_positions: np.ndarray | None = None
        self._joint_velocities: np.ndarray | None = None
        self._joint_received = 0.0
        self._motion_command_count = 0
        self._motion_goal_count = 0
        # True only after both software gates have each returned two
        # successful false acknowledgements.  Keep this independent from the
        # current step record: a later step can fail before it has a
        # postaction record even though the previous step closed both gates.
        self._gates_closed_proven = False
        self._arm_status: AgxArmStatus | None = None
        self._arm_status_received = 0.0
        self._controller_diagnostic: dict[str, Any] | None = None
        self._controller_diagnostic_received = 0.0
        self._active_nbv_configuration: dict[str, Any] | None = None
        self._configure_call_count = 0
        self._session_ledger: MotionSessionLedger | None = None
        self._authorization_plan_sha256: str | None = None
        self._authorization_receipt_path: Path | None = None
        self._audit: dict[str, Any] = {}
        self._session_raw_observation_ids: set[str] = set()
        self._session_raw_scenes: set[str] = set()
        self._session_raw_stamps_ns: set[int] = set()
        self._session_aggregate_ids: set[str] = set()
        self.create_subscription(
            Observation, self.raw_topic, self._on_observation, state_qos
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_state,
            live_qos,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("motion_command_topic").value),
            self._on_motion_command,
            live_qos,
        )
        self.create_subscription(
            AgxArmStatus,
            str(self.get_parameter("arm_status_topic").value),
            self._on_arm_status,
            live_qos,
        )
        self.create_subscription(
            DiagnosticArray,
            str(self.get_parameter("controller_diagnostics_topic").value),
            self._on_controller_diagnostic,
            live_qos,
        )
        self._corrected_publisher = self.create_publisher(
            Observation, self.corrected_topic, state_qos
        )
        self._capture_client = self.create_client(
            CaptureObservation,
            str(self.get_parameter("capture_service").value),
        )
        self._configure_client = self.create_client(
            ConfigureNBV,
            str(self.get_parameter("configure_service").value),
        )
        self._evaluate_candidates_client = self.create_client(
            EvaluateViewCandidates,
            str(self.get_parameter("evaluate_candidates_service").value),
        )
        self._compute_client = ActionClient(
            self,
            ComputeNextView,
            str(self.get_parameter("compute_action").value),
        )
        self._solve_client = self.create_client(
            SolveIK, str(self.get_parameter("solve_ik_service").value)
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=False
        )

    def _on_observation(self, message: Observation) -> None:
        key = (str(message.scene_id), str(message.observation_id))
        if all(key) and key not in self._observations:
            self._observations[key] = (copy.deepcopy(message), time.monotonic())
            if len(self._observations) > 64:
                oldest = min(self._observations, key=lambda item: self._observations[item][1])
                del self._observations[oldest]

    def _on_joint_state(self, message: JointState) -> None:
        try:
            positions = ordered_joint_positions(
                message.name, message.position, label="measured joint state"
            )
            velocities = ordered_joint_positions(
                message.name, message.velocity, label="measured joint velocity"
            )
        except ValueError:
            return
        self._joint_positions = positions
        self._joint_velocities = velocities
        self._joint_received = time.monotonic()

    def _on_motion_command(self, _message: JointState) -> None:
        self._motion_command_count += 1

    def _on_arm_status(self, message: AgxArmStatus) -> None:
        self._arm_status = copy.deepcopy(message)
        self._arm_status_received = time.monotonic()

    @staticmethod
    def _diagnostic_value(text: str) -> Any:
        normalized = str(text).strip()
        if normalized.lower() == "true":
            return True
        if normalized.lower() == "false":
            return False
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            return normalized

    def _on_controller_diagnostic(self, message: DiagnosticArray) -> None:
        for status in message.status:
            if status.hardware_id != "nero" or not status.name.endswith(": safety"):
                continue
            self._controller_diagnostic = {
                "name": str(status.name),
                "level": wire_uint8(status.level, "DiagnosticStatus.level"),
                "message": str(status.message),
                "values": {
                    str(item.key): self._diagnostic_value(item.value)
                    for item in status.values
                },
            }
            self._controller_diagnostic_received = time.monotonic()

    def _spin_until(
        self, predicate: Callable[[], bool], timeout: float, label: str
    ) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=0.02)
        raise SupervisorError(f"timed out waiting for {label}")

    def _wait_future(self, future: Any, timeout: float, label: str) -> Any:
        self._spin_until(future.done, timeout, label)
        error = future.exception()
        if error is not None:
            raise SupervisorError(f"{label} failed: {error}")
        return future.result()

    def _checkpoint_execution_audit(self, status: str) -> None:
        """Atomically persist the exact session state before/after each goal."""
        if not self.execute_requested:
            return
        self._audit["status"] = status
        self._audit["last_checkpoint_utc"] = _utc_now()
        self._audit["motion_goal_count"] = self._motion_goal_count
        self._audit["motion_commands_observed"] = self._motion_command_count
        _write_json_atomic(self.output_path, self._audit)

    def _consume_session_authorization(self) -> None:
        """Persist a plan-SHA receipt before the first motion Goal is sent."""
        if self._authorization_plan_sha256 is None:
            raise SupervisorError("execution authorization plan SHA is unavailable")
        if self._authorization_receipt_path is None:
            raise SupervisorError("execution authorization receipt path is unavailable")
        receipt = {
            "schema": "strawberry_real_nbv_authorization_receipt/v1",
            "plan_sha256": self._authorization_plan_sha256,
            "max_motion_steps": self.max_motion_steps,
            "consumed_before_first_goal_utc": _utc_now(),
            "reusable": False,
        }
        _write_json_atomic(self._authorization_receipt_path, receipt)
        self._audit["authorization_receipt"] = {
            **receipt,
            "path": str(self._authorization_receipt_path),
        }

    def _register_session_batch(
        self, batch: dict[str, Any], label: str
    ) -> dict[str, Any]:
        """Reject reuse of any raw or aggregate Observation within a session."""
        raw_ids = set(batch.get("member_observation_ids", ()))
        raw_scenes = set(batch.get("raw_scenes", ()))
        raw_stamps = set(batch.get("member_stamps_ns", ()))
        aggregate_id = str(batch.get("aggregate_observation_id", "")).strip()
        if not (
            len(raw_ids) == len(raw_scenes) == len(raw_stamps) == 5
            and aggregate_id
        ):
            raise SupervisorError(f"{label} batch identity evidence is incomplete")
        overlap = {
            "observation_ids": sorted(
                raw_ids & self._session_raw_observation_ids
            ),
            "raw_scenes": sorted(raw_scenes & self._session_raw_scenes),
            "stamps_ns": sorted(raw_stamps & self._session_raw_stamps_ns),
            "aggregate_observation_id": (
                [aggregate_id]
                if aggregate_id in self._session_aggregate_ids
                else []
            ),
        }
        if any(overlap.values()):
            raise SupervisorError(
                f"{label} reuses an Observation identity from this session"
            )
        self._session_raw_observation_ids.update(raw_ids)
        self._session_raw_scenes.update(raw_scenes)
        self._session_raw_stamps_ns.update(int(value) for value in raw_stamps)
        self._session_aggregate_ids.add(aggregate_id)
        return {
            "label": label,
            "raw_capture_count": 5,
            "aggregate_observation_id": aggregate_id,
            "overlap": overlap,
            "all_session_identity_sets_disjoint": True,
        }

    def _capture(self, scene_id: str) -> tuple[Observation, dict[str, Any]]:
        if not self._capture_client.wait_for_service(timeout_sec=3.0):
            raise SupervisorError("CaptureObservation service is unavailable")
        request = CaptureObservation.Request()
        request.scene_id = scene_id
        request.not_before = self.get_clock().now().to_msg()
        request.timeout = _duration_message(
            float(self.get_parameter("capture_timeout_sec").value)
        )
        request.discard_frames = int(
            self.get_parameter("capture_discard_frames").value
        )
        request.require_color = True
        request.require_mask = True
        # The adapter's fixed camera_session/I pose is deliberately ignored.
        request.require_pose = False
        requested_not_before_ns = stamp_nanoseconds(
            request.not_before, "Capture not_before"
        )
        response = self._wait_future(
            self._capture_client.call_async(request),
            self.operation_timeout,
            "CaptureObservation response",
        )
        if not response.success or int(response.code) != CaptureObservation.Response.SUCCESS:
            raise SupervisorError(
                f"CaptureObservation rejected code={response.code}: {response.reason}"
            )
        if not str(response.observation_id).strip():
            raise SupervisorError("CaptureObservation returned an empty observation_id")
        response_stamp_ns = stamp_nanoseconds(
            response.stamp, "CaptureObservation response stamp"
        )
        if response_stamp_ns < requested_not_before_ns:
            raise SupervisorError("CaptureObservation violated not_before")
        key = (scene_id, str(response.observation_id))
        self._spin_until(
            lambda: key in self._observations,
            2.0,
            "the captured canonical Observation topic sample",
        )
        observation = self._observations[key][0]
        observation_stamp_ns = stamp_nanoseconds(
            observation.header.stamp, "Observation stamp"
        )
        if observation_stamp_ns != response_stamp_ns:
            raise SupervisorError("Capture response stamp does not match Observation")
        self._validate_real_observation(observation)
        return observation, {
            "scene_id": scene_id,
            "observation_id": str(response.observation_id),
            "stamp_ns": response_stamp_ns,
            "not_before_ns": requested_not_before_ns,
            "discard_frames": int(request.discard_frames),
            "require_color": True,
            "require_mask": True,
            "require_pose": False,
            "response_code": int(response.code),
            "response_reason": str(response.reason),
        }

    def _capture_five_frame_aggregate(
        self,
        aggregate_scene_id: str,
        batch_label: str,
        *,
        target_configuration: dict[str, Any] | None = None,
    ) -> tuple[Observation, np.ndarray, dict[str, Any]]:
        """Capture exactly five unselected frames and aggregate deterministically."""
        members: list[Observation] = []
        member_captures: list[dict[str, Any]] = []
        member_poses: list[np.ndarray] = []
        member_tf_audits: list[dict[str, Any]] = []
        member_target_audits: list[dict[str, Any]] = []
        member_target_centres: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        mask_frames: list[np.ndarray] = []
        camera_signatures: list[tuple[Any, ...]] = []
        member_grids: list[tuple[int, int]] = []

        for index in range(AGGREGATE_CAPTURE_COUNT):
            raw_scene = (
                f"{self.capture_session_id}_{batch_label}_raw_{index + 1:02d}"
            )
            observation, capture_audit = self._capture(raw_scene)
            pose, tf_audit = self._exact_base_camera(observation)
            target, target_audit = self._estimate_target(
                observation,
                pose,
                configuration=target_configuration,
                # A five-frame aggregate is specifically allowed to recover
                # pixels missing from as many as two member frames.  Member
                # centres are diagnostic only; the resulting aggregate is
                # still required to pass the profile-bound session gate.
                minimum_pixels=1,
            )
            session_minimum = int(
                self.get_parameter("minimum_target_pixels").value
            )
            target_audit["diagnostic_only"] = True
            target_audit["session_minimum_target_pixels"] = session_minimum
            target_audit["meets_session_minimum"] = bool(
                target_audit["mask_pixels"] >= session_minimum
                and target_audit["valid_mask_pixels"] >= session_minimum
                and target_audit["retained_pixels"] >= session_minimum
            )
            members.append(observation)
            member_captures.append(capture_audit)
            member_poses.append(pose)
            member_tf_audits.append(tf_audit)
            member_target_centres.append(target)
            member_target_audits.append(target_audit)
            depth = decode_depth_32fc1(observation.depth)
            mask = decode_mask_mono8(observation.target_mask)
            grid = depth.shape
            if mask.shape != grid or (
                int(observation.color.height), int(observation.color.width)
            ) != grid or (
                int(observation.camera_info.height),
                int(observation.camera_info.width),
            ) != grid:
                raise SupervisorError(
                    "color/depth/mask/CameraInfo grids differ within a raw capture"
                )
            depth_frames.append(depth)
            mask_frames.append(mask)
            camera_signatures.append(_camera_info_signature(observation.camera_info))
            member_grids.append(grid)

        ids = [str(message.observation_id) for message in members]
        raw_scenes = [str(message.scene_id) for message in members]
        stamps_ns = [stamp_nanoseconds(message.header.stamp) for message in members]
        if (
            len(set(ids)) != AGGREGATE_CAPTURE_COUNT
            or len(set(raw_scenes)) != AGGREGATE_CAPTURE_COUNT
        ):
            raise SupervisorError("five-frame batch contains duplicate IDs or raw scenes")
        if (
            len(set(stamps_ns)) != AGGREGATE_CAPTURE_COUNT
            or stamps_ns != sorted(stamps_ns)
        ):
            raise SupervisorError(
                "five-frame batch exposure timestamps are not unique and increasing"
            )
        reference_signature = camera_signatures[2]
        if any(signature != reference_signature for signature in camera_signatures):
            raise SupervisorError("CameraInfo changed within the five-frame batch")
        if len(set(member_grids)) != 1:
            raise SupervisorError("image shape changed within the five-frame batch")

        maximum_translation = 0.0
        maximum_rotation = 0.0
        maximum_translation_pair = (0, 0)
        maximum_rotation_pair = (0, 0)
        for first in range(AGGREGATE_CAPTURE_COUNT):
            for second in range(first + 1, AGGREGATE_CAPTURE_COUNT):
                motion = camera_motion(member_poses[first], member_poses[second])
                if motion.translation_m > maximum_translation:
                    maximum_translation = motion.translation_m
                    maximum_translation_pair = (first, second)
                if motion.rotation_rad > maximum_rotation:
                    maximum_rotation = motion.rotation_rad
                    maximum_rotation_pair = (first, second)
        translation_limit = float(
            self.get_parameter("max_batch_camera_translation_span_m").value
        )
        rotation_limit = math.radians(
            float(self.get_parameter("max_batch_camera_rotation_span_deg").value)
        )
        if maximum_translation > translation_limit + 1.0e-9:
            raise SupervisorError(
                "camera moved too far within five-frame aggregate: "
                f"{maximum_translation:.9f} m"
            )
        if maximum_rotation > rotation_limit + 1.0e-9:
            raise SupervisorError(
                "camera rotated too far within five-frame aggregate: "
                f"{math.degrees(maximum_rotation):.6f} deg"
            )

        aggregation = aggregate_five_frame_depth_mask(depth_frames, mask_frames)
        reference_index = AGGREGATE_CAPTURE_COUNT // 2
        reference = members[reference_index]
        reference_pose = member_poses[reference_index]
        digest = five_frame_identity_sha256(raw_scenes, ids, stamps_ns)
        signature_json = json.dumps(
            reference_signature, separators=(",", ":"), ensure_ascii=True
        )
        payload_identity = hashlib.sha256()
        payload_identity.update(digest.encode("ascii"))
        payload_identity.update(signature_json.encode("utf-8"))
        payload_identity.update(np.asarray(reference_pose, dtype="<f8").tobytes())
        payload_identity.update(
            np.asarray(aggregation.depth_m, dtype="<f4").tobytes()
        )
        payload_identity.update(
            np.asarray(aggregation.mask, dtype=np.uint8).tobytes()
        )
        payload_digest = payload_identity.hexdigest()
        aggregate = copy.deepcopy(reference)
        aggregate.scene_id = aggregate_scene_id
        aggregate.observation_id = f"agg5_{digest[:24]}"
        aggregate.source_type = Observation.SOURCE_REAL
        aggregate.source_name = (
            f"{reference.source_name}|fixed5_median_majority|{digest[:12]}"
        )
        aggregate.depth.encoding = "32FC1"
        aggregate.depth.is_bigendian = 0
        aggregate.depth.height, aggregate.depth.width = aggregation.depth_m.shape
        aggregate.depth.step = int(aggregate.depth.width) * 4
        aggregate.depth.data = np.asarray(
            aggregation.depth_m, dtype="<f4"
        ).tobytes()
        aggregate.target_mask.header = copy.deepcopy(reference.color.header)
        aggregate.target_mask.encoding = "mono8"
        aggregate.target_mask.is_bigendian = 0
        aggregate.target_mask.height, aggregate.target_mask.width = (
            aggregation.mask.shape
        )
        aggregate.target_mask.step = int(aggregate.target_mask.width)
        aggregate.target_mask.data = np.asarray(
            aggregation.mask, dtype=np.uint8
        ).tobytes()
        aggregate.valid_depth_fraction = float(
            np.count_nonzero(np.isfinite(aggregation.depth_m))
            / aggregation.depth_m.size
        )
        _fill_pose_stamped(
            aggregate.camera_pose,
            reference_pose,
            self.base_frame,
            aggregate.header.stamp,
        )
        aggregate.pose_valid = True
        self._validate_real_observation(aggregate)

        finite_support = aggregation.finite_support
        mask_votes = aggregation.mask_votes
        target_stack = np.stack(member_target_centres, axis=0)
        target_median = np.median(target_stack, axis=0)
        target_deviations = np.linalg.norm(target_stack - target_median, axis=1)
        audit = {
            "contract": "fixed five-frame finite-median depth and 3-of-5 mask",
            "selection_performed": False,
            "capture_count": AGGREGATE_CAPTURE_COUNT,
            "raw_scenes": raw_scenes,
            "member_observation_ids": ids,
            "member_stamps_ns": stamps_ns,
            "member_captures": member_captures,
            "member_exact_tfs": member_tf_audits,
            "member_targets": member_target_audits,
            "member_target_centres_m": [
                array_list(value) for value in member_target_centres
            ],
            "member_target_median_m": array_list(target_median),
            "member_target_max_deviation_m": float(np.max(target_deviations)),
            "camera_info_signature_sha256": hashlib.sha256(
                signature_json.encode("utf-8")
            ).hexdigest(),
            "camera_info_exactly_equal": True,
            "pose_span": {
                "max_translation_m": maximum_translation,
                "max_rotation_deg": math.degrees(maximum_rotation),
                "max_translation_pair_zero_based": list(
                    maximum_translation_pair
                ),
                "max_rotation_pair_zero_based": list(maximum_rotation_pair),
                "translation_limit_m": translation_limit,
                "rotation_limit_deg": math.degrees(rotation_limit),
            },
            "reference_member_index_zero_based": reference_index,
            "reference_observation_id": ids[reference_index],
            "aggregate_scene_id": aggregate.scene_id,
            "aggregate_observation_id": aggregate.observation_id,
            "aggregate_identity_contract": (
                "ordered compact sort_keys canonical JSON of five "
                "scene_id/observation_id/stamp_ns records"
            ),
            "aggregate_identity_sha256": digest,
            "aggregate_payload_sha256": payload_digest,
            "aggregate_stamp_ns": stamp_nanoseconds(aggregate.header.stamp),
            "aggregate_source_name": aggregate.source_name,
            "aggregate_counts": aggregation.audit_counts,
            "depth_support": {
                "minimum_finite_votes": AGGREGATE_MIN_FINITE_COUNT,
                "valid_output_pixels": int(
                    np.count_nonzero(
                        finite_support >= AGGREGATE_MIN_FINITE_COUNT
                    )
                ),
                "insufficient_output_pixels": int(
                    np.count_nonzero(
                        finite_support < AGGREGATE_MIN_FINITE_COUNT
                    )
                ),
                "support_histogram_0_to_5": [
                    int(np.count_nonzero(finite_support == value))
                    for value in range(6)
                ],
            },
            "mask_votes": {
                "majority_votes_required": AGGREGATE_MASK_MAJORITY_COUNT,
                "foreground_output_pixels": int(
                    np.count_nonzero(
                        mask_votes >= AGGREGATE_MASK_MAJORITY_COUNT
                    )
                ),
                "vote_histogram_0_to_5": [
                    int(np.count_nonzero(mask_votes == value))
                    for value in range(6)
                ],
            },
            "aggregate_valid_depth_fraction": float(
                aggregate.valid_depth_fraction
            ),
        }
        return aggregate, reference_pose, audit

    def _validate_real_observation(self, observation: Observation) -> None:
        if wire_uint8(
            observation.source_type, "Observation.source_type"
        ) != Observation.SOURCE_REAL:
            raise SupervisorError("captured Observation source_type is not SOURCE_REAL")
        if not str(observation.source_name).strip():
            raise SupervisorError("captured Observation source_name is empty")
        if str(observation.header.frame_id) != self.camera_frame:
            raise SupervisorError("captured Observation optical frame is unexpected")
        stamp_ns = stamp_nanoseconds(observation.header.stamp, "Observation stamp")
        for label, field in (
            ("depth", observation.depth),
            ("camera_info", observation.camera_info),
        ):
            if str(field.header.frame_id) != self.camera_frame:
                raise SupervisorError(f"{label} frame does not match optical frame")
            if stamp_nanoseconds(field.header.stamp, f"{label} stamp") != stamp_ns:
                raise SupervisorError(f"{label} stamp does not match exposure stamp")
        for label, field in (
            ("color", observation.color),
            ("target_mask", observation.target_mask),
        ):
            if str(field.header.frame_id) != self.camera_frame:
                raise SupervisorError(f"{label} frame does not match optical frame")
            skew = abs(
                stamp_nanoseconds(field.header.stamp, f"{label} stamp") - stamp_ns
            ) / 1.0e9
            if skew > 0.005 + 1.0e-9:
                raise SupervisorError(f"{label} exposure skew exceeds 5 ms")
        fraction = float(observation.valid_depth_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise SupervisorError("Observation valid_depth_fraction is invalid")

    def _exact_base_camera(
        self, observation: Observation
    ) -> tuple[np.ndarray, dict[str, Any]]:
        requested_ns = stamp_nanoseconds(observation.header.stamp, "exposure stamp")
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame,
                self.link_frame,
                Time.from_msg(observation.header.stamp),
                timeout=Duration(seconds=self.tf_timeout),
            )
        except TransformException as error:
            raise SupervisorError(
                f"exact exposure-time TF {self.base_frame}<-{self.link_frame} "
                f"is unavailable: {error}"
            ) from error
        returned_ns = stamp_nanoseconds(transform.header.stamp, "returned TF stamp")
        if returned_ns != requested_ns:
            raise SupervisorError(
                "TF lookup did not return the exact exposure timestamp: "
                f"requested={requested_ns}, returned={returned_ns}"
            )
        base_link7 = _strict_transform_matrix(transform, "T_base_link7")
        base_camera = base_link7 @ self.report.transform_link7_camera_optical
        # matrix_to_pose_components performs a second rigid-transform check.
        matrix_to_pose_components(base_camera)
        return base_camera, {
            "requested_stamp_ns": requested_ns,
            "returned_stamp_ns": returned_ns,
            "stamp_difference_ns": returned_ns - requested_ns,
            "T_base_link7": array_list(base_link7),
            "T_link7_camera_optical": array_list(
                self.report.transform_link7_camera_optical
            ),
            "T_base_camera_optical": array_list(base_camera),
        }

    def _estimate_target(
        self,
        observation: Observation,
        base_camera: np.ndarray,
        *,
        configuration: dict[str, Any] | None = None,
        minimum_pixels: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        depth = decode_depth_32fc1(observation.depth)
        mask = decode_mask_mono8(observation.target_mask)
        intrinsic = camera_matrix(observation.camera_info, depth.shape)
        if mask.shape != depth.shape:
            raise SupervisorError("target mask and depth dimensions differ")
        if configuration is None:
            depth_min_m = float(self.get_parameter("depth_min_m").value)
            depth_max_m = float(self.get_parameter("depth_max_m").value)
        else:
            depth_min_m = float(configuration["depth_min_m"])
            depth_max_m = float(configuration["depth_max_m"])
        estimate = estimate_target_center(
            depth,
            mask,
            intrinsic,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            minimum_pixels=(
                int(self.get_parameter("minimum_target_pixels").value)
                if minimum_pixels is None
                else int(minimum_pixels)
            ),
        )
        centre_base = transform_point(base_camera, estimate.camera_xyz_m)
        audit = _target_audit(estimate, centre_base)
        audit["depth_range_m"] = [depth_min_m, depth_max_m]
        audit["depth_range_source"] = (
            "runtime_preview_parameters"
            if configuration is None
            else "sha_frozen_ConfigureNBV"
        )
        return centre_base, audit

    def _configure(
        self,
        target_center: np.ndarray,
        current_camera: np.ndarray,
        *,
        frozen_configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Configure a fresh map from preview values or exact frozen semantics."""
        if self._configure_call_count != 0:
            raise SupervisorError(
                "ConfigureNBV may be called only once per persistent-map session"
            )
        if not self._configure_client.wait_for_service(timeout_sec=5.0):
            raise SupervisorError("ConfigureNBV service is unavailable")
        if frozen_configuration is None:
            map_size = float(self.get_parameter("map_size_m").value)
            roi_size = float(self.get_parameter("target_roi_size_m").value)
            half_extent = float(
                self.get_parameter("observation_bound_half_extent_m").value
            )
            current_position = current_camera[:3, 3]
            values = {
                "scene_id": self.scene_id,
                "world_frame": self.base_frame,
                "target_center_m": array_list(target_center),
                "map_size_m": [map_size] * 3,
                "target_roi_size_m": [roi_size] * 3,
                "observation_min_m": array_list(
                    current_position - half_extent
                ),
                "observation_max_m": array_list(
                    current_position + half_extent
                ),
                "voxel_size_m": float(
                    self.get_parameter("voxel_size_m").value
                ),
                "depth_min_m": float(self.get_parameter("depth_min_m").value),
                "depth_max_m": float(self.get_parameter("depth_max_m").value),
                "samples_per_ray": 128,
                "optimization_steps": 10,
                "max_step_m": float(self.get_parameter("max_step_m").value),
                "random_seed": 0,
            }
            configuration_source = "preview_from_aggregate"
        else:
            values = copy.deepcopy(frozen_configuration)
            configuration_source = "exact_sha_frozen_v3"
        try:
            evidence = normalize_nbv_configuration(
                values,
                max_step_ceiling_m=(
                    self.motion_limits.maximum_camera_step_m
                ),
            )
        except (TypeError, ValueError) as error:
            raise SupervisorError(
                f"ConfigureNBV request semantics are invalid: {error}"
            ) from error
        configuration = evidence.request
        if frozen_configuration is not None and configuration != frozen_configuration:
            raise SupervisorError("frozen ConfigureNBV semantics changed on normalization")
        if (
            configuration["scene_id"] != self.scene_id
            or configuration["world_frame"] != self.base_frame
        ):
            raise SupervisorError("ConfigureNBV frozen scene/world does not match session")
        current_position = current_camera[:3, 3]
        if np.any(
            current_position
            < np.asarray(configuration["observation_min_m"]) - 1.0e-9
        ) or np.any(
            current_position
            > np.asarray(configuration["observation_max_m"]) + 1.0e-9
        ):
            raise SupervisorError(
                "fresh aggregate camera lies outside ConfigureNBV observation bounds"
            )

        request = ConfigureNBV.Request()
        request.scene_id = configuration["scene_id"]
        request.world_frame = configuration["world_frame"]
        request.target_center.x, request.target_center.y, request.target_center.z = (
            configuration["target_center_m"]
        )
        request.map_size.x, request.map_size.y, request.map_size.z = configuration[
            "map_size_m"
        ]
        (
            request.target_roi_size.x,
            request.target_roi_size.y,
            request.target_roi_size.z,
        ) = configuration["target_roi_size_m"]
        (
            request.observation_min.x,
            request.observation_min.y,
            request.observation_min.z,
        ) = configuration["observation_min_m"]
        (
            request.observation_max.x,
            request.observation_max.y,
            request.observation_max.z,
        ) = configuration["observation_max_m"]
        request.voxel_size = configuration["voxel_size_m"]
        request.depth_min = configuration["depth_min_m"]
        request.depth_max = configuration["depth_max_m"]
        request.samples_per_ray = configuration["samples_per_ray"]
        request.optimization_steps = configuration["optimization_steps"]
        request.max_step = configuration["max_step_m"]
        request.random_seed = configuration["random_seed"]
        response = self._wait_future(
            self._configure_client.call_async(request),
            self.operation_timeout,
            "ConfigureNBV response",
        )
        if not response.success or int(response.code) != ConfigureNBV.Response.SUCCESS:
            raise SupervisorError(
                f"ConfigureNBV rejected code={response.code}: {response.reason}"
            )
        self._active_nbv_configuration = copy.deepcopy(configuration)
        self._configure_call_count += 1
        return {
            "request_semantics": copy.deepcopy(configuration),
            "request_sha256": evidence.sha256,
            "derived_voxel_grid": {
                "dimensions": list(evidence.voxel_dimensions),
                "origin_m": array_list(evidence.map_origin_m),
            },
            "configuration_source": configuration_source,
            "response_code": int(response.code),
            "response_reason": str(response.reason),
            "configure_call_count": self._configure_call_count,
            "ordering_note": "configured before final capture; configure clears NBV cache",
        }

    def _correct_observation(
        self, observation: Observation, base_camera: np.ndarray
    ) -> Observation:
        corrected = copy.deepcopy(observation)
        corrected.source_type = Observation.SOURCE_REAL
        _fill_pose_stamped(
            corrected.camera_pose,
            base_camera,
            self.base_frame,
            corrected.header.stamp,
        )
        corrected.pose_valid = True
        return corrected

    def _publish_and_compute(
        self,
        corrected: Observation,
        *,
        require_strict_improvement_proxy: bool = True,
    ) -> tuple[NextView, dict[str, Any]]:
        if self._active_nbv_configuration is None:
            raise SupervisorError("Gradient-NBV map has not been configured")
        # A supervisor is short-lived and creates this publisher immediately
        # before sending a large Observation.  DDS discovery can take longer
        # than construction of the message, especially after several rapid
        # preview runs.  Publishing before the transient-local subscriber is
        # matched makes wait_for_all_acked fail nondeterministically even
        # though the Gradient-NBV action server is healthy.
        self._spin_until(
            lambda: self._corrected_publisher.get_subscription_count() >= 1,
            5.0,
            "a matched Gradient-NBV Observation subscriber",
        )
        self._corrected_publisher.publish(corrected)
        acknowledged = self._corrected_publisher.wait_for_all_acked(
            Duration(seconds=5.0)
        )
        if not acknowledged:
            raise SupervisorError(
                "corrected Observation was not acknowledged on the dedicated topic"
            )
        if not self._compute_client.wait_for_server(timeout_sec=5.0):
            raise SupervisorError("ComputeNextView action is unavailable")
        goal = ComputeNextView.Goal()
        goal.scene_id = corrected.scene_id
        goal.observation_id = corrected.observation_id
        goal_handle = self._wait_future(
            self._compute_client.send_goal_async(goal),
            self.operation_timeout,
            "ComputeNextView goal acceptance",
        )
        if not goal_handle.accepted:
            raise SupervisorError("ComputeNextView goal was rejected")
        wrapped = self._wait_future(
            goal_handle.get_result_async(),
            self.operation_timeout,
            "ComputeNextView result",
        )
        if int(wrapped.status) != GoalStatus.STATUS_SUCCEEDED:
            raise SupervisorError(
                f"ComputeNextView action status is {int(wrapped.status)}, not SUCCEEDED"
            )
        next_view = wrapped.result.next_view
        if (
            str(next_view.scene_id) != corrected.scene_id
            or str(next_view.observation_id) != corrected.observation_id
        ):
            raise SupervisorError("NextView identifiers do not bind to final Observation")
        if not next_view.success or int(next_view.code) != NextView.SUCCESS:
            raise SupervisorError(
                f"NextView failed code={next_view.code}: {next_view.reason}"
            )
        if str(next_view.pose.header.frame_id) != self.base_frame:
            raise SupervisorError("NextView pose frame is not base_link")
        if stamp_nanoseconds(next_view.pose.header.stamp, "NextView pose stamp") != (
            stamp_nanoseconds(corrected.header.stamp, "corrected Observation stamp")
        ):
            raise SupervisorError("NextView pose stamp does not match Observation")
        numeric = np.asarray(
            (
                next_view.gain,
                next_view.coverage,
                next_view.compute_time_ms,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(numeric)) or not 0.0 <= numeric[1] <= 1.0:
            raise SupervisorError("NextView numerical diagnostics are invalid")
        target = _strict_pose_matrix(next_view.pose, "NextView pose")
        current = _strict_pose_matrix(corrected.camera_pose, "Observation pose")
        original_target = target.copy()
        target, projected_overshoot = project_numerical_step_overshoot(
            current,
            target,
            max_step_m=float(self._active_nbv_configuration["max_step_m"]),
        )
        if projected_overshoot > 0.0:
            _fill_pose_stamped(
                next_view.pose, target, self.base_frame, corrected.header.stamp
            )
        if require_strict_improvement_proxy:
            raw_motion = validate_raw_next_view(
                current,
                target,
                float(next_view.gain),
                max_step_m=float(
                    self._active_nbv_configuration["max_step_m"]
                ),
            )
            improvement_kind = "core_invariant_inference"
        else:
            if not math.isfinite(float(next_view.gain)):
                raise SupervisorError("post-motion NextView gain is not finite")
            raw_motion = camera_motion(current, target)
            improvement_kind = "post_update_diagnostic_only"
        return next_view, {
            "action_status": int(wrapped.status),
            "scene_id": str(next_view.scene_id),
            "observation_id": str(next_view.observation_id),
            "success": bool(next_view.success),
            "code": int(next_view.code),
            "reason": str(next_view.reason),
            "T_base_camera_raw": array_list(target),
            "T_base_camera_unprojected": array_list(original_target),
            "translation_projection_correction_m": projected_overshoot,
            "translation_m": raw_motion.translation_m,
            "rotation_deg": math.degrees(raw_motion.rotation_rad),
            "planned_gain": float(next_view.gain),
            "coverage": float(next_view.coverage),
            "voxel_counts": {
                "total": int(next_view.total_voxel_count),
                "observed": int(next_view.observed_voxel_count),
                "occupied": int(next_view.occupied_voxel_count),
                "unknown": int(next_view.unknown_voxel_count),
            },
            "optimization_iterations": int(next_view.optimization_iterations),
            "compute_time_ms": float(next_view.compute_time_ms),
            "gain_improvement_evidence": {
                "kind": improvement_kind,
                "numerical_gain_difference_available": False,
                "conditions": [
                    "NextView planned_gain is finite and positive",
                    "raw camera translation exceeds 1 mm deadband",
                    "GradientNBVCore changes position only for strict gain improvement",
                ],
                "note": (
                    "The public v1 NextView interface exposes planned gain but not "
                    "current gain; this is not a measured numerical difference."
                ),
            },
        }

    def _fresh_joints(self) -> np.ndarray:
        requested = time.monotonic()
        self._spin_until(
            lambda: self._joint_received >= requested,
            3.0,
            "a fresh complete joint1..joint7 sample",
        )
        assert self._joint_positions is not None
        return self._joint_positions.copy()

    def _solve_candidates(
        self,
        current_camera: np.ndarray,
        raw_target: np.ndarray,
        stamp: Any,
        observation_id: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if self.motion_profile == LARGE_MOTION_PROFILE:
            return self._solve_reachable_view_lattice(
                current_camera, raw_target, stamp, observation_id
            )
        if not self._solve_client.wait_for_service(timeout_sec=5.0):
            raise SupervisorError("SolveIK service is unavailable")
        alphas = tuple(
            float(value) for value in self.get_parameter("candidate_alphas").value
        )
        candidates = independently_segmented_camera_candidates(
            current_camera, raw_target, alphas
        )
        max_rotation = math.radians(
            float(self.get_parameter("max_selected_camera_rotation_deg").value)
        )
        max_joint_delta = float(
            self.get_parameter("max_ik_joint_delta_rad").value
        )
        if self._active_nbv_configuration is None:
            raise SupervisorError("Gradient-NBV map has not been configured")
        observation_min = np.asarray(
            self._active_nbv_configuration["observation_min_m"], dtype=float
        )
        observation_max = np.asarray(
            self._active_nbv_configuration["observation_max_m"], dtype=float
        )
        records: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for translation_alpha, rotation_alpha, camera_target, motion in candidates:
            link7_target = camera_target_to_link7(
                camera_target, self.report.transform_link7_camera_optical
            )
            request = SolveIK.Request()
            _fill_pose_stamped(
                request.target_pose, link7_target, self.base_frame, stamp
            )
            request.controlled_frame = "link7"
            current_joints = self._fresh_joints()
            response = self._wait_future(
                self._solve_client.call_async(request),
                self.operation_timeout,
                (
                    "SolveIK translation_alpha="
                    f"{translation_alpha:g}, rotation_alpha={rotation_alpha:g}"
                ),
            )
            result = response.result
            record: dict[str, Any] = {
                # ``alpha`` stays the translation factor for v3 compatibility.
                "alpha": translation_alpha,
                "translation_alpha": translation_alpha,
                "rotation_alpha": rotation_alpha,
                "segmentation_contract": "independent_translation_and_rotation",
                "T_base_camera": array_list(camera_target),
                "T_base_link7": array_list(link7_target),
                "camera_translation_m": motion.translation_m,
                "camera_rotation_deg": math.degrees(motion.rotation_rad),
                "ik_success": bool(result.success),
                "ik_code": int(result.code),
                "ik_reason": str(result.reason),
                "position_error_m": float(result.position_error_m),
                "orientation_error_rad": float(result.orientation_error_rad),
                "sigma_min": float(result.sigma_min),
                "condition_number": float(result.condition_number),
                "reported_max_joint_delta_rad": float(result.max_joint_delta_rad),
                "solution_names": list(result.solution_joint_state.name),
                "solution_positions_rad": [
                    float(value) for value in result.solution_joint_state.position
                ],
                "accepted": False,
                "rejection": "",
            }
            try:
                if not result.success or int(result.code) != IKResult.SUCCESS:
                    raise ValueError("SolveIK did not return exact SUCCESS")
                if not camera_step_above_minimum(
                    motion.translation_m, self.motion_limits
                ):
                    raise ValueError(
                        "candidate translation is below the selected motion "
                        "profile minimum"
                    )
                if (
                    motion.translation_m
                    > self.motion_limits.maximum_camera_step_m + 1.0e-9
                ):
                    raise ValueError(
                        "candidate translation exceeds the selected motion "
                        "profile maximum"
                    )
                if motion.rotation_rad > max_rotation + 1.0e-9:
                    raise ValueError(
                        "candidate single-step camera rotation exceeds "
                        f"{math.degrees(max_rotation):.3f} degrees"
                    )
                if np.any(camera_target[:3, 3] < observation_min - 1.0e-9) or np.any(
                    camera_target[:3, 3] > observation_max + 1.0e-9
                ):
                    raise ValueError("candidate lies outside configured observation bounds")
                validation = validate_ik_solution(
                    current_joints=current_joints,
                    solution_names=result.solution_joint_state.name,
                    solution_positions=result.solution_joint_state.position,
                    reported_max_joint_delta_rad=result.max_joint_delta_rad,
                    position_error_m=result.position_error_m,
                    orientation_error_rad=result.orientation_error_rad,
                    sigma_min=result.sigma_min,
                    condition_number=result.condition_number,
                    max_joint_delta_rad=max_joint_delta,
                    max_position_error_m=(
                        self.motion_limits.maximum_ik_position_error_m
                    ),
                )
                record["independent_max_joint_delta_rad"] = (
                    validation.max_joint_delta_rad
                )
                record["minimum_joint_limit_clearance_rad"] = (
                    validation.minimum_joint_limit_clearance_rad
                )
                record["limiting_joint_name"] = validation.limiting_joint_name
                record["accepted"] = True
                if selected is None:
                    selected = record
            except ValueError as error:
                record["rejection"] = str(error)
            records.append(record)
        return selected, records

    def _solve_reachable_view_lattice(
        self,
        current_camera: np.ndarray,
        raw_target: np.ndarray,
        stamp: Any,
        observation_id: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """IK-filter a finite view lattice, then select by map information."""
        if self._active_nbv_configuration is None:
            raise SupervisorError("Gradient-NBV map has not been configured")
        if not self._solve_client.wait_for_service(timeout_sec=5.0):
            raise SupervisorError("SolveIK service is unavailable")
        configuration = self._active_nbv_configuration
        lattice = reachable_view_lattice(
            current_camera,
            raw_target,
            configuration["target_center_m"],
            configuration["observation_min_m"],
            configuration["observation_max_m"],
        )
        max_rotation = math.radians(
            float(self.get_parameter("max_selected_camera_rotation_deg").value)
        )
        max_joint_delta = float(
            self.get_parameter("max_ik_joint_delta_rad").value
        )
        records: list[dict[str, Any]] = []
        accepted_records: list[dict[str, Any]] = []
        for candidate in lattice:
            link7_target = camera_target_to_link7(
                candidate.pose, self.report.transform_link7_camera_optical
            )
            request = SolveIK.Request()
            _fill_pose_stamped(
                request.target_pose, link7_target, self.base_frame, stamp
            )
            request.controlled_frame = "link7"
            current_joints = self._fresh_joints()
            response = self._wait_future(
                self._solve_client.call_async(request),
                self.operation_timeout,
                f"SolveIK reachable lattice candidate {candidate.candidate_index}",
            )
            result = response.result
            record: dict[str, Any] = {
                "candidate_index": candidate.candidate_index,
                "direction_index": candidate.direction_index,
                "direction_world": array_list(candidate.direction_world),
                "radius_m": candidate.radius_m,
                "alpha": 1.0,
                "translation_alpha": 1.0,
                "rotation_alpha": 1.0,
                "segmentation_contract": "reachable_view_lattice",
                "T_base_camera": array_list(candidate.pose),
                "T_base_link7": array_list(link7_target),
                "camera_translation_m": candidate.motion.translation_m,
                "camera_rotation_deg": math.degrees(candidate.motion.rotation_rad),
                "ik_success": bool(result.success),
                "ik_code": int(result.code),
                "ik_reason": str(result.reason),
                "position_error_m": float(result.position_error_m),
                "orientation_error_rad": float(result.orientation_error_rad),
                "sigma_min": float(result.sigma_min),
                "condition_number": float(result.condition_number),
                "reported_max_joint_delta_rad": float(result.max_joint_delta_rad),
                "solution_names": list(result.solution_joint_state.name),
                "solution_positions_rad": [
                    float(value) for value in result.solution_joint_state.position
                ],
                "accepted": False,
                "rejection": "",
                "ik_gate_passed": False,
                "gain_scored": False,
            }
            try:
                if not result.success or int(result.code) != IKResult.SUCCESS:
                    raise ValueError("SolveIK did not return exact SUCCESS")
                if not camera_step_above_minimum(
                    candidate.motion.translation_m, self.motion_limits
                ):
                    raise ValueError("candidate translation is inside motion deadband")
                if (
                    candidate.motion.translation_m
                    > self.motion_limits.maximum_camera_step_m + 1.0e-9
                ):
                    raise ValueError("candidate translation exceeds profile maximum")
                if candidate.motion.rotation_rad > max_rotation + 1.0e-9:
                    raise ValueError("candidate camera rotation exceeds profile maximum")
                validation = validate_ik_solution(
                    current_joints=current_joints,
                    solution_names=result.solution_joint_state.name,
                    solution_positions=result.solution_joint_state.position,
                    reported_max_joint_delta_rad=result.max_joint_delta_rad,
                    position_error_m=result.position_error_m,
                    orientation_error_rad=result.orientation_error_rad,
                    sigma_min=result.sigma_min,
                    condition_number=result.condition_number,
                    max_joint_delta_rad=max_joint_delta,
                    max_position_error_m=(
                        self.motion_limits.maximum_ik_position_error_m
                    ),
                )
                record["independent_max_joint_delta_rad"] = (
                    validation.max_joint_delta_rad
                )
                record["minimum_joint_limit_clearance_rad"] = (
                    validation.minimum_joint_limit_clearance_rad
                )
                record["limiting_joint_name"] = validation.limiting_joint_name
                record["ik_gate_passed"] = True
                accepted_records.append(record)
            except ValueError as error:
                record["rejection"] = str(error)
            records.append(record)

        if not accepted_records:
            return None, records
        if not self._evaluate_candidates_client.wait_for_service(timeout_sec=5.0):
            raise SupervisorError("EvaluateViewCandidates service is unavailable")
        score_request = EvaluateViewCandidates.Request()
        score_request.scene_id = self.scene_id
        score_request.observation_id = str(observation_id)
        for record in accepted_records:
            pose = PoseStamped()
            _fill_pose_stamped(
                pose,
                np.asarray(record["T_base_camera"], dtype=float),
                self.base_frame,
                stamp,
            )
            score_request.candidate_poses.append(pose)
        score_response = self._wait_future(
            self._evaluate_candidates_client.call_async(score_request),
            max(self.operation_timeout, 60.0),
            "read-only reachable candidate gain scoring",
        )
        if not score_response.success or int(score_response.code) != 0:
            raise SupervisorError(
                "EvaluateViewCandidates failed: "
                f"code={score_response.code}, reason={score_response.reason}"
            )
        gains = [float(value) for value in score_response.candidate_gains]
        current_gain = float(score_response.current_gain)
        if len(gains) != len(accepted_records) or not math.isfinite(current_gain):
            raise SupervisorError("candidate gain response has invalid dimensions")
        translations = [
            float(record["camera_translation_m"]) for record in accepted_records
        ]
        clearances = [
            float(record["minimum_joint_limit_clearance_rad"])
            for record in accepted_records
        ]
        selection = select_near_best_reachable_candidate(
            current_gain=current_gain,
            candidate_gains=gains,
            candidate_translations_m=translations,
            joint_limit_clearances_rad=clearances,
        )
        tolerance = max(1.0e-9, abs(current_gain) * 1.0e-6)
        for record, gain in zip(accepted_records, gains):
            if not math.isfinite(gain):
                raise SupervisorError("candidate gain response contains non-finite data")
            improvement = gain - current_gain
            record["gain_scored"] = True
            record["current_gain"] = current_gain
            record["candidate_gain"] = gain
            record["gain_improvement"] = improvement
            if improvement <= tolerance:
                record["rejection"] = "candidate does not improve map information gain"
        if selection is None:
            return None, records
        selected = accepted_records[selection.selected_index]
        selected["accepted"] = True
        selected["selection_contract"] = (
            "IK-safe candidates with positive gain; within 90% of the best "
            "gain improvement, choose the largest camera translation"
        )
        selected["best_gain_improvement"] = selection.best_gain_improvement
        selected["near_best_gain_floor"] = selection.near_best_gain_floor
        selected["near_best_candidate_count"] = len(selection.near_best_indices)
        selected["reachable_candidate_count"] = len(accepted_records)
        selected["useful_candidate_count"] = len(selection.useful_indices)
        return selected, records

    def _validate_controller_parameters(self) -> dict[str, Any]:
        service_name = str(
            self.get_parameter("controller_parameters_service").value
        )
        client = self.create_client(GetParameters, service_name)
        if not client.wait_for_service(timeout_sec=5.0):
            raise SupervisorError("nero_control parameter service is unavailable")
        names = [
            "simulation_mode",
            "first_motion_test_mode",
            "precision_test_mode",
            "precision_max_joint_delta_rad",
            "ik_position_tolerance_m",
            "precision_max_ik_position_error_m",
            "precision_final_position_tolerance_m",
            "verified_driver_speed_percent",
            "execution_enabled_on_start",
        ]
        request = GetParameters.Request()
        request.names = names
        response = self._wait_future(
            client.call_async(request), 5.0, "nero_control safety parameters"
        )
        if len(response.values) != len(names):
            raise SupervisorError("nero_control safety parameter response is incomplete")
        values = dict(zip(names, response.values))
        for name in (
            "simulation_mode",
            "first_motion_test_mode",
            "precision_test_mode",
            "execution_enabled_on_start",
        ):
            if values[name].type != ParameterType.PARAMETER_BOOL:
                raise SupervisorError(f"nero_control parameter {name} is not boolean")
        speed = values["verified_driver_speed_percent"]
        if speed.type != ParameterType.PARAMETER_INTEGER:
            raise SupervisorError("verified_driver_speed_percent is not integer")
        precision_joint_delta = values["precision_max_joint_delta_rad"]
        if precision_joint_delta.type != ParameterType.PARAMETER_DOUBLE:
            raise SupervisorError("precision_max_joint_delta_rad is not a double")
        for name in (
            "ik_position_tolerance_m",
            "precision_max_ik_position_error_m",
            "precision_final_position_tolerance_m",
        ):
            if values[name].type != ParameterType.PARAMETER_DOUBLE:
                raise SupervisorError(f"nero_control parameter {name} is not a double")
        decoded = {
            "simulation_mode": bool(values["simulation_mode"].bool_value),
            "first_motion_test_mode": bool(
                values["first_motion_test_mode"].bool_value
            ),
            "precision_test_mode": bool(values["precision_test_mode"].bool_value),
            "precision_max_joint_delta_rad": float(
                precision_joint_delta.double_value
            ),
            "ik_position_tolerance_m": float(
                values["ik_position_tolerance_m"].double_value
            ),
            "precision_max_ik_position_error_m": float(
                values["precision_max_ik_position_error_m"].double_value
            ),
            "precision_final_position_tolerance_m": float(
                values["precision_final_position_tolerance_m"].double_value
            ),
            "verified_driver_speed_percent": int(speed.integer_value),
            "execution_enabled_on_start": bool(
                values["execution_enabled_on_start"].bool_value
            ),
        }
        return _validate_controller_parameter_values(decoded, self.motion_limits)

    def _validate_command_publishers(self) -> dict[str, Any]:
        command_topic = str(self.get_parameter("motion_command_topic").value)
        publishers = self.get_publishers_info_by_topic(command_topic)
        if len(publishers) != 1:
            raise SupervisorError(
                f"{command_topic} must have exactly one publisher; got {len(publishers)}"
            )
        publisher = publishers[0]
        if publisher.node_name != "nero_control" or publisher.node_namespace != "/":
            raise SupervisorError(
                "the sole move_j publisher must be exactly /nero_control"
            )
        forbidden_counts = {
            topic: len(self.get_publishers_info_by_topic(topic))
            for topic in FORBIDDEN_COMMAND_TOPICS
        }
        active = {topic: count for topic, count in forbidden_counts.items() if count}
        if active:
            raise SupervisorError(f"forbidden vendor command publishers exist: {active}")
        return {
            "command_topic": command_topic,
            "publisher": "/nero_control",
            "publisher_count": 1,
            "forbidden_publisher_counts": forbidden_counts,
        }

    def _wait_controller_closed(self, after: float) -> dict[str, Any]:
        self._spin_until(
            lambda: self._controller_diagnostic_received >= after,
            3.0,
            "a fresh controller diagnostic after gate closure",
        )
        diagnostic = copy.deepcopy(self._controller_diagnostic)
        if diagnostic is None:
            raise SupervisorError("controller diagnostic is unavailable")
        values = diagnostic["values"]
        expected = {
            "mode": "real",
            "execution_enabled": False,
            "precision_test_mode": True,
            "feedback_fresh": True,
        }
        for key, value in expected.items():
            if values.get(key) != value:
                raise SupervisorError(
                    f"controller diagnostic {key}={values.get(key)!r}; expected {value!r}"
                )
        # This controller explicitly reports that environment collision
        # checking is absent.  Execution relies on the operator clearance lock.
        if values.get("environment_collision_checking") is not False:
            raise SupervisorError("unexpected environment-collision diagnostic value")
        return diagnostic

    def _fresh_arm_health(self) -> dict[str, Any]:
        requested = time.monotonic()
        self._spin_until(
            lambda: self._arm_status_received >= requested,
            3.0,
            "fresh arm health",
        )
        status = self._arm_status
        if status is None:
            raise SupervisorError("arm status is unavailable")
        decoded = {
            "ctrl_mode": wire_uint8(status.ctrl_mode, "ctrl_mode"),
            "arm_status": wire_uint8(status.arm_status, "arm_status"),
            "motion_status": wire_uint8(status.motion_status, "motion_status"),
            "err_status": int(status.err_status),
            "joint_angle_limit": [
                wire_bool(value, "joint_angle_limit")
                for value in status.joint_angle_limit
            ],
            "communication_status_joint": [
                wire_bool(value, "communication_status_joint")
                for value in status.communication_status_joint
            ],
        }
        if decoded["ctrl_mode"] != 1:
            raise SupervisorError("arm is not in CAN control mode")
        if decoded["arm_status"] != 0 or decoded["err_status"] != 0:
            raise SupervisorError(f"arm reports a fault: {decoded}")
        if decoded["motion_status"] != 0:
            raise SupervisorError("arm has not reported target-reached/stationary status")
        if len(decoded["joint_angle_limit"]) != 7 or any(
            decoded["joint_angle_limit"]
        ):
            raise SupervisorError("arm joint-limit flags are incomplete or asserted")
        if len(decoded["communication_status_joint"]) != 7 or any(
            decoded["communication_status_joint"]
        ):
            raise SupervisorError("arm communication flags are incomplete or asserted")
        return decoded

    def _wait_stationary(
        self,
        duration_sec: float = 0.5,
        *,
        command_count_must_remain: int | None = None,
    ) -> np.ndarray:
        deadline = time.monotonic() + 4.0
        stationary_since: float | None = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            now = time.monotonic()
            if (
                command_count_must_remain is not None
                and self._motion_command_count != command_count_must_remain
            ):
                raise SupervisorError(
                    "a new move_j command appeared after both gates were closed"
                )
            if (
                self._joint_positions is None
                or self._joint_velocities is None
                or now - self._joint_received > FEEDBACK_MAX_AGE_SEC
            ):
                stationary_since = None
                continue
            speed = float(np.max(np.abs(self._joint_velocities)))
            if speed > STATIONARY_SPEED_RAD_SEC:
                stationary_since = None
                continue
            if stationary_since is None:
                stationary_since = now
            if now - stationary_since >= duration_sec:
                return self._joint_positions.copy()
        raise SupervisorError("arm was not continuously stationary for 0.5 seconds")

    def _set_gate(
        self,
        client: Any,
        service_name: str,
        enabled: bool,
        records: list[dict[str, Any]],
    ) -> None:
        started = time.monotonic()
        record = {
            "service": service_name,
            "requested": bool(enabled),
            "started_monotonic_ns": time.monotonic_ns(),
            "success": False,
            "message": "",
        }
        records.append(record)
        if not client.wait_for_service(timeout_sec=3.0):
            record["message"] = "service unavailable"
            raise SupervisorError(f"gate service {service_name} is unavailable")
        request = SetBool.Request()
        request.data = bool(enabled)
        response = self._wait_future(
            client.call_async(request), 3.0, f"set {service_name}={enabled}"
        )
        record["completed_monotonic_ns"] = time.monotonic_ns()
        record["duration_sec"] = time.monotonic() - started
        record["success"] = bool(response.success)
        record["message"] = str(response.message)
        if not response.success:
            raise SupervisorError(
                f"gate {service_name} rejected {enabled}: {response.message}"
            )

    def _close_gates_strict(
        self,
        controller_client: Any,
        driver_client: Any,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failures: list[str] = []
        for _attempt in range(2):
            for client, name in (
                (
                    controller_client,
                    str(self.get_parameter("controller_gate_service").value),
                ),
                (driver_client, str(self.get_parameter("driver_gate_service").value)),
            ):
                try:
                    self._set_gate(client, name, False, records)
                except Exception as error:
                    failures.append(str(error))
        closed_at = time.monotonic()
        if failures:
            raise GateClosureError(
                "FAILED TO PROVE COMMAND GATES CLOSED; ONSITE OPERATOR MUST STOP "
                "THE SYSTEM AND CUT CONTROL-BOX POWER IF ANY MOTION CONTINUES: "
                + "; ".join(failures)
            )
        try:
            diagnostic = self._wait_controller_closed(closed_at)
            command_baseline = self._motion_command_count
            final_joints = self._wait_stationary(
                command_count_must_remain=command_baseline
            )
            arm_health = self._fresh_arm_health()
            if self._motion_command_count != command_baseline:
                raise SupervisorError(
                    "a new move_j command appeared during post-close arm-health check"
                )
        except Exception as error:
            raise GateClosureError(
                "GATE FALSE ACKS RECEIVED BUT CLOSED/STATIC STATE COULD NOT BE "
                "PROVED; ONSITE OPERATOR MUST STOP THE SYSTEM: " + str(error)
            ) from error
        return {
            "driver_false_ack_count": 2,
            "controller_false_ack_count": 2,
            "controller_diagnostic": diagnostic,
            "arm_health": arm_health,
            "post_close_command_count": command_baseline,
            "stationary_joint_positions_rad": array_list(final_joints),
        }

    def _solve_execution_target(
        self,
        target_link7: np.ndarray,
        stamp: Any,
    ) -> tuple[Any, np.ndarray, dict[str, Any]]:
        current_joints = self._fresh_joints()
        request = SolveIK.Request()
        _fill_pose_stamped(request.target_pose, target_link7, self.base_frame, stamp)
        request.controlled_frame = "link7"
        response = self._wait_future(
            self._solve_client.call_async(request),
            self.operation_timeout,
            "fresh execution SolveIK",
        )
        result = response.result
        if not result.success or int(result.code) != IKResult.SUCCESS:
            raise SupervisorError(
                f"fresh execution SolveIK failed code={result.code}: {result.reason}"
            )
        validation = validate_ik_solution(
            current_joints=current_joints,
            solution_names=result.solution_joint_state.name,
            solution_positions=result.solution_joint_state.position,
            reported_max_joint_delta_rad=result.max_joint_delta_rad,
            position_error_m=result.position_error_m,
            orientation_error_rad=result.orientation_error_rad,
            sigma_min=result.sigma_min,
            condition_number=result.condition_number,
            max_joint_delta_rad=float(
                self.get_parameter("max_ik_joint_delta_rad").value
            ),
            max_position_error_m=(
                self.motion_limits.maximum_ik_position_error_m
            ),
            reported_delta_tolerance_rad=0.002,
        )
        record = {
            "success": True,
            "code": int(result.code),
            "reason": str(result.reason),
            "position_error_m": validation.position_error_m,
            "orientation_error_rad": validation.orientation_error_rad,
            "sigma_min": validation.sigma_min,
            "condition_number": validation.condition_number,
            "reported_max_joint_delta_rad": validation.reported_max_joint_delta_rad,
            "independent_max_joint_delta_rad": validation.max_joint_delta_rad,
            "minimum_joint_limit_clearance_rad": (
                validation.minimum_joint_limit_clearance_rad
            ),
            "limiting_joint_name": validation.limiting_joint_name,
            "solution_names": list(result.solution_joint_state.name),
            "solution_positions_rad": [
                float(value) for value in result.solution_joint_state.position
            ],
            "start_joint_positions_rad": array_list(current_joints),
        }
        return result, current_joints, record

    def _execute_one_goal(
        self,
        target_link7: np.ndarray,
        target_stamp: Any,
        planned_start_joints: np.ndarray,
        *,
        step_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._motion_goal_count >= self.max_motion_steps:
            raise SupervisorError("session motion-goal limit is already exhausted")
        record_owner = self._audit if step_audit is None else step_audit
        driver_name = str(self.get_parameter("driver_gate_service").value)
        controller_name = str(
            self.get_parameter("controller_gate_service").value
        )
        driver_client = self.create_client(SetBool, driver_name)
        controller_client = self.create_client(SetBool, controller_name)
        records: list[dict[str, Any]] = []
        record_owner["gate_calls"] = records

        # Establish both gates closed and fresh static state before any open.
        self._gates_closed_proven = False
        record_owner["preopen_gate_closure"] = self._close_gates_strict(
            controller_client, driver_client, records
        )
        self._gates_closed_proven = True
        record_owner["controller_parameters"] = self._validate_controller_parameters()
        record_owner["command_publishers"] = self._validate_command_publishers()
        record_owner["preopen_arm_health"] = self._fresh_arm_health()
        preopen_joints = self._wait_stationary()
        start_drift = float(np.max(np.abs(preopen_joints - planned_start_joints)))
        record_owner["plan_start_joint_drift_rad"] = start_drift
        if start_drift > 0.002:
            raise SupervisorError(
                f"joint state drifted {start_drift:.6f} rad after SolveIK; replan"
            )
        # Re-check the unique publisher immediately before opening.
        self._validate_command_publishers()

        move_client = ActionClient(
            self,
            MoveToPose,
            str(self.get_parameter("move_action").value),
        )
        if not move_client.wait_for_server(timeout_sec=5.0):
            raise SupervisorError("MoveToPose action is unavailable")
        goal = MoveToPose.Goal()
        _fill_pose_stamped(goal.target_pose, target_link7, self.base_frame, target_stamp)
        goal.controlled_frame = "link7"
        goal.timeout.sec = 15
        action_error: Exception | None = None
        wrapped = None
        try:
            # Action-server discovery may have taken several seconds.  Refresh
            # every dynamic safety fact at the final possible instant before
            # the driver gate is opened.
            final_check_started = time.monotonic()
            final_publishers = self._validate_command_publishers()
            final_arm_health = self._fresh_arm_health()
            final_start_joints = self._wait_stationary()
            final_start_drift = float(
                np.max(np.abs(final_start_joints - planned_start_joints))
            )
            if final_start_drift > 0.002:
                raise SupervisorError(
                    "joint state drifted during final pre-gate checks; replan"
                )
            final_closed_diagnostic = self._wait_controller_closed(
                final_check_started
            )
            record_owner["final_check_immediately_before_driver_open"] = {
                "command_publishers": final_publishers,
                "arm_health": final_arm_health,
                "stationary_joint_positions_rad": array_list(final_start_joints),
                "plan_start_joint_drift_rad": final_start_drift,
                "controller_closed_diagnostic": final_closed_diagnostic,
                "completed_monotonic_ns": time.monotonic_ns(),
            }
            # From this point until strict closure succeeds, no audit may
            # claim that both gates are closed.
            self._gates_closed_proven = False
            self._set_gate(driver_client, driver_name, True, records)
            self._set_gate(controller_client, controller_name, True, records)
            if self._motion_goal_count == 0:
                self._consume_session_authorization()
            self._motion_goal_count += 1
            goal_handle = self._wait_future(
                move_client.send_goal_async(goal),
                5.0,
                "one session MoveToPose goal acceptance",
            )
            if not goal_handle.accepted:
                raise SupervisorError("MoveToPose goal was rejected")
            wrapped = self._wait_future(
                goal_handle.get_result_async(), 20.0, "one session MoveToPose result"
            )
        except Exception as error:
            action_error = error
        finally:
            # Closure failure has priority and is never reduced to a warning.
            self._gates_closed_proven = False
            record_owner["postaction_gate_closure"] = self._close_gates_strict(
                controller_client, driver_client, records
            )
            self._gates_closed_proven = True
        if action_error is not None:
            raise action_error
        if wrapped is None:
            raise SupervisorError("MoveToPose returned no result")
        result = wrapped.result
        if int(wrapped.status) != GoalStatus.STATUS_SUCCEEDED:
            raise SupervisorError(
                f"MoveToPose action status={int(wrapped.status)}, not SUCCEEDED"
            )
        if (
            not result.ik_result.success
            or int(result.ik_result.code) != IKResult.SUCCESS
        ):
            raise SupervisorError(
                "MoveToPose IK/execution result failed: "
                f"code={result.ik_result.code} {result.ik_result.reason}"
            )
        action_ik = validate_ik_solution(
            current_joints=final_start_joints,
            solution_names=result.ik_result.solution_joint_state.name,
            solution_positions=result.ik_result.solution_joint_state.position,
            reported_max_joint_delta_rad=result.ik_result.max_joint_delta_rad,
            position_error_m=result.ik_result.position_error_m,
            orientation_error_rad=result.ik_result.orientation_error_rad,
            sigma_min=result.ik_result.sigma_min,
            condition_number=result.ik_result.condition_number,
            max_joint_delta_rad=float(
                self.get_parameter("max_ik_joint_delta_rad").value
            ),
            max_position_error_m=(
                self.motion_limits.maximum_ik_position_error_m
            ),
            reported_delta_tolerance_rad=0.002,
        )
        final_values = np.asarray(
            (result.final_position_error_m, result.final_orientation_error_rad),
            dtype=float,
        )
        if not np.all(np.isfinite(final_values)) or np.any(final_values < 0.0):
            raise SupervisorError("MoveToPose returned invalid final errors")
        if (
            final_values[0]
            > self.motion_limits.maximum_final_position_error_m
            or final_values[1] > math.radians(2.0)
        ):
            raise SupervisorError(
                "MoveToPose final residual exceeds selected motion profile"
            )
        return {
            "goal_count": self._motion_goal_count,
            "action_status": int(wrapped.status),
            "ik_code": int(result.ik_result.code),
            "ik_reason": str(result.ik_result.reason),
            "ik_independent_max_joint_delta_rad": action_ik.max_joint_delta_rad,
            "ik_reported_max_joint_delta_rad": (
                action_ik.reported_max_joint_delta_rad
            ),
            "ik_sigma_min": action_ik.sigma_min,
            "ik_condition_number": action_ik.condition_number,
            "final_position_error_m": float(final_values[0]),
            "final_orientation_error_rad": float(final_values[1]),
            "motion_command_count_total": self._motion_command_count,
        }

    def _prepare_dynamic_session_step(
        self,
        *,
        step_index: int,
        selected: dict[str, Any],
        current_camera: np.ndarray,
        session_initial_camera: np.ndarray,
        frozen_configuration: dict[str, Any],
        planned_gain: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Recheck one map-derived step immediately before its gate boundary."""
        camera_target = validate_rigid_transform(
            np.asarray(selected["T_base_camera"], dtype=float),
            f"step {step_index} selected camera",
        )
        link7_target = camera_target_to_link7(
            camera_target, self.report.transform_link7_camera_optical
        )
        selected_link7 = validate_rigid_transform(
            np.asarray(selected["T_base_link7"], dtype=float),
            f"step {step_index} selected link7",
        )
        if not np.allclose(
            link7_target, selected_link7, rtol=0.0, atol=1.0e-9
        ):
            raise SupervisorError(
                f"step {step_index} camera/link7 targets are inconsistent"
            )
        cumulative = camera_motion(session_initial_camera, camera_target)
        if (
            cumulative.translation_m
            > self.motion_limits.maximum_session_translation_m + 1.0e-9
        ):
            raise SupervisorError(
                f"step {step_index} planned cumulative translation exceeds "
                "the selected motion profile"
            )
        if (
            cumulative.rotation_rad
            > self.motion_limits.maximum_session_rotation_rad + 1.0e-9
        ):
            raise SupervisorError(
                f"step {step_index} planned cumulative rotation exceeds "
                "the selected motion profile"
            )

        pre_observation, pre_capture = self._capture(
            f"{self.capture_session_id}_step_{step_index:02d}_preflight"
        )
        pre_camera, pre_tf = self._exact_base_camera(pre_observation)
        pre_target, pre_target_audit = self._estimate_target(
            pre_observation,
            pre_camera,
            configuration=frozen_configuration,
        )
        start_drift = camera_motion(current_camera, pre_camera)
        if start_drift.translation_m > float(
            self.get_parameter("max_plan_start_translation_drift_m").value
        ) or math.degrees(start_drift.rotation_rad) > float(
            self.get_parameter("max_plan_start_rotation_drift_deg").value
        ):
            raise SupervisorError(
                f"step {step_index} camera moved after its five-frame plan"
            )
        frozen_target = np.asarray(
            frozen_configuration["target_center_m"], dtype=float
        )
        target_drift = float(np.linalg.norm(pre_target - frozen_target))
        if target_drift > float(self.get_parameter("max_target_drift_m").value):
            raise SupervisorError(
                f"step {step_index} lost or moved the fixed red target"
            )
        # The map-derived target was computed from ``current_camera``.  A fresh
        # preflight TF can differ by a few micrometres because of encoder
        # quantisation even while the arm is stationary.  If that moves an
        # otherwise valid 5 mm endpoint just beyond the limit, shorten the
        # still-unfrozen dynamic endpoint along the same straight translation
        # path.  The adjusted matrix is the one audited, solved and executed.
        camera_target, start_projection = project_numerical_step_overshoot(
            pre_camera,
            camera_target,
            max_step_m=float(frozen_configuration["max_step_m"]),
            overshoot_tolerance_m=float(
                self.get_parameter("max_plan_start_translation_drift_m").value
            ),
        )
        link7_target = camera_target_to_link7(
            camera_target, self.report.transform_link7_camera_optical
        )
        immediate = validate_raw_next_view(
            pre_camera,
            camera_target,
            planned_gain,
            max_step_m=float(frozen_configuration["max_step_m"]),
        )
        maximum_rotation = math.radians(
            float(self.get_parameter("max_selected_camera_rotation_deg").value)
        )
        if immediate.rotation_rad > maximum_rotation + 1.0e-9:
            raise SupervisorError(
                f"step {step_index} immediate rotation exceeds 10 degrees"
            )
        _result, planned_joints, fresh_ik = self._solve_execution_target(
            link7_target, pre_observation.header.stamp
        )
        selected_solution = np.asarray(
            selected["solution_positions_rad"], dtype=float
        )
        fresh_solution = np.asarray(
            fresh_ik["solution_positions_rad"], dtype=float
        )
        solution_drift = float(
            np.max(np.abs(fresh_solution - selected_solution))
        )
        if solution_drift > float(
            self.get_parameter("max_plan_solution_drift_rad").value
        ):
            raise SupervisorError(
                f"step {step_index} fresh SolveIK solution drifted from plan"
            )
        selected_for_execution = copy.deepcopy(selected)
        selected_for_execution["T_base_camera"] = array_list(camera_target)
        selected_for_execution["T_base_link7"] = array_list(link7_target)
        selected_for_execution["camera_translation_m"] = immediate.translation_m
        selected_for_execution[
            "preflight_translation_projection_correction_m"
        ] = start_projection
        return link7_target, planned_joints, {
            "target_source": "persistent Gradient-NBV map after prior step",
            "selected_candidate": selected_for_execution,
            "T_base_camera": array_list(camera_target),
            "T_base_link7": array_list(link7_target),
            "planned_start_camera": array_list(pre_camera),
            "preflight_translation_projection_correction_m": start_projection,
            "planned_gain": float(planned_gain),
            "planned_cumulative_motion": {
                "translation_m": cumulative.translation_m,
                "rotation_deg": math.degrees(cumulative.rotation_rad),
            },
            "preflight_capture": pre_capture,
            "preflight_tf": pre_tf,
            "preflight_target": pre_target_audit,
            "preflight_target_to_frozen_center_m": target_drift,
            "preflight_start_drift": {
                "translation_m": start_drift.translation_m,
                "rotation_deg": math.degrees(start_drift.rotation_rad),
            },
            "immediate_camera_motion": {
                "translation_m": immediate.translation_m,
                "rotation_deg": math.degrees(immediate.rotation_rad),
            },
            "fresh_execution_ik": fresh_ik,
            "fresh_solution_drift_from_candidate_rad": solution_drift,
        }

    def run_once(self) -> dict[str, Any]:
        audit: dict[str, Any] = {
            "schema": "strawberry_real_nbv_once_supervisor/v1",
            "started_utc": _utc_now(),
            "status": "running",
            "safe_to_execute": False,
            "execute_requested": self.execute_requested,
            "max_motion_steps": self.max_motion_steps,
            "session_policy": copy.deepcopy(self.session_policy),
            "motion_api_owned": bool(self.execute_requested),
            "command_publisher_owned": False,
            "scene_id": self.scene_id,
            "capture_session_id": self.capture_session_id,
            "formal_handeye": {
                "path": str(self.report.path),
                "sha256": self.report.sha256,
                "session_id": self.report.session_id,
                "sample_count": self.report.sample_count,
                "T_link7_camera_optical": array_list(
                    self.report.transform_link7_camera_optical
                ),
            },
            "interfaces": {
                "raw_observation_topic": self.raw_topic,
                "corrected_observation_topic": self.corrected_topic,
                "gradient_runtime": (
                    "separate .venv-nbv ROS process; this node imports no Torch/core"
                ),
            },
        }
        self._audit = audit
        execution_plan = None
        frozen_configuration = None
        if self.execute_requested:
            if self.get_parameter(
                "operator_workspace_clearance_confirmed"
            ).value is not True:
                raise SupervisorError(
                    "execute=true requires operator_workspace_clearance_confirmed=true"
                )
            token = str(
                self.get_parameter("execution_authorization_token").value
            )
            expected_token = _authorization_token(
                self.max_motion_steps, self.motion_profile
            )
            if token != expected_token:
                raise SupervisorError(
                    "execution authorization token is missing or incorrect"
                )
            execution_plan, plan_sha, plan_path = _load_execution_plan(
                str(self.get_parameter("execution_plan_path").value),
                str(self.get_parameter("execution_plan_sha256").value),
                self.report.sha256,
            )
            _require_v3_execution_plan(execution_plan)
            bound_session_policy = _validate_bound_session_policy(
                execution_plan,
                self.max_motion_steps,
                motion_profile=self.motion_profile,
                minimum_target_pixels=self.minimum_target_pixels,
                coverage_target=self.coverage_target,
                coverage_plateau_delta=self.coverage_plateau_delta,
                coverage_plateau_patience=self.coverage_plateau_patience,
            )
            receipt_parameter = str(
                self.get_parameter("authorization_receipt_path").value
            ).strip()
            receipt_path = (
                Path(receipt_parameter).expanduser().resolve()
                if receipt_parameter
                else plan_path.with_name(plan_path.name + ".consumed.json")
            )
            if receipt_path.is_file():
                raise SupervisorError(
                    "this plan has an authorization-consumption receipt and "
                    "cannot be executed again"
                )
            self._authorization_plan_sha256 = plan_sha
            self._authorization_receipt_path = receipt_path
            previous_output = Path(self.output_path).expanduser().resolve()
            if previous_output.is_file():
                try:
                    previous_audit = json.loads(
                        previous_output.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    previous_audit = {}
                if (
                    previous_audit.get("authorization_consumed") is True
                    and previous_audit.get("execution_authorization", {}).get(
                        "plan_sha256"
                    )
                    == plan_sha
                ):
                    raise SupervisorError(
                        "this plan authorization was already consumed successfully"
                    )
            frozen_configuration = copy.deepcopy(
                execution_plan["nbv_configuration"]
            )
            requested_scene_id = self.scene_id
            self.scene_id = str(frozen_configuration["scene_id"])
            audit["scene_id"] = self.scene_id
            audit["execution_scene_binding"] = {
                "requested_runtime_scene_id": requested_scene_id,
                "frozen_configure_scene_id": self.scene_id,
                "capture_session_id": self.capture_session_id,
                "corrected_aggregate_uses_frozen_scene": True,
                "raw_captures_use_independent_session_prefix": True,
            }
            audit["execution_authorization"] = {
                "operator_workspace_clearance_confirmed": True,
                "authorization_token_matched": True,
                "authorization_token_kind": (
                    "one_step"
                    if self.max_motion_steps == 1
                    else "bounded_convergence_session"
                ),
                "bound_session_policy": bound_session_policy,
                "authorization_receipt_path": str(receipt_path),
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha,
                "bound_scene_id": str(execution_plan["scene_id"]),
                "bound_observation_id": str(
                    execution_plan["observation"]["observation_id"]
                ),
                "bound_selected_T_base_camera": execution_plan[
                    "selected_candidate"
                ]["T_base_camera"],
                "environment_collision_checking": False,
                "manual_clearance_lock": True,
            }
        command_count_at_start = self._motion_command_count
        bootstrap, bootstrap_camera, audit["bootstrap_batch"] = (
            self._capture_five_frame_aggregate(
                self.scene_id + "_bootstrap_aggregate",
                "bootstrap",
                target_configuration=frozen_configuration,
            )
        )
        audit["bootstrap_session_identity"] = self._register_session_batch(
            audit["bootstrap_batch"], "bootstrap"
        )
        if execution_plan is not None:
            audit["bootstrap_capture_independence"] = (
                _verify_new_aggregate_batch(
                    audit["bootstrap_batch"], execution_plan, "bootstrap"
                )
            )
        audit["bootstrap_tf"] = audit["bootstrap_batch"]["member_exact_tfs"][2]
        target_center, audit["bootstrap_target"] = self._estimate_target(
            bootstrap,
            bootstrap_camera,
            configuration=frozen_configuration,
        )
        if execution_plan is not None:
            frozen_current = validate_rigid_transform(
                np.asarray(execution_plan["exact_tf"]["T_base_camera_optical"]),
                "frozen current camera",
            )
            fresh_start = camera_motion(frozen_current, bootstrap_camera)
            frozen_target_center = np.asarray(
                frozen_configuration["target_center_m"], dtype=float
            )
            fresh_target_drift = float(
                np.linalg.norm(target_center - frozen_target_center)
            )
            audit["preconfigure_frozen_scene_validation"] = {
                "fresh_start_translation_drift_m": fresh_start.translation_m,
                "fresh_start_rotation_drift_deg": math.degrees(
                    fresh_start.rotation_rad
                ),
                "fresh_target_center_drift_m": fresh_target_drift,
                "start_translation_limit_m": float(
                    self.get_parameter(
                        "max_plan_start_translation_drift_m"
                    ).value
                ),
                "start_rotation_limit_deg": float(
                    self.get_parameter("max_plan_start_rotation_drift_deg").value
                ),
                "target_center_limit_m": float(
                    self.get_parameter(
                        "max_frozen_target_center_drift_m"
                    ).value
                ),
                "passed": False,
            }
            if fresh_start.translation_m > float(
                self.get_parameter("max_plan_start_translation_drift_m").value
            ) or math.degrees(fresh_start.rotation_rad) > float(
                self.get_parameter("max_plan_start_rotation_drift_deg").value
            ):
                raise SupervisorError(
                    "fresh camera start differs from frozen v3 configuration"
                )
            if fresh_target_drift > float(
                self.get_parameter("max_frozen_target_center_drift_m").value
            ):
                raise SupervisorError(
                    "fresh red target differs from frozen v3 configuration"
                )
            audit["preconfigure_frozen_scene_validation"]["passed"] = True
        audit["configuration"] = self._configure(
            target_center,
            bootstrap_camera,
            frozen_configuration=frozen_configuration,
        )

        final_observation, final_camera, audit["planning_batch"] = (
            self._capture_five_frame_aggregate(
                self.scene_id,
                "planning",
                target_configuration=frozen_configuration,
            )
        )
        audit["planning_session_identity"] = self._register_session_batch(
            audit["planning_batch"], "planning"
        )
        if execution_plan is not None:
            audit["planning_capture_independence"] = _verify_new_aggregate_batch(
                audit["planning_batch"], execution_plan, "planning"
            )
            audit["new_execution_batch_pair_independence"] = (
                _verify_new_batch_pair(
                    audit["bootstrap_batch"], audit["planning_batch"]
                )
            )
        audit["final_tf"] = audit["planning_batch"]["member_exact_tfs"][2]
        final_target, audit["final_target"] = self._estimate_target(
            final_observation,
            final_camera,
            configuration=frozen_configuration,
        )
        target_drift = float(np.linalg.norm(final_target - target_center))
        audit["target_center_drift_m"] = target_drift
        if target_drift > float(
            self.get_parameter("max_frozen_target_center_drift_m").value
        ):
            raise SupervisorError(
                f"target centre changed by {target_drift:.6f} m between captures"
            )
        if frozen_configuration is not None:
            planning_frozen_target_drift = float(
                np.linalg.norm(
                    final_target
                    - np.asarray(
                        frozen_configuration["target_center_m"], dtype=float
                    )
                )
            )
            audit["planning_target_to_frozen_config_drift_m"] = (
                planning_frozen_target_drift
            )
            if planning_frozen_target_drift > float(
                self.get_parameter("max_frozen_target_center_drift_m").value
            ):
                raise SupervisorError(
                    "planning red target differs from frozen v3 configuration"
                )
        bootstrap_camera_drift = camera_motion(bootstrap_camera, final_camera)
        audit["bootstrap_to_final_camera_motion"] = {
            "translation_m": bootstrap_camera_drift.translation_m,
            "rotation_deg": math.degrees(bootstrap_camera_drift.rotation_rad),
        }
        corrected = self._correct_observation(final_observation, final_camera)
        audit["corrected_observation"] = {
            "scene_id": corrected.scene_id,
            "observation_id": corrected.observation_id,
            "source_type": wire_uint8(
                corrected.source_type, "Observation.source_type"
            ),
            "source_name": corrected.source_name,
            "optical_frame": corrected.header.frame_id,
            "world_frame": corrected.camera_pose.header.frame_id,
            "stamp_ns": stamp_nanoseconds(corrected.header.stamp),
            "pose_valid": bool(corrected.pose_valid),
            "T_base_camera_optical": array_list(final_camera),
        }
        next_view, audit["raw_next_view"] = self._publish_and_compute(corrected)
        initial_coverage = float(next_view.coverage)
        if (
            self.coverage_target is not None
            and initial_coverage + 1.0e-12 >= self.coverage_target
        ):
            audit["initial_stop_condition"] = {
                "reason": "configured coverage target was already reached",
                "coverage": initial_coverage,
                "coverage_target": self.coverage_target,
                "motion_goal_sent": False,
            }
            raise SupervisorError("COVERAGE_TARGET_ALREADY_REACHED")
        raw_target = _strict_pose_matrix(next_view.pose, "raw NextView")
        selected, audit["ik_candidates"] = self._solve_candidates(
            final_camera,
            raw_target,
            final_observation.header.stamp,
            corrected.observation_id,
        )
        audit["selected_candidate"] = selected
        preview_command_count = (
            self._motion_command_count - command_count_at_start
        )
        audit["preview_motion_commands_observed"] = preview_command_count
        if preview_command_count != 0:
            raise SupervisorError(
                "motion command traffic was observed during the read-only preview"
            )
        # A preview must provide one executable segment before it can be frozen.
        # During execution the exact first target is already bound by the
        # reviewed artifact SHA. A fresh optimizer run is scene-health
        # diagnostic evidence only: single-view NBV can legitimately choose a
        # different member of a near-symmetric set of useful directions. Fresh
        # captures may never replace the frozen target, while target identity,
        # exact start pose and a fresh IK of that frozen target remain hard gates.
        if selected is None and not self.execute_requested:
            raise SupervisorError("NO_REACHABLE_SEGMENTED_VIEW")
        if not self.execute_requested:
            assert selected is not None
            freeze_check_started = time.monotonic()
            freeze_diagnostic = self._wait_controller_closed(
                freeze_check_started
            )
            freeze_publishers = self._validate_command_publishers()
            freeze_arm_health = self._fresh_arm_health()
            freeze_joints = self._wait_stationary()
            if self._motion_command_count - command_count_at_start != 0:
                raise SupervisorError(
                    "motion command appeared during freeze-candidate checks"
                )
            audit["freeze_candidate_safety"] = {
                "controller_diagnostic": freeze_diagnostic,
                "command_publishers": freeze_publishers,
                "arm_health": freeze_arm_health,
                "stationary_joint_positions_rad": array_list(freeze_joints),
                "motion_command_count": 0,
            }
            audit["execution_plan_candidate"] = {
                "schema_version": AGGREGATE_PLAN_V3,
                "status": "passed",
                "scene_id": self.scene_id,
                "session_policy": copy.deepcopy(self.session_policy),
                "nbv_configuration": copy.deepcopy(
                    audit["configuration"]["request_semantics"]
                ),
                "nbv_configuration_sha256": audit["configuration"][
                    "request_sha256"
                ],
                "derived_voxel_grid": copy.deepcopy(
                    audit["configuration"]["derived_voxel_grid"]
                ),
                "handeye_report": {
                    "path": str(self.report.path),
                    "sha256": self.report.sha256,
                    "session_id": self.report.session_id,
                    "sample_count": self.report.sample_count,
                    "T_link7_camera_optical": array_list(
                        self.report.transform_link7_camera_optical
                    ),
                },
                "exact_tf": {
                    "T_base_camera_optical": array_list(final_camera),
                    "requested_exposure_stamp_ns": stamp_nanoseconds(
                        final_observation.header.stamp
                    ),
                    "returned_tf_stamp_ns": audit["final_tf"][
                        "returned_stamp_ns"
                    ],
                },
                "observation": {
                    "scene_id": final_observation.scene_id,
                    "observation_id": final_observation.observation_id,
                    "member_observation_ids": audit["planning_batch"][
                        "member_observation_ids"
                    ],
                    "exposure_stamp_ns": stamp_nanoseconds(
                        final_observation.header.stamp
                    ),
                    "source_name": final_observation.source_name,
                    "aggregation_contract": audit["planning_batch"]["contract"],
                },
                "target": {
                    "center_base_link_m": array_list(target_center),
                    "planning_batch_center_base_link_m": array_list(final_target),
                },
                "nbv": {
                    "planned_gain": float(next_view.gain),
                    "coverage": float(next_view.coverage),
                    "strict_gain_evidence": "core_invariant_nonzero_translation",
                    "raw_translation_m": audit["raw_next_view"]["translation_m"],
                    "raw_rotation_deg": audit["raw_next_view"]["rotation_deg"],
                },
                "selected_candidate": {
                    "T_base_camera": selected["T_base_camera"],
                    "T_base_link7": selected["T_base_link7"],
                    "alpha": selected["alpha"],
                    "translation_alpha": selected["translation_alpha"],
                    "rotation_alpha": selected["rotation_alpha"],
                    "segmentation_contract": selected[
                        "segmentation_contract"
                    ],
                    "reachable_view_selection": (
                        _candidate_selection_evidence(selected)
                    ),
                    "translation_m": selected["camera_translation_m"],
                    "rotation_deg": selected["camera_rotation_deg"],
                },
                "ik": {
                    "success": selected["ik_success"],
                    "code": selected["ik_code"],
                    "reason": selected["ik_reason"],
                    "solution_joint_names": selected["solution_names"],
                    "solution_joint_positions": selected[
                        "solution_positions_rad"
                    ],
                    "max_joint_delta_rad": selected[
                        "independent_max_joint_delta_rad"
                    ],
                    "position_error_m": selected["position_error_m"],
                    "orientation_error_rad": selected["orientation_error_rad"],
                    "sigma_min": selected["sigma_min"],
                    "condition_number": selected["condition_number"],
                },
                "aggregation": {
                    "contract": (
                        "fixed five-frame finite-median depth and 3-of-5 mask"
                    ),
                    "selection_performed": False,
                    "capture_count_per_batch": AGGREGATE_CAPTURE_COUNT,
                    "bootstrap_member_ids": audit["bootstrap_batch"][
                        "member_observation_ids"
                    ],
                    "bootstrap_raw_scenes": audit["bootstrap_batch"][
                        "raw_scenes"
                    ],
                    "bootstrap_member_stamps_ns": audit["bootstrap_batch"][
                        "member_stamps_ns"
                    ],
                    "bootstrap_aggregate_observation_id": audit[
                        "bootstrap_batch"
                    ]["aggregate_observation_id"],
                    "bootstrap_aggregate_identity_sha256": audit[
                        "bootstrap_batch"
                    ]["aggregate_identity_sha256"],
                    "planning_member_ids": audit["planning_batch"][
                        "member_observation_ids"
                    ],
                    "planning_raw_scenes": audit["planning_batch"][
                        "raw_scenes"
                    ],
                    "planning_member_stamps_ns": audit["planning_batch"][
                        "member_stamps_ns"
                    ],
                    "planning_aggregate_observation_id": audit[
                        "planning_batch"
                    ]["aggregate_observation_id"],
                    "planning_aggregate_identity_sha256": audit[
                        "planning_batch"
                    ]["aggregate_identity_sha256"],
                    "bootstrap_pose_span": audit["bootstrap_batch"]["pose_span"],
                    "planning_pose_span": audit["planning_batch"]["pose_span"],
                },
                "safety": {
                    "passed": True,
                    "controller_execution_enabled": False,
                    "controller_diagnostic_fresh_after_solve": True,
                    "motion_command_count_observed": 0,
                    "gate_clients_created": False,
                    "motion_action_clients_created": False,
                    "command_publishers_created": False,
                    "arm_health_passed": True,
                    "stationary_passed": True,
                },
            }
            audit["status"] = "passed_preview_only"
            audit["reason"] = (
                "real Observation, exact exposure TF, formal hand-eye, Gradient-NBV, "
                "frozen ConfigureNBV v3 semantics, segmented pose conversion, and "
                "SolveIK passed; no motion was requested"
            )
            audit["safe_to_execute"] = False
            audit["diagnostic_note"] = (
                "Preview evidence is not execution authorization. Run execute=true "
                "only with the exact accepted artifact SHA, onsite clearance, and "
                "the matching bounded-session authorization token."
            )
            audit["freeze_instruction"] = (
                "Freeze the complete output JSON bytes with sha256sum; execution "
                "must receive that path and digest, and will use a new independent "
                "two-batch five-frame corroboration run with the exact same "
                "ConfigureNBV request and derived map origin."
            )
            audit["finished_utc"] = _utc_now()
            return audit

        assert execution_plan is not None
        artifact_current = validate_rigid_transform(
            np.asarray(execution_plan["exact_tf"]["T_base_camera_optical"]),
            "artifact current camera",
        )
        artifact_target = validate_rigid_transform(
            np.asarray(execution_plan["selected_candidate"]["T_base_camera"]),
            "artifact selected camera",
        )
        start_drift = camera_motion(artifact_current, final_camera)
        translation_threshold = float(
            self.get_parameter("max_plan_target_translation_drift_m").value
        )
        rotation_threshold = float(
            self.get_parameter("max_plan_target_rotation_drift_deg").value
        )
        median_depth_change = abs(
            float(audit["bootstrap_target"]["median_depth_m"])
            - float(audit["final_target"]["median_depth_m"])
        )
        consistency_audit: dict[str, Any] = {
            "start_translation_drift_m": start_drift.translation_m,
            "start_rotation_drift_deg": math.degrees(start_drift.rotation_rad),
            "artifact_alpha": float(execution_plan["selected_candidate"]["alpha"]),
            "translation_corroboration_threshold_m": translation_threshold,
            "rotation_corroboration_threshold_deg": rotation_threshold,
            "bootstrap_to_final_target_center_drift_m": float(
                audit["target_center_drift_m"]
            ),
            "bootstrap_to_final_median_depth_change_m": median_depth_change,
            "fresh_optimizer_candidate_available": selected is not None,
            "role": (
                "fresh optimizer output is diagnostic only; it is neither an "
                "execution gate nor a replacement for the SHA-frozen target"
            ),
            "corroboration_required_for_motion": False,
            "frozen_target_changed": False,
            "gate_clients_created_at_this_point": False,
        }
        if selected is None:
            consistency_audit.update(
                {
                    "classification": "fresh_optimizer_has_no_safe_segment",
                    "translation_corroboration_passed": False,
                    "rotation_corroboration_passed": False,
                }
            )
        else:
            live_target = validate_rigid_transform(
                np.asarray(selected["T_base_camera"]), "live selected camera"
            )
            consistency = compare_gradient_candidates(
                artifact_current,
                artifact_target,
                final_camera,
                live_target,
            )
            target_drift = consistency.target_motion
            translation_agrees = (
                target_drift.translation_m <= translation_threshold
            )
            rotation_agrees = (
                math.degrees(target_drift.rotation_rad) <= rotation_threshold
            )
            consistency_audit.update(
                {
                    "selected_translation_drift_m": target_drift.translation_m,
                    "selected_rotation_drift_deg": math.degrees(
                        target_drift.rotation_rad
                    ),
                    "live_alpha": float(selected["alpha"]),
                    "artifact_selected_step_m": consistency.artifact_step_m,
                    "live_selected_step_m": consistency.live_step_m,
                    "translation_direction_disagreement_deg": math.degrees(
                        consistency.translation_direction_disagreement_rad
                    ),
                    "bounded_search_triangle_limit_m": consistency.triangle_bound_m,
                    "translation_corroboration_passed": translation_agrees,
                    "rotation_corroboration_passed": rotation_agrees,
                    "classification": (
                        "corroborated"
                        if translation_agrees and rotation_agrees
                        else "alternate_valid_single_view_nbv_direction"
                    ),
                }
            )
        audit["artifact_consistency"] = consistency_audit
        if start_drift.translation_m > float(
            self.get_parameter("max_plan_start_translation_drift_m").value
        ) or math.degrees(start_drift.rotation_rad) > float(
            self.get_parameter("max_plan_start_rotation_drift_deg").value
        ):
            raise SupervisorError("fresh exact camera pose drifted from bound artifact")
        artifact_target_center = np.asarray(
            execution_plan["target"]["center_base_link_m"], dtype=float
        )
        if artifact_target_center.shape != (3,) or not np.all(
            np.isfinite(artifact_target_center)
        ):
            raise SupervisorError("artifact target centre is invalid")
        artifact_target_center_drift = float(
            np.linalg.norm(artifact_target_center - target_center)
        )
        audit["artifact_target_center_drift_m"] = artifact_target_center_drift
        if artifact_target_center_drift > float(
            self.get_parameter("max_frozen_target_center_drift_m").value
        ):
            raise SupervisorError("fresh red target centre differs from artifact")

        # The SHA-frozen camera target is the *only* pose that may cross the
        # motion boundary.  The fresh live replan above is consistency evidence,
        # never a replacement target outside the reviewed bytes.
        execution_camera_target = artifact_target
        execution_link7_target = camera_target_to_link7(
            execution_camera_target, self.report.transform_link7_camera_optical
        )
        artifact_link7 = validate_rigid_transform(
            np.asarray(execution_plan["selected_candidate"]["T_base_link7"]),
            "artifact selected link7",
        )
        if not np.allclose(
            execution_link7_target, artifact_link7, rtol=0.0, atol=1.0e-9
        ):
            raise SupervisorError(
                "artifact camera/link7 targets are inconsistent with formal hand-eye"
            )
        audit["execution_target_binding"] = {
            "source": "exact SHA-frozen selected_candidate",
            "T_base_camera": array_list(execution_camera_target),
            "T_base_link7": array_list(execution_link7_target),
            "live_replan_is_consistency_only": True,
        }
        execution_pre_observation, audit["execution_pre_capture"] = self._capture(
            self.scene_id + "_execution_preflight"
        )
        execution_pre_camera, audit["execution_pre_tf"] = self._exact_base_camera(
            execution_pre_observation
        )
        execution_pre_target, audit["execution_pre_target"] = self._estimate_target(
            execution_pre_observation,
            execution_pre_camera,
            configuration=frozen_configuration,
        )
        pre_start_drift = camera_motion(artifact_current, execution_pre_camera)
        audit["execution_pre_artifact_start_drift"] = {
            "translation_m": pre_start_drift.translation_m,
            "rotation_deg": math.degrees(pre_start_drift.rotation_rad),
        }
        if pre_start_drift.translation_m > float(
            self.get_parameter("max_plan_start_translation_drift_m").value
        ) or math.degrees(pre_start_drift.rotation_rad) > float(
            self.get_parameter("max_plan_start_rotation_drift_deg").value
        ):
            raise SupervisorError(
                "immediate pre-execution exact camera pose drifted from artifact"
            )
        pre_target_drift = float(
            np.linalg.norm(execution_pre_target - artifact_target_center)
        )
        audit["execution_pre_target_center_drift_m"] = pre_target_drift
        if pre_target_drift > float(
            self.get_parameter("max_frozen_target_center_drift_m").value
        ):
            raise SupervisorError("red target changed before execution")
        immediate_motion = validate_raw_next_view(
            execution_pre_camera,
            execution_camera_target,
            float(next_view.gain),
            max_step_m=float(frozen_configuration["max_step_m"]),
        )
        if immediate_motion.rotation_rad > math.radians(
            float(self.get_parameter("max_selected_camera_rotation_deg").value)
        ) + 1.0e-9:
            raise SupervisorError("immediate target rotation exceeds single-step gate")
        audit["immediate_preexecution_camera_motion"] = {
            "translation_m": immediate_motion.translation_m,
            "rotation_deg": math.degrees(immediate_motion.rotation_rad),
        }
        _ik_result, planned_joints, audit["fresh_execution_ik"] = (
            self._solve_execution_target(
                execution_link7_target, execution_pre_observation.header.stamp
            )
        )
        preview_solution = np.asarray(
            execution_plan["ik"]["solution_joint_positions"], dtype=float
        )
        fresh_solution = np.asarray(
            audit["fresh_execution_ik"]["solution_positions_rad"], dtype=float
        )
        solution_drift = float(np.max(np.abs(fresh_solution - preview_solution)))
        audit["fresh_solution_drift_from_preview_rad"] = solution_drift
        if solution_drift > float(
            self.get_parameter("max_plan_solution_drift_rad").value
        ):
            raise SupervisorError("fresh SolveIK solution drifted from frozen artifact")

        if self._configure_call_count != 1:
            raise SupervisorError("persistent-map session must configure exactly once")
        ledger = MotionSessionLedger(
            max_motion_steps=self.max_motion_steps,
            initial_camera=artifact_current,
            initial_coverage=float(next_view.coverage),
            motion_limits=self.motion_limits,
            minimum_target_pixels=int(
                self.get_parameter("minimum_target_pixels").value
            ),
            coverage_plateau_delta=self.coverage_plateau_delta,
            coverage_plateau_patience=self.coverage_plateau_patience,
            coverage_target=self.coverage_target,
        )
        self._session_ledger = ledger
        audit["persistent_map_session"] = {
            "scene_id": self.scene_id,
            "configure_request_sha256": audit["configuration"]["request_sha256"],
            "configure_call_count": self._configure_call_count,
            "map_origin_m": audit["configuration"]["derived_voxel_grid"][
                "origin_m"
            ],
            "configured_once": True,
        }
        first_step = {
            "step_index": 1,
            "target_source": "exact SHA-frozen first-step preview target",
            "target_binding": copy.deepcopy(audit["execution_target_binding"]),
            "selected_candidate": copy.deepcopy(
                execution_plan["selected_candidate"]
            ),
            "fresh_execution_ik": copy.deepcopy(audit["fresh_execution_ik"]),
            "fresh_solution_drift_from_preview_rad": solution_drift,
            "preflight_capture": copy.deepcopy(audit["execution_pre_capture"]),
            "preflight_tf": copy.deepcopy(audit["execution_pre_tf"]),
            "planned_start_camera": array_list(execution_pre_camera),
            "preflight_target": copy.deepcopy(audit["execution_pre_target"]),
            "preflight_target_to_frozen_center_m": pre_target_drift,
            "immediate_camera_motion": copy.deepcopy(
                audit["immediate_preexecution_camera_motion"]
            ),
            "plan_sha256_binding": plan_sha,
            "dynamic_target_sha256": None,
        }
        first_budget_step = ledger.validate_next_planned_step(
            planned_start_camera=execution_pre_camera,
            planned_camera=execution_camera_target,
        )
        first_step["session_motion_budget_preflight"] = {
            "translation_m": first_budget_step.translation_m,
            "rotation_deg": math.degrees(first_budget_step.rotation_rad),
            "passed_before_gate_open": True,
        }
        audit["motion_steps"] = [first_step]
        audit["active_step_index"] = 1
        audit["authorization_consumed"] = False
        self._checkpoint_execution_audit("ready_to_open_gates_for_step_1")
        command_count_before_step = self._motion_command_count
        first_step["execution"] = self._execute_one_goal(
            execution_link7_target,
            self.get_clock().now().to_msg(),
            planned_joints,
            step_audit=first_step,
        )
        first_step["motion_command_count_step"] = (
            self._motion_command_count - command_count_before_step
        )
        audit["execution"] = copy.deepcopy(first_step["execution"])
        audit["authorization_consumed"] = True
        self._checkpoint_execution_audit("step_1_gates_closed")

        current_planned_camera = execution_camera_target
        while ledger.motion_goal_count < self.max_motion_steps:
            step_index = ledger.motion_goal_count + 1
            step_record = audit["motion_steps"][step_index - 1]

            # Perception resumes only after _execute_one_goal has obtained two
            # false acknowledgements per gate, a fresh controller=false
            # diagnostic, no new command, and continuous stationary feedback.
            settle_delay = float(
                self.get_parameter("post_motion_camera_settle_delay_sec").value
            )
            step_record["post_motion_camera_settle_delay_sec"] = settle_delay
            if settle_delay > 0.0:
                time.sleep(settle_delay)
            post_observation, post_camera, post_batch = (
                self._capture_five_frame_aggregate(
                    self.scene_id,
                    f"post_step_{step_index:02d}",
                    target_configuration=frozen_configuration,
                )
            )
            step_record["post_batch"] = post_batch
            step_record["post_session_identity"] = self._register_session_batch(
                post_batch, f"post step {step_index}"
            )
            step_record["post_tf"] = post_batch["member_exact_tfs"][2]
            post_target, step_record["post_target"] = self._estimate_target(
                post_observation,
                post_camera,
                configuration=frozen_configuration,
            )
            frozen_target_center = np.asarray(
                frozen_configuration["target_center_m"], dtype=float
            )
            post_target_drift = float(
                np.linalg.norm(post_target - frozen_target_center)
            )
            step_record["post_target_center_drift_m"] = post_target_drift
            if post_target_drift > float(
                self.get_parameter("max_target_drift_m").value
            ):
                raise SupervisorError(
                    f"step {step_index} post-motion red target was lost or moved"
                )
            achieved_error = camera_motion(current_planned_camera, post_camera)
            step_record["post_pose_error"] = {
                "translation_m": achieved_error.translation_m,
                "rotation_deg": math.degrees(achieved_error.rotation_rad),
            }
            if achieved_error.translation_m > float(
                self.get_parameter("post_target_position_tolerance_m").value
            ) or math.degrees(achieved_error.rotation_rad) > float(
                self.get_parameter("post_target_rotation_tolerance_deg").value
            ):
                raise SupervisorError(
                    f"step {step_index} exact camera pose missed its selected target"
                )
            post_corrected = self._correct_observation(
                post_observation, post_camera
            )
            post_view, post_map_update = self._publish_and_compute(
                post_corrected,
                require_strict_improvement_proxy=False,
            )
            step_record["post_map_update"] = post_map_update
            closed = step_record.get("postaction_gate_closure", {})
            gates_closed = (
                closed.get("driver_false_ack_count") == 2
                and closed.get("controller_false_ack_count") == 2
                and closed.get("controller_diagnostic", {})
                .get("values", {})
                .get("execution_enabled")
                is False
            )
            evidence = ledger.record_closed_step(
                planned_start_camera=np.asarray(
                    step_record["planned_start_camera"], dtype=float
                ),
                planned_camera=current_planned_camera,
                actual_camera=post_camera,
                coverage_after=float(post_view.coverage),
                target_valid_mask_depth_pixels=int(
                    step_record["post_target"]["valid_mask_pixels"]
                ),
                reported_motion_goal_count=self._motion_goal_count,
                gates_closed=gates_closed,
            )
            step_record["coverage"] = {
                "before": evidence.coverage_before,
                "after": evidence.coverage_after,
                "delta": evidence.coverage_delta,
                "nondecreasing": True,
                "plateau_delta_threshold": self.coverage_plateau_delta,
                "plateau_patience": self.coverage_plateau_patience,
                "plateau_count": evidence.coverage_plateau_count,
                "coverage_target": self.coverage_target,
                "coverage_target_reached": evidence.coverage_target_reached,
                "converged": evidence.converged,
            }
            step_record["cumulative_camera_motion"] = {
                "planned_step_translation_m": (
                    evidence.planned_step_translation_m
                ),
                "planned_step_rotation_deg": math.degrees(
                    evidence.planned_step_rotation_rad
                ),
                "actual_step_translation_m": evidence.actual_step_translation_m,
                "actual_step_rotation_deg": math.degrees(
                    evidence.actual_step_rotation_rad
                ),
                "planned_translation_m": (
                    evidence.planned_cumulative_translation_m
                ),
                "planned_rotation_deg": math.degrees(
                    evidence.planned_cumulative_rotation_rad
                ),
                "actual_translation_m": evidence.actual_cumulative_translation_m,
                "actual_rotation_deg": math.degrees(
                    evidence.actual_cumulative_rotation_rad
                ),
                "translation_limit_m": (
                    self.motion_limits.maximum_session_translation_m
                ),
                "rotation_limit_deg": math.degrees(
                    self.motion_limits.maximum_session_rotation_rad
                ),
            }
            step_record["completed_gates_closed"] = True
            audit["motion_goal_count"] = self._motion_goal_count
            audit["active_step_index"] = None
            audit["session_progress"] = ledger.summary()
            self._checkpoint_execution_audit(
                f"step_{step_index}_closed_and_map_updated"
            )

            if evidence.converged or step_index >= self.max_motion_steps:
                break

            next_step_index = step_index + 1
            audit["active_step_index"] = next_step_index
            raw_target = _strict_pose_matrix(
                post_view.pose, f"step {next_step_index} raw NextView"
            )
            raw_gain = float(post_view.gain)
            raw_motion = camera_motion(post_camera, raw_target)
            if not math.isfinite(raw_gain):
                raise SupervisorError(
                    f"step {next_step_index} NextView gain is not finite"
                )
            if raw_gain <= 0.0:
                convergence_reason = (
                    "Gradient-NBV found no positive-gain next view in the "
                    "current persistent map"
                )
                ledger.mark_converged(convergence_reason)
                step_record["next_view_convergence"] = {
                    "next_step_index": next_step_index,
                    "reason": convergence_reason,
                    "raw_translation_m": raw_motion.translation_m,
                    "raw_rotation_deg": math.degrees(raw_motion.rotation_rad),
                    "planned_gain": raw_gain,
                    "source_observation_id": str(post_view.observation_id),
                    "source_map_coverage": float(post_view.coverage),
                    "motion_goal_sent": False,
                }
                audit["session_progress"] = ledger.summary()
                audit["active_step_index"] = None
                self._checkpoint_execution_audit(
                    f"converged_before_step_{next_step_index}_no_positive_gain"
                )
                break
            if not camera_step_above_minimum(
                raw_motion.translation_m, self.motion_limits
            ):
                convergence_reason = (
                    "Gradient-NBV requested no additional camera translation "
                    "above the selected motion profile minimum"
                )
                ledger.mark_converged(convergence_reason)
                step_record["next_view_convergence"] = {
                    "next_step_index": next_step_index,
                    "reason": convergence_reason,
                    "raw_translation_m": raw_motion.translation_m,
                    "raw_rotation_deg": math.degrees(raw_motion.rotation_rad),
                    "planned_gain": raw_gain,
                    "source_observation_id": str(post_view.observation_id),
                    "source_map_coverage": float(post_view.coverage),
                    "motion_goal_sent": False,
                }
                audit["session_progress"] = ledger.summary()
                audit["active_step_index"] = None
                self._checkpoint_execution_audit(
                    f"converged_before_step_{next_step_index}_deadband"
                )
                break
            validate_raw_next_view(
                post_camera,
                raw_target,
                raw_gain,
                max_step_m=float(frozen_configuration["max_step_m"]),
            )
            dynamic_selected, dynamic_candidates = self._solve_candidates(
                post_camera,
                raw_target,
                post_observation.header.stamp,
                post_corrected.observation_id,
            )
            if dynamic_selected is None:
                reachable_count = sum(
                    bool(candidate.get("ik_gate_passed", False))
                    for candidate in dynamic_candidates
                )
                useful_count = sum(
                    bool(candidate.get("ik_gate_passed", False))
                    and bool(candidate.get("gain_scored", False))
                    and float(candidate.get("gain_improvement", 0.0)) > 0.0
                    for candidate in dynamic_candidates
                )
                convergence_reason = (
                    "no positive-gain reachable candidate remained in the "
                    "predeclared multi-radius search lattice"
                )
                ledger.mark_converged(convergence_reason)
                step_record["next_view_convergence"] = {
                    "next_step_index": next_step_index,
                    "reason": convergence_reason,
                    "raw_translation_m": raw_motion.translation_m,
                    "raw_rotation_deg": math.degrees(raw_motion.rotation_rad),
                    "planned_gain": raw_gain,
                    "source_observation_id": str(post_view.observation_id),
                    "source_map_coverage": float(post_view.coverage),
                    "candidate_count": len(dynamic_candidates),
                    "reachable_candidate_count": reachable_count,
                    "useful_candidate_count": useful_count,
                    "ik_candidates": dynamic_candidates,
                    "motion_goal_sent": False,
                }
                audit["session_progress"] = ledger.summary()
                audit["active_step_index"] = None
                self._checkpoint_execution_audit(
                    f"converged_before_step_{next_step_index}_no_reachable_gain"
                )
                break
            dynamic_camera_target = validate_rigid_transform(
                np.asarray(dynamic_selected["T_base_camera"], dtype=float),
                f"step {next_step_index} selected camera",
            )
            try:
                dynamic_budget_step = ledger.validate_next_planned_step(
                    planned_start_camera=post_camera,
                    planned_camera=dynamic_camera_target,
                )
            except MotionBudgetExhausted:
                convergence_reason = (
                    "the next useful view would exceed the audited cumulative "
                    "camera-motion envelope"
                )
                ledger.mark_converged(convergence_reason)
                step_record["next_view_convergence"] = {
                    "next_step_index": next_step_index,
                    "reason": convergence_reason,
                    "raw_translation_m": raw_motion.translation_m,
                    "raw_rotation_deg": math.degrees(raw_motion.rotation_rad),
                    "planned_gain": raw_gain,
                    "source_observation_id": str(post_view.observation_id),
                    "source_map_coverage": float(post_view.coverage),
                    "motion_goal_sent": False,
                }
                audit["session_progress"] = ledger.summary()
                audit["active_step_index"] = None
                self._checkpoint_execution_audit(
                    f"converged_before_step_{next_step_index}_motion_envelope"
                )
                break
            dynamic_link7, dynamic_joints, dynamic_audit = (
                self._prepare_dynamic_session_step(
                    step_index=next_step_index,
                    selected=dynamic_selected,
                    current_camera=post_camera,
                    session_initial_camera=artifact_current,
                    frozen_configuration=frozen_configuration,
                    planned_gain=float(post_view.gain),
                )
            )
            target_payload = {
                "step_index": next_step_index,
                "configure_request_sha256": audit["configuration"][
                    "request_sha256"
                ],
                "source_observation_id": str(post_view.observation_id),
                "T_base_camera": dynamic_audit["T_base_camera"],
                "T_base_link7": dynamic_audit["T_base_link7"],
                "alpha": float(dynamic_selected["alpha"]),
                "translation_alpha": float(
                    dynamic_selected["translation_alpha"]
                ),
                "rotation_alpha": float(dynamic_selected["rotation_alpha"]),
                "segmentation_contract": dynamic_selected[
                    "segmentation_contract"
                ],
                "reachable_view_selection": (
                    _candidate_selection_evidence(dynamic_selected)
                ),
            }
            target_digest = hashlib.sha256(
                json.dumps(
                    target_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            next_step = {
                "step_index": next_step_index,
                **dynamic_audit,
                "source_map_observation_id": str(post_view.observation_id),
                "source_map_coverage": float(post_view.coverage),
                "ik_candidates": dynamic_candidates,
                "dynamic_target_payload": target_payload,
                "dynamic_target_sha256": target_digest,
            }
            next_step["session_motion_budget_preflight"] = {
                "translation_m": dynamic_budget_step.translation_m,
                "rotation_deg": math.degrees(dynamic_budget_step.rotation_rad),
                "passed_before_gate_open": True,
            }
            audit["motion_steps"].append(next_step)
            self._checkpoint_execution_audit(
                f"dynamic_target_step_{next_step_index}_frozen_before_gates"
            )
            command_count_before_step = self._motion_command_count
            next_step["execution"] = self._execute_one_goal(
                dynamic_link7,
                self.get_clock().now().to_msg(),
                dynamic_joints,
                step_audit=next_step,
            )
            next_step["motion_command_count_step"] = (
                self._motion_command_count - command_count_before_step
            )
            audit["authorization_consumed"] = True
            current_planned_camera = validate_rigid_transform(
                np.asarray(dynamic_audit["T_base_camera"], dtype=float),
                f"step {next_step_index} planned camera",
            )
            self._checkpoint_execution_audit(
                f"step_{next_step_index}_gates_closed"
            )

        audit["session_progress"] = ledger.summary()
        audit["motion_goal_count"] = self._motion_goal_count
        audit["configure_call_count"] = self._configure_call_count
        audit["second_motion_requested"] = self._motion_goal_count >= 2
        audit["safe_to_execute"] = False
        audit["authorization_consumed"] = self._motion_goal_count > 0
        audit["one_shot_consumed"] = (
            self.max_motion_steps == 1 and self._motion_goal_count == 1
        )
        if self._motion_goal_count != ledger.motion_goal_count:
            raise SupervisorError("session goal count differs from closed-step ledger")
        if self._configure_call_count != 1:
            raise SupervisorError("NBV map was unexpectedly reconfigured")
        if self.max_motion_steps == 1:
            audit["status"] = "executed_once_gates_closed_post_map_updated"
            audit["reason"] = (
                "one artifact-bound NBV segment executed; both gates were closed "
                "and a five-frame post-motion update was fused into the same map"
            )
        elif audit["session_progress"]["scientific_acceptance_passed"]:
            audit["status"] = "executed_session_scientific_acceptance_passed"
            audit["reason"] = (
                "bounded convergence session completed with persistent-map "
                "coverage meeting all scientific acceptance thresholds"
            )
        elif audit["session_progress"]["converged_early"]:
            audit["status"] = "executed_session_converged_scientific_acceptance_not_met"
            convergence_reason = audit["session_progress"].get(
                "convergence_reason"
            )
            if convergence_reason:
                audit["reason"] = (
                    "session stopped safely after reaching a configured stop "
                    f"condition ({convergence_reason}); scientific "
                    "acceptance was not met"
                )
            else:
                audit["reason"] = (
                    "session stopped safely at its bounded motion limit; "
                    "scientific acceptance was not met"
                )
        else:
            audit["status"] = "executed_session_scientific_acceptance_not_met"
            audit["reason"] = (
                "the authorized motion session ended safely, but its coverage "
                "did not meet the scientific acceptance thresholds"
            )
        audit["finished_utc"] = _utc_now()
        return audit


def main(args=None) -> None:
    """Run once, always write an audit when node construction succeeds."""
    rclpy.init(args=args)
    node: RealNBVSupervisor | None = None
    audit: dict[str, Any] | None = None
    output_path = DEFAULT_OUTPUT_PATH
    exit_error: Exception | None = None
    try:
        node = RealNBVSupervisor()
        output_path = node.output_path
        audit = node.run_once()
    except Exception as error:  # write the refusal before returning non-zero
        exit_error = error
        audit = copy.deepcopy(node._audit) if node is not None else {}
        if node is not None and node._session_ledger is not None:
            gates_closed = _failure_gates_closed(node, error)
            node._session_ledger.abort(
                str(error), gates_closed=gates_closed
            )
            audit["session_progress"] = node._session_ledger.summary()
        audit.update(
            {
                "schema": "strawberry_real_nbv_once_supervisor/v1",
                "status": "failed",
                "safe_to_execute": False,
                "reason": str(error),
                "finished_utc": _utc_now(),
                "execute_requested": bool(
                    node.execute_requested if node is not None else False
                ),
                "motion_commands_observed": int(
                    node._motion_command_count if node is not None else 0
                ),
                "motion_goal_count": int(
                    node._motion_goal_count if node is not None else 0
                ),
                "terminated_after_motion_goal_count": int(
                    node._motion_goal_count if node is not None else 0
                ),
                "authorization_consumed": bool(
                    node is not None and node._motion_goal_count > 0
                ),
            }
        )
        if isinstance(error, GateClosureError):
            audit["onsite_stop_required"] = True
            audit["post_capture_attempted_after_close_failure"] = False
        if node is not None:
            output_path = node.output_path
            audit["scene_id"] = node.scene_id
            audit["formal_handeye"] = {
                "path": str(node.report.path),
                "sha256": node.report.sha256,
                "session_id": node.report.session_id,
            }
    finally:
        try:
            _write_json_atomic(output_path, audit or {})
            print(json.dumps(audit or {}, ensure_ascii=False, indent=2, sort_keys=True))
            print(f"audit_json={Path(output_path).expanduser().resolve()}")
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    if exit_error is not None:
        raise SystemExit(f"real NBV preview failed: {exit_error}")


if __name__ == "__main__":
    main()
