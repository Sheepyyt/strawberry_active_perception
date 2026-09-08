"""Pure validation helpers for the bounded real-camera NBV supervisor.

This module deliberately depends only on NumPy and the bridge transform
helpers.  In particular, it does not import ROS, Torch, OpenCV, a robot
driver, or the Gradient-NBV implementation.  The ROS supervisor uses these
functions at every untrusted boundary and the unit tests exercise the same
checks without a running ROS graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .transforms import (
    interpolate_transform,
    validate_rigid_transform,
)


MIN_CAMERA_STEP_M = 0.001
MAX_CAMERA_STEP_M = 0.005
MAX_SELECTED_CAMERA_ROTATION_RAD = math.radians(10.0)
MAX_IK_JOINT_DELTA_RAD = 0.08
MAX_IK_POSITION_ERROR_M = 0.003
MAX_IK_ORIENTATION_ERROR_RAD = math.radians(2.0)
MIN_IK_SIGMA = 0.10
MAX_IK_CONDITION = 20.0
DEFAULT_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
ALLOWED_SESSION_MOTION_STEPS = (1, 3)
MAX_SESSION_TRANSLATION_M = 0.015
MAX_SESSION_ROTATION_RAD = math.radians(30.0)
CONVERGENCE_COVERAGE_DELTA = 0.005
SCIENCE_STEP_COVERAGE_DELTA = 0.01
SCIENCE_FINAL_COVERAGE_DELTA = 0.20
NERO_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
_EPS = 1.0e-12


@dataclass(frozen=True)
class TargetEstimate:
    """Robust target centre estimate and its auditable pixel statistics."""

    camera_xyz_m: np.ndarray
    mask_pixels: int
    valid_mask_pixels: int
    retained_pixels: int
    median_depth_m: float
    depth_mad_m: float


@dataclass(frozen=True)
class DepthMaskAggregation:
    """Deterministic pixel-wise aggregation of exactly five observations.

    ``finite_support`` and ``mask_votes`` retain the per-pixel evidence used
    to form the two aggregate images.  ``audit_counts`` intentionally returns
    only built-in Python integers so it can be included in a JSON audit
    without a NumPy-specific encoder.
    """

    depth_m: np.ndarray
    mask: np.ndarray
    finite_support: np.ndarray
    mask_votes: np.ndarray
    input_frame_count: int
    pixel_count: int
    finite_depth_sample_count: int
    output_finite_depth_pixel_count: int
    output_nan_depth_pixel_count: int
    foreground_mask_sample_count: int
    output_foreground_mask_pixel_count: int

    @property
    def audit_counts(self) -> dict[str, int]:
        """Return scalar aggregation evidence suitable for JSON output."""
        return {
            "input_frame_count": int(self.input_frame_count),
            "pixel_count": int(self.pixel_count),
            "finite_depth_sample_count": int(
                self.finite_depth_sample_count
            ),
            "output_finite_depth_pixel_count": int(
                self.output_finite_depth_pixel_count
            ),
            "output_nan_depth_pixel_count": int(
                self.output_nan_depth_pixel_count
            ),
            "foreground_mask_sample_count": int(
                self.foreground_mask_sample_count
            ),
            "output_foreground_mask_pixel_count": int(
                self.output_foreground_mask_pixel_count
            ),
        }


@dataclass(frozen=True)
class CameraMotion:
    """Geodesic camera-pose displacement."""

    translation_m: float
    rotation_rad: float


@dataclass(frozen=True)
class IKValidation:
    """Independently recomputed IK safety metrics."""

    max_joint_delta_rad: float
    reported_max_joint_delta_rad: float
    position_error_m: float
    orientation_error_rad: float
    sigma_min: float
    condition_number: float


@dataclass(frozen=True)
class GradientCandidateConsistency:
    """Separate optimizer-direction repeatability from endpoint safety."""

    start_motion: CameraMotion
    target_motion: CameraMotion
    artifact_step_m: float
    live_step_m: float
    translation_direction_disagreement_rad: float
    triangle_bound_m: float


@dataclass(frozen=True)
class NBVConfigurationEvidence:
    """Canonical ConfigureNBV request plus its derived voxel-grid geometry."""

    request: dict[str, Any]
    voxel_dimensions: tuple[int, int, int]
    map_origin_m: np.ndarray
    sha256: str


@dataclass(frozen=True)
class SessionStepEvidence:
    """Validated outcome of one closed-gate physical NBV movement."""

    step_index: int
    coverage_before: float
    coverage_after: float
    coverage_delta: float
    planned_step_translation_m: float
    planned_step_rotation_rad: float
    actual_step_translation_m: float
    actual_step_rotation_rad: float
    planned_cumulative_translation_m: float
    planned_cumulative_rotation_rad: float
    actual_cumulative_translation_m: float
    actual_cumulative_rotation_rad: float
    target_valid_mask_depth_pixels: int
    converged: bool


class MotionSessionLedger:
    """Fail-closed, ROS-free accounting for a one- or three-step session.

    The supervisor owns all hardware interactions.  This ledger independently
    enforces the cumulative pose, coverage, target-pixel, goal-count, and
    closed-gate invariants after each completed physical step.
    """

    def __init__(
        self,
        *,
        max_motion_steps: int,
        initial_camera: np.ndarray,
        initial_coverage: float,
        minimum_target_pixels: int = 200,
    ) -> None:
        if (
            isinstance(max_motion_steps, bool)
            or max_motion_steps not in ALLOWED_SESSION_MOTION_STEPS
        ):
            raise ValueError("max_motion_steps must be exactly 1 or 3")
        coverage = float(initial_coverage)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("initial coverage must be finite and in [0, 1]")
        if (
            isinstance(minimum_target_pixels, bool)
            or int(minimum_target_pixels) < 200
        ):
            raise ValueError("minimum target pixels must be at least 200")
        self.max_motion_steps = int(max_motion_steps)
        self.initial_camera = validate_rigid_transform(
            initial_camera, "session initial camera"
        )
        self.initial_coverage = coverage
        self.minimum_target_pixels = int(minimum_target_pixels)
        self.steps: list[SessionStepEvidence] = []
        self.termination_reason: str | None = None
        self.gates_closed_at_termination: bool | None = None
        self._last_planned_camera = self.initial_camera.copy()
        self._last_actual_camera = self.initial_camera.copy()
        self._planned_translation_total_m = 0.0
        self._planned_rotation_total_rad = 0.0
        self._actual_translation_total_m = 0.0
        self._actual_rotation_total_rad = 0.0

    @property
    def motion_goal_count(self) -> int:
        return len(self.steps)

    @property
    def current_coverage(self) -> float:
        if not self.steps:
            return self.initial_coverage
        return self.steps[-1].coverage_after

    def record_closed_step(
        self,
        *,
        planned_camera: np.ndarray,
        actual_camera: np.ndarray,
        coverage_after: float,
        target_valid_mask_depth_pixels: int,
        reported_motion_goal_count: int,
        gates_closed: bool,
    ) -> SessionStepEvidence:
        """Record exactly one completed goal only after both gates are closed."""
        if self.termination_reason is not None:
            raise ValueError("terminated session cannot accept another goal")
        expected_index = len(self.steps) + 1
        if expected_index > self.max_motion_steps:
            raise ValueError("session motion-goal limit is already exhausted")
        if reported_motion_goal_count != expected_index:
            raise ValueError("motion goal count is not sequential")
        if gates_closed is not True:
            raise ValueError("both command gates must be proven closed")
        if (
            isinstance(target_valid_mask_depth_pixels, bool)
            or int(target_valid_mask_depth_pixels) < self.minimum_target_pixels
        ):
            raise ValueError("red target has fewer than 200 valid mask-depth pixels")

        planned_transform = validate_rigid_transform(
            planned_camera, "planned step camera"
        )
        actual_transform = validate_rigid_transform(
            actual_camera, "actual step camera"
        )
        planned_step = camera_motion(
            self._last_planned_camera, planned_transform
        )
        actual_step = camera_motion(self._last_actual_camera, actual_transform)
        if not (
            planned_step.translation_m > MIN_CAMERA_STEP_M
            and planned_step.translation_m <= MAX_CAMERA_STEP_M + 1.0e-9
        ):
            raise ValueError("planned camera step must be in (1, 5] mm")
        if (
            planned_step.rotation_rad
            > MAX_SELECTED_CAMERA_ROTATION_RAD + 1.0e-9
        ):
            raise ValueError("planned camera step rotation exceeds 10 degrees")

        planned_translation_total = (
            self._planned_translation_total_m + planned_step.translation_m
        )
        planned_rotation_total = (
            self._planned_rotation_total_rad + planned_step.rotation_rad
        )
        actual_translation_total = (
            self._actual_translation_total_m + actual_step.translation_m
        )
        actual_rotation_total = (
            self._actual_rotation_total_rad + actual_step.rotation_rad
        )
        for label, translation_total, rotation_total in (
            (
                "planned",
                planned_translation_total,
                planned_rotation_total,
            ),
            ("actual", actual_translation_total, actual_rotation_total),
        ):
            if translation_total > MAX_SESSION_TRANSLATION_M + 1.0e-9:
                raise ValueError(
                    f"{label} cumulative camera translation exceeds 15 mm"
                )
            if rotation_total > MAX_SESSION_ROTATION_RAD + 1.0e-9:
                raise ValueError(
                    f"{label} cumulative camera rotation exceeds 30 degrees"
                )

        after = float(coverage_after)
        before = self.current_coverage
        if not math.isfinite(after) or not 0.0 <= after <= 1.0:
            raise ValueError("coverage must be finite and in [0, 1]")
        delta = after - before
        if delta < -1.0e-6:
            raise ValueError("NBV map coverage decreased")
        evidence = SessionStepEvidence(
            step_index=expected_index,
            coverage_before=before,
            coverage_after=after,
            coverage_delta=delta,
            planned_step_translation_m=planned_step.translation_m,
            planned_step_rotation_rad=planned_step.rotation_rad,
            actual_step_translation_m=actual_step.translation_m,
            actual_step_rotation_rad=actual_step.rotation_rad,
            planned_cumulative_translation_m=planned_translation_total,
            planned_cumulative_rotation_rad=planned_rotation_total,
            actual_cumulative_translation_m=actual_translation_total,
            actual_cumulative_rotation_rad=actual_rotation_total,
            target_valid_mask_depth_pixels=int(target_valid_mask_depth_pixels),
            converged=delta < CONVERGENCE_COVERAGE_DELTA,
        )
        self._last_planned_camera = planned_transform.copy()
        self._last_actual_camera = actual_transform.copy()
        self._planned_translation_total_m = planned_translation_total
        self._planned_rotation_total_rad = planned_rotation_total
        self._actual_translation_total_m = actual_translation_total
        self._actual_rotation_total_rad = actual_rotation_total
        self.steps.append(evidence)
        return evidence

    def abort(self, reason: str, *, gates_closed: bool) -> None:
        """Latch a terminal failure so no later goal can be recorded."""
        text = str(reason).strip()
        if not text:
            raise ValueError("session termination reason must be non-empty")
        if self.termination_reason is None:
            self.termination_reason = text
            self.gates_closed_at_termination = bool(gates_closed)

    def summary(self) -> dict[str, Any]:
        """Return deterministic scientific and authorization-consumption facts."""
        deltas = [step.coverage_delta for step in self.steps]
        final_coverage = self.current_coverage
        science_required = self.max_motion_steps == 3
        science_passed = (
            len(self.steps) >= 2
            and sum(delta >= SCIENCE_STEP_COVERAGE_DELTA for delta in deltas) >= 2
            and final_coverage - self.initial_coverage
            >= SCIENCE_FINAL_COVERAGE_DELTA
        )
        return {
            "requested_max_motion_steps": self.max_motion_steps,
            "motion_goal_count": len(self.steps),
            "initial_coverage": self.initial_coverage,
            "final_coverage": final_coverage,
            "total_coverage_delta": final_coverage - self.initial_coverage,
            "per_step_coverage_delta": deltas,
            "steps_with_at_least_one_percentage_point_gain": sum(
                delta >= SCIENCE_STEP_COVERAGE_DELTA for delta in deltas
            ),
            "converged_early": bool(self.steps and self.steps[-1].converged),
            "scientific_acceptance_required": science_required,
            "scientific_acceptance_passed": (
                science_passed if science_required else None
            ),
            "authorization_consumed": bool(self.steps),
            "terminated": self.termination_reason is not None,
            "termination_reason": self.termination_reason,
            "gates_closed_at_termination": self.gates_closed_at_termination,
        }


CONFIGURE_NBV_REQUEST_KEYS = (
    "scene_id",
    "world_frame",
    "target_center_m",
    "map_size_m",
    "target_roi_size_m",
    "observation_min_m",
    "observation_max_m",
    "voxel_size_m",
    "depth_min_m",
    "depth_max_m",
    "samples_per_ray",
    "optimization_steps",
    "max_step_m",
    "random_seed",
)


def normalize_nbv_configuration(
    values: Mapping[str, Any],
) -> NBVConfigurationEvidence:
    """Validate and canonicalize every ConfigureNBV request field.

    The three ROS ``float32`` fields are rounded through NumPy float32 before
    hashing, matching the values received by the Gradient-NBV service.  Map
    dimensions and origin reproduce the core's ceil-based voxel construction.
    """
    if not isinstance(values, Mapping):
        raise ValueError("ConfigureNBV semantics must be a mapping")
    missing = [key for key in CONFIGURE_NBV_REQUEST_KEYS if key not in values]
    if missing:
        raise ValueError(f"ConfigureNBV semantics are missing {missing}")
    scene_id = values["scene_id"]
    world_frame = values["world_frame"]
    if (
        not isinstance(scene_id, str)
        or not scene_id.strip()
        or not isinstance(world_frame, str)
        or not world_frame.strip()
    ):
        raise ValueError("ConfigureNBV scene_id/world_frame must be non-empty strings")

    vectors: dict[str, np.ndarray] = {}
    for key in (
        "target_center_m",
        "map_size_m",
        "target_roi_size_m",
        "observation_min_m",
        "observation_max_m",
    ):
        vector = np.asarray(values[key], dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"ConfigureNBV {key} must contain three finite values")
        vectors[key] = vector
    if np.any(vectors["map_size_m"] <= 0.0):
        raise ValueError("ConfigureNBV map_size_m must be positive")
    if np.any(vectors["target_roi_size_m"] <= 0.0) or np.any(
        vectors["target_roi_size_m"] > vectors["map_size_m"]
    ):
        raise ValueError("ConfigureNBV target ROI must be positive and fit in map")
    if np.any(vectors["observation_min_m"] >= vectors["observation_max_m"]):
        raise ValueError("ConfigureNBV observation bounds are invalid")

    def wire_float32(key: str) -> float:
        try:
            result = float(np.float32(values[key]))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"ConfigureNBV {key} is not a scalar") from error
        if not math.isfinite(result):
            raise ValueError(f"ConfigureNBV {key} must be finite float32")
        return result

    voxel_size = wire_float32("voxel_size_m")
    depth_min = wire_float32("depth_min_m")
    depth_max = wire_float32("depth_max_m")
    max_step = wire_float32("max_step_m")
    if voxel_size <= 0.0 or not 0.0 <= depth_min < depth_max:
        raise ValueError("ConfigureNBV voxel/depth interval is invalid")
    if not 0.0 < max_step <= MAX_CAMERA_STEP_M + 1.0e-9:
        raise ValueError("ConfigureNBV max_step_m is outside (0, 5 mm]")

    integers: dict[str, int] = {}
    for key, minimum in (
        ("samples_per_ray", 2),
        ("optimization_steps", 1),
        ("random_seed", 0),
    ):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"ConfigureNBV {key} must be an integer")
        integer = int(value)
        if integer < minimum or integer > 0xFFFFFFFF:
            raise ValueError(f"ConfigureNBV {key} is outside uint32 bounds")
        integers[key] = integer

    ratios = vectors["map_size_m"] / voxel_size
    if not np.all(np.isfinite(ratios)) or np.any(ratios > 20_000_000.0):
        raise ValueError("ConfigureNBV voxel dimensions are invalid")
    dimensions = tuple(int(math.ceil(float(value))) for value in ratios)
    if math.prod(dimensions) > 20_000_000:
        raise ValueError("ConfigureNBV voxel grid exceeds 20M cells")
    origin = (
        vectors["target_center_m"]
        - np.asarray(dimensions, dtype=np.float64) * voxel_size / 2.0
    )
    request = {
        "scene_id": scene_id,
        "world_frame": world_frame,
        **{
            key: [float(item) for item in vectors[key]]
            for key in (
                "target_center_m",
                "map_size_m",
                "target_roi_size_m",
                "observation_min_m",
                "observation_max_m",
            )
        },
        "voxel_size_m": voxel_size,
        "depth_min_m": depth_min,
        "depth_max_m": depth_max,
        "samples_per_ray": integers["samples_per_ray"],
        "optimization_steps": integers["optimization_steps"],
        "max_step_m": max_step,
        "random_seed": integers["random_seed"],
    }
    payload = (
        "strawberry_configure_nbv_request/v1\n"
        + json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    ).encode("utf-8")
    return NBVConfigurationEvidence(
        request=request,
        voxel_dimensions=dimensions,
        map_origin_m=origin,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def five_frame_identity_sha256(
    scene_ids: Sequence[str],
    observation_ids: Sequence[str],
    stamps_ns: Sequence[int],
) -> str:
    """Hash the ordered identities of one fixed, unselected five-frame batch.

    The canonical payload is a compact, key-sorted UTF-8 JSON array of five
    ``scene_id``/``observation_id``/``stamp_ns`` records.  Both preview
    creation and execution-plan loading call this function, so the audit does
    not merely trust a digest-shaped string stored in the artifact.
    """
    scenes = tuple(scene_ids)
    observations = tuple(observation_ids)
    if any(not isinstance(value, str) for value in scenes) or any(
        not isinstance(value, str) for value in observations
    ):
        raise ValueError("five-frame scene and observation IDs must be strings")
    stamps: tuple[int, ...] = tuple(stamps_ns)
    if not (
        len(scenes)
        == len(observations)
        == len(stamps)
        == 5
    ):
        raise ValueError("five-frame identity requires exactly five records")
    if (
        any(not value.strip() for value in scenes + observations)
        or len(set(scenes)) != 5
        or len(set(observations)) != 5
    ):
        raise ValueError("five-frame scene and observation IDs must be unique")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in stamps
    ):
        raise ValueError("five-frame stamps must be integer nanoseconds")
    canonical_stamps = tuple(int(value) for value in stamps)
    if (
        any(value <= 0 for value in canonical_stamps)
        or len(set(canonical_stamps)) != 5
        or tuple(sorted(canonical_stamps)) != canonical_stamps
    ):
        raise ValueError("five-frame stamps must be positive, unique, and increasing")
    records = [
        {
            "observation_id": observation,
            "scene_id": scene,
            "stamp_ns": stamp,
        }
        for scene, observation, stamp in zip(
            scenes, observations, canonical_stamps
        )
    ]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wire_uint8(value: Any, label: str = "uint8") -> int:
    """Decode Jazzy Python uint8 values represented as int or one byte."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        if len(payload) != 1:
            raise ValueError(f"{label} byte representation must have length one")
        return payload[0]
    decoded = int(value)
    if not 0 <= decoded <= 255:
        raise ValueError(f"{label} must be in [0, 255]")
    return decoded


def wire_bool(value: Any, label: str = "bool") -> bool:
    """Decode ROS booleans defensively when a binding yields one byte."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        decoded = wire_uint8(value, label)
        if decoded not in (0, 1):
            raise ValueError(f"{label} byte value must be 0 or 1")
        return decoded == 1
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{label} must be boolean or a zero/one byte")


def stamp_nanoseconds(stamp: Any, label: str = "stamp") -> int:
    """Decode a non-zero, structurally valid ROS-like timestamp."""
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError(f"{label} is not a valid non-negative ROS timestamp")
    result = seconds * 1_000_000_000 + nanoseconds
    if result == 0:
        raise ValueError(f"{label} must not be zero")
    return result


def require_unit_quaternion(
    quaternion_xyzw: Sequence[float],
    label: str = "quaternion",
    tolerance: float = 1.0e-3,
) -> np.ndarray:
    """Reject rather than silently normalize a malformed wire quaternion."""
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{label} must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} norm must be 1 (got {norm:.9g})")
    return quaternion.copy()


def decode_depth_32fc1(image: Any) -> np.ndarray:
    """Decode a ROS-like 32FC1 image, including padded rows and endianness."""
    if str(image.encoding).upper() != "32FC1":
        raise ValueError(f"depth encoding must be 32FC1, got {image.encoding!r}")
    height = int(image.height)
    width = int(image.width)
    step = int(image.step)
    if height <= 0 or width <= 0 or step < width * 4 or step % 4:
        raise ValueError("depth dimensions/step are invalid")
    payload = memoryview(bytes(image.data))
    required = height * step
    if payload.nbytes != required:
        raise ValueError(
            f"depth payload has {payload.nbytes} bytes; expected exactly {required}"
        )
    big_endian = wire_uint8(image.is_bigendian, "Image.is_bigendian") != 0
    dtype = np.dtype(">f4" if big_endian else "<f4")
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(height, step)
    contiguous = rows[:, : width * 4].copy()
    result = contiguous.view(dtype).reshape(height, width).astype(np.float64)
    return result


def decode_mask_mono8(image: Any) -> np.ndarray:
    """Decode a canonical 0/255 mono8 target mask with padded rows."""
    if str(image.encoding).lower() != "mono8":
        raise ValueError(f"mask encoding must be mono8, got {image.encoding!r}")
    height = int(image.height)
    width = int(image.width)
    step = int(image.step)
    if height <= 0 or width <= 0 or step < width:
        raise ValueError("mask dimensions/step are invalid")
    payload = np.frombuffer(bytes(image.data), dtype=np.uint8)
    required = height * step
    if payload.size != required:
        raise ValueError(
            f"mask payload has {payload.size} bytes; expected exactly {required}"
        )
    result = payload.reshape(height, step)[:, :width].copy()
    if not np.all((result == 0) | (result == 255)):
        raise ValueError("mask pixels must be exactly 0 or 255")
    return result


def aggregate_depth_mask(
    depth_frames: Sequence[np.ndarray],
    mask_frames: Sequence[np.ndarray],
    minimum_finite_count: int = 3,
    mask_majority_count: int = 3,
) -> DepthMaskAggregation:
    """Aggregate exactly five registered depth/mask frames pixel by pixel.

    Non-finite depth samples are missing evidence.  A depth pixel is the
    median of its finite samples only when it has the requested support
    (three by default), and is NaN otherwise.  A mask pixel is foreground
    when at least the requested number of the five strict 0/255 masks vote
    for it (three by default).  No input frame is selected or discarded.
    """
    depths = tuple(
        np.asarray(frame, dtype=np.float64) for frame in depth_frames
    )
    masks = tuple(np.asarray(frame) for frame in mask_frames)
    if len(depths) != 5 or len(masks) != 5:
        raise ValueError(
            "depth_frames and mask_frames must each contain exactly 5 frames"
        )
    for value, label in (
        (minimum_finite_count, "minimum_finite_count"),
        (mask_majority_count, "mask_majority_count"),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{label} must be an integer in [1, 5]")
        if not 1 <= int(value) <= 5:
            raise ValueError(f"{label} must be an integer in [1, 5]")

    shape = depths[0].shape
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("aggregation frames must be non-empty 2-D arrays")
    for index, frame in enumerate(depths):
        if frame.ndim != 2 or frame.shape != shape:
            raise ValueError(
                f"depth frame {index} shape {frame.shape} does not match {shape}"
            )
    canonical_masks: list[np.ndarray] = []
    for index, frame in enumerate(masks):
        if frame.ndim != 2 or frame.shape != shape:
            raise ValueError(
                f"mask frame {index} shape {frame.shape} does not match {shape}"
            )
        if not np.all((frame == 0) | (frame == 255)):
            raise ValueError(f"mask frame {index} pixels must be exactly 0 or 255")
        canonical_masks.append(frame.astype(np.uint8, copy=False))

    depth_stack = np.stack(depths, axis=0)
    finite = np.isfinite(depth_stack)
    finite_support = np.sum(finite, axis=0, dtype=np.uint8)

    # Replacing missing samples by +Inf puts every finite value first.  With
    # only five frames, spelling out the five possible finite counts avoids
    # nanmedian warnings and makes the exact even-count median auditable.
    ordered = np.sort(np.where(finite, depth_stack, np.inf), axis=0)
    aggregated_depth = np.full(shape, np.nan, dtype=np.float64)
    for count in range(1, 6):
        selected = finite_support == count
        if not np.any(selected):
            continue
        if count % 2:
            median = ordered[count // 2]
        else:
            lower = ordered[count // 2 - 1]
            upper = ordered[count // 2]
            # Half before addition prevents overflow for two large finite
            # same-sign values while preserving the ordinary median.
            median = lower * 0.5 + upper * 0.5
        aggregated_depth[selected] = median[selected]
    aggregated_depth[finite_support < int(minimum_finite_count)] = np.nan

    mask_stack = np.stack(canonical_masks, axis=0)
    mask_votes = np.sum(mask_stack == 255, axis=0, dtype=np.uint8)
    aggregated_mask = np.where(
        mask_votes >= int(mask_majority_count), 255, 0
    ).astype(np.uint8)

    finite_output = np.isfinite(aggregated_depth)
    pixel_count = int(aggregated_depth.size)
    return DepthMaskAggregation(
        depth_m=aggregated_depth,
        mask=aggregated_mask,
        finite_support=finite_support,
        mask_votes=mask_votes,
        input_frame_count=5,
        pixel_count=pixel_count,
        finite_depth_sample_count=int(
            np.sum(finite_support, dtype=np.int64)
        ),
        output_finite_depth_pixel_count=int(np.count_nonzero(finite_output)),
        output_nan_depth_pixel_count=int(
            pixel_count - np.count_nonzero(finite_output)
        ),
        foreground_mask_sample_count=int(
            np.sum(mask_votes, dtype=np.int64)
        ),
        output_foreground_mask_pixel_count=int(
            np.count_nonzero(aggregated_mask == 255)
        ),
    )


# Keep the fully descriptive spelling available to callers that introduced
# the helper while its public name was being finalized.
aggregate_five_frame_depth_mask = aggregate_depth_mask


def camera_matrix(camera_info: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    """Validate the registered pinhole CameraInfo and return its K matrix."""
    height, width = expected_shape
    if (int(camera_info.height), int(camera_info.width)) != (height, width):
        raise ValueError("CameraInfo grid does not match depth/mask")
    intrinsic = np.asarray(camera_info.k, dtype=float)
    if intrinsic.shape != (9,) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("CameraInfo.k must contain nine finite values")
    intrinsic = intrinsic.reshape(3, 3)
    if (
        intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1.0e-9)
    ):
        raise ValueError("CameraInfo.k is not a valid pinhole matrix")
    distortion = np.asarray(camera_info.d, dtype=float)
    if not np.all(np.isfinite(distortion)) or not np.all(distortion == 0.0):
        raise ValueError("canonical CameraInfo distortion must be finite and zero")
    return intrinsic


def estimate_target_center(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    *,
    depth_min_m: float = 0.20,
    depth_max_m: float = 2.50,
    minimum_pixels: int = 200,
) -> TargetEstimate:
    """Robustly back-project the masked target centre in optical coordinates.

    A median/MAD depth trim removes isolated mask or depth-edge contamination;
    the retained 3-D points are then reduced component-wise by median.
    """
    depth = np.asarray(depth_m, dtype=float)
    target_mask = np.asarray(mask)
    k = np.asarray(intrinsic, dtype=float)
    if depth.ndim != 2 or target_mask.shape != depth.shape:
        raise ValueError("depth and mask must be same-sized 2-D arrays")
    if k.shape != (3, 3) or not np.all(np.isfinite(k)):
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    if (
        not math.isfinite(depth_min_m)
        or not math.isfinite(depth_max_m)
        or not 0.0 <= depth_min_m < depth_max_m
    ):
        raise ValueError("depth bounds are invalid")
    if isinstance(minimum_pixels, bool) or minimum_pixels < 1:
        raise ValueError("minimum_pixels must be positive")
    is_target = target_mask == 255
    mask_pixels = int(np.count_nonzero(is_target))
    if mask_pixels < minimum_pixels:
        raise ValueError(
            f"target mask has {mask_pixels} pixels; need at least {minimum_pixels}"
        )
    valid = (
        is_target
        & np.isfinite(depth)
        & (depth >= depth_min_m)
        & (depth <= depth_max_m)
    )
    rows, columns = np.nonzero(valid)
    values = depth[rows, columns]
    valid_count = int(values.size)
    if valid_count < minimum_pixels:
        raise ValueError(
            f"only {valid_count} target pixels have valid depth; "
            f"need at least {minimum_pixels}"
        )
    median_depth = float(np.median(values))
    mad = float(np.median(np.abs(values - median_depth)))
    trim_radius = max(3.0 * 1.4826 * mad, 0.005)
    retained = np.abs(values - median_depth) <= trim_radius
    if int(np.count_nonzero(retained)) < minimum_pixels:
        raise ValueError("robust target-depth trim retained too few pixels")
    rows = rows[retained].astype(float)
    columns = columns[retained].astype(float)
    z = values[retained]
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsic focal lengths must be positive")
    points = np.column_stack(
        ((columns - cx) * z / fx, (rows - cy) * z / fy, z)
    )
    centre = np.median(points, axis=0)
    if not np.all(np.isfinite(centre)) or centre[2] <= 0.0:
        raise ValueError("estimated target centre is invalid")
    return TargetEstimate(
        camera_xyz_m=centre,
        mask_pixels=mask_pixels,
        valid_mask_pixels=valid_count,
        retained_pixels=int(points.shape[0]),
        median_depth_m=median_depth,
        depth_mad_m=mad,
    )


def transform_point(
    transform_parent_child: np.ndarray,
    point_child: Sequence[float],
) -> np.ndarray:
    """Transform one finite 3-D point after rigid-transform validation."""
    transform = validate_rigid_transform(transform_parent_child, "transform")
    point = np.asarray(point_child, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point must contain three finite values")
    return transform[:3, :3] @ point + transform[:3, 3]


def camera_motion(current: np.ndarray, target: np.ndarray) -> CameraMotion:
    """Return Euclidean translation and shortest SO(3) rotation."""
    first = validate_rigid_transform(current, "current_camera_pose")
    second = validate_rigid_transform(target, "target_camera_pose")
    translation = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation = float(math.acos(cosine))
    return CameraMotion(translation, rotation)


def compare_gradient_candidates(
    artifact_current: np.ndarray,
    artifact_target: np.ndarray,
    live_current: np.ndarray,
    live_target: np.ndarray,
) -> GradientCandidateConsistency:
    """Measure repeatability without treating it as frozen-target safety.

    Two independently optimized translations can point in different directions
    while both remain inside the same bounded search ball.  Their endpoint
    separation is therefore diagnostic; motion safety must instead validate
    the one frozen endpoint from the actual current pose.
    """
    artifact_start = validate_rigid_transform(
        artifact_current, "artifact_current"
    )
    artifact_end = validate_rigid_transform(artifact_target, "artifact_target")
    live_start = validate_rigid_transform(live_current, "live_current")
    live_end = validate_rigid_transform(live_target, "live_target")
    start_motion = camera_motion(artifact_start, live_start)
    target_motion = camera_motion(artifact_end, live_end)
    artifact_vector = artifact_end[:3, 3] - artifact_start[:3, 3]
    live_vector = live_end[:3, 3] - live_start[:3, 3]
    artifact_step = float(np.linalg.norm(artifact_vector))
    live_step = float(np.linalg.norm(live_vector))
    if artifact_step <= _EPS or live_step <= _EPS:
        raise ValueError("candidate translations must both be non-zero")
    cosine = float(
        np.clip(
            np.dot(artifact_vector, live_vector) / (artifact_step * live_step),
            -1.0,
            1.0,
        )
    )
    direction_disagreement = float(math.acos(cosine))
    # Triangle inequality: different bounded directions may be separated by
    # both step lengths plus the measured difference in their start points.
    bound = artifact_step + start_motion.translation_m + live_step
    if target_motion.translation_m > bound + 1.0e-9:
        raise ValueError("candidate endpoint separation violates triangle bound")
    return GradientCandidateConsistency(
        start_motion=start_motion,
        target_motion=target_motion,
        artifact_step_m=artifact_step,
        live_step_m=live_step,
        translation_direction_disagreement_rad=direction_disagreement,
        triangle_bound_m=bound,
    )


def validate_raw_next_view(
    current: np.ndarray,
    target: np.ndarray,
    gain: float,
    *,
    max_step_m: float = MAX_CAMERA_STEP_M,
) -> CameraMotion:
    """Validate the raw Gradient-NBV result and its strict-gain proxy.

    The public NextView message exposes planned gain, not current gain.  The
    Gradient-NBV core only changes position after a strict gain improvement.
    Therefore finite positive gain plus a non-deadband translation is recorded
    as a *core-invariant inference*, never as a measured numerical difference.
    """
    planned_gain = float(gain)
    if not math.isfinite(planned_gain) or planned_gain <= 0.0:
        raise ValueError("NextView planned gain must be finite and positive")
    motion = camera_motion(current, target)
    if motion.translation_m <= MIN_CAMERA_STEP_M + _EPS:
        raise ValueError(
            "raw NextView camera translation must exceed the 1 mm deadband"
        )
    if motion.translation_m > float(max_step_m) + 1.0e-9:
        raise ValueError(
            f"raw NextView camera translation {motion.translation_m:.9f} m "
            f"exceeds {float(max_step_m):.9f} m"
        )
    return motion


def segmented_camera_candidates(
    current: np.ndarray,
    raw_target: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> tuple[tuple[float, np.ndarray, CameraMotion], ...]:
    """Build audited translation+slerp candidates toward one raw NBV pose."""
    if not alphas:
        raise ValueError("at least one alpha is required")
    previous = math.inf
    output = []
    for value in alphas:
        alpha = float(value)
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("candidate alphas must be finite and in (0, 1]")
        if alpha >= previous:
            raise ValueError("candidate alphas must be strictly descending")
        previous = alpha
        candidate = interpolate_transform(current, raw_target, alpha)
        output.append((alpha, candidate, camera_motion(current, candidate)))
    return tuple(output)


def ordered_joint_positions(
    names: Sequence[str],
    positions: Sequence[float],
    *,
    label: str,
) -> np.ndarray:
    """Require exactly one complete joint1..joint7 set and return that order."""
    if len(names) != len(set(names)) or len(names) != len(positions):
        raise ValueError(f"{label} has duplicate names or inconsistent lengths")
    mapping = dict(zip(names, positions))
    if set(mapping) != set(NERO_JOINT_NAMES):
        raise ValueError(f"{label} must contain exactly joint1 through joint7")
    ordered = np.asarray([mapping[name] for name in NERO_JOINT_NAMES], dtype=float)
    if not np.all(np.isfinite(ordered)):
        raise ValueError(f"{label} positions must be finite")
    return ordered


def validate_ik_solution(
    *,
    current_joints: Sequence[float],
    solution_names: Sequence[str],
    solution_positions: Sequence[float],
    reported_max_joint_delta_rad: float,
    position_error_m: float,
    orientation_error_rad: float,
    sigma_min: float,
    condition_number: float,
    max_joint_delta_rad: float = MAX_IK_JOINT_DELTA_RAD,
    reported_delta_tolerance_rad: float = 1.0e-4,
) -> IKValidation:
    """Apply independent precision, continuity, and singularity gates."""
    current = np.asarray(current_joints, dtype=float)
    if current.shape != (7,) or not np.all(np.isfinite(current)):
        raise ValueError("current_joints must contain seven finite values")
    solution = ordered_joint_positions(
        solution_names, solution_positions, label="IK solution"
    )
    values = np.asarray(
        (
            reported_max_joint_delta_rad,
            position_error_m,
            orientation_error_rad,
            sigma_min,
            condition_number,
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("IK diagnostics must all be finite")
    reported, position_error, orientation_error, sigma, condition = values
    if np.any(values[:3] < 0.0) or sigma < 0.0 or condition < 0.0:
        raise ValueError("IK diagnostics cannot be negative")
    independent_delta = float(np.max(np.abs(solution - current)))
    limit = float(max_joint_delta_rad)
    if reported > limit + 1.0e-9 or independent_delta > limit + 1.0e-9:
        raise ValueError(
            "IK joint delta exceeds limit: "
            f"reported={reported:.6f}, independent={independent_delta:.6f}, "
            f"limit={limit:.6f} rad"
        )
    tolerance = float(reported_delta_tolerance_rad)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "reported_delta_tolerance_rad must be finite and non-negative"
        )
    if abs(reported - independent_delta) > tolerance:
        raise ValueError(
            "reported max_joint_delta_rad does not match the returned solution"
        )
    if position_error > MAX_IK_POSITION_ERROR_M + 1.0e-12:
        raise ValueError("IK position residual exceeds 3 mm")
    if orientation_error > MAX_IK_ORIENTATION_ERROR_RAD + 1.0e-12:
        raise ValueError("IK orientation residual exceeds 2 degrees")
    if sigma < MIN_IK_SIGMA - 1.0e-12:
        raise ValueError("IK sigma_min is below 0.10")
    if condition > MAX_IK_CONDITION + 1.0e-12:
        raise ValueError("IK condition_number exceeds 20")
    return IKValidation(
        max_joint_delta_rad=independent_delta,
        reported_max_joint_delta_rad=float(reported),
        position_error_m=float(position_error),
        orientation_error_rad=float(orientation_error),
        sigma_min=float(sigma),
        condition_number=float(condition),
    )


def array_list(value: np.ndarray) -> list[Any]:
    """Convert a finite NumPy array to a JSON-safe nested list."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("cannot serialize non-finite array")
    return array.tolist()
