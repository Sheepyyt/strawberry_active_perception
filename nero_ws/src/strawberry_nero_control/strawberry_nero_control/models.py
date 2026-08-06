"""Pure-Python data models shared by NERO IK and trajectory code."""

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Tuple


NERO_JOINT_NAMES: Tuple[str, ...] = tuple(
    f'joint{index}' for index in range(1, 8)
)

# Safe, tested pose supplied by the project owner.  Values are radians and
# follow NERO's joint1 ... joint7 ordering.
READY_JOINT_POSITIONS: Tuple[float, ...] = (
    0.0,
    -1.2698666571660342,
    0.0,
    1.8844843532558375,
    0.0,
    -0.00003490658503988659,
    0.0,
)

# pyAgxArm's NERO preset limits.  IK uses their intersection with the URDF
# limits, then shrinks both ends by IKConfig.joint_margin_rad.
NERO_SDK_JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-2.705261, 2.705261),
    (-1.745330, 1.745330),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
)

# The checked-in NERO URDF is slightly more restrictive than the SDK preset
# for every differing endpoint.  This constant lets the standalone trajectory
# generator remain safe even when it is not constructed from a solver.
NERO_URDF_JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-2.705260, 2.705260),
    (-1.740000, 1.740000),
    (-2.750000, 2.750000),
    (-1.010000, 2.140000),
    (-2.750000, 2.750000),
    (-0.730000, 0.950000),
    (-1.5707963, 1.5707963),
)


class IKErrorCode(IntEnum):
    """IK status values kept identical to ``IKResult.msg`` constants."""

    SUCCESS = 0
    ALREADY_AT_TARGET = 1
    INVALID_TARGET = 10
    TF_UNAVAILABLE = 11
    FEEDBACK_STALE = 12
    UNREACHABLE = 20
    JOINT_LIMIT_VIOLATION = 21
    NEAR_SINGULARITY = 22
    DISCONTINUOUS_SOLUTION = 23
    JOINT_DELTA_TOO_LARGE = 24
    TRAJECTORY_LIMIT_VIOLATION = 30
    TRACKING_ERROR = 31
    TIMEOUT = 32
    CANCELED = 33
    DRIVER_FAULT = 34
    INTERNAL_ERROR = 255


class TrajectoryErrorCode(IntEnum):
    """Structured trajectory generation status."""

    SUCCESS = 0
    INVALID_INPUT = 1
    JOINT_LIMIT = 2
    DURATION_TOO_SHORT = 3
    DURATION_TOO_LONG = 4
    NUMERICAL_ERROR = 5


@dataclass(frozen=True)
class IKConfig:
    """Numerical and safety settings for one Placo IK request."""

    position_weight: float = 1.0
    orientation_weight: float = 0.3
    posture_weight: float = 0.001
    regularization: float = 1.0e-6
    max_iterations: int = 200
    timeout_s: float = 0.020
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = math.radians(2.0)
    position_deadband_m: float = 0.001
    orientation_deadband_rad: float = math.radians(0.5)
    joint_margin_rad: float = math.radians(2.0)
    solver_dt_s: float = 0.020
    solver_velocity_limit_rad_s: float = 0.30
    max_joint_delta_rad: float = 0.35
    singular_sigma_min: float = 0.03
    singular_condition_max: float = 100.0


@dataclass(frozen=True)
class SingularityMetrics:
    """Jacobian conditioning at one robot configuration."""

    sigma_min: float = 0.0
    condition_number: float = math.inf


@dataclass(frozen=True)
class IKResult:
    """ROS-independent result returned by :class:`PlacoIKSolver`."""

    success: bool
    error_code: IKErrorCode
    message: str
    joint_positions: Tuple[float, ...] = field(default_factory=tuple)
    position_error_m: float = math.inf
    orientation_error_rad: float = math.inf
    solve_time_ms: float = 0.0
    iterations: int = 0
    sigma_min: float = 0.0
    condition_number: float = math.inf
    max_joint_delta_rad: float = math.inf


@dataclass(frozen=True)
class TrajectoryConfig:
    """Sampling and motion limits for a minimum-jerk trajectory."""

    frequency_hz: float = 50.0
    max_velocity_rad_s: float = 0.30
    max_acceleration_rad_s2: float = 0.50
    min_duration_s: float = 0.20
    max_duration_s: float = 30.0


@dataclass(frozen=True)
class TrajectoryPoint:
    """One complete seven-joint trajectory sample."""

    time_from_start_s: float
    positions: Tuple[float, ...]
    velocities: Tuple[float, ...]
    accelerations: Tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryResult:
    """ROS-independent output from :class:`TrajectoryGenerator`."""

    success: bool
    error_code: TrajectoryErrorCode
    message: str
    points: Tuple[TrajectoryPoint, ...] = field(default_factory=tuple)
    duration_s: float = 0.0
    peak_velocity_rad_s: float = 0.0
    peak_acceleration_rad_s2: float = 0.0
