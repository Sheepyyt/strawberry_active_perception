"""Continuous, safety-checked NERO inverse kinematics using Placo only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import threading
import time
from typing import Any, Optional

import numpy as np

from .models import (
    IKConfig,
    IKErrorCode,
    IKResult,
    NERO_JOINT_NAMES,
    NERO_SDK_JOINT_LIMITS,
    SingularityMetrics,
)

try:
    import placo
except ImportError:  # pragma: no cover - exercised only outside sap-core.
    placo = None


def transform_from_pose(
    position_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """
    Build a homogeneous transform from a position and normalized quaternion.

    The quaternion is normalized here instead of trusting a ROS caller.  Both
    ``q`` and ``-q`` produce the same rotation, so the representation cannot
    introduce a long-way-around orientation discontinuity.
    """
    position = np.asarray(position_xyz, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError('position_xyz must contain three finite values')
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError('quaternion_xyzw must contain four finite values')

    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError('quaternion norm must be greater than zero')
    x, y, z, w = quaternion / norm
    rotation = np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _orientation_error(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Return the shortest angular distance between two rotation matrices."""
    relative = rotation_a.T @ rotation_b
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


class PlacoIKSolver:
    """Solve NERO end-effector poses while preserving solution continuity."""

    def __init__(
        self,
        urdf_path: str | Path,
        config: Optional[IKConfig] = None,
        sdk_joint_limits: Optional[
            Mapping[str, Sequence[float]] | Sequence[Sequence[float]]
        ] = None,
    ) -> None:
        """Load NERO and configure conservative joint and velocity limits."""
        if placo is None:
            raise RuntimeError(
                'Placo is unavailable; activate the sap-core environment'
            )
        self.config = config or IKConfig()
        self._validate_config(self.config)
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f'NERO URDF not found: {self.urdf_path}')
        self.joint_names = NERO_JOINT_NAMES
        self._lock = threading.RLock()
        self.robot = self._load_robot(self.urdf_path)

        actual_names = tuple(str(name) for name in self.robot.joint_names())
        if actual_names != self.joint_names:
            raise ValueError(
                'NERO URDF joint order mismatch: '
                f'expected {self.joint_names}, got {actual_names}'
            )
        if 'link7' not in tuple(str(name) for name in self.robot.frame_names()):
            raise ValueError('NERO URDF does not contain the link7 frame')

        sdk_limits = self._normalise_sdk_limits(sdk_joint_limits)
        safe_limits = []
        for index, joint_name in enumerate(self.joint_names):
            urdf_lower, urdf_upper = np.asarray(
                self.robot.get_joint_limits(joint_name), dtype=float
            )
            sdk_lower, sdk_upper = sdk_limits[index]
            lower = max(float(urdf_lower), sdk_lower)
            upper = min(float(urdf_upper), sdk_upper)
            lower += self.config.joint_margin_rad
            upper -= self.config.joint_margin_rad
            if lower >= upper:
                raise ValueError(
                    f'empty safe interval for {joint_name}: '
                    f'[{lower}, {upper}]'
                )
            safe_limits.append((lower, upper))
            self.robot.set_joint_limits(joint_name, lower, upper)
            self.robot.set_velocity_limit(
                joint_name,
                self.config.solver_velocity_limit_rad_s,
            )
        self._safe_joint_limits = np.asarray(safe_limits, dtype=float)

        # Placo creates a floating base even for this fixed-base URDF.  Keep it
        # at identity, and every request masks its six solver DoFs.
        self.robot.set_T_world_fbase(np.eye(4, dtype=float))
        self.robot.update_kinematics()

    @staticmethod
    def _validate_config(config: IKConfig) -> None:
        """Reject settings that could disable a safety invariant."""
        positive_values = (
            config.position_weight,
            config.orientation_weight,
            config.posture_weight,
            config.regularization,
            config.timeout_s,
            config.position_tolerance_m,
            config.orientation_tolerance_rad,
            config.position_deadband_m,
            config.orientation_deadband_rad,
            config.joint_margin_rad,
            config.solver_dt_s,
            config.solver_velocity_limit_rad_s,
            config.max_joint_delta_rad,
            config.singular_sigma_min,
            config.singular_condition_max,
        )
        if not all(np.isfinite(value) and value > 0.0
                   for value in positive_values):
            raise ValueError('all IKConfig numeric limits must be positive')
        if config.max_iterations <= 0:
            raise ValueError('max_iterations must be positive')
        if config.position_deadband_m > config.position_tolerance_m:
            raise ValueError('position deadband cannot exceed tolerance')
        if (
            config.orientation_deadband_rad
            > config.orientation_tolerance_rad
        ):
            raise ValueError('orientation deadband cannot exceed tolerance')

    @staticmethod
    def _load_robot(urdf_path: Path) -> Any:
        """Load a URDF and resolve its AgileX package URI without ROS APIs."""
        urdf_content = urdf_path.read_text(encoding='utf-8')
        package_token = 'package://agx_arm_description/'
        if package_token in urdf_content:
            package_root = next(
                (
                    parent for parent in urdf_path.parents
                    if parent.name == 'agx_arm_description'
                ),
                None,
            )
            if package_root is None:
                raise ValueError(
                    'cannot resolve package://agx_arm_description in URDF'
                )
            replacement = f'file://{package_root.as_posix()}/'
            urdf_content = urdf_content.replace(
                package_token,
                replacement,
            )
        return placo.RobotWrapper(
            str(urdf_path),
            placo.Flags.ignore_collisions,
            urdf_content,
        )

    def _normalise_sdk_limits(
        self,
        limits: Optional[
            Mapping[str, Sequence[float]] | Sequence[Sequence[float]]
        ],
    ) -> tuple[tuple[float, float], ...]:
        """Convert optional SDK limits to joint-order tuples."""
        if limits is None:
            values = NERO_SDK_JOINT_LIMITS
        elif isinstance(limits, Mapping):
            missing = set(self.joint_names).difference(limits)
            if missing:
                raise ValueError(
                    f'SDK limits missing joints: {sorted(missing)}'
                )
            values = tuple(limits[name] for name in self.joint_names)
        else:
            values = tuple(limits)
        array = np.asarray(values, dtype=float)
        if array.shape != (7, 2) or not np.all(np.isfinite(array)):
            raise ValueError('SDK limits must be a finite 7 by 2 array')
        if np.any(array[:, 0] >= array[:, 1]):
            raise ValueError('every SDK lower limit must be below its upper')
        return tuple((float(row[0]), float(row[1])) for row in array)

    @property
    def safe_joint_limits(self) -> np.ndarray:
        """Return a defensive copy of the seven conservative joint ranges."""
        return self._safe_joint_limits.copy()

    def _joint_vector(self, joints: Sequence[float]) -> np.ndarray:
        """Validate and copy one complete NERO joint vector."""
        vector = np.asarray(joints, dtype=float)
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise ValueError('joint vector must contain seven finite values')
        return vector.copy()

    def _set_joint_vector(self, joints: np.ndarray) -> None:
        """Set Placo state from feedback, including zero measured velocity."""
        self.robot.set_T_world_fbase(np.eye(4, dtype=float))
        for joint_name, value in zip(self.joint_names, joints):
            self.robot.set_joint(joint_name, float(value))
            self.robot.set_joint_velocity(joint_name, 0.0)
        self.robot.update_kinematics()

    def _get_joint_vector(self) -> np.ndarray:
        """Read the seven actuated joints from Placo in driver order."""
        return np.asarray([
            self.robot.get_joint(name) for name in self.joint_names
        ], dtype=float)

    def _validate_frame(self, controlled_frame: str) -> None:
        """Ensure the requested controlled frame is in the loaded model."""
        if not isinstance(controlled_frame, str) or not controlled_frame:
            raise ValueError('controlled_frame must be a non-empty string')
        frames = tuple(str(name) for name in self.robot.frame_names())
        if controlled_frame not in frames:
            raise ValueError(
                f'controlled frame is absent from URDF: {controlled_frame}'
            )

    @staticmethod
    def _target_transform(target_transform: np.ndarray) -> np.ndarray:
        """Validate one rigid homogeneous transform."""
        target = np.asarray(target_transform, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError('target transform must be a finite 4 by 4 array')
        if not np.allclose(target[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError('target transform has an invalid final row')
        rotation = target[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError('target orientation is not orthonormal')
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError('target orientation determinant is not +1')
        return target.copy()

    def _metrics_from_current_state(
        self,
        controlled_frame: str,
    ) -> SingularityMetrics:
        """Compute six-dimensional Jacobian conditioning for seven joints."""
        jacobian = np.asarray(
            self.robot.frame_jacobian(
                controlled_frame,
                'local_world_aligned',
            ),
            dtype=float,
        )
        columns = [
            self.robot.get_joint_v_offset(name)
            for name in self.joint_names
        ]
        actuated_jacobian = jacobian[:, columns]
        singular_values = np.linalg.svd(
            actuated_jacobian,
            compute_uv=False,
        )
        if singular_values.size < 6 or not np.all(np.isfinite(singular_values)):
            return SingularityMetrics()
        sigma_min = float(singular_values[-1])
        if sigma_min <= np.finfo(float).eps:
            condition = float('inf')
        else:
            condition = float(singular_values[0] / sigma_min)
        return SingularityMetrics(sigma_min, condition)

    def configuration_metrics(
        self,
        joints: Sequence[float],
        controlled_frame: str = 'link7',
    ) -> SingularityMetrics:
        """Evaluate singularity metrics without running inverse kinematics."""
        with self._lock:
            vector = self._joint_vector(joints)
            self._validate_frame(controlled_frame)
            self._set_joint_vector(vector)
            return self._metrics_from_current_state(controlled_frame)

    def forward_kinematics(
        self,
        joints: Sequence[float],
        controlled_frame: str = 'link7',
    ) -> np.ndarray:
        """Return ``T_world_frame`` for one complete joint vector."""
        with self._lock:
            vector = self._joint_vector(joints)
            self._validate_frame(controlled_frame)
            self._set_joint_vector(vector)
            return np.asarray(
                self.robot.get_T_world_frame(controlled_frame),
                dtype=float,
            ).copy()

    def _is_near_singular(self, metrics: SingularityMetrics) -> bool:
        """Apply both configured singularity rejection tests."""
        return (
            metrics.sigma_min < self.config.singular_sigma_min
            or metrics.condition_number > self.config.singular_condition_max
        )

    def _pose_error(
        self,
        controlled_frame: str,
        target: np.ndarray,
    ) -> tuple[float, float]:
        """Measure current frame position and shortest orientation error."""
        current = np.asarray(
            self.robot.get_T_world_frame(controlled_frame),
            dtype=float,
        )
        position_error = float(
            np.linalg.norm(current[:3, 3] - target[:3, 3])
        )
        orientation_error = _orientation_error(
            current[:3, :3],
            target[:3, :3],
        )
        return position_error, orientation_error

    def _result(
        self,
        success: bool,
        error_code: IKErrorCode,
        message: str,
        joints: Sequence[float] = (),
        position_error_m: float = float('inf'),
        orientation_error_rad: float = float('inf'),
        solve_time_ms: float = 0.0,
        iterations: int = 0,
        metrics: Optional[SingularityMetrics] = None,
        max_joint_delta_rad: float = float('inf'),
    ) -> IKResult:
        """Construct one consistently typed IK result."""
        metrics = metrics or SingularityMetrics()
        return IKResult(
            success=success,
            error_code=error_code,
            message=message,
            joint_positions=tuple(float(value) for value in joints),
            position_error_m=float(position_error_m),
            orientation_error_rad=float(orientation_error_rad),
            solve_time_ms=float(solve_time_ms),
            iterations=int(iterations),
            sigma_min=float(metrics.sigma_min),
            condition_number=float(metrics.condition_number),
            max_joint_delta_rad=float(max_joint_delta_rad),
        )

    def solve(
        self,
        target_transform: np.ndarray,
        current_joints: Sequence[float],
        controlled_frame: str = 'link7',
    ) -> IKResult:
        """
        Solve one target from fresh, measured joint feedback.

        No previous virtual solution is used as an initial condition.  A soft
        posture task anchors the redundant seven-DoF arm to the measured
        branch, while joint/velocity limits and final continuity checks reject
        unsafe alternatives.
        """
        with self._lock:
            try:
                current = self._joint_vector(current_joints)
                target = self._target_transform(target_transform)
                self._validate_frame(controlled_frame)
            except (TypeError, ValueError) as error:
                return self._result(
                    False,
                    IKErrorCode.INVALID_TARGET,
                    str(error),
                )

            below = current < self._safe_joint_limits[:, 0]
            above = current > self._safe_joint_limits[:, 1]
            if np.any(below | above):
                indices = np.flatnonzero(below | above)
                names = ', '.join(self.joint_names[index] for index in indices)
                return self._result(
                    False,
                    IKErrorCode.JOINT_LIMIT_VIOLATION,
                    f'current feedback is outside safe limits: {names}',
                    current,
                )

            self._set_joint_vector(current)
            initial_metrics = self._metrics_from_current_state(
                controlled_frame
            )
            initial_position_error, initial_orientation_error = (
                self._pose_error(controlled_frame, target)
            )
            if self._is_near_singular(initial_metrics):
                return self._result(
                    False,
                    IKErrorCode.NEAR_SINGULARITY,
                    'current configuration is too close to a singularity',
                    current,
                    initial_position_error,
                    initial_orientation_error,
                    metrics=initial_metrics,
                    max_joint_delta_rad=0.0,
                )
            if (
                initial_position_error <= self.config.position_deadband_m
                and initial_orientation_error
                <= self.config.orientation_deadband_rad
            ):
                return self._result(
                    True,
                    IKErrorCode.ALREADY_AT_TARGET,
                    'target change is inside the Cartesian deadband',
                    current,
                    initial_position_error,
                    initial_orientation_error,
                    metrics=initial_metrics,
                    max_joint_delta_rad=0.0,
                )

            solver = placo.KinematicsSolver(self.robot)
            solver.mask_fbase(True)
            solver.dt = self.config.solver_dt_s
            solver.enable_joint_limits(True)
            solver.enable_velocity_limits(True)

            frame_task = solver.add_frame_task(controlled_frame, target)
            frame_task.configure(
                'end_effector_pose',
                'soft',
                self.config.position_weight,
                self.config.orientation_weight,
            )
            posture_task = solver.add_joints_task()
            posture_task.set_joints(dict(zip(self.joint_names, current)))
            posture_task.configure(
                'measured_posture_anchor',
                'soft',
                self.config.posture_weight,
            )
            solver.add_regularization_task(self.config.regularization)

            start_time = time.perf_counter()
            iterations = 0
            timed_out = False
            position_error = initial_position_error
            orientation_error = initial_orientation_error
            try:
                for iterations in range(1, self.config.max_iterations + 1):
                    if time.perf_counter() - start_time >= self.config.timeout_s:
                        timed_out = True
                        break
                    solver.solve(True)
                    self.robot.update_kinematics()
                    position_error, orientation_error = self._pose_error(
                        controlled_frame,
                        target,
                    )
                    elapsed = time.perf_counter() - start_time
                    if elapsed > self.config.timeout_s:
                        timed_out = True
                        break
                    if (
                        position_error <= self.config.position_tolerance_m
                        and orientation_error
                        <= self.config.orientation_tolerance_rad
                    ):
                        break
            except Exception as error:  # Placo exposes C++ solver exceptions.
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                candidate = self._get_joint_vector()
                return self._result(
                    False,
                    IKErrorCode.INTERNAL_ERROR,
                    f'Placo solver failed: {error}',
                    candidate,
                    position_error,
                    orientation_error,
                    elapsed_ms,
                    iterations,
                    initial_metrics,
                    float(np.max(np.abs(candidate - current))),
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            candidate = self._get_joint_vector()
            max_delta = float(np.max(np.abs(candidate - current)))
            final_metrics = self._metrics_from_current_state(controlled_frame)
            if timed_out:
                return self._result(
                    False,
                    IKErrorCode.TIMEOUT,
                    f'IK exceeded {self.config.timeout_s * 1000.0:.1f} ms',
                    candidate,
                    position_error,
                    orientation_error,
                    elapsed_ms,
                    iterations,
                    final_metrics,
                    max_delta,
                )
            if (
                position_error > self.config.position_tolerance_m
                or orientation_error > self.config.orientation_tolerance_rad
            ):
                return self._result(
                    False,
                    IKErrorCode.UNREACHABLE,
                    'IK residual exceeds 2 mm or 2 degrees',
                    candidate,
                    position_error,
                    orientation_error,
                    elapsed_ms,
                    iterations,
                    final_metrics,
                    max_delta,
                )
            outside_lower = np.any(
                candidate < self._safe_joint_limits[:, 0] - 1.0e-9
            )
            outside_upper = np.any(
                candidate > self._safe_joint_limits[:, 1] + 1.0e-9
            )
            if outside_lower or outside_upper:
                return self._result(
                    False,
                    IKErrorCode.JOINT_LIMIT_VIOLATION,
                    'Placo candidate violates a conservative joint limit',
                    candidate,
                    position_error,
                    orientation_error,
                    elapsed_ms,
                    iterations,
                    final_metrics,
                    max_delta,
                )
            if self._is_near_singular(final_metrics):
                return self._result(
                    False,
                    IKErrorCode.NEAR_SINGULARITY,
                    'IK candidate is too close to a singularity',
                    candidate,
                    position_error,
                    orientation_error,
                    elapsed_ms,
                    iterations,
                    final_metrics,
                    max_delta,
                )
            if max_delta > self.config.max_joint_delta_rad:
                return self._result(
                    False,
                    IKErrorCode.JOINT_DELTA_TOO_LARGE,
                    'IK candidate changes at least one joint by more than '
                    f'{self.config.max_joint_delta_rad:.3f} rad',
                    candidate,
                    position_error,
                    orientation_error,
                    elapsed_ms,
                    iterations,
                    final_metrics,
                    max_delta,
                )
            return self._result(
                True,
                IKErrorCode.SUCCESS,
                'IK solution accepted',
                candidate,
                position_error,
                orientation_error,
                elapsed_ms,
                iterations,
                final_metrics,
                max_delta,
            )
