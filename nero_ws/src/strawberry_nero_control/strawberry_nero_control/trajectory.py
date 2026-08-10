"""Limit-aware 50 Hz minimum-jerk trajectories for all seven NERO joints."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Optional

import numpy as np

from .models import (
    IKErrorCode,
    JointLimitRecoveryConfig,
    JointLimitRecoveryPlan,
    NERO_SDK_JOINT_LIMITS,
    NERO_URDF_JOINT_LIMITS,
    TrajectoryConfig,
    TrajectoryErrorCode,
    TrajectoryPoint,
    TrajectoryResult,
)


# Exact extrema of s'(u) and |s''(u)| for
# s(u) = 10u^3 - 15u^4 + 6u^5, u in [0, 1].
_MINIMUM_JERK_PEAK_VELOCITY = 1.875
_MINIMUM_JERK_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)


class TrajectoryGenerator:
    """Generate smooth, stationary-endpoint seven-joint trajectories."""

    def __init__(
        self,
        config: Optional[TrajectoryConfig] = None,
        joint_limits: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Configure sampling, motion bounds and conservative joint limits."""
        self.config = config or TrajectoryConfig()
        self._validate_config(self.config)
        if joint_limits is None:
            urdf = np.asarray(NERO_URDF_JOINT_LIMITS, dtype=float)
            sdk = np.asarray(NERO_SDK_JOINT_LIMITS, dtype=float)
            margin = math.radians(2.0)
            limits = np.column_stack((
                np.maximum(urdf[:, 0], sdk[:, 0]) + margin,
                np.minimum(urdf[:, 1], sdk[:, 1]) - margin,
            ))
        else:
            limits = np.asarray(joint_limits, dtype=float)
        if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
            raise ValueError('joint_limits must be a finite 7 by 2 array')
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError('joint lower limits must be below upper limits')
        self._joint_limits = limits.copy()

    @staticmethod
    def _validate_config(config: TrajectoryConfig) -> None:
        """Ensure every trajectory timing and motion bound is usable."""
        values = (
            config.frequency_hz,
            config.max_velocity_rad_s,
            config.max_acceleration_rad_s2,
            config.min_duration_s,
            config.max_duration_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError('trajectory configuration must be positive')
        if config.min_duration_s > config.max_duration_s:
            raise ValueError('minimum duration cannot exceed maximum duration')

    @property
    def joint_limits(self) -> np.ndarray:
        """Return a defensive copy of trajectory position bounds."""
        return self._joint_limits.copy()

    @staticmethod
    def _vector(values: Sequence[float]) -> np.ndarray:
        """Validate one complete seven-joint position vector."""
        vector = np.asarray(values, dtype=float)
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise ValueError('joint vector must contain seven finite values')
        return vector.copy()

    def _failure(
        self,
        code: TrajectoryErrorCode,
        message: str,
        duration_s: float = 0.0,
        peak_velocity_rad_s: float = 0.0,
        peak_acceleration_rad_s2: float = 0.0,
    ) -> TrajectoryResult:
        """Construct a trajectory refusal with no executable points."""
        return TrajectoryResult(
            success=False,
            error_code=code,
            message=message,
            duration_s=float(duration_s),
            peak_velocity_rad_s=float(peak_velocity_rad_s),
            peak_acceleration_rad_s2=float(
                peak_acceleration_rad_s2
            ),
        )

    def _required_duration(self, displacement: np.ndarray) -> float:
        """Find the shortest duration satisfying exact quintic extrema."""
        largest_delta = float(np.max(np.abs(displacement)))
        velocity_duration = (
            _MINIMUM_JERK_PEAK_VELOCITY
            * largest_delta
            / self.config.max_velocity_rad_s
        )
        acceleration_duration = math.sqrt(
            _MINIMUM_JERK_PEAK_ACCELERATION
            * largest_delta
            / self.config.max_acceleration_rad_s2
        )
        return max(
            self.config.min_duration_s,
            velocity_duration,
            acceleration_duration,
        )

    def _quantized_duration(self, duration_s: float) -> tuple[float, int]:
        """Round a duration upward to an integer number of 50 Hz periods."""
        intervals = max(
            1,
            int(math.ceil(
                duration_s * self.config.frequency_hz - 1.0e-12
            )),
        )
        return intervals / self.config.frequency_hz, intervals

    def generate(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        duration_s: Optional[float] = None,
    ) -> TrajectoryResult:
        """
        Generate a complete 50 Hz quintic trajectory.

        If ``duration_s`` is omitted, the shortest sample-aligned duration that
        satisfies the configured velocity and acceleration limits is selected.
        An explicit duration is never silently stretched; an unsafe request is
        rejected instead.
        """
        try:
            start_vector = self._vector(start)
            goal_vector = self._vector(goal)
        except (TypeError, ValueError) as error:
            return self._failure(
                TrajectoryErrorCode.INVALID_INPUT,
                str(error),
            )

        for label, vector in (
            ('start', start_vector),
            ('goal', goal_vector),
        ):
            invalid = (
                (vector < self._joint_limits[:, 0])
                | (vector > self._joint_limits[:, 1])
            )
            if np.any(invalid):
                indices = ', '.join(
                    str(index + 1) for index in np.flatnonzero(invalid)
                )
                return self._failure(
                    TrajectoryErrorCode.JOINT_LIMIT,
                    f'{label} violates safe joint limits at joints {indices}',
                )

        return self._generate_validated_vectors(
            start_vector,
            goal_vector,
            duration_s,
        )

    def _generate_validated_vectors(
        self,
        start_vector: np.ndarray,
        goal_vector: np.ndarray,
        duration_s: Optional[float],
    ) -> TrajectoryResult:
        """Generate samples after the caller has validated both endpoints."""
        displacement = goal_vector - start_vector
        required_duration = self._required_duration(displacement)
        if duration_s is None:
            requested_duration = required_duration
        else:
            try:
                requested_duration = float(duration_s)
            except (TypeError, ValueError):
                return self._failure(
                    TrajectoryErrorCode.INVALID_INPUT,
                    'duration_s must be a finite positive value',
                )
            if (
                not np.isfinite(requested_duration)
                or requested_duration <= 0.0
            ):
                return self._failure(
                    TrajectoryErrorCode.INVALID_INPUT,
                    'duration_s must be a finite positive value',
                )
            if requested_duration + 1.0e-12 < required_duration:
                return self._failure(
                    TrajectoryErrorCode.DURATION_TOO_SHORT,
                    'explicit duration would exceed velocity or acceleration '
                    f'limits; at least {required_duration:.3f} s is required',
                    requested_duration,
                )

        quantized_duration, intervals = self._quantized_duration(
            requested_duration
        )
        if quantized_duration > self.config.max_duration_s + 1.0e-12:
            return self._failure(
                TrajectoryErrorCode.DURATION_TOO_LONG,
                'safe trajectory duration exceeds configured maximum',
                quantized_duration,
            )

        largest_delta = float(np.max(np.abs(displacement)))
        peak_velocity = (
            _MINIMUM_JERK_PEAK_VELOCITY
            * largest_delta
            / quantized_duration
        )
        peak_acceleration = (
            _MINIMUM_JERK_PEAK_ACCELERATION
            * largest_delta
            / (quantized_duration * quantized_duration)
        )
        tolerance = 1.0e-10
        if (
            peak_velocity > self.config.max_velocity_rad_s + tolerance
            or peak_acceleration
            > self.config.max_acceleration_rad_s2 + tolerance
        ):
            return self._failure(
                TrajectoryErrorCode.NUMERICAL_ERROR,
                'quantized trajectory exceeds a configured motion limit',
                quantized_duration,
                peak_velocity,
                peak_acceleration,
            )

        points = []
        zero = np.zeros(7, dtype=float)
        for index in range(intervals + 1):
            u = index / intervals
            blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            blend_velocity = (
                30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
            )
            blend_acceleration = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
            positions = start_vector + displacement * blend
            velocities = (
                displacement * blend_velocity / quantized_duration
            )
            accelerations = (
                displacement
                * blend_acceleration
                / (quantized_duration * quantized_duration)
            )
            if index == 0:
                positions = start_vector
                velocities = zero
                accelerations = zero
            elif index == intervals:
                positions = goal_vector
                velocities = zero
                accelerations = zero
            points.append(TrajectoryPoint(
                time_from_start_s=index / self.config.frequency_hz,
                positions=tuple(float(value) for value in positions),
                velocities=tuple(float(value) for value in velocities),
                accelerations=tuple(float(value) for value in accelerations),
            ))

        return TrajectoryResult(
            success=True,
            error_code=TrajectoryErrorCode.SUCCESS,
            message='minimum-jerk trajectory accepted',
            points=tuple(points),
            duration_s=quantized_duration,
            peak_velocity_rad_s=peak_velocity,
            peak_acceleration_rad_s2=peak_acceleration,
        )

    def generate_limit_recovery(
        self,
        start: Sequence[float],
        raw_joint_limits: Sequence[Sequence[float]],
        config: Optional[JointLimitRecoveryConfig] = None,
    ) -> JointLimitRecoveryPlan:
        """
        Plan a strictly inward recovery from a small joint-limit violation.

        The caller cannot supply a target.  Only joints outside the conservative
        safe interval may move, and every sample must reduce (or preserve) its
        violation.  Phase A is a single small target inside the raw URDF/SDK
        envelope; Phase B is the normal minimum-jerk trajectory into the
        conservative safe interval.
        """
        settings = config or JointLimitRecoveryConfig()
        try:
            start_vector = self._vector(start)
            raw_limits = np.asarray(raw_joint_limits, dtype=float)
            if raw_limits.shape != (7, 2) or not np.all(np.isfinite(raw_limits)):
                raise ValueError('raw_joint_limits must be a finite 7 by 2 array')
            if np.any(raw_limits[:, 0] >= raw_limits[:, 1]):
                raise ValueError('raw joint lower limits must be below upper limits')
            values = (
                settings.raw_interior_margin_rad,
                settings.safe_interior_margin_rad,
                settings.max_raw_start_violation_rad,
                settings.max_safe_start_violation_rad,
                settings.max_ingress_delta_rad,
                settings.max_total_delta_rad,
            )
            if not all(np.isfinite(value) and value > 0.0 for value in values):
                raise ValueError('joint-limit recovery settings must be positive')
            if np.any(self._joint_limits[:, 0] < raw_limits[:, 0]):
                raise ValueError('safe lower limits must lie inside raw limits')
            if np.any(self._joint_limits[:, 1] > raw_limits[:, 1]):
                raise ValueError('safe upper limits must lie inside raw limits')
            if np.any(
                raw_limits[:, 0] + settings.raw_interior_margin_rad
                >= raw_limits[:, 1] - settings.raw_interior_margin_rad
            ):
                raise ValueError('raw recovery margin leaves an empty interval')
            if np.any(
                self._joint_limits[:, 0] + settings.safe_interior_margin_rad
                >= self._joint_limits[:, 1] - settings.safe_interior_margin_rad
            ):
                raise ValueError('safe recovery margin leaves an empty interval')
        except (TypeError, ValueError) as error:
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.INVALID_TARGET,
                str(error),
            )

        safe_lower = self._joint_limits[:, 0]
        safe_upper = self._joint_limits[:, 1]
        raw_lower = raw_limits[:, 0]
        raw_upper = raw_limits[:, 1]
        below_safe = start_vector < safe_lower
        above_safe = start_vector > safe_upper
        recovering = below_safe | above_safe

        safe_violation = np.maximum(
            np.maximum(safe_lower - start_vector, start_vector - safe_upper),
            0.0,
        )
        raw_violation = np.maximum(
            np.maximum(raw_lower - start_vector, start_vector - raw_upper),
            0.0,
        )
        max_safe_violation = float(np.max(safe_violation))
        max_raw_violation = float(np.max(raw_violation))
        start_tuple = tuple(float(value) for value in start_vector)
        recovering_indices = tuple(
            int(index + 1) for index in np.flatnonzero(recovering)
        )

        if not np.any(recovering):
            return JointLimitRecoveryPlan(
                True,
                IKErrorCode.ALREADY_AT_TARGET,
                'all joints are already inside conservative safe limits',
                already_safe=True,
                start_positions=start_tuple,
                ingress_positions=start_tuple,
                target_positions=start_tuple,
            )
        if max_raw_violation > settings.max_raw_start_violation_rad + 1.0e-12:
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.JOINT_LIMIT_VIOLATION,
                'raw-limit violation is too large for automatic recovery: '
                f'{max_raw_violation:.4f}rad > '
                f'{settings.max_raw_start_violation_rad:.4f}rad',
                start_positions=start_tuple,
                recovering_joint_indices=recovering_indices,
                max_raw_violation_rad=max_raw_violation,
                max_safe_violation_rad=max_safe_violation,
            )
        if max_safe_violation > settings.max_safe_start_violation_rad + 1.0e-12:
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.JOINT_LIMIT_VIOLATION,
                'safe-limit violation is too large for automatic recovery: '
                f'{max_safe_violation:.4f}rad > '
                f'{settings.max_safe_start_violation_rad:.4f}rad',
                start_positions=start_tuple,
                recovering_joint_indices=recovering_indices,
                max_raw_violation_rad=max_raw_violation,
                max_safe_violation_rad=max_safe_violation,
            )

        ingress = start_vector.copy()
        target = start_vector.copy()
        ingress[below_safe] = np.maximum(
            start_vector[below_safe],
            raw_lower[below_safe] + settings.raw_interior_margin_rad,
        )
        ingress[above_safe] = np.minimum(
            start_vector[above_safe],
            raw_upper[above_safe] - settings.raw_interior_margin_rad,
        )
        target[below_safe] = (
            safe_lower[below_safe] + settings.safe_interior_margin_rad
        )
        target[above_safe] = (
            safe_upper[above_safe] - settings.safe_interior_margin_rad
        )

        ingress_delta = float(np.max(np.abs(ingress - start_vector)))
        total_delta = float(np.max(np.abs(target - start_vector)))
        common = dict(
            start_positions=start_tuple,
            ingress_positions=tuple(float(value) for value in ingress),
            target_positions=tuple(float(value) for value in target),
            recovering_joint_indices=recovering_indices,
            max_raw_violation_rad=max_raw_violation,
            max_safe_violation_rad=max_safe_violation,
            max_joint_delta_rad=total_delta,
        )
        if ingress_delta > settings.max_ingress_delta_rad + 1.0e-12:
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.JOINT_DELTA_TOO_LARGE,
                'phase-A ingress delta is too large: '
                f'{ingress_delta:.4f}rad > '
                f'{settings.max_ingress_delta_rad:.4f}rad',
                **common,
            )
        if total_delta > settings.max_total_delta_rad + 1.0e-12:
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.JOINT_DELTA_TOO_LARGE,
                'total recovery delta is too large: '
                f'{total_delta:.4f}rad > '
                f'{settings.max_total_delta_rad:.4f}rad',
                **common,
            )

        phase_a = None
        if ingress_delta > 1.0e-12:
            phase_a = self._generate_validated_vectors(
                start_vector,
                ingress,
                None,
            )
            if not phase_a.success:
                return JointLimitRecoveryPlan(
                    False,
                    IKErrorCode.TRAJECTORY_LIMIT_VIOLATION,
                    f'cannot generate phase-A recovery: {phase_a.message}',
                    phase_a_trajectory=phase_a,
                    **common,
                )
        common['phase_a_trajectory'] = phase_a

        phase_b = self._generate_validated_vectors(ingress, target, None)
        if not phase_b.success:
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.TRAJECTORY_LIMIT_VIOLATION,
                f'cannot generate phase-B recovery: {phase_b.message}',
                phase_b_trajectory=phase_b,
                **common,
            )

        samples_list = [start_vector]
        if phase_a is None:
            samples_list.append(ingress)
        else:
            samples_list.extend(
                np.asarray(point.positions) for point in phase_a.points[1:]
            )
        samples_list.extend(
            np.asarray(point.positions) for point in phase_b.points[1:]
        )
        samples = np.asarray(samples_list)
        for index in np.flatnonzero(recovering):
            differences = np.diff(samples[:, index])
            if below_safe[index] and np.any(differences < -1.0e-12):
                return JointLimitRecoveryPlan(
                    False,
                    IKErrorCode.DISCONTINUOUS_SOLUTION,
                    f'joint{index + 1} recovery is not monotonically inward',
                    phase_b_trajectory=phase_b,
                    **common,
                )
            if above_safe[index] and np.any(differences > 1.0e-12):
                return JointLimitRecoveryPlan(
                    False,
                    IKErrorCode.DISCONTINUOUS_SOLUTION,
                    f'joint{index + 1} recovery is not monotonically inward',
                    phase_b_trajectory=phase_b,
                    **common,
                )
        fixed = np.flatnonzero(~recovering)
        if fixed.size and not np.allclose(
            samples[:, fixed],
            start_vector[fixed],
            atol=1.0e-12,
            rtol=0.0,
        ):
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.DISCONTINUOUS_SOLUTION,
                'a joint that was already safe would move during recovery',
                phase_b_trajectory=phase_b,
                **common,
            )

        violation_history = np.maximum(
            np.maximum(safe_lower - samples, samples - safe_upper),
            0.0,
        )
        if np.any(np.diff(violation_history, axis=0) > 1.0e-12):
            return JointLimitRecoveryPlan(
                False,
                IKErrorCode.DISCONTINUOUS_SOLUTION,
                'a recovery sample would increase a joint-limit violation',
                phase_b_trajectory=phase_b,
                **common,
            )

        return JointLimitRecoveryPlan(
            True,
            IKErrorCode.SUCCESS,
            'two-stage joint-limit recovery accepted',
            phase_b_trajectory=phase_b,
            **common,
        )


class JointLimitRecoveryMonitor:
    """Reject measured motion that violates an accepted inward-only plan."""

    def __init__(
        self,
        plan: JointLimitRecoveryPlan,
        safe_joint_limits: Sequence[Sequence[float]],
        *,
        fixed_joint_tolerance_rad: float,
        progress_tolerance_rad: float,
    ) -> None:
        """Store the measured start and permitted direction for every joint."""
        limits = np.asarray(safe_joint_limits, dtype=float)
        start = np.asarray(plan.start_positions, dtype=float)
        recovering = np.asarray(plan.recovering_joint_indices, dtype=int) - 1
        if limits.shape != (7, 2) or start.shape != (7,):
            raise ValueError('recovery monitor requires seven-joint inputs')
        if recovering.size == 0 or np.any((recovering < 0) | (recovering >= 7)):
            raise ValueError('recovery monitor requires valid recovering joints')
        if len({int(index) for index in recovering}) != len(recovering):
            raise ValueError('recovering joint indices must be unique')
        tolerances = (fixed_joint_tolerance_rad, progress_tolerance_rad)
        if not all(np.isfinite(value) and value > 0.0 for value in tolerances):
            raise ValueError('recovery monitor tolerances must be positive')

        self._limits = limits.copy()
        self._start = start.copy()
        self._recovering = recovering
        self._fixed = np.asarray(
            [index for index in range(7) if index not in recovering],
            dtype=int,
        )
        self._below = start < limits[:, 0]
        self._above = start > limits[:, 1]
        if np.any(~(self._below[recovering] | self._above[recovering])):
            raise ValueError('every recovering joint must start outside safe limits')
        self._fixed_tolerance = float(fixed_joint_tolerance_rad)
        self._progress_tolerance = float(progress_tolerance_rad)
        self._best_inward = start.copy()
        self._minimum_violation = self._violation(start)

    def _violation(self, positions: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.maximum(
                self._limits[:, 0] - positions,
                positions - self._limits[:, 1],
            ),
            0.0,
        )

    def validate(self, positions: Sequence[float]) -> Optional[str]:
        """Return a refusal reason, or None after accepting this feedback."""
        measured = np.asarray(positions, dtype=float)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            return '恢复期间收到不完整或非有限的关节反馈'

        if self._fixed.size:
            fixed_error = float(
                np.max(np.abs(measured[self._fixed] - self._start[self._fixed]))
            )
            if fixed_error > self._fixed_tolerance:
                return (
                    '恢复期间原本安全的关节发生非预期移动：'
                    f'{fixed_error:.4f}rad'
                )
            fixed_below = measured[self._fixed] < self._limits[self._fixed, 0]
            fixed_above = measured[self._fixed] > self._limits[self._fixed, 1]
            if np.any(fixed_below | fixed_above):
                return '恢复期间原本安全的关节离开了安全范围'

        for index in self._recovering:
            if (
                self._below[index]
                and measured[index]
                < self._best_inward[index] - self._progress_tolerance
            ):
                return f'joint{index + 1} 在恢复期间向错误方向移动'
            if (
                self._above[index]
                and measured[index]
                > self._best_inward[index] + self._progress_tolerance
            ):
                return f'joint{index + 1} 在恢复期间向错误方向移动'

        violation = self._violation(measured)
        if np.any(
            violation[self._recovering]
            > self._minimum_violation[self._recovering]
            + self._progress_tolerance
        ):
            return '恢复期间真实关节的安全限位越界量增大'

        for index in self._recovering:
            if self._below[index]:
                self._best_inward[index] = max(
                    self._best_inward[index], measured[index]
                )
            else:
                self._best_inward[index] = min(
                    self._best_inward[index], measured[index]
                )
        self._minimum_violation = np.minimum(
            self._minimum_violation,
            violation,
        )
        return None
