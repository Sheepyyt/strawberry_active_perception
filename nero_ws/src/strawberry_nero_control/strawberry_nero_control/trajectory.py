"""Limit-aware 50 Hz minimum-jerk trajectories for all seven NERO joints."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Optional

import numpy as np

from .models import (
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
