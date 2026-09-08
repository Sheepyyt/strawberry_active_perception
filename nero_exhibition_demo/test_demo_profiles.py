#!/usr/bin/env python3
"""CAN-free regression tests for the independent exhibition profiles."""

from __future__ import annotations

import math
import unittest

import numpy as np

from strawberry_nero_control.models import (
    READY_JOINT_POSITIONS,
    TrajectoryConfig,
)
from strawberry_nero_control.real_smoke_test import nero_urdf_path
from strawberry_nero_control.trajectory import TrajectoryGenerator

from demo_profiles import (
    MAX_CONDITION_NUMBER,
    MAX_LEG_DURATION_S,
    MAX_LEG_JOINT_DELTA_RAD,
    MAX_PREVIEW_ORIENTATION_ERROR_RAD,
    MAX_PREVIEW_POSITION_ERROR_M,
    MIN_JOINT_LIMIT_MARGIN_RAD,
    MIN_SIGMA,
    joint_limit_margin,
    make_profile_solver,
    plan_profile,
    profile_targets,
)


class ExhibitionProfileTests(unittest.TestCase):
    """Verify amplitude, IK, limits and every minimum-jerk segment."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = make_profile_solver(nero_urdf_path(), timeout_s=0.20)

    def test_profiles_are_large_complete_and_return_to_ready(self) -> None:
        ready = np.asarray(READY_JOINT_POSITIONS, dtype=float)
        for mode, expected_count in (("poses", 16), ("trajectory", 16)):
            with self.subTest(mode=mode):
                targets = profile_targets(mode)
                self.assertEqual(len(targets), expected_count)
                values = np.asarray(
                    [target.joints for target in targets],
                    dtype=float,
                )
                self.assertEqual(values.shape, (expected_count, 7))
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.allclose(
                    values[-1], ready, rtol=0.0, atol=1.0e-12
                ))
                self.assertGreaterEqual(float(np.ptp(values[:, 0])), 2.0)
                for row in values:
                    self.assertGreaterEqual(
                        joint_limit_margin(row, self.solver.safe_joint_limits),
                        MIN_JOINT_LIMIT_MARGIN_RAD,
                    )

    def test_every_profile_leg_passes_strict_offline_gates(self) -> None:
        for mode in ("poses", "trajectory"):
            with self.subTest(mode=mode):
                plan = plan_profile(mode, self.solver)
                self.assertGreaterEqual(plan.xyz_span_m[1], 0.55)
                self.assertLessEqual(plan.main_x_range_m[1], -0.18)
                self.assertLess(plan.main_x_range_m[0], -0.35)
                self.assertEqual(len(plan.targets), len(profile_targets(mode)))
                self.assertGreater(plan.predicted_motion_duration_s, 20.0)
                for leg in plan.targets:
                    self.assertLessEqual(
                        leg.max_joint_delta_rad,
                        MAX_LEG_JOINT_DELTA_RAD,
                    )
                    self.assertLessEqual(
                        leg.position_error_m,
                        MAX_PREVIEW_POSITION_ERROR_M,
                    )
                    self.assertLessEqual(
                        leg.orientation_error_rad,
                        MAX_PREVIEW_ORIENTATION_ERROR_RAD,
                    )
                    self.assertGreaterEqual(leg.sigma_min, MIN_SIGMA)
                    self.assertLessEqual(
                        leg.condition_number,
                        MAX_CONDITION_NUMBER,
                    )
                    self.assertGreaterEqual(
                        leg.joint_limit_margin_rad,
                        MIN_JOINT_LIMIT_MARGIN_RAD,
                    )
                    self.assertLessEqual(
                        leg.trajectory_duration_s,
                        MAX_LEG_DURATION_S,
                    )
                    self.assertTrue(np.allclose(
                        leg.transform[3],
                        (0.0, 0.0, 0.0, 1.0),
                        rtol=0.0,
                        atol=1.0e-12,
                    ))
                    rotation = leg.transform[:3, :3]
                    self.assertTrue(np.allclose(
                        rotation.T @ rotation,
                        np.eye(3),
                        rtol=0.0,
                        atol=1.0e-9,
                    ))
                    self.assertAlmostEqual(
                        float(np.linalg.det(rotation)), 1.0, places=9
                    )

    def test_all_generated_samples_are_bounded_and_monotonic(self) -> None:
        generator = TrajectoryGenerator(
            TrajectoryConfig(
                frequency_hz=50.0,
                max_velocity_rad_s=0.30,
                max_acceleration_rad_s2=0.50,
            ),
            joint_limits=self.solver.safe_joint_limits,
        )
        for mode in ("poses", "trajectory"):
            plan = plan_profile(mode, self.solver)
            previous = np.asarray(READY_JOINT_POSITIONS, dtype=float)
            for index, leg in enumerate(plan.targets):
                with self.subTest(mode=mode, leg=index):
                    goal = np.asarray(leg.predicted_joints, dtype=float)
                    result = generator.generate(previous, goal)
                    self.assertTrue(result.success, result.message)
                    self.assertLessEqual(result.peak_velocity_rad_s, 0.30)
                    self.assertLessEqual(result.peak_acceleration_rad_s2, 0.50)
                    positions = np.asarray(
                        [point.positions for point in result.points],
                        dtype=float,
                    )
                    velocities = np.asarray(
                        [result.points[0].velocities,
                         result.points[-1].velocities],
                        dtype=float,
                    )
                    accelerations = np.asarray(
                        [result.points[0].accelerations,
                         result.points[-1].accelerations],
                        dtype=float,
                    )
                    self.assertTrue(np.array_equal(positions[0], previous))
                    self.assertTrue(np.array_equal(positions[-1], goal))
                    self.assertTrue(np.array_equal(velocities, np.zeros((2, 7))))
                    self.assertTrue(np.array_equal(
                        accelerations, np.zeros((2, 7))
                    ))
                    self.assertTrue(np.all(
                        positions >= self.solver.safe_joint_limits[:, 0]
                    ))
                    self.assertTrue(np.all(
                        positions <= self.solver.safe_joint_limits[:, 1]
                    ))
                    displacement = goal - previous
                    for joint in range(7):
                        differences = np.diff(positions[:, joint])
                        if math.isclose(displacement[joint], 0.0, abs_tol=1e-12):
                            self.assertTrue(np.allclose(
                                differences, 0.0, rtol=0.0, atol=1e-12
                            ))
                        elif displacement[joint] > 0.0:
                            self.assertTrue(np.all(differences >= -1e-12))
                        else:
                            self.assertTrue(np.all(differences <= 1e-12))
                    previous = goal


if __name__ == "__main__":
    unittest.main()
