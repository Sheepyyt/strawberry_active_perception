"""Unit tests for the ROS-independent Placo NERO IK core."""

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('placo')

from strawberry_nero_control.ik_core import (  # noqa: E402
    PlacoIKSolver,
    transform_from_pose,
)
from strawberry_nero_control.models import (  # noqa: E402
    IKErrorCode,
    NERO_JOINT_NAMES,
    READY_JOINT_POSITIONS,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
NERO_URDF = (
    SOURCE_ROOT
    / 'agx_arm_ros'
    / 'src'
    / 'agx_arm_description'
    / 'agx_arm_urdf'
    / 'nero'
    / 'urdf'
    / 'nero_description.urdf'
)


@pytest.fixture(scope='module')
def solver():
    """Create one real Placo solver for all kinematics tests."""
    return PlacoIKSolver(NERO_URDF)


def test_loads_nero_and_applies_two_degree_margin(solver):
    """The safe limits are the URDF/SDK intersection minus two degrees."""
    assert solver.joint_names == NERO_JOINT_NAMES
    limits = solver.safe_joint_limits
    assert limits.shape == (7, 2)
    assert limits[0, 0] == pytest.approx(-2.70526 + math.radians(2.0))
    assert limits[1, 1] == pytest.approx(1.74 - math.radians(2.0))


def test_ready_pose_is_valid_and_well_conditioned(solver):
    """The user-tested ready pose stays well away from a singularity."""
    ready = np.asarray(READY_JOINT_POSITIONS)
    assert np.all(ready > solver.safe_joint_limits[:, 0])
    assert np.all(ready < solver.safe_joint_limits[:, 1])
    metrics = solver.configuration_metrics(ready)
    assert metrics.sigma_min == pytest.approx(0.18559, rel=2.0e-3)
    assert metrics.condition_number == pytest.approx(9.4814, rel=2.0e-3)


def test_home_pose_is_rejected_as_singular(solver):
    """All-zero home must not pass the singularity safety gate."""
    home = np.zeros(7)
    target = solver.forward_kinematics(home)
    metrics = solver.configuration_metrics(home)
    assert metrics.sigma_min < solver.config.singular_sigma_min

    result = solver.solve(target, home)

    assert not result.success
    assert result.error_code == IKErrorCode.NEAR_SINGULARITY


def test_reachable_nearby_target_converges_from_measured_state(solver):
    """A nearby FK-generated pose is recovered with tight residuals."""
    ready = np.asarray(READY_JOINT_POSITIONS)
    target_joints = ready + np.array([
        0.010,
        -0.020,
        0.015,
        0.020,
        -0.010,
        0.010,
        0.005,
    ])
    target = solver.forward_kinematics(target_joints)

    result = solver.solve(target, ready)

    assert result.success, result.message
    assert result.error_code == IKErrorCode.SUCCESS
    assert len(result.joint_positions) == 7
    assert result.position_error_m <= solver.config.position_tolerance_m
    assert result.orientation_error_rad <= solver.config.orientation_tolerance_rad
    assert result.max_joint_delta_rad <= solver.config.max_joint_delta_rad
    assert result.iterations <= solver.config.max_iterations
    assert result.solve_time_ms <= solver.config.timeout_s * 1000.0


def test_optional_posture_reference_selects_ready_redundant_branch(solver):
    """Centering can prefer ready while still initializing from measurement."""
    measured = np.array([
        0.167761,
        -1.695081,
        -0.041015,
        2.020620,
        -0.023440,
        0.033772,
        0.003735,
    ])
    ready = np.asarray(READY_JOINT_POSITIONS)
    predicted = measured.copy()

    for index in range(1, 11):
        reference = measured + (ready - measured) * index / 10.0
        target = solver.forward_kinematics(reference)
        result = solver.solve(
            target,
            predicted,
            posture_reference_joints=reference,
        )
        assert result.success, result.message
        assert result.max_joint_delta_rad <= 0.06
        predicted = np.asarray(result.joint_positions)

    assert np.max(np.abs(predicted - ready)) < 0.005


def test_target_inside_deadband_does_not_create_motion(solver):
    """An unchanged target returns the measured joints without solving."""
    ready = np.asarray(READY_JOINT_POSITIONS)
    target = solver.forward_kinematics(ready)

    result = solver.solve(target, ready)

    assert result.success
    assert result.error_code == IKErrorCode.ALREADY_AT_TARGET
    assert result.joint_positions == pytest.approx(tuple(ready))
    assert result.iterations == 0


def test_unreachable_target_is_refused(solver):
    """A pose many metres outside the workspace cannot produce a command."""
    ready = np.asarray(READY_JOINT_POSITIONS)
    target = np.eye(4)
    target[:3, 3] = (5.0, 5.0, 5.0)

    result = solver.solve(target, ready)

    assert not result.success
    assert result.error_code == IKErrorCode.UNREACHABLE
    assert result.position_error_m > solver.config.position_tolerance_m


def test_joint_jump_is_rejected_after_other_checks_pass(solver):
    """A valid solution is rejected when its measured-state jump is too big."""
    strict_config = replace(solver.config, max_joint_delta_rad=0.005)
    strict_solver = PlacoIKSolver(NERO_URDF, strict_config)
    ready = np.asarray(READY_JOINT_POSITIONS)
    target_joints = ready + np.array([
        0.010,
        -0.020,
        0.015,
        0.020,
        -0.010,
        0.010,
        0.005,
    ])
    target = strict_solver.forward_kinematics(target_joints)

    result = strict_solver.solve(target, ready)

    assert not result.success
    assert result.error_code == IKErrorCode.JOINT_DELTA_TOO_LARGE
    assert result.max_joint_delta_rad > strict_config.max_joint_delta_rad
    assert result.position_error_m <= strict_config.position_tolerance_m


def test_invalid_pose_and_current_limit_are_structured_failures(solver):
    """Malformed targets and unsafe feedback never raise into the ROS layer."""
    ready = np.asarray(READY_JOINT_POSITIONS)
    invalid_target = np.eye(4)
    invalid_target[0, 0] = 2.0
    invalid = solver.solve(invalid_target, ready)
    assert invalid.error_code == IKErrorCode.INVALID_TARGET

    target = solver.forward_kinematics(ready)
    unsafe = ready.copy()
    unsafe[1] = solver.safe_joint_limits[1, 0] - 0.01
    limited = solver.solve(target, unsafe)
    assert limited.error_code == IKErrorCode.JOINT_LIMIT_VIOLATION


def test_pose_conversion_normalizes_quaternion_and_rejects_zero():
    """Quaternion scale/sign cannot change the represented orientation."""
    positive = transform_from_pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 2.0))
    negative = transform_from_pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, -2.0))
    assert np.allclose(positive, negative)
    assert np.allclose(positive[:3, 3], (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        transform_from_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
