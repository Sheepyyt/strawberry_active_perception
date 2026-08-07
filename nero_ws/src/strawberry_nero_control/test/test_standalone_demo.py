"""Tests for the ROS-free Placo and trajectory demonstration."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")

from strawberry_nero_control.models import IKErrorCode  # noqa: E402
from strawberry_nero_control.standalone_demo import (  # noqa: E402
    default_urdf_path,
    plan_standalone_motion,
)


NERO_URDF = default_urdf_path()


def test_default_urdf_is_the_checked_out_nero_model():
    """The standalone demo resolves the driver source tree without ROS."""
    assert NERO_URDF.is_file()
    assert NERO_URDF.name == "nero_description.urdf"
    assert "agx_arm_description" in NERO_URDF.parts


def test_default_offset_produces_safe_smooth_motion():
    """The documented 20 mm move passes IK and trajectory safety gates."""
    plan = plan_standalone_motion(NERO_URDF)
    result = plan.ik_result
    trajectory = plan.trajectory_result

    assert result.success, result.message
    assert result.error_code == IKErrorCode.SUCCESS
    assert result.position_error_m <= 0.002
    assert result.orientation_error_rad <= np.radians(2.0)
    assert result.max_joint_delta_rad <= 0.35
    assert trajectory is not None and trajectory.success
    assert trajectory.points[0].positions == pytest.approx(plan.start_joints)
    assert trajectory.points[-1].positions == pytest.approx(
        result.joint_positions
    )
    assert trajectory.peak_velocity_rad_s <= 0.30
    assert trajectory.peak_acceleration_rad_s2 <= 0.50


def test_zero_offset_stays_inside_deadband_without_trajectory():
    """An unchanged target is reported instead of faking an animation."""
    plan = plan_standalone_motion(NERO_URDF, (0.0, 0.0, 0.0))

    assert plan.ik_result.success
    assert plan.ik_result.error_code == IKErrorCode.ALREADY_AT_TARGET
    assert plan.trajectory_result is None


def test_unreachable_offset_is_refused_without_trajectory():
    """A many-metre target never reaches visualization or execution."""
    plan = plan_standalone_motion(NERO_URDF, (5.0, 0.0, 0.0))

    assert not plan.ik_result.success
    assert plan.ik_result.error_code == IKErrorCode.UNREACHABLE
    assert plan.trajectory_result is None


def test_invalid_offset_is_rejected_before_solver_use():
    """Malformed Cartesian input cannot create a partial plan."""
    with pytest.raises(ValueError):
        plan_standalone_motion(Path(NERO_URDF), (0.0, float("nan"), 0.0))
