"""Unit tests for the operator-supplied Cartesian pose demo."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from strawberry_nero_control.pose_demo import absolute_target_transform
from strawberry_nero_control.pose_demo import minimum_joint_limit_margin
from strawberry_nero_control.pose_demo import relative_target_transform
from strawberry_nero_control.pose_demo import validate_demo_delta


def test_absolute_target_normalizes_quaternion():
    """Quaternion magnitude does not change the requested orientation."""
    quaternion = Rotation.from_euler("z", 30.0, degrees=True).as_quat()
    target = absolute_target_transform(
        [0.1, -0.2, 0.4],
        quaternion_xyzw=quaternion * 4.0,
    )

    np.testing.assert_allclose(target[:3, 3], [0.1, -0.2, 0.4])
    np.testing.assert_allclose(
        target[:3, :3],
        Rotation.from_quat(quaternion).as_matrix(),
    )


def test_absolute_target_rejects_zero_quaternion():
    """An orientation must not silently default from an invalid quaternion."""
    with pytest.raises(ValueError, match="不能全为零"):
        absolute_target_transform(
            [0.1, -0.2, 0.4],
            quaternion_xyzw=[0.0, 0.0, 0.0, 0.0],
        )


def test_relative_base_translation_uses_base_axes():
    """A base-frame X request changes only the absolute X coordinate."""
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler(
        "z", 90.0, degrees=True
    ).as_matrix()
    current[:3, 3] = [0.2, 0.3, 0.4]

    target = relative_target_transform(
        current,
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        "base",
    )

    np.testing.assert_allclose(target[:3, 3], [0.21, 0.3, 0.4])


def test_relative_tool_translation_uses_rotated_tool_axes():
    """A tool-frame X request follows link7 X after its current rotation."""
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler(
        "z", 90.0, degrees=True
    ).as_matrix()

    target = relative_target_transform(
        current,
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        "tool",
    )

    np.testing.assert_allclose(target[:3, 3], [0.0, 0.01, 0.0], atol=1e-12)


def test_demo_delta_enforces_deadband_and_outer_bounds():
    """Tiny requests and unverified large steps are both refused."""
    current = np.eye(4)
    tiny = current.copy()
    tiny[0, 3] = 0.0005
    with pytest.raises(ValueError, match="死区"):
        validate_demo_delta(current, tiny)

    far = current.copy()
    far[0, 3] = 0.081
    with pytest.raises(ValueError, match="80 mm"):
        validate_demo_delta(current, far)

    rotated = current.copy()
    rotated[:3, :3] = Rotation.from_euler(
        "x", 30.1, degrees=True
    ).as_matrix()
    with pytest.raises(ValueError, match="30°"):
        validate_demo_delta(current, rotated)

    accepted = current.copy()
    accepted[:3, 3] = [0.06, 0.0, 0.0]
    accepted[:3, :3] = Rotation.from_euler(
        "z", 20.0, degrees=True
    ).as_matrix()
    position, orientation = validate_demo_delta(current, accepted)
    assert position == pytest.approx(0.06)
    assert orientation == pytest.approx(math.radians(20.0))


def test_minimum_joint_limit_margin_detects_boundary_targets():
    """A valid but boundary-hugging IK solution has almost no usable margin."""
    limits = np.column_stack((-np.ones(7), np.ones(7)))
    centered = np.zeros(7)
    near_lower = centered.copy()
    near_lower[1] = -0.999

    assert minimum_joint_limit_margin(centered, limits) == pytest.approx(1.0)
    assert minimum_joint_limit_margin(near_lower, limits) == pytest.approx(0.001)
