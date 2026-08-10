"""Tests for the deterministic formal 30-pose acceptance dataset."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")

from strawberry_nero_control.acceptance_dataset import (  # noqa: E402
    ACCEPTANCE_MAX_JOINT_DELTA_RAD,
    ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD,
    ACCEPTANCE_TARGET_COUNT,
    deterministic_joint_offsets,
    plan_acceptance_suite,
    write_acceptance_target_set,
)
from strawberry_nero_control.ik_core import PlacoIKSolver  # noqa: E402
from strawberry_nero_control.models import READY_JOINT_POSITIONS  # noqa: E402
from strawberry_nero_control.trajectory import (  # noqa: E402
    TrajectoryGenerator,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
NERO_URDF = (
    SOURCE_ROOT
    / "agx_arm_ros"
    / "src"
    / "agx_arm_description"
    / "agx_arm_urdf"
    / "nero"
    / "urdf"
    / "nero_description.urdf"
)


@pytest.fixture(scope="module")
def acceptance_plan():
    """Plan the fixed dataset once for all assertions."""
    solver = PlacoIKSolver(NERO_URDF)
    trajectory = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
    return plan_acceptance_suite(solver, trajectory, READY_JOINT_POSITIONS)


def test_joint_offsets_are_fixed_balanced_and_small():
    offsets_a = deterministic_joint_offsets()
    offsets_b = deterministic_joint_offsets()

    assert len(offsets_a) == ACCEPTANCE_TARGET_COUNT
    for first, second in zip(offsets_a, offsets_b):
        np.testing.assert_array_equal(first, second)
        assert first.shape == (7,)
        assert np.max(np.abs(first)) <= 0.110
    stacked = np.vstack(offsets_a)
    assert np.all(np.min(stacked, axis=0) < 0.0)
    assert np.all(np.max(stacked, axis=0) > 0.0)


def test_all_30_targets_and_returns_pass_continuity(acceptance_plan):
    assert acceptance_plan.passed
    assert len(acceptance_plan.targets) == ACCEPTANCE_TARGET_COUNT
    assert acceptance_plan.dataset_id.startswith("week1-30-v2-")
    assert len({target.target_id for target in acceptance_plan.targets}) == 30

    for target in acceptance_plan.targets:
        assert target.accepted, target.reason
        assert target.outbound_trajectory.success
        assert target.return_trajectory.success
        assert (
            target.outbound_ik.max_joint_delta_rad
            <= ACCEPTANCE_MAX_JOINT_DELTA_RAD
        )
        assert (
            target.predicted_return_joint_error_rad
            <= ACCEPTANCE_MAX_RETURN_JOINT_ERROR_RAD
        )
        assert target.position_offset_m > 0.010
        assert target.orientation_offset_rad > math.radians(3.0)

    assert max(
        target.position_offset_m for target in acceptance_plan.targets
    ) > 0.040
    assert max(
        target.orientation_offset_rad for target in acceptance_plan.targets
    ) > math.radians(14.0)


def test_target_set_writer_uses_stable_visible_filenames(
    tmp_path,
    acceptance_plan,
):
    csv_path, json_path = write_acceptance_target_set(
        acceptance_plan,
        tmp_path,
    )

    assert csv_path.name == "target_set_30.csv"
    assert json_path.name == "target_set_30.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == acceptance_plan.dataset_id
    assert payload["target_count"] == 30
    assert payload["target_attempts_required"] == 90
    assert payload["passed_offline_precheck"] is True
    assert len(payload["targets"]) == 30
