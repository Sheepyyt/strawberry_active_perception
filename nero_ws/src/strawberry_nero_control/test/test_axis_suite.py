"""Tests for the fixed six-direction Placo validation suite."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")

from strawberry_nero_control.axis_suite import (  # noqa: E402
    AXIS_DIRECTIONS,
    AXIS_TEST_DISTANCE_M,
    AxisReportWriter,
    axis_target,
    case_to_row,
    plan_axis_case,
    plan_axis_suite,
)
from strawberry_nero_control.ik_core import PlacoIKSolver  # noqa: E402
from strawberry_nero_control.models import (  # noqa: E402
    READY_JOINT_POSITIONS,
)
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
LIMIT_NEAR_ANCHOR = np.array([
    -0.017174,
    -1.695081,
    0.072047,
    2.099055,
    -0.179891,
    -0.029566,
    -0.199282,
])


@pytest.fixture(scope="module")
def suite_plan():
    """Plan once from the user-verified central ready neighborhood."""
    solver = PlacoIKSolver(NERO_URDF)
    trajectory = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
    return plan_axis_suite(solver, trajectory, READY_JOINT_POSITIONS)


def test_axis_target_is_frozen_to_locked_direction_distances():
    """No caller can turn the staged test into an arbitrary displacement."""
    anchor = np.eye(4)
    anchor[:3, 3] = [0.1, 0.2, 0.3]

    for direction in AXIS_DIRECTIONS:
        target = axis_target(anchor, direction)
        expected = anchor[:3, 3].copy()
        expected[direction.axis_index] += (
            direction.sign * direction.distance_m
        )
        np.testing.assert_allclose(target[:3, 3], expected)
        np.testing.assert_allclose(target[:3, :3], anchor[:3, :3])

    with pytest.raises(ValueError):
        axis_target(anchor, AXIS_DIRECTIONS[0], 0.005)

    assert [direction.distance_m for direction in AXIS_DIRECTIONS] == [
        AXIS_TEST_DISTANCE_M,
        AXIS_TEST_DISTANCE_M,
        AXIS_TEST_DISTANCE_M,
        AXIS_TEST_DISTANCE_M,
        AXIS_TEST_DISTANCE_M,
        AXIS_TEST_DISTANCE_M,
    ]


def test_ready_anchor_accepts_all_six_out_and_return_pairs(suite_plan):
    """The central ready neighborhood passes every continuity gate."""
    assert suite_plan.passed
    assert [case.direction.label for case in suite_plan.cases] == [
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
    ]
    for case in suite_plan.cases:
        assert case.accepted, case.reason
        assert case.outbound_trajectory.success
        assert case.return_trajectory.success
        assert case.outbound_ik.max_joint_delta_rad <= 0.08
        assert case.return_ik.max_joint_delta_rad <= 0.08
        assert case.predicted_return_joint_error_rad <= 0.02


def test_sequential_pairs_refresh_their_anchor_from_each_return():
    """A small return mismatch is not reused as the next Cartesian target."""
    solver = PlacoIKSolver(NERO_URDF)
    trajectory = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
    current = np.asarray(READY_JOINT_POSITIONS, dtype=float)

    for direction in AXIS_DIRECTIONS:
        pair = plan_axis_case(
            solver,
            trajectory,
            current,
            direction,
        )
        assert pair.accepted, f"{direction.label}: {pair.reason}"
        assert pair.outbound_ik.position_error_m <= 0.002
        assert pair.return_ik.position_error_m <= 0.002
        current = np.asarray(pair.return_ik.joint_positions)


def test_limit_near_anchor_is_refused_before_fifteen_mm_suite():
    """Being inside limits alone is not enough for the larger axis suite."""
    solver = PlacoIKSolver(NERO_URDF)
    trajectory = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
    plan = plan_axis_suite(solver, trajectory, LIMIT_NEAR_ANCHOR)

    assert not plan.passed
    refused = [case.direction.label for case in plan.cases if not case.accepted]
    assert "-X" in refused
    assert "+Y" in refused


def test_report_writer_persists_complete_csv_and_json(tmp_path, suite_plan):
    """Both machine-readable reports contain all six deterministic rows."""
    rows = [
        case_to_row(index, case)
        for index, case in enumerate(suite_plan.cases)
    ]
    writer = AxisReportWriter(tmp_path, "offline")
    writer.write(
        rows,
        anchor_joints=suite_plan.anchor_joints,
        completed=True,
        passed=True,
        message="test passed",
    )

    with writer.csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    payload = json.loads(writer.json_path.read_text(encoding="utf-8"))

    assert len(csv_rows) == 6
    assert payload["completed"] is True
    assert payload["passed"] is True
    assert payload["directions_recorded"] == 6
    assert [row["direction"] for row in payload["rows"]] == [
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
    ]
