"""Safety and persistence tests for formal Week-1 acceptance tooling."""

import json
from pathlib import Path
from types import SimpleNamespace

from builtin_interfaces.msg import Time
import numpy as np
import pytest

pytest.importorskip("placo")

from strawberry_nero_control.acceptance_dataset import (  # noqa: E402
    ACCEPTANCE_TOTAL_TARGET_ATTEMPTS,
    plan_acceptance_suite,
)
from strawberry_nero_control.ik_core import PlacoIKSolver  # noqa: E402
from strawberry_nero_control.models import READY_JOINT_POSITIONS  # noqa: E402
from strawberry_nero_control.trajectory import (  # noqa: E402
    TrajectoryGenerator,
)
from strawberry_nero_control.week1_acceptance import (  # noqa: E402
    ExecutionObservation,
    Week1AcceptanceNode,
    _argument_parser,
    _empty_attempt_row,
    _record_recovered_return,
    create_real_session,
    load_real_session,
    promote_axis_result,
    save_real_session,
    write_final_real_report,
)
from strawberry_nero_interfaces.msg import IKResult  # noqa: E402


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
    solver = PlacoIKSolver(NERO_URDF)
    trajectory = TrajectoryGenerator(joint_limits=solver.safe_joint_limits)
    return plan_acceptance_suite(solver, trajectory, READY_JOINT_POSITIONS)


def test_real_session_is_hidden_resumable_and_structurally_checked(
    tmp_path,
    acceptance_plan,
):
    path = create_real_session(acceptance_plan, tmp_path)
    session = load_real_session(path)

    assert session["dataset_id"] == acceptance_plan.dataset_id
    assert session["next_attempt_index"] == 0
    assert not session["blocked"]
    assert len(session["targets"]) == 30
    assert session["rows"] == []

    session["rows"].append(_empty_attempt_row(0))
    session["next_attempt_index"] = 1
    save_real_session(session)
    assert load_real_session(path)["next_attempt_index"] == 1

    damaged = json.loads(path.read_text(encoding="utf-8"))
    damaged["next_attempt_index"] = 2
    path.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(Exception, match="进度与结果行数"):
        load_real_session(path)


def test_explicit_anchor_return_repairs_only_a_partial_return(
    tmp_path,
    acceptance_plan,
):
    session_path = create_real_session(acceptance_plan, tmp_path)
    session = load_real_session(session_path)
    row = _empty_attempt_row(0)
    row.update({
        "target_success": True,
        "target_code": 0,
        "target_reason": "目标已稳定到达",
        "stop_stage": "return_execution",
    })
    session["rows"] = [row]
    session["next_attempt_index"] = 1
    session["blocked"] = True
    session["blocked_reason"] = "client drift check"
    save_real_session(session)

    repaired = _record_recovered_return(
        session,
        {
            "return_success": True,
            "return_code": 0,
            "return_reason": "显式返回锚点",
            "return_solve_time_ms": 0.5,
            "return_position_error_m": 0.0002,
            "return_orientation_error_rad": 0.0003,
            "return_sigma_min": 0.18,
            "return_condition_number": 9.5,
            "return_max_joint_delta_rad": 0.07,
            "return_response_latency_s": 0.12,
            "return_motion_start_detected": True,
            "return_total_duration_s": 1.0,
        },
        acceptance_plan.anchor_joints + 0.0001,
    )

    assert repaired
    saved = load_real_session(session_path)
    saved_row = saved["rows"][-1]
    assert saved_row["return_success"]
    assert saved_row["attempt_accepted"]
    assert saved_row["return_recovered_after_stop"]
    assert saved_row["stop_stage"] == "return_recovered_after_stop"
    assert saved["blocked"]
    assert len(saved["recovery_events"]) == 1


def _successful_final_session(acceptance_plan):
    rows = []
    for attempt_index in range(ACCEPTANCE_TOTAL_TARGET_ATTEMPTS):
        row = _empty_attempt_row(attempt_index)
        row.update({
            "target_success": True,
            "target_code": 0,
            "target_reason": "目标已稳定到达",
            "target_solve_time_ms": 0.5,
            "target_position_error_m": 0.001,
            "target_orientation_error_rad": 0.001,
            "target_sigma_min": 0.18,
            "target_condition_number": 9.5,
            "target_max_joint_delta_rad": 0.05,
            "target_response_latency_s": 0.10,
            "target_motion_start_detected": True,
            "target_total_duration_s": 1.0,
            "return_success": True,
            "return_code": 0,
            "return_reason": "目标已稳定到达",
            "return_solve_time_ms": 0.4,
            "return_position_error_m": 0.0005,
            "return_orientation_error_rad": 0.0005,
            "return_sigma_min": 0.18,
            "return_condition_number": 9.5,
            "return_max_joint_delta_rad": 0.05,
            "return_response_latency_s": 0.09,
            "return_motion_start_detected": True,
            "return_total_duration_s": 1.0,
            "return_joint_error_rad": 0.001,
            "attempt_accepted": True,
        })
        rows.append(row)
    return {
        "dataset_version": "week1-30-v2",
        "dataset_id": acceptance_plan.dataset_id,
        "anchor_joints_rad": acceptance_plan.anchor_joints.tolist(),
        "anchor_transform": acceptance_plan.anchor_transform.tolist(),
        "targets": [
            {
                "index": target.index,
                "target_id": target.target_id,
                "target_transform": target.target_transform.tolist(),
                "reference_joints_rad": target.reference_joints.tolist(),
                "position_offset_m": target.position_offset_m,
                "orientation_offset_rad": target.orientation_offset_rad,
            }
            for target in acceptance_plan.targets
        ],
        "rows": rows,
        "blocked": False,
        "operator_safe_batches": [{
            "no_collision_confirmed": True,
            "start_attempt": 0,
            "end_attempt_exclusive": 90,
        }],
    }


def test_completed_real_report_contains_only_stable_summary_and_details(
    tmp_path,
    acceptance_plan,
):
    session = _successful_final_session(acceptance_plan)
    csv_path, json_path, summary = write_final_real_report(
        session,
        tmp_path,
    )

    assert csv_path.name == "real_30x3.csv"
    assert json_path.name == "real_30x3.json"
    assert summary["passed"]
    assert summary["target_attempts"] == 90
    assert summary["target_success_rate"] == 1.0
    assert summary["response_latency_measurements"] == 90
    assert len(summary["target_definitions"]) == 30
    assert summary["acceptance"][
        "all_targets_change_orientation_by_at_least_3_deg"
    ]
    assert summary["acceptance"][
        "dataset_reaches_at_least_30_mm_translation"
    ]
    assert summary["acceptance"][
        "dataset_reaches_at_least_10_deg_orientation"
    ]
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert "rows" not in persisted
    assert persisted["artifacts"]["details_csv"] == "real_30x3.csv"


def test_missing_latency_or_operator_confirmation_fails_formal_report(
    tmp_path,
    acceptance_plan,
):
    session = _successful_final_session(acceptance_plan)
    session["rows"][0]["target_response_latency_s"] = ""
    session["rows"][0]["target_motion_start_detected"] = False
    session["operator_safe_batches"] = []

    _, _, summary = write_final_real_report(session, tmp_path)

    assert not summary["passed"]
    assert not summary["acceptance"][
        "response_latency_p95_at_most_200_ms"
    ]
    assert not summary["acceptance"][
        "operator_confirmed_zero_collisions_for_every_batch"
    ]


def test_promote_axis_rejects_preview_and_keeps_only_stable_paths(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "completed": True,
        "passed": True,
        "rows": [{"index": index} for index in range(6)],
        "csv_path": "/hidden/temporary.csv",
        "json_path": "/hidden/temporary.json",
    }), encoding="utf-8")

    csv_path, json_path = promote_axis_result(source, tmp_path / "visible")

    assert csv_path.name == "axis_15mm.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "csv_path" not in payload
    assert "json_path" not in payload
    assert payload["artifacts"]["details_csv"] == "axis_15mm.csv"

    source.write_text(json.dumps({
        "completed": False,
        "passed": False,
        "rows": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        promote_axis_result(source, tmp_path / "visible2")


def test_real_batch_is_preview_only_by_default_and_capped_at_five_targets():
    parser = _argument_parser()
    preview = parser.parse_args([
        "real-batch",
        "--session",
        "/tmp/session.json",
    ])
    assert not preview.execute
    assert preview.batch_targets == 5

    with pytest.raises(SystemExit):
        parser.parse_args([
            "real-batch",
            "--session",
            "/tmp/session.json",
            "--batch-targets",
            "6",
        ])


def _valid_ros_ik_result():
    result = IKResult()
    result.success = True
    result.code = IKResult.SUCCESS
    result.reason = "IK solution accepted"
    result.solution_joint_state.name = [
        f"joint{index}" for index in range(1, 8)
    ]
    result.solution_joint_state.position = [0.01] * 7
    result.position_error_m = 0.0005
    result.orientation_error_rad = 0.001
    result.solve_time_ms = 0.5
    result.sigma_min = 0.18
    result.condition_number = 9.5
    result.max_joint_delta_rad = 0.05
    return result


def test_batch_requires_both_confirmations_and_checkpoints_each_pair(
    tmp_path,
    acceptance_plan,
):
    session_path = create_real_session(acceptance_plan, tmp_path)
    session = load_real_session(session_path)
    state = SimpleNamespace(positions=np.asarray(READY_JOINT_POSITIONS))
    execution_result = SimpleNamespace(
        ik_result=_valid_ros_ik_result(),
        final_position_error_m=0.0005,
        final_orientation_error_rad=0.001,
    )

    class Harness:
        calls = []
        last_execution_observation = ExecutionObservation(
            response_latency_s=0.1,
            total_duration_s=1.0,
            motion_start_detected=True,
        )

        @staticmethod
        def measured_state():
            return state

        @staticmethod
        def _require_session_anchor(_session, _state):
            pass

        @staticmethod
        def _solve_pose_preview(
            _pose,
            label,
            posture_reference=None,
        ):
            Harness.calls.append(("preview", label))
            if "回程" in label:
                assert posture_reference is not None
            return _valid_ros_ik_result()

        @staticmethod
        def execute_placo(_pose, _state, **kwargs):
            Harness.calls.append(("execute", kwargs["label"]))
            return execution_result

        @staticmethod
        def get_clock():
            return SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: Time())
            )

    answers = iter(("RUN_BATCH", "BATCH_SAFE"))
    result = Week1AcceptanceNode.execute_batch(
        Harness(),
        session,
        0,
        3,
        lambda _prompt: next(answers),
        False,
        tmp_path / "visible",
    )

    assert result is None
    reloaded = load_real_session(session_path)
    assert reloaded["next_attempt_index"] == 3
    assert len(reloaded["rows"]) == 3
    assert all(row["attempt_accepted"] for row in reloaded["rows"])
    assert len(reloaded["operator_safe_batches"]) == 1
    assert len([call for call in Harness.calls if call[0] == "execute"]) == 6


def test_wrong_batch_confirmation_executes_nothing(tmp_path, acceptance_plan):
    session = load_real_session(create_real_session(acceptance_plan, tmp_path))

    class Harness:
        @staticmethod
        def execute_placo(*_args, **_kwargs):
            pytest.fail("wrong confirmation must not execute")

    with pytest.raises(Exception, match="确认词不匹配"):
        Week1AcceptanceNode.execute_batch(
            Harness(),
            session,
            0,
            3,
            lambda _prompt: "WRONG",
            False,
            tmp_path / "visible",
        )
