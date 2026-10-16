"""The human report must preserve the distinction between motion and mapping."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("result_report", ROOT / "tools/demo/result_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _mapped_step():
    return {"step_index": 1, "execution": {"action_status": 4},
            "post_map_update": {"observation_id": "post1"},
            "coverage": {"before": .1, "after": .2, "delta": .1},
            "cumulative_camera_motion": {"planned_step_translation_m": .1, "actual_step_translation_m": .098}}


def test_startup_recovery_is_not_counted_as_nbv_motion(tmp_path):
    (tmp_path / "recovery").mkdir()
    (tmp_path / "recovery/execution.json").write_text(json.dumps({"executed": True, "max_joint_delta_rad": .1}))
    document = {"status": "failed", "motion_goal_count": 0,
                "reason": "raw NextView camera translation 0.100008714 m exceeds 0.100000001 m beyond the numerical tolerance",
                "raw_next_view": {"coverage": .088944},
                "selected_candidate": {"camera_translation_m": .1}}
    data = report.build_overview(tmp_path, {}, document)
    assert data["startup_recovery_executed"] is True
    assert data["completed_nbv_actions"] == data["mapped_motion_steps"] == 0
    assert data["final_roi_ray_coverage"] is None
    assert data["coverage_gain_pp"] is None
    assert "100.008714 mm" in data["stop_explanation"]
    assert data["motion_rows"][0]["result"] == "计划已生成，未执行"


def test_action_without_post_observation_stays_separate_from_mapped_step(tmp_path):
    document = {"status": "failed", "motion_goal_count": 2,
                "session_progress": {"motion_goal_count": 1},
                "motion_steps": [_mapped_step(), {"step_index": 2, "execution": {"action_status": 4},
                                                     "selected_candidate": {"camera_translation_m": .075}}]}
    data = report.build_overview(tmp_path, {}, document)
    assert data["completed_nbv_actions"] == 2
    assert data["mapped_motion_steps"] == 1
    assert data["coverage_gain_pp"] == 10
    assert data["motion_rows"][1]["actual_mm"] is None
    assert data["motion_rows"][1]["coverage_gain_pp"] is None


def test_terminal_candidate_counts_are_not_taken_from_initial_selection(tmp_path):
    initial = {"ik_gate_passed": True, "gain_scored": True, "gain_improvement": .2}
    terminal = [{"target_view_distance_gate_passed": False},
                {"ik_requested": True, "ik_gate_passed": False, "rejection": "joint limit clearance too small"},
                {"ik_gate_passed": True, "gain_scored": True, "gain_improvement": -.01}]
    step = _mapped_step()
    step["next_view_convergence"] = {"ik_candidates": terminal}
    document = {"status": "executed_session_converged", "motion_goal_count": 1,
                "reason": "no positive-gain reachable candidate remained",
                "ik_candidates": [initial], "motion_steps": [step]}
    data = report.build_overview(tmp_path, {}, document)
    assert data["candidate_phase"] == "停止前最后一次选点"
    assert data["candidates"]["total"] == 3
    assert data["candidates"]["reachable"] == 1
    assert data["candidates"]["positive"] == 0
    assert sum(data["candidates"]["categories"].values()) == 3
    assert "本轮候选集合" in data["stop_explanation"]


def test_report_escapes_raw_reason_and_keeps_evidence_collapsed(tmp_path):
    for name in ("execution.json", "run.json"):
        (tmp_path / name).touch()
    document = {"status": "failed", "reason": "<script>unexpected()</script>"}
    page = report.render_index(tmp_path, {}, document, ['<li><a href="execution.json">执行记录</a></li>'])
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert '<details id="evidence">' in page
    assert '<details id="evidence" open' not in page
    assert page.split('<details id="evidence">')[0].count('<a ') <= 4
    assert "不是草莓表面的完整度" in page


def test_report_does_not_turn_depth_failure_into_target_disappearance(tmp_path):
    document = {"status": "failed", "reason": "target mask has 0 pixels; need at least 1"}
    data = report.build_overview(tmp_path, {}, document)
    assert "不能断定草莓已经离开画面" in data["stop_explanation"]


def test_text_reports_normal_loop_separately_from_unmet_scientific_acceptance(tmp_path):
    step = _mapped_step()
    step["next_view_convergence"] = {"ik_candidates": [
        {"ik_gate_passed": True, "gain_scored": True, "gain_improvement": -.01},
        {"target_view_distance_gate_passed": False},
    ]}
    document = {"status": "executed_session_converged_scientific_acceptance_not_met",
                "reason": "no positive-gain reachable candidate remained",
                "motion_goal_count": 1, "motion_steps": [step],
                "session_progress": {"scientific_acceptance_passed": False, "coverage_target": .8}}
    text = report.render_text(tmp_path, {}, document)
    assert "闭环已正常运行" in text
    assert "1 次运动确认到位，1 次运动后更新地图" in text
    assert "10.00% → 20.00%，提升 10.00 个百分点" in text
    assert "共检查 2 个候选，1 个通过可达性检查，0 个有正收益" in text
    assert "实测 98.00 mm" in text
    assert "当前 20.00%，尚未达到" in text
    assert "不等于科研验收通过" in text
    assert "不是草莓表面完整度" in text
    assert "<html" not in text and "<p>" not in text


def test_text_does_not_fill_missing_measured_motion_from_planned_distance(tmp_path):
    document = {"status": "failed", "motion_goal_count": 2, "motion_steps": [
        _mapped_step(), {"step_index": 2, "execution": {"action_status": 4},
                         "selected_candidate": {"camera_translation_m": .075}},
    ]}
    text = report.render_text(tmp_path, {}, document)
    assert "2 次运动确认到位，1 次运动后更新地图" in text
    assert "第 2 步：计划 75.00 mm；实测位移未记录；没有本步地图增量" in text
    assert "闭环已正常运行" not in text


def test_text_keeps_startup_recovery_out_of_nbv_counts(tmp_path):
    (tmp_path / "recovery").mkdir()
    (tmp_path / "recovery/execution.json").write_text(json.dumps({"executed": True, "max_joint_delta_rad": .1}))
    text = report.render_text(tmp_path, {}, {"status": "failed", "motion_goal_count": 0})
    assert "0 次运动确认到位，0 次运动后更新地图" in text
    assert "启动恢复：已执行，最大关节变化约 5.73°" in text
    assert "不计入 NBV 运动次数" in text
