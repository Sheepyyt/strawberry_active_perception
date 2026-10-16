"""Chinese, evidence-derived overview for a local demonstration result.

This module only reads artifacts and returns HTML/data. It never controls the
robot, rewrites execution evidence, or changes acceptance thresholds.
"""
from __future__ import annotations

from collections import Counter
import html
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _read(directory: Path, relative: str) -> dict:
    for path in (directory / relative, directory / "evidence" / relative):
        try:
            result = json.loads(path.read_text())
            if isinstance(result, dict):
                return result
        except (OSError, ValueError):
            pass
    return {}


def candidate_counts(records: Sequence[Mapping]) -> dict:
    """Count the logged gates without conflating IK failure and lack of gain."""
    counts = Counter()
    reachable = scored = positive = 0
    for record in records:
        reachable += bool(record.get("ik_gate_passed"))
        scored += bool(record.get("gain_scored"))
        gain = _number(record.get("gain_improvement"))
        current = _number(record.get("current_gain")) or 0.0
        improves = gain is not None and gain > max(1e-9, abs(current) * 1e-6)
        positive += bool(record.get("gain_scored") and improves)
        reason = f"{record.get('rejection', '')} {record.get('ik_reason', '')}".lower()
        if record.get("target_view_distance_gate_passed") is False:
            label = "相机距离过近"
        elif record.get("depth_gate_passed") is False:
            label = "超出深度工作范围"
        elif record.get("gain_scored"):
            label = "有正收益" if improves else "可达，但没有新增收益"
        elif record.get("ik_gate_passed"):
            label = "可达，尚未评分"
        elif "clearance" in reason or "margin" in reason or "限位余量" in reason:
            label = "关节限位余量不足"
        elif "joint delta" in reason or "candidate changes" in reason or "关节变化" in reason:
            label = "关节变化未通过检查"
        elif "residual" in reason:
            label = "逆运动学精度不足"
        elif "singular" in reason or "奇异" in reason:
            label = "接近奇异姿态"
        elif "rotation" in reason and "maximum" in reason:
            label = "相机转角未通过检查"
        elif record.get("ik_requested"):
            label = "逆运动学未通过"
        else:
            label = "其他候选检查未通过"
        counts[label] += 1
    return {"total": len(records), "reachable": reachable, "scored": scored,
            "positive": positive, "categories": dict(counts)}


def build_overview(directory: Path, summary: Mapping, document: Mapping) -> dict:
    """Keep startup recovery, NBV actions, and map updates as separate facts."""
    directory = Path(directory)
    steps = [step for step in document.get("motion_steps", []) if isinstance(step, dict)]
    completed = sum(step.get("execution", {}).get("action_status") == 4 for step in steps)
    motion_goals = int(document.get("motion_goal_count", 0))
    mapped = [step for step in steps if "coverage" in step and "post_map_update" in step]
    progress = document.get("session_progress") or {}
    initial = _number(progress.get("initial_coverage"))
    if initial is None:
        initial = _number(mapped[0].get("coverage", {}).get("before")) if mapped else _number(document.get("raw_next_view", {}).get("coverage"))
    final = _number(mapped[-1].get("coverage", {}).get("after")) if mapped else None
    recovery = _read(directory, "recovery/report.json")
    recovery_action = _read(directory, "recovery/execution.json")
    recovery_executed = bool(recovery_action.get("executed") or recovery.get("executed"))
    recovery_delta = _number(recovery_action.get("max_joint_delta_rad"))
    status = str(summary.get("status") or document.get("status") or "unknown")
    reason = str(summary.get("reason") or document.get("reason") or progress.get("termination_reason") or "")
    terminal = next((step.get("next_view_convergence") for step in reversed(steps)
                     if isinstance(step.get("next_view_convergence"), dict)), None)
    terminal_records = terminal.get("ik_candidates", []) if terminal else []
    records = terminal_records or document.get("ik_candidates", [])
    candidate_phase = "停止前最后一次选点" if terminal_records else "首步候选评估"
    counts = candidate_counts(records) if records else None
    selected = document.get("selected_candidate") or {}
    selected_mm = _number(selected.get("camera_translation_m", selected.get("translation_m")))
    selected_mm = selected_mm * 1000 if selected_mm is not None else None

    if status == "cleanup_unverified":
        headline = "本轮已停止，停止状态尚未完整确认"
    elif completed == 0 and motion_goals > 0:
        headline = "本轮已请求主动感知运动，尚未确认完成"
    elif completed == 0:
        headline = "本轮执行了启动恢复，主动感知尚未移动" if recovery_executed else "本轮尚未执行主动感知运动"
    elif status == "failed":
        headline = f"完成 {completed} 次主动感知运动后停止"
    else:
        headline = f"完成 {completed} 次主动感知运动，{len(mapped)} 次运动后更新地图"

    excess = re.search(r"camera translation ([0-9.eE+-]+) m exceeds ([0-9.eE+-]+) m", reason)
    if excess:
        measured, limit = map(float, excess.groups())
        stop_explanation = (
            f"执行前步长复核未通过：按刷新后的相机起点计算，计划位移为 {measured * 1000:.6f} mm，"
            f"比本轮 {limit * 1000:.3f} mm 边界多 {max(0.0, measured-limit) * 1000:.6f} mm，超过当时程序使用的数值容差。"
        )
        if motion_goals == 0:
            stop_explanation += "本轮没有发出主动感知运动。"
    elif "no positive-gain reachable candidate" in reason:
        stop_explanation = "在最后一轮搜索的候选视角中，没有同时通过可达性检查且能增加地图信息的视角，所以停止。"
        if counts and terminal_records:
            stop_explanation += f"共检查 {counts['total']} 个候选，{counts['reachable']} 个通过可达性检查，{counts['positive']} 个有正收益。"
        stop_explanation += "这个结论只针对本轮候选集合。"
    elif "mask has 0 pixels" in reason or "depth" in reason.lower() and status == "failed":
        stop_explanation = "后续观测未提供足够的有效目标深度，程序停止继续运动。仅凭这个错误不能断定草莓已经离开画面。"
    elif "coverage" in reason.lower() and "target" in reason.lower() and progress.get("coverage_target_reached"):
        stop_explanation = "ROI 射线覆盖率已达到本轮设置的停止目标。"
    elif "plateau" in reason.lower():
        stop_explanation = "连续观测带来的覆盖率提升低于本轮阈值，程序按已有停止规则结束。"
    elif status == "passed_preview_only":
        stop_explanation = "本轮只完成预览，没有请求主动感知运动。"
    elif status == "initialization_only":
        stop_explanation = "本轮只完成设备与草莓检查，尚未开始主动感知。"
    else:
        stop_explanation = reason or "当前记录没有给出停止原因。"

    rows = []
    for i, step in enumerate(steps, 1):
        motion = step.get("cumulative_camera_motion") or {}
        chosen = step.get("selected_candidate") or {}
        planned = _number(motion.get("planned_step_translation_m", chosen.get("camera_translation_m", chosen.get("translation_m"))))
        actual = _number(motion.get("actual_step_translation_m"))
        gain = _number(step.get("coverage", {}).get("delta"))
        mapped_here = step in mapped
        done = step.get("execution", {}).get("action_status") == 4
        rows.append({"step": step.get("step_index", i), "planned_mm": planned * 1000 if planned is not None else None,
                     "actual_mm": actual * 1000 if actual is not None else None,
                     "coverage_gain_pp": gain * 100 if gain is not None else None,
                     "result": "到位并更新地图" if mapped_here else "运动到位，缺少后续地图" if done else "未确认完成运动"})
    if not rows and selected_mm is not None:
        rows.append({"step": 1, "planned_mm": selected_mm, "actual_mm": None,
                     "coverage_gain_pp": None, "result": "计划已生成，未执行" if motion_goals == 0 else "已请求运动，结果未记录"})
    return {
        "schema": "strawberry_demo_human_overview/v1", "run": directory.name,
        "headline": headline, "status": status, "stop_explanation": stop_explanation,
        "raw_reason": reason, "startup_recovery_executed": recovery_executed,
        "startup_recovery_max_joint_delta_deg": math.degrees(recovery_delta) if recovery_delta is not None else None,
        "nbv_motion_goals": motion_goals, "completed_nbv_actions": completed,
        "mapped_motion_steps": len(mapped), "initial_roi_ray_coverage": initial,
        "final_roi_ray_coverage": final,
        "coverage_gain_pp": (final-initial)*100 if final is not None and initial is not None else None,
        "scientific_acceptance_passed": progress.get("scientific_acceptance_passed"),
        "coverage_target": _number(progress.get("coverage_target")),
        "candidate_phase": candidate_phase, "candidates": counts, "motion_rows": rows,
    }


def render_text(directory: Path, summary: Mapping, document: Mapping) -> str:
    """Return the same evidence as readable terminal/TXT text, without HTML."""
    data = build_overview(Path(directory), summary, document)
    normal_end = (data["status"].startswith("executed_")
                  and data["completed_nbv_actions"] > 0
                  and data["completed_nbv_actions"] == data["mapped_motion_steps"])
    conclusion = "主动感知闭环已正常运行，并按停止条件结束。" if normal_end else data["headline"] + "。"
    lines = ["草莓主动感知 · 本轮结果", f"运行：{data['run']}", "", f"结论：{conclusion}",
             f"主动感知（NBV）：{data['completed_nbv_actions']} 次运动确认到位，"
             f"{data['mapped_motion_steps']} 次运动后更新地图（共请求 {data['nbv_motion_goals']} 次运动）。"]
    if data["startup_recovery_executed"]:
        delta = data["startup_recovery_max_joint_delta_deg"]
        detail = f"，最大关节变化约 {delta:.2f}°" if delta is not None else ""
        lines.append(f"启动恢复：已执行{detail}；这是起始关节向内恢复，不计入 NBV 运动次数。")
    else:
        lines.append("启动恢复：本轮记录中没有恢复动作，与 NBV 运动次数分开统计。")

    initial, final = data["initial_roi_ray_coverage"], data["final_roi_ray_coverage"]
    if initial is not None and final is not None:
        lines += ["", f"ROI 射线覆盖率：{initial*100:.2f}% → {final*100:.2f}%，"
                  f"提升 {data['coverage_gain_pp']:.2f} 个百分点。"]
    elif initial is not None:
        lines += ["", f"ROI 射线覆盖率：只有起始观测 {initial*100:.2f}%；没有运动后地图，无法比较提升。"]
    else:
        lines += ["", "ROI 射线覆盖率：本轮尚无可用的覆盖率记录。"]
    lines.append("该指标表示草莓附近 ROI 中被相机射线观测过的体素比例，不是草莓表面完整度。")
    lines += ["", "为何停止：" + data["stop_explanation"]]

    if data["motion_rows"]:
        lines += ["", "逐步执行记录（相机平移）："]
        for row in data["motion_rows"]:
            planned = f"计划 {row['planned_mm']:.2f} mm" if row["planned_mm"] is not None else "计划位移未记录"
            actual = f"实测 {row['actual_mm']:.2f} mm" if row["actual_mm"] is not None else "实测位移未记录"
            gain = (f"ROI 增加 {row['coverage_gain_pp']:.2f} 个百分点"
                    if row["coverage_gain_pp"] is not None else "没有本步地图增量")
            lines.append(f"  第 {row['step']} 步：{planned}；{actual}；{gain}；{row['result']}。")
        lines.append("计划位移不代表已执行；缺少实测值的步骤不会按计划值补算。")

    acceptance = data["scientific_acceptance_passed"]
    if acceptance is not None:
        lines.append("")
        if acceptance:
            lines.append("科研验收：本轮程序记录的验收条件已满足。")
        else:
            lines.append("科研验收：本轮程序记录的验收条件尚未满足。")
            target = data["coverage_target"]
            if target is not None and final is not None and final < target:
                lines.append(f"本轮覆盖率目标为 {target*100:.2f}%，当前 {final*100:.2f}%，尚未达到。")
            if normal_end:
                lines.append("闭环正常结束表示观测、选点和执行流程已经跑通；不等于科研验收通过。")
    return "\n".join(lines) + "\n"


def render_index(directory: Path, summary: Mapping, document: Mapping, links: Sequence[str] = ()) -> str:
    """Return one Chinese overview; the caller owns writing and hashing files."""
    directory = Path(directory)
    data = build_overview(directory, summary, document)
    esc = lambda value: html.escape(str(value), quote=True)
    fmt = lambda value, suffix="": "—" if value is None else f"{value:.2f}{suffix}"
    gain = fmt(data["coverage_gain_pp"], " 个百分点")
    cards = [("启动关节恢复", "已执行" if data["startup_recovery_executed"] else "未执行"),
             ("主动感知运动到位", f"{data['completed_nbv_actions']} 次"),
             ("运动后完成建图", f"{data['mapped_motion_steps']} 次"), ("ROI 覆盖率新增", gain)]
    style = """body{margin:0;background:#f5f7fa;color:#152638;font:16px/1.7 system-ui,sans-serif}
main{max-width:1080px;margin:auto;padding:36px 24px}h1{font-size:30px;line-height:1.35;margin:12px 0}h2{font-size:21px;margin:0 0 12px}
p{margin:10px 0}.muted,figcaption{color:#566579}.label{font-size:13px;color:#566579}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}
.card,section,details{background:white;border:1px solid #e0e6ee;border-radius:12px;padding:20px}.card strong{display:block;font-size:23px;margin-top:8px}.card{padding:16px}
section{margin:18px 0}.notice{border-left:4px solid #d69b3a;background:#fff8e9;padding:12px 16px}.links{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}a{color:#1c5ab8}
table{width:100%;border-collapse:collapse;font-size:15px}th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #e4e9f0}th{font-weight:600;color:#566579}
.scroll{overflow-x:auto}figure{margin:20px 0 0}img{display:block;width:100%;height:auto;border-radius:8px}figcaption{font-size:14px;margin-top:8px}details{margin:16px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}ul{padding-left:22px}.chips{display:flex;gap:8px;flex-wrap:wrap}.chips span{background:#edf3fa;border-radius:6px;padding:5px 10px;font-size:14px}
@media(max-width:700px){main{padding:20px 14px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}.card strong{font-size:20px}}@media print{body{background:white}main{padding:0}details,.links{display:none}section{break-inside:avoid}}
"""
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>草莓主动感知 · 本轮结果</title><style>{style}</style></head><body><main>'
    page += f'<div class="label">草莓主动感知 · 本轮结果</div><h1>{esc(data["headline"])}</h1><p class="muted">{esc(directory.name)}</p>'
    page += '<div class="cards">' + ''.join(f'<div class="card"><span class="label">{esc(label)}</span><strong>{esc(value)}</strong></div>' for label, value in cards) + '</div>'
    page += '<section><h2>这轮具体发生了什么</h2>'
    if data["startup_recovery_executed"]:
        delta = data["startup_recovery_max_joint_delta_deg"]
        detail = f"，最大关节变化约 {delta:.2f}°" if delta is not None else ""
        page += f'<p>先执行了一次启动关节恢复{detail}。这是把起始越界关节向内调整，单独记录，不计作主动感知的下一视角运动。</p>'
    page += f'<p>{esc(data["stop_explanation"])}</p>'
    initial, final = data["initial_roi_ray_coverage"], data["final_roi_ray_coverage"]
    if final is not None and initial is not None:
        page += f'<p>运动前后 ROI 射线覆盖率：<strong>{initial*100:.2f}% → {final*100:.2f}%</strong>，增加 <strong>{gain}</strong>。</p>'
    elif initial is not None:
        page += f'<p>仅有起始观测的 ROI 射线覆盖率：<strong>{initial*100:.2f}%</strong>。没有运动后地图，因此没有前后提升可比较。</p>'
    page += '<p class="muted">ROI 射线覆盖率表示草莓附近目标区域有多少体素被相机射线观测过；它不是草莓表面的完整度。</p>'
    if data["scientific_acceptance_passed"] is False:
        page += '<p class="notice">本轮程序记录的科研验收条件尚未满足。运动、地图更新和覆盖率变化按实际完成情况展示。</p>'
    page += '</section>'
    if data["motion_rows"]:
        page += '<section><h2>每一步做了多少运动</h2><div class="scroll"><table><thead><tr><th>NBV 步骤</th><th>计划相机平移</th><th>实测相机平移</th><th>新增 ROI 覆盖</th><th>结果</th></tr></thead><tbody>'
        for row in data["motion_rows"]:
            page += '<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in (row["step"], fmt(row["planned_mm"], " mm"), fmt(row["actual_mm"], " mm"), fmt(row["coverage_gain_pp"], " 个百分点"), row["result"])) + '</tr>'
        page += '</tbody></table></div><p class="muted">“—”表示没有对应实测值。计划位移不等于已经执行的位移。</p></section>'
    counts = data["candidates"]
    if counts:
        page += f'<section><h2>算法比较了哪些选择</h2><p>{esc(data["candidate_phase"])}：检查 <strong>{counts["total"]}</strong> 个候选，<strong>{counts["reachable"]}</strong> 个通过可达性检查，<strong>{counts["positive"]}</strong> 个有正收益。</p>'
        if counts["positive"] and data["nbv_motion_goals"] == 0 and data["status"] == "failed":
            page += '<p>本轮已有可达且有收益的视角；停止发生在随后执行复核阶段，不能把它解释成算法没有找到有用视角。</p>'
        page += '<div class="chips">' + ''.join(f'<span>{esc(label)}：{count}</span>' for label, count in counts["categories"].items()) + '</div></section>'

    charts = [
        ("figures/session/01_coverage_curve.png", "覆盖率是否提高", "横轴是完成观测的步骤，纵轴是 ROI 射线覆盖率；相邻点的差值是该次观测带来的提升。图上的停止目标是预设阈值，并不代表每轮都能达到。"),
        ("figures/observations/observation_contact_sheet.png", "相机实际看到了什么", "画面按本轮执行观测顺序排列，可对照草莓是否仍在画面及分割位置是否稳定；深度有效性以统计记录为准。" + ("本轮只有起始观测，没有运动后画面。" if not data["mapped_motion_steps"] else "只有已完成采集的观测会出现在这里。")),
        ("figures/session/02_camera_trajectory_3d.png", "相机去了哪里", "位置与朝向使用机械臂基座坐标系。轨迹仅包含完成运动后观测的步骤；若某次运动后没有有效观测，不能从这张图读取那一段实测路径。"),
    ]
    available = [item for item in charts if (directory / item[0]).is_file()]
    if not (directory / "figures/observations/observation_contact_sheet.png").is_file() and (directory / "initialization/overlay.png").is_file():
        available.append(("initialization/overlay.png", "启动时识别到的草莓", "这是启动检查时的分割画面，用来确认本轮选中的目标；不代表运动后的观测。"))
    if (directory / "figures/voxel/voxel_cloud_final_3d.png").is_file():
        available.append(("figures/voxel/voxel_cloud_final_3d.png", "本轮形成的三维地图", "红色点是分割支持的目标，灰色点是测得的表面，浅蓝色是已观测空间，绿色框是目标 ROI。单视角只能记录当前可见部分；这张图不是完整草莓模型。"))
    if available:
        page += '<section><h2>看这几张图就够了</h2>'
        for path, title, caption in available:
            page += f'<figure><h3>{esc(title)}</h3><img loading="lazy" src="{esc(path)}" alt="{esc(title)}"><figcaption>{esc(caption)}</figcaption></figure>'
        page += '</section>'
    if summary.get("render_errors"):
        page += '<p class="notice">部分图表生成失败；上面的运动次数和停止原因来自执行记录。技术详情见原始证据中的 render.json。</p>'
    primary = [("figures/session/index.html", "其他图表"), ("execution.json", "本轮执行记录"), ("run.json", "本轮配置记录")]
    page += '<nav class="links">' + ''.join(f'<a href="{esc(path)}">{esc(label)}</a>' for path, label in primary if (directory / path).is_file()) + '<a href="#evidence">原始证据</a></nav>'
    page += '<details id="evidence"><summary>展开原始证据与技术记录</summary><p>以下文件保留用于复查和复现，日常查看上面的摘要即可。</p>'
    page += '<h3>原始状态与停止原因</h3><pre>' + esc(json.dumps({"status": data["status"], "reason": data["raw_reason"]}, ensure_ascii=False, indent=2)) + '</pre>'
    page += '<h3>文件清单</h3><ul>' + ''.join(links) + '</ul></details></main></body></html>'
    return page
