"""Generate a reproducible, offline weekly report from one real run."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .documents import write_documents
from .metrics import analyze_run
from .plots import render_figures


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, content) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    # UTF-8 BOM lets desktop spreadsheet software recognize Chinese headings.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([title for key, title in columns])
        for row in rows:
            writer.writerow([row.get(key) for key, title in columns])


def export_tables(analysis: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Keep candidate rows once; original transforms remain in SHA-linked input.
    serialized = {**analysis, "rounds": [
        {key: value for key, value in row.items() if key != "candidate_rows"}
        for row in analysis["rounds"]
    ]}
    _json(output / "analysis.json", serialized)
    _csv(output / "视角统计.csv", analysis["views"], [
        ("index", "视角序号_0为初始"), ("observation_id", "观测编号"),
        ("coverage", "ROI覆盖率_0到1"), ("coverage_gain_pp", "覆盖率增加_百分点"),
        ("roi_observed_voxels", "ROI已观测体素数"),
        ("target_voxels", "目标支持体素数"), ("target_added", "新标作目标体素数"),
        ("target_removed", "不再标作目标体素数"), ("mask_pixels", "掩膜像素数"),
        ("valid_depth_pixels", "掩膜内有效深度像素数"),
        ("valid_depth_fraction", "掩膜内有效深度比例_0到1"),
        ("median_depth_mm", "近层中位深度_mm"), ("depth_mad_mm", "近层深度MAD_mm"),
        ("actual_translation_mm", "实际相机平移_mm"), ("position_error_mm", "到位位置误差_mm"),
    ])
    common = [("round_index", "决策轮次"), ("label", "决策来源")]
    by_radius = [{"round_index": item["round_index"], "label": item["label"], **row}
                 for item in analysis["rounds"] for row in item["radii"]]
    _csv(output / "候选半径统计.csv", by_radius, common + [
        ("radius_mm", "半径_mm"), ("total", "候选总数"), ("reachable", "已证实可达数"),
        ("scored", "已评分数"), ("positive", "正收益数"),
    ])
    thresholds = [{"round_index": item["round_index"], "label": item["label"], **row}
                  for item in analysis["rounds"] for row in item["thresholds"]]
    _csv(output / "距离门限统计.csv", thresholds, common + [
        ("minimum_depth_m", "目标光轴深度门限_m"), ("depth_pass", "距离通过数"),
        ("reachable", "保留的已证实可达数"), ("scored", "保留的已评分数"),
        ("positive", "保留的正收益数"), ("best_gain_improvement", "最佳已评分收益差_空值为无评分"),
    ])
    _csv(output / "候选明细.csv", analysis["candidate_rows"], [
        ("round_index", "决策轮次"), ("source", "源记录"), ("candidate_index", "候选编号"),
        ("radius_mm", "半径_mm"), ("target_depth_m", "预测目标光轴深度_m"),
        ("ik_gate_passed", "IK可达检查通过"), ("gain_scored", "收益已评分"),
        ("current_gain", "当前收益分数"), ("candidate_gain", "候选收益分数"),
        ("gain_improvement", "收益差"), ("gain_tolerance", "正收益判定容差"),
        ("positive", "有正收益_未评分为空"),
    ])
    (output / "统计口径.txt").write_text(
        "数据来自一轮真机运行，按执行记录的 scene_id 与 observation_id 配对。\n"
        "初始观测作为基线；其目标体素新增/消失均填0，不代表初始地图没有目标。\n"
        "覆盖率分母为目标ROI体素数，不是全地图体素数，也不是草莓表面面积。\n"
        "目标支持体素来自语义分类；集合新增减去消失才等于净增。\n"
        "有效深度比例按掩膜内有限且位于本轮冻结深度范围的像素计算。\n"
        "近层中位深度和MAD读取实际定位证据，不把整个掩膜混合深度代作近层统计。\n"
        "首步候选读取SHA绑定预演；后续读取实际选点与终止记录，不混入首步执行复核。\n"
        "收益差是规划器分数，不是覆盖率百分点；正收益使用原程序的数值容差。\n"
        "距离门限分析仅事后筛选已记录候选，不补造被拒绝候选的IK或收益。\n"
        "CSV空值表示无对应记录，不能当作0。门限变化结果不是新策略真机轨迹。\n",
        encoding="utf-8",
    )


def generate_weekly(run: Path, output: Path) -> list[Path]:
    run, output = run.resolve(), output.resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("周报输出目录必须与原始运行目录分开")
    print("正在离线核对执行观测、地图和候选来源……", flush=True)
    analysis = analyze_run(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".weekly-build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        export_tables(analysis, stage / "data")
        print("正在生成三张分析图及三页 Word / PDF……", flush=True)
        figures = render_figures(analysis, stage / "figures")
        documents = write_documents(analysis, figures, stage)
        for source, expected in analysis["source_hashes"].items():
            if _sha(Path(source)) != expected:
                raise RuntimeError(f"分析期间输入文件发生变化，未发布本次结果：{source}")
        outputs = {str(path.relative_to(stage)): _sha(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        code = {str(path.resolve()): _sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {
            "schema": "strawberry_weekly_analysis/v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": analysis["run_id"], "scene_id": analysis["scene_id"],
            "input_sha256": analysis["source_hashes"], "code_sha256": code, "output_sha256": outputs,
            "reproduce_command": f"./demo_operator.sh analyze {run}",
            "scope": "one real run; offline descriptive analysis; no new robot execution or policy rollout",
        }
        _json(stage / "data/manifest.json", manifest)
        final_documents = [output / path.relative_to(stage) for path in documents]
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                destination = output / path.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
    summary = analysis["summary"]
    print(f"分析完成：{len(analysis['views'])} 个执行观测，{summary['completed_actions']} 次运动。", flush=True)
    print(f"ROI覆盖率 {summary['initial_coverage']*100:.2f}% → {summary['final_coverage']*100:.2f}%；"
          f"目标支持体素 {summary['initial_target_voxels']} → {summary['final_target_voxels']}。", flush=True)
    for path in final_documents:
        print(f"可直接打开：{path}", flush=True)
    print(f"配图和统计表：{output}", flush=True)
    return final_documents
