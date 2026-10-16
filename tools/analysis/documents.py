"""Create a three-page Chinese Word/PDF weekly report from saved analysis only.

The temporary FODT and LibreOffice profile never leave a temporary directory.
No ROS, model environment, hardware API, or network operation is used here.
"""
from __future__ import annotations

import base64
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from PIL import Image


NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "xlink": "http://www.w3.org/1999/xlink",
}
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)


def _q(name: str) -> str:
    prefix, local = name.split(":", 1)
    return f"{{{NS[prefix]}}}{local}"


def _element(parent, tag: str, attrs: dict | None = None, text: str | None = None):
    element = ET.SubElement(parent, _q(tag), {_q(key): str(value) for key, value in (attrs or {}).items()})
    element.text = text
    return element


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _number(value, digits=0, suffix="") -> str:
    number = _finite(value)
    return "未记录" if number is None else f"{number:.{digits}f}{suffix}"


def _percent(value, digits=2) -> str:
    number = _finite(value)
    return "未记录" if number is None else f"{number * 100:.{digits}f}%"


def _range(views: list[dict], key: str, digits=0, suffix="") -> str:
    values = [_finite(view.get(key)) for view in views]
    values = [value for value in values if value is not None]
    if not values:
        return "未记录"
    return f"{min(values):.{digits}f}–{max(values):.{digits}f}{suffix}"


def _local_time(value) -> str:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return stamp.strftime("%Y-%m-%d %H:%M") + "（时区未记录）"
        return stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return "时间未记录"


def _stop_reason(value) -> str:
    reason = str(value or "未记录停止原因")
    if any(part in reason.lower() for part in ("no_reachable_positive", "no_positive_gain", "no reachable", "no positive-gain", "no remaining positive")):
        return "已检查的候选中没有可达且有正收益的新视角"
    if "coverage_target" in reason.lower():
        return "达到配置的 ROI 射线覆盖率停止目标"
    if "plateau" in reason.lower():
        return "连续观测的信息增量低于配置的平台期阈值"
    return reason[:180]


def _sensitivity(rounds: list[dict]) -> str:
    """Describe a saved-candidate threshold change without inventing trials."""
    for decision in rounds:
        rows = sorted(
            (row for row in decision.get("thresholds", []) if _finite(row.get("minimum_depth_m")) is not None),
            key=lambda row: float(row["minimum_depth_m"]),
        )
        for low in rows:
            if (_finite(low.get("positive")) or 0) <= 0:
                continue
            for high in rows:
                if float(high["minimum_depth_m"]) <= float(low["minimum_depth_m"]):
                    continue
                if _finite(high.get("positive")) != 0:
                    continue
                label = decision.get("label") or f"第 {decision.get('round_index', '?')} 轮"
                return (
                    f"{label}：最小目标深度门限从 {_number(low['minimum_depth_m'], 2, ' m')} "
                    f"提高到 {_number(high['minimum_depth_m'], 2, ' m')} 时，已保存候选中的可达数 "
                    f"{_number(low.get('reachable'))} → {_number(high.get('reachable'))}，"
                    f"正收益数 {_number(low.get('positive'))} → {_number(high.get('positive'))}。"
                    "门限变化会明显改变当前可选集合；这只是同一冻结状态下的离线筛选。"
                )
    return "图 3 比较已保存候选在不同深度门限下的保留情况。这里只重筛选同一冻结状态，没有重新规划、执行或采集新视角，不能推断新门限的闭环成绩。"


def _target_changes(views: list[dict]) -> str:
    steps = views[1:]
    added = [_finite(view.get("target_added")) for view in steps]
    removed = [_finite(view.get("target_removed")) for view in steps]
    if not steps or any(value is None for value in added + removed):
        return ""
    return (f"逐步新增合计 {_number(sum(added))}，移除 {_number(sum(removed))}，"
            f"净增 {_number(sum(added) - sum(removed))}。")


def _perception_trend(views: list[dict]) -> str:
    if not views:
        return ""
    first_depth = _finite(views[0].get("median_depth_mm"))
    last_depth = _finite(views[-1].get("median_depth_mm"))
    fractions = [_finite(view.get("valid_depth_fraction")) for view in views]
    statements = []
    if first_depth is not None and last_depth is not None:
        statements.append(f"近层中位深度由 {_number(first_depth)} → {_number(last_depth)} mm")
    if all(value is not None for value in fractions):
        statements.append(f"有效深度比例 {_percent(fractions[0])} → {_percent(fractions[-1])}，最低 {_percent(min(fractions))}")
        if any(later < earlier for earlier, later in zip(fractions, fractions[1:])):
            if first_depth is not None and last_depth is not None and last_depth < first_depth:
                statements.append("本案例中，靠近目标未伴随该比例的持续改善")
            else:
                statements.append("本案例的该比例未呈持续改善")
    return "；".join(statements) + "。" if statements else ""


def _candidate_progress(rounds: list[dict], completed_actions) -> str:
    counts = []
    for decision in rounds:
        values = [_finite(row.get("positive")) for row in decision.get("radii", [])]
        counts.append(_number(sum(values)) if values and all(value is not None for value in values) else "未记录")
    if not counts:
        return ""
    statement = "各轮正收益候选数：" + " → ".join(counts) + "。"
    selected = [decision for decision in rounds if _finite(decision.get("selected_radius_mm")) is not None]
    if not selected or _finite(completed_actions) != len(selected):
        return statement
    statement += "实际所选半径：" + " / ".join(_number(item["selected_radius_mm"]) for item in selected) + " mm。"
    maxima = []
    for decision in selected:
        radii = [_finite(row.get("radius_mm")) for row in decision.get("radii", [])
                 if (_finite(row.get("positive")) or 0) > 0]
        maxima.append(max(radii) if radii and all(value is not None for value in radii) else None)
    if len(maxima) >= 3 and all(value is not None for value in maxima) and max(maxima[-2:]) < max(maxima[:-2]):
        statement += ("后两次选点的正收益候选最大半径依次为 "
                      + " / ".join(_number(value) for value in maxima[-2:])
                      + " mm；后段仅余小半径候选，对应动作均已完成。")
    return statement


def _paragraph(body, text: str, style="Body"):
    lines = text.split("\n")
    paragraph = _element(body, "text:p", {"text:style-name": style}, lines[0])
    for line in lines[1:]:
        _element(paragraph, "text:line-break").tail = line
    return paragraph


def _heading(body, text: str, *, page_break=False):
    return _paragraph(body, text, "PageHeading" if page_break else "Heading")


def _figure(body, path: Path, caption: str, max_height: float):
    with Image.open(path) as source:
        width, height = source.size
    factor = min(17.4 / width, max_height / height)
    paragraph = _paragraph(body, "", "Figure")
    frame = _element(paragraph, "draw:frame", {
        "draw:name": path.stem,
        "text:anchor-type": "as-char",
        "svg:width": f"{width * factor:.3f}cm",
        "svg:height": f"{height * factor:.3f}cm",
    })
    image = _element(frame, "draw:image")
    _element(image, "office:binary-data", text=base64.b64encode(path.read_bytes()).decode("ascii"))
    _paragraph(body, caption, "Caption")


def _style(styles, name: str, size: float, color="#253548", *, bold=False, before="0cm", after="0.16cm", line="140%", break_before=False):
    style = _element(styles, "style:style", {"style:name": name, "style:family": "paragraph"})
    properties = {"fo:margin-top": before, "fo:margin-bottom": after, "fo:line-height": line,
                  "fo:orphans": "2", "fo:widows": "2"}
    if break_before:
        properties["fo:break-before"] = "page"
    if name in ("Heading", "PageHeading", "Title", "Subheading"):
        properties["fo:keep-with-next"] = "always"
    if name in ("Figure", "Caption", "Footer"):
        properties["fo:text-align"] = "center"
    _element(style, "style:paragraph-properties", properties)
    _element(style, "style:text-properties", {
        "style:font-name": "NotoSansCJK", "style:font-name-asian": "NotoSansCJK",
        "fo:font-size": f"{size}pt", "style:font-size-asian": f"{size}pt",
        "fo:color": color, "fo:font-weight": "bold" if bold else "normal",
        "style:font-weight-asian": "bold" if bold else "normal",
        "fo:language": "zh", "fo:country": "CN", "style:language-asian": "zh", "style:country-asian": "CN",
    })


def _make_fodt(analysis: dict, figures: list[Path], path: Path, *, compact=False) -> None:
    root = ET.Element(_q("office:document"), {_q("office:version"): "1.2", _q("office:mimetype"): "application/vnd.oasis.opendocument.text"})
    fonts = _element(root, "office:font-face-decls")
    _element(fonts, "style:font-face", {"style:name": "NotoSansCJK", "svg:font-family": "'Noto Sans CJK SC'"})
    styles = _element(root, "office:styles")
    body_size = 9.8 if compact else 10.5
    _style(styles, "Body", body_size)
    _style(styles, "Title", 20, "#123D57", bold=True, after="0.18cm", line="120%")
    _style(styles, "Meta", 8.7, "#667788", after="0.25cm", line="125%")
    _style(styles, "Heading", 12.5, "#123D57", bold=True, before="0.18cm", after="0.14cm")
    _style(styles, "PageHeading", 14, "#123D57", bold=True, after="0.20cm", break_before=True)
    _style(styles, "Subheading", 11, "#136873", bold=True, after="0.12cm")
    _style(styles, "Metric", 11.2, "#123D57", bold=True, before="0.12cm", after="0.20cm", line="145%")
    _style(styles, "Figure", 1, after="0.08cm", line="100%")
    _style(styles, "Caption", 8.4, "#667788", after="0.20cm", line="125%")
    _style(styles, "Note", 8.7, "#667788", before="0.12cm", line="135%")
    _style(styles, "Footer", 8, "#82919D", before="0cm", after="0cm", line="110%")
    automatic = _element(root, "office:automatic-styles")
    page_layout = _element(automatic, "style:page-layout", {"style:name": "A4"})
    _element(page_layout, "style:page-layout-properties", {
        "fo:page-width": "21cm", "fo:page-height": "29.7cm", "style:print-orientation": "portrait",
        "fo:margin-top": "1.4cm", "fo:margin-bottom": "1.4cm", "fo:margin-left": "1.7cm", "fo:margin-right": "1.7cm",
    })
    footer_style = _element(page_layout, "style:footer-style")
    _element(footer_style, "style:header-footer-properties", {"fo:min-height": "0.55cm", "fo:margin-top": "0.2cm"})
    master_styles = _element(root, "office:master-styles")
    master = _element(master_styles, "style:master-page", {"style:name": "Standard", "style:page-layout-name": "A4"})
    footer = _element(master, "style:footer")
    footer_text = _paragraph(footer, "主动感知复现 · 单轮真实案例的离线复盘     ", "Footer")
    page = _element(footer_text, "text:page-number", {"text:select-page": "current"}, "1")
    page.tail = " / "
    _element(footer_text, "text:page-count", text="3")
    body = _element(_element(root, "office:body"), "office:text")
    summary = analysis.get("summary", {})
    views = analysis.get("views", [])
    rounds = analysis.get("rounds", [])
    run_id = str(analysis.get("run_id", "未记录"))
    max_height = 8.5 if compact else 9.6

    _paragraph(body, "草莓主动感知｜周进展", "Title")
    _paragraph(body, f"NERO · Gemini 2 XL · YOLO11 · Placo    采集：{_local_time(summary.get('started_utc'))}（北京时间）\n案例：{run_id}", "Meta")
    _heading(body, "一、本周完成")
    _paragraph(body, "贯通准备、初始关节检查与按需恢复、主动感知闭环、证据保存及支撑后失能收尾。每轮依据当前关节反馈和 RGB-D 观测初始化目标，并检查本轮起始状态是否可用。")
    _paragraph(body, "本次选取一轮真实运行，联合复盘地图增量、分割内深度质量与候选筛选；启动关节恢复与 NBV 动作分开计数。")
    _heading(body, "二、本轮结果")
    _paragraph(body, f"NBV 动作 {_number(summary.get('completed_actions'))} 次   ·   运动后地图更新 {_number(summary.get('map_updates'))} 次   ·   执行观测 {len(views)} 份（含初始）\nROI 射线覆盖率 {_percent(summary.get('initial_coverage'))} → {_percent(summary.get('final_coverage'))}   ·   增加 {_number(summary.get('coverage_gain_pp'), 2)} 个百分点", "Metric")
    _figure(body, figures[0], "图 1｜信息增量。ROI 射线覆盖率不是果实完整度；目标支持体素代表正向语义证据，不等同于已重建的真实表面。", max_height)
    _paragraph(body, f"目标支持体素：{_number(summary.get('initial_target_voxels'))} → {_number(summary.get('final_target_voxels'))}。{_target_changes(views)}停止原因：{_stop_reason(summary.get('stop_reason'))}。")
    target = _percent(summary.get("coverage_target"), 0)
    acceptance = summary.get("scientific_acceptance_passed")
    verdict = "通过" if acceptance is True else "未通过" if acceptance is False else "未记录"
    _paragraph(body, f"本轮配置覆盖率目标为 {target}；预设科研验收：{verdict}。提前停止需结合停因解释，完成动作数不能单独证明算法效果。", "Note")

    _heading(body, "二、本轮结果｜分割与深度质量", page_break=True)
    _paragraph(body, "逐视角检查 mask、有效深度比例和近层深度离散程度，说明本轮输入质量的实际变化。")
    _figure(body, figures[1], "图 2｜感知质量。有效深度比例针对全部 mask；中位深度及 MAD 则取自审计记录的近深度层，MAD 为中位绝对偏差。", max_height)
    trend = _perception_trend(views)
    if trend:
        _paragraph(body, trend)
    _paragraph(body, f"mask 像素范围：{_range(views, 'mask_pixels')}；有效深度像素：{_range(views, 'valid_depth_pixels')}；近深度层 MAD：{_range(views, 'depth_mad_mm', 1, ' mm')}。")
    _paragraph(body, f"运动后实际相机平移范围：{_range(views[1:], 'actual_translation_mm', 1, ' mm')}；到位位置误差范围：{_range(views[1:], 'position_error_mm', 2, ' mm')}。这些误差反映本轮运动执行情况，不包含外部测量的绝对定位精度。")
    _heading(body, "三、问题与诊断")
    _paragraph(body, "• 目标支持体素含未判为占据的体素；新增与移除反映地图分类变化，不是经过真值验证的表面增量。")
    _paragraph(body, "• mask 边缘和背景混入仍需结合原图检查。有效深度比例不等于分割准确率；无正收益也只针对本轮已检查的候选集合。")

    _heading(body, "三、问题与诊断｜候选分析", page_break=True)
    _figure(body, figures[2], "图 3｜候选数量、可达性与深度门限敏感性。收益是规划器信息增益分数，不能换算成覆盖率百分点。", max_height)
    candidate_progress = _candidate_progress(rounds, summary.get("completed_actions"))
    if candidate_progress:
        _paragraph(body, candidate_progress)
    _paragraph(body, _sensitivity(rounds))
    if rounds:
        last = rounds[-1]
        radii = last.get("radii", [])
        if radii:
            totals = {key: sum(int(row.get(key, 0) or 0) for row in radii) for key in ("total", "reachable", "positive")}
            ending = "本轮因此停止新增运动。" if last.get("selected_radius_mm") is None and totals["positive"] == 0 else ""
            _paragraph(body, f"最后一次保存的搜索：{totals['total']} 个候选、{totals['reachable']} 个可达、{totals['positive']} 个正收益。{ending}")
    _heading(body, "四、下一步")
    _paragraph(body, "1. 保持软硬件与分析口径一致，改变起始关节姿态和草莓位置，记录多轮完整结果及拒绝原因，再统计重复运行表现。")
    _paragraph(body, "2. 针对接近工作距离边界、反光与背景混入，核查原始 RGB-D 和 mask；补充少量人工标注后，再评价分割与目标深度质量。")
    _paragraph(body, "3. 将候选门限敏感性结论转成下一轮实验假设；只有完成新的预览、真机执行及观测后，才能评价新策略的闭环效果。")
    _paragraph(body, "结论边界：本报告是一轮真实案例的离线分析，不代表重复成功率、分割准确率或算法优越性。候选敏感性固定地图、起姿与已有候选，仅用于解释当前决策，不能作为新策略的真机成绩。", "Note")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _convert(executable: str, source: Path, target_format: str, directory: Path, profile: Path) -> Path:
    result = subprocess.run([
        executable, f"-env:UserInstallation={profile.as_uri()}", "--headless", "--nologo", "--nodefault", "--norestore",
        "--convert-to", target_format, "--outdir", str(directory), str(source),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90, check=False)
    destination = directory / f"{source.stem}.{target_format.split(':', 1)[0]}"
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"LibreOffice 转换失败: {result.stdout[-2000:]}")
    return destination


def _page_count(path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if not executable:
        raise RuntimeError("需要 pdfinfo 验证周报页数")
    result = subprocess.run([executable, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=20, check=True, env={**os.environ, "LC_ALL": "C"})
    match = re.search(r"^Pages:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("无法确认 PDF 页数")
    return int(match.group(1))


def write_documents(analysis: dict, figures: list[Path], output: Path) -> list[Path]:
    """Write exactly one 3-page DOCX/PDF pair with three embedded figures."""
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError("本机需要 LibreOffice 生成 Word 和 PDF")
    if len(figures) != 3:
        raise ValueError("周报需要按信息增量、感知质量、候选分析顺序提供三张图")
    figures = [Path(figure).expanduser().resolve() for figure in figures]
    for figure in figures:
        if not figure.is_file():
            raise FileNotFoundError(figure)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(analysis.get("run_id", "本轮"))).strip("._")[:70] or "本轮"
    document_name = f"草莓主动感知周报_{identifier}"
    with tempfile.TemporaryDirectory(prefix="strawberry_weekly_documents_") as temporary:
        work = Path(temporary)
        for attempt in range(2):
            attempt_dir = work / str(attempt)
            attempt_dir.mkdir()
            fodt = attempt_dir / "weekly_report.fodt"
            _make_fodt(analysis, figures, fodt, compact=bool(attempt))
            profile = attempt_dir / "lo_profile"
            docx = _convert(executable, fodt, "docx:Office Open XML Text", attempt_dir, profile)
            pdf = _convert(executable, docx, "pdf:writer_pdf_Export", attempt_dir, profile)
            pages = _page_count(pdf)
            if pages == 3:
                break
        else:
            raise RuntimeError(f"周报排版得到 {pages} 页，未满足目标三页，未发布文件")
        with ZipFile(docx) as archive:
            media = [name for name in archive.namelist() if name.startswith("word/media/")]
            if len(media) < 3:
                raise RuntimeError("Word 文档没有完整内嵌三张图，未发布文件")
        targets = [output / f"{document_name}.docx", output / f"{document_name}.pdf"]
        for source, destination in zip((docx, pdf), targets):
            staging = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, staging)
            staging.replace(destination)
    return targets
