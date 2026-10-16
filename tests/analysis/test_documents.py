"""Offline document contract: embedded figures, page bounds and honest metrics."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("analysis_documents", ROOT / "tools/analysis/documents.py")
documents = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(documents)


def test_export_three_page_embedded_figures_and_evidence_boundaries(tmp_path):
    if not all(shutil.which(name) for name in ("libreoffice", "pdfinfo", "pdftotext")):
        pytest.skip("LibreOffice and Poppler are required for the document integration test")
    figures = []
    for index, color in enumerate(("#5A8097", "#188B71", "#C66C28")):
        path = tmp_path / f"figure_{index}.png"
        Image.new("RGB", (1200, 520 + index * 40), color).save(path)
        figures.append(path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in figures}
    analysis = {
        "run_id": "independent_case", "summary": {
            "completed_actions": 2, "map_updates": 2, "mapped_observations": 3,
            "initial_coverage": .11, "final_coverage": .29, "coverage_gain_pp": 18.,
            "initial_target_voxels": 17, "final_target_voxels": 31,
            "coverage_target": .9, "scientific_acceptance_passed": False,
            "started_utc": "2026-10-16T12:34:00Z", "stop_reason": "no_positive_gain",
        },
        "views": [
            {"actual_translation_mm": 0, "position_error_mm": None},
            {"actual_translation_mm": 7.2, "position_error_mm": .3},
            {"actual_translation_mm": 19.8, "position_error_mm": .7},
        ],
        "rounds": [{
            "round_index": 2, "label": "测试决策", "radii": [
                {"total": 57, "reachable": 12, "positive": 0},
            ], "thresholds": [
                {"minimum_depth_m": .37, "reachable": 23, "positive": 8},
                {"minimum_depth_m": .46, "reachable": 6, "positive": 0},
            ],
        }],
    }
    output = tmp_path / "output"
    docx, pdf = documents.write_documents(analysis, figures, output)
    assert {path.suffix for path in output.iterdir()} == {".docx", ".pdf"}
    assert documents._page_count(pdf) == 3
    with ZipFile(docx) as archive:
        assert len([name for name in archive.namelist() if name.startswith("word/media/")]) == 3
    text = subprocess.run(["pdftotext", str(pdf), "-"], check=True, capture_output=True,
                          text=True, timeout=20).stdout
    compact = "".join(text.split())
    for expected in (
        "运动后地图更新2次", "执行观测3份（含初始）", "11.00%→29.00%",
        "增加18.00个百分点", "目标支持体素：17→31", "7.2–19.8mm",
        "0.37m", "0.46m", "23→6", "8→0", "近深度层", "不代表重复成功率",
        "不能作为新策略的真机成绩",
    ):
        assert expected in compact
    assert "0.0–19.8mm" not in compact  # Initial baseline is not an executed motion.
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in figures}


def test_missing_figure_does_not_publish_partial_document(tmp_path):
    if not shutil.which("libreoffice"):
        pytest.skip("LibreOffice availability is checked before input validation")
    with pytest.raises(FileNotFoundError):
        documents.write_documents({}, [tmp_path / f"missing_{i}.png" for i in range(3)], tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_result_sentences_follow_input_deltas_and_selected_candidates():
    views = [
        {"median_depth_mm": 510, "valid_depth_fraction": .95},
        {"median_depth_mm": 470, "valid_depth_fraction": .72, "target_added": 11, "target_removed": 1},
        {"median_depth_mm": 460, "valid_depth_fraction": .86, "target_added": 9, "target_removed": 2},
    ]
    assert documents._target_changes(views) == "逐步新增合计 20，移除 3，净增 17。"
    trend = documents._perception_trend(views)
    assert "510 → 460 mm" in trend
    assert "95.00% → 86.00%，最低 72.00%" in trend
    assert "靠近目标未伴随该比例的持续改善" in trend
    rounds = [
        {"selected_radius_mm": radius, "radii": [{"radius_mm": radius, "positive": positive}]}
        for radius, positive in ((80, 2), (50, 3), (12, 2), (7, 1))
    ] + [{"radii": [{"radius_mm": 80, "positive": 0}]}]
    candidates = documents._candidate_progress(rounds, 4)
    assert "2 → 3 → 2 → 1 → 0" in candidates
    assert "80 / 50 / 12 / 7 mm" in candidates
    assert "最大半径依次为 12 / 7 mm" in candidates
    assert "对应动作均已完成" in candidates


def test_missing_or_improving_evidence_does_not_invent_results():
    assert documents._target_changes([{}, {}]) == ""
    assert documents._perception_trend([{}, {}]) == ""
    assert documents._candidate_progress([], None) == ""
    trend = documents._perception_trend([
        {"median_depth_mm": 510, "valid_depth_fraction": .70},
        {"median_depth_mm": 470, "valid_depth_fraction": .85},
    ])
    assert "未" not in trend
    candidates = documents._candidate_progress([
        {"selected_radius_mm": 10, "radii": [{"radius_mm": 10, "positive": 2}]},
    ], 0)
    assert "实际所选" not in candidates
    assert "已完成" not in candidates
