"""Tests for the formal deterministic 100-target offline benchmark."""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("placo")

from strawberry_nero_control.offline_benchmark import (  # noqa: E402
    FORMAL_OFFLINE_SAMPLES,
    run_benchmark,
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


def test_formal_offline_benchmark_passes_and_replaces_stable_artifacts(tmp_path):
    summary = run_benchmark(
        NERO_URDF,
        tmp_path,
        FORMAL_OFFLINE_SAMPLES,
        20260806,
        0.15,
    )

    assert summary["passed"]
    assert summary["samples"] == 100
    assert summary["successes"] >= 99
    assert summary["ik_time_p95_ms"] <= 20.0
    assert all(summary["negative_checks"].values())
    assert Path(summary["csv_path"]).name == "offline_100.csv"
    assert Path(summary["json_path"]).name == "offline_100.json"

    with Path(summary["csv_path"]).open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))
    payload = json.loads(
        Path(summary["json_path"]).read_text(encoding="utf-8")
    )
    assert len(rows) == 100
    assert all(row["trajectory_success"] == "True" for row in rows)
    assert payload["passed"] is True
    assert "csv_path" not in payload
    assert payload["artifacts"]["details_csv"] == "offline_100.csv"
