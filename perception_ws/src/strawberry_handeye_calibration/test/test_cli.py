"""Smoke test the two disk-only entry points through their public APIs."""

import json

from strawberry_handeye_calibration.calibration import calibrate_eye_in_hand
from strawberry_handeye_calibration.fixture import make_synthetic_dataset
from strawberry_handeye_calibration.schema import load_dataset, save_dataset, write_json_atomic


def test_fixture_dataset_and_report_are_plain_json(tmp_path) -> None:
    dataset_path = tmp_path / "fixture.json"
    report_path = tmp_path / "report.json"
    save_dataset(make_synthetic_dataset(sample_count=10), dataset_path)
    dataset = load_dataset(dataset_path)
    report = calibrate_eye_in_hand(dataset)
    write_json_atomic(report, report_path)
    recovered = json.loads(report_path.read_text(encoding="utf-8"))
    assert recovered["success"]
    assert recovered["report_version"] == "strawberry_handeye_report/v1"
    assert len(recovered["T_link7_camera_optical"]) == 4
    assert recovered["metrics"]["holdout"]["sample_ids"]
