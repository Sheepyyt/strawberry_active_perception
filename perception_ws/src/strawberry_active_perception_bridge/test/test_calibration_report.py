"""Unit tests for the hand-eye report trust boundary."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from strawberry_active_perception_bridge.calibration_report import (
    REPORT_VERSION,
    load_verified_handeye_report,
)


def _document() -> dict[str, object]:
    transform = [
        [0.0, 0.0, 1.0, 0.064],
        [0.0, 1.0, 0.0, 0.002],
        [-1.0, 0.0, 0.0, 0.001],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "report_version": REPORT_VERSION,
        "success": True,
        "safe_for_robot_use": True,
        "status": "passed",
        "sample_count": 30,
        "split_count": 20,
        "successful_split_count": 20,
        "successful_split_fraction": 1.0,
        "session_id": "fixture_session",
        "T_link7_camera_optical": transform,
        "diagnostic_medoid_T_link7_camera_optical": transform,
        "transform_convention": {
            "matrix_layout": "row-major homogeneous 4x4",
            "units": "metres and radians unless a field name states otherwise",
            "camera_optical_axes": "+X right, +Y down, +Z forward",
            "result_T_link7_camera_optical": (
                "maps a camera optical point into link7; "
                "p_link7 = T_link7_camera_optical p_camera"
            ),
        },
    }


def _write_report(tmp_path, document):
    path = tmp_path / "report.json"
    payload = json.dumps(document, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_loads_exactly_hash_bound_safe_report(tmp_path) -> None:
    path, digest = _write_report(tmp_path, _document())
    report = load_verified_handeye_report(path, digest)
    assert report.sha256 == digest
    assert report.session_id == "fixture_session"
    assert report.sample_count == 30
    np.testing.assert_array_equal(
        report.transform_link7_camera_optical,
        _document()["T_link7_camera_optical"],
    )


def test_rejects_wrong_digest_before_trusting_json(tmp_path) -> None:
    path, _digest = _write_report(tmp_path, _document())
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_verified_handeye_report(path, "0" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("success", False, "success"),
        ("safe_for_robot_use", False, "safe_for_robot_use"),
        ("status", "failed", "status"),
        ("successful_split_count", 19, "every configured split"),
    ],
)
def test_rejects_failed_report_gate(tmp_path, field, value, message) -> None:
    document = _document()
    document[field] = value
    path, digest = _write_report(tmp_path, document)
    with pytest.raises(ValueError, match=message):
        load_verified_handeye_report(path, digest)


def test_rejects_reflection_even_when_report_claims_safe(tmp_path) -> None:
    document = _document()
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    document["T_link7_camera_optical"] = reflection.tolist()
    document["diagnostic_medoid_T_link7_camera_optical"] = reflection.tolist()
    path, digest = _write_report(tmp_path, document)
    with pytest.raises(ValueError, match="determinant"):
        load_verified_handeye_report(path, digest)


def test_rejects_result_that_differs_from_report_medoid(tmp_path) -> None:
    document = _document()
    medoid = np.asarray(document["T_link7_camera_optical"], dtype=float)
    medoid[0, 3] += 0.001
    document["diagnostic_medoid_T_link7_camera_optical"] = medoid.tolist()
    path, digest = _write_report(tmp_path, document)
    with pytest.raises(ValueError, match="does not exactly match"):
        load_verified_handeye_report(path, digest)


def test_rejects_non_integer_minimum_sample_count(tmp_path) -> None:
    path, digest = _write_report(tmp_path, _document())
    with pytest.raises(ValueError, match="positive integer"):
        load_verified_handeye_report(path, digest, 29.5)
