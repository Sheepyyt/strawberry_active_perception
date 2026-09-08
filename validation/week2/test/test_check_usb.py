"""Pure unit tests for read-only USB/sysfs classification."""

from __future__ import annotations

from pathlib import Path
import sys


WEEK2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK2_DIR))

import check_usb  # noqa: E402


def _write_device(root: Path, name: str, **fields: str) -> Path:
    device = root / name
    device.mkdir()
    for field, value in fields.items():
        (device / field).write_text(value + "\n", encoding="utf-8")
    return device


def test_speed_classification_keeps_480m_conditional():
    result = check_usb.classify_speed(480.0)
    assert result["classification"] == "high_speed_480_conditional"
    assert result["conditional"] is True
    assert result["superspeed"] is False
    assert check_usb.classify_speed(5000.0)["classification"] == "superspeed"
    assert check_usb.classify_speed(None)["classification"] == "unknown"


def test_discovery_reads_versions_and_redacts_serial(tmp_path):
    _write_device(
        tmp_path,
        "1-3.3",
        idVendor="2bc5",
        idProduct="0671",
        manufacturer="Orbbec",
        product="Orbbec XL",
        serial="example-serial",
        speed="480",
        version="2.00",
        bcdDevice="0409",
        busnum="1",
        devnum="11",
    )
    _write_device(tmp_path, "2-1", idVendor="1234", speed="5000")

    devices = check_usb.discover_orbbec_devices(tmp_path)
    assert len(devices) == 1
    device = devices[0]
    assert device["vendor_id"] == "2bc5"
    assert device["negotiated_speed_mbps"] == 480.0
    assert device["usb_spec_version"] == "2.00"
    assert device["usb_device_release_bcd"] == "0409"
    assert device["serial"] is None
    assert len(device["serial_sha256_12"]) == 12
    assert device["transport"]["conditional"] is True


def test_overall_assessment_is_conservative():
    no_devices = check_usb.overall_transport_assessment([])
    assert no_devices["status"] == "not_detected"

    devices = [
        {"transport": check_usb.classify_speed(5000.0)},
        {"transport": check_usb.classify_speed(480.0)},
    ]
    assessment = check_usb.overall_transport_assessment(devices)
    assert assessment["status"] == "high_speed_480_conditional"
    assert assessment["conditional"] is True


def test_report_can_be_built_without_external_commands(tmp_path):
    report = check_usb.build_report(
        sysfs_root=tmp_path,
        include_serial=False,
        collect_lsusb=False,
    )
    assert report["status"] == "complete"
    assert report["gate_evaluation"] == "not_evaluated"
    assert report["commands"] == {}
