#!/usr/bin/env python3
"""Read-only Orbbec USB topology and sysfs transport audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Sequence


SCHEMA_VERSION = 1
ORBBEC_VENDOR_ID = "2bc5"
SYSFS_FIELDS = (
    "manufacturer",
    "product",
    "serial",
    "idVendor",
    "idProduct",
    "speed",
    "version",
    "bcdDevice",
    "busnum",
    "devnum",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _parse_speed_mbps(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        speed = float(value)
    except ValueError:
        return None
    return speed if speed >= 0.0 else None


def classify_speed(speed_mbps: float | None) -> dict[str, Any]:
    """Classify negotiated speed without treating USB 480M as a false pass."""
    if speed_mbps is None:
        return {
            "classification": "unknown",
            "conditional": False,
            "superspeed": False,
            "reason": "sysfs did not expose a numeric negotiated USB speed",
        }
    if speed_mbps >= 5_000.0:
        return {
            "classification": "superspeed",
            "conditional": False,
            "superspeed": True,
            "reason": "negotiated speed is at least 5 Gbit/s",
        }
    if speed_mbps >= 480.0:
        return {
            "classification": "high_speed_480_conditional",
            "conditional": True,
            "superspeed": False,
            "reason": (
                "480 Mbit/s is USB 2.0 High-Speed, not SuperSpeed; it is only "
                "conditionally usable at the tested low-bandwidth stream profile "
                "after cold-start and 30-minute stability gates pass"
            ),
        }
    return {
        "classification": "below_high_speed",
        "conditional": False,
        "superspeed": False,
        "reason": "negotiated speed is below USB 2.0 High-Speed",
    }


def discover_orbbec_devices(
    sysfs_root: Path,
    *,
    include_serial: bool = False,
) -> list[dict[str, Any]]:
    """Read Orbbec USB device attributes from a sysfs-compatible tree."""
    devices: list[dict[str, Any]] = []
    try:
        candidates = sorted(sysfs_root.iterdir())
    except (FileNotFoundError, PermissionError, OSError):
        return devices
    for candidate in candidates:
        vendor = _read_text(candidate / "idVendor")
        if vendor is None or vendor.lower() != ORBBEC_VENDOR_ID:
            continue
        values = {field: _read_text(candidate / field) for field in SYSFS_FIELDS}
        serial = values.pop("serial")
        speed_mbps = _parse_speed_mbps(values.pop("speed"))
        device: dict[str, Any] = {
            "sysfs_name": candidate.name,
            "sysfs_path": str(candidate),
            "manufacturer": values["manufacturer"],
            "product": values["product"],
            "vendor_id": values["idVendor"],
            "product_id": values["idProduct"],
            "negotiated_speed_mbps": speed_mbps,
            # Exact sysfs meanings: USB spec version and device release number.
            # Neither is mislabeled as camera firmware.
            "usb_spec_version": values["version"],
            "usb_device_release_bcd": values["bcdDevice"],
            "bus_number": values["busnum"],
            "device_number": values["devnum"],
            "serial_present": bool(serial),
            "serial": serial if include_serial else None,
            "serial_sha256_12": (
                hashlib.sha256(serial.encode("utf-8")).hexdigest()[:12]
                if serial
                else None
            ),
            "transport": classify_speed(speed_mbps),
        }
        devices.append(device)
    return devices


def run_command(command: Sequence[str]) -> dict[str, Any]:
    """Run one read-only command and retain stdout/stderr and return code."""
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": list(command),
            "available": False,
            "return_code": None,
            "stdout": "",
            "stderr": f"{command[0]} was not found on PATH",
        }
    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return {
        "command": list(command),
        "available": True,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def overall_transport_assessment(devices: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all discovered Orbbec links conservatively."""
    if not devices:
        return {
            "status": "not_detected",
            "conditional": False,
            "reason": "no USB device with Orbbec vendor ID 2bc5 was found",
        }
    classifications = [device["transport"]["classification"] for device in devices]
    if all(value == "superspeed" for value in classifications):
        return {
            "status": "superspeed",
            "conditional": False,
            "reason": "all detected Orbbec devices negotiated at least 5 Gbit/s",
        }
    if any(value == "below_high_speed" for value in classifications):
        return {
            "status": "below_high_speed",
            "conditional": False,
            "reason": "at least one Orbbec device negotiated below 480 Mbit/s",
        }
    if any(value == "unknown" for value in classifications):
        return {
            "status": "unknown",
            "conditional": False,
            "reason": "at least one Orbbec negotiated speed is unknown",
        }
    return {
        "status": "high_speed_480_conditional",
        "conditional": True,
        "reason": (
            "at least one Orbbec device is limited to 480 Mbit/s; accept only "
            "for the exact low-bandwidth profile that passes the cold-start and "
            "30-minute stream gates, and do not report it as SuperSpeed"
        ),
    }


def build_report(
    *,
    sysfs_root: Path,
    include_serial: bool,
    collect_lsusb: bool,
) -> dict[str, Any]:
    devices = discover_orbbec_devices(sysfs_root, include_serial=include_serial)
    commands = (
        {
            "lsusb": run_command(("lsusb",)),
            "lsusb_tree": run_command(("lsusb", "-t")),
        }
        if collect_lsusb
        else {}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "check_usb.py",
        "generated_utc": utc_now(),
        "status": "complete",
        "gate_evaluation": "not_evaluated",
        "host": {
            "hostname": socket.gethostname(),
            "kernel": platform.release(),
        },
        "sysfs_root": str(sysfs_root),
        "orbbec_vendor_id": ORBBEC_VENDOR_ID,
        "device_count": len(devices),
        "devices": devices,
        "overall_transport": overall_transport_assessment(devices),
        "commands": commands,
        "notes": [
            "This tool performs read-only sysfs reads and lsusb invocations.",
            "usb_device_release_bcd is the USB bcdDevice value, not a claimed "
            "camera firmware version.",
            "Serial values are redacted by default; hashes permit comparison "
            "without committing the device serial.",
        ],
    }


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys/bus/usb/devices"))
    parser.add_argument("--include-serial", action="store_true")
    parser.add_argument(
        "--skip-lsusb",
        action="store_true",
        help="skip command capture (useful only for deterministic unit tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    document = build_report(
        sysfs_root=arguments.sysfs_root,
        include_serial=arguments.include_serial,
        collect_lsusb=not arguments.skip_lsusb,
    )
    write_json_atomic(arguments.output, document)
    print(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
