"""Fail-closed loading of a formally accepted hand-eye report."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re

import numpy as np

from .transforms import validate_rigid_transform


REPORT_VERSION = "strawberry_handeye_stability_report/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EXPECTED_AXES = "+X right, +Y down, +Z forward"
_EXPECTED_RESULT_CONVENTION = (
    "maps a camera optical point into link7; "
    "p_link7 = T_link7_camera_optical p_camera"
)


@dataclass(frozen=True)
class VerifiedHandeyeReport:
    """The minimal immutable calibration material consumed by the bridge."""

    path: Path
    sha256: str
    report_version: str
    session_id: str
    sample_count: int
    transform_link7_camera_optical: np.ndarray


def _require_exact_true(document: dict[str, object], key: str) -> None:
    if document.get(key) is not True:
        raise ValueError(f"hand-eye report field {key!r} must be exactly true")


def load_verified_handeye_report(
    path: str | Path,
    expected_sha256: str,
    minimum_sample_count: int = 30,
) -> VerifiedHandeyeReport:
    """Load a report only when its bytes and all safety gates are accepted.

    The expected digest is mandatory.  It binds the runtime transform to the
    exact reviewed report rather than merely to a mutable file path.
    """
    digest_text = str(expected_sha256).strip().lower()
    if _SHA256_PATTERN.fullmatch(digest_text) is None:
        raise ValueError("handeye_report_sha256 must be 64 lowercase hex digits")
    if (
        isinstance(minimum_sample_count, bool)
        or not isinstance(minimum_sample_count, int)
        or minimum_sample_count < 1
    ):
        raise ValueError("minimum_sample_count must be a positive integer")

    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise ValueError(f"hand-eye report does not exist: {report_path}")
    payload = report_path.read_bytes()
    actual_digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_digest, digest_text):
        raise ValueError(
            "hand-eye report SHA256 mismatch: "
            f"expected {digest_text}, got {actual_digest}"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"hand-eye report is not valid UTF-8 JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("hand-eye report root must be a JSON object")
    if document.get("report_version") != REPORT_VERSION:
        raise ValueError(
            "unsupported hand-eye report_version: "
            f"{document.get('report_version')!r}"
        )
    _require_exact_true(document, "success")
    _require_exact_true(document, "safe_for_robot_use")
    if document.get("status") != "passed":
        raise ValueError("hand-eye report status must be exactly 'passed'")

    sample_count = document.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("hand-eye report sample_count must be an integer")
    if sample_count < minimum_sample_count:
        raise ValueError(
            f"hand-eye report has {sample_count} samples; "
            f"at least {minimum_sample_count} are required"
        )
    split_count = document.get("split_count")
    successful_count = document.get("successful_split_count")
    if (
        isinstance(split_count, bool)
        or not isinstance(split_count, int)
        or split_count < 1
        or isinstance(successful_count, bool)
        or not isinstance(successful_count, int)
        or successful_count != split_count
        or document.get("successful_split_fraction") != 1.0
    ):
        raise ValueError("hand-eye report did not pass every configured split")

    convention = document.get("transform_convention")
    if not isinstance(convention, dict):
        raise ValueError("hand-eye report transform_convention is missing")
    if convention.get("matrix_layout") != "row-major homogeneous 4x4":
        raise ValueError("hand-eye report matrix layout is not row-major 4x4")
    if convention.get("units") != "metres and radians unless a field name states otherwise":
        raise ValueError("hand-eye report transform units are unexpected")
    if convention.get("camera_optical_axes") != _EXPECTED_AXES:
        raise ValueError("hand-eye report camera optical axes are unexpected")
    if convention.get("result_T_link7_camera_optical") != _EXPECTED_RESULT_CONVENTION:
        raise ValueError("hand-eye report transform direction is unexpected")

    try:
        transform = validate_rigid_transform(
            np.asarray(document["T_link7_camera_optical"], dtype=float),
            "T_link7_camera_optical",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid T_link7_camera_optical: {error}") from error

    # The stability report promotes its medoid as the formal result.  Requiring
    # byte-level numerical agreement closes a second, ambiguous transform path.
    try:
        diagnostic_medoid = validate_rigid_transform(
            np.asarray(
                document["diagnostic_medoid_T_link7_camera_optical"],
                dtype=float,
            ),
            "diagnostic_medoid_T_link7_camera_optical",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid diagnostic medoid transform: {error}") from error
    if not np.array_equal(transform, diagnostic_medoid):
        raise ValueError("formal transform does not exactly match the diagnostic medoid")

    session_id = document.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("hand-eye report session_id must be a non-empty string")
    return VerifiedHandeyeReport(
        path=report_path,
        sha256=actual_digest,
        report_version=REPORT_VERSION,
        session_id=session_id.strip(),
        sample_count=sample_count,
        transform_link7_camera_optical=transform,
    )
