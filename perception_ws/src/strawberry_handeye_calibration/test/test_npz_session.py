"""Tests for fail-closed conversion of read-only capture NPZs."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from strawberry_handeye_calibration.npz_session import (
    CAPTURE_SCHEMA_VERSION,
    CaptureSessionImportError,
    load_capture_session,
)


def _capture_arrays(sample_id: str, index: int = 0) -> dict[str, np.ndarray]:
    columns, rows = 3, 2
    square = 0.03
    objects = np.zeros((columns * rows, 3), dtype=np.float32)
    objects[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objects[:, :2] *= np.float32(square)
    K = np.array(
        [[500.0, 0.0, 6.0], [0.0, 500.0, 5.0], [0.0, 0.0, 1.0]]
    )
    D = np.zeros(5)
    T_camera_board = np.eye(4)
    T_camera_board[:3, 3] = (0.0, 0.0, 1.0)
    corners, _ = cv2.projectPoints(
        objects.astype(np.float64),
        np.zeros(3),
        T_camera_board[:3, 3],
        K,
        D,
    )
    T_base_link = np.eye(4)
    angle = np.deg2rad(float(index) * 4.0)
    T_base_link[:3, :3] = (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    T_base_link[0, 3] = index * 0.005
    seconds, nanoseconds = 100 + index, 123_000_000
    return {
        "schema_version": np.array(CAPTURE_SCHEMA_VERSION),
        "sample_id": np.array(sample_id),
        "timestamp_sec": np.array(seconds + nanoseconds * 1.0e-9),
        "scene_id": np.array("test_session"),
        "observation_id": np.array(f"obs-{index}"),
        "source_name": np.array("test_camera"),
        "source_type": np.array(1, dtype=np.uint8),
        "stamp": np.array((seconds, nanoseconds), dtype=np.int64),
        "frame_id": np.array("camera_optical"),
        "rgb": np.zeros((10, 12, 3), dtype=np.uint8),
        "depth_m": np.ones((10, 12), dtype=np.float32),
        "K": K,
        "D": D,
        "valid_depth_fraction": np.array(1.0, dtype=np.float32),
        "board_squares": np.array((columns + 1, rows + 1), dtype=np.int32),
        "board_inner_corners": np.array((columns, rows), dtype=np.int32),
        "square_size_m": np.array(square),
        "checkerboard_corners_px": corners.reshape(-1, 2),
        "checkerboard_object_points_m": objects.astype(np.float64),
        "checkerboard_outer_corners_px": np.ones((4, 2)),
        "checkerboard_minimum_border_margin_px": np.array(1.0),
        "checkerboard_reprojection_rms_px": np.array(0.0),
        "corner_count": np.array(columns * rows, dtype=np.int32),
        "reprojection_rms_px": np.array(0.0),
        "T_camera_checkerboard": T_camera_board,
        "base_frame": np.array("base_link"),
        "link_frame": np.array("link7"),
        "T_base_link7": T_base_link,
        "tf_lookup_mode": np.array("latest_while_stationary"),
        "joint_names": np.array([f"joint{i}" for i in range(1, 8)]),
        "joint_positions": np.full(7, index * 0.04),
        "joint_selection_mode": np.array("nearest_tf_stamp_while_stationary"),
        "joint_maximum_position_span_rad": np.array(0.0001),
        "motion_commands_sent": np.array(0, dtype=np.uint8),
    }


def _save(directory: Path, sample_id: str, index: int = 0) -> Path:
    path = directory / f"{sample_id}.npz"
    np.savez_compressed(path, **_capture_arrays(sample_id, index))
    return path


def test_import_preserves_core_transforms_and_capture_audit(tmp_path) -> None:
    _save(tmp_path, "pose_001", 1)
    _save(tmp_path, "pose_002", 2)
    dataset = load_capture_session(tmp_path)
    assert dataset.session_id == "test_session"
    assert [sample.sample_id for sample in dataset.samples] == [
        "pose_001",
        "pose_002",
    ]
    assert dataset.camera_frame == "camera_optical"
    assert dataset.checkerboard_columns == 3
    assert len(dataset.metadata["capture_audit"]) == 2
    assert all(
        len(item["sha256"]) == 64
        and item["motion_commands_sent"] == 0
        for item in dataset.metadata["capture_audit"]
    )


def test_explicit_subset_is_immutable_and_ordered(tmp_path) -> None:
    _save(tmp_path, "pose_001", 1)
    _save(tmp_path, "pose_002", 2)
    dataset = load_capture_session(tmp_path, sample_ids=("pose_002",))
    assert [sample.sample_id for sample in dataset.samples] == ["pose_002"]
    with pytest.raises(CaptureSessionImportError, match="plain non-empty"):
        load_capture_session(tmp_path, sample_ids=("../pose_001",))


def test_nonzero_motion_audit_and_object_dtype_fail_closed(tmp_path) -> None:
    arrays = _capture_arrays("pose_001")
    arrays["motion_commands_sent"] = np.array(1, dtype=np.uint8)
    np.savez_compressed(tmp_path / "pose_001.npz", **arrays)
    with pytest.raises(CaptureSessionImportError, match="motion command"):
        load_capture_session(tmp_path)

    (tmp_path / "pose_001.npz").unlink()
    arrays = _capture_arrays("pose_001")
    arrays["unsafe"] = np.array({"pickle": True}, dtype=object)
    np.savez_compressed(tmp_path / "pose_001.npz", **arrays)
    with pytest.raises(CaptureSessionImportError, match="non-pickle NPZ"):
        load_capture_session(tmp_path)


def test_changed_intrinsics_and_corrupt_pnp_evidence_are_rejected(tmp_path) -> None:
    _save(tmp_path, "pose_001", 1)
    arrays = _capture_arrays("pose_002", 2)
    arrays["K"] = arrays["K"].copy()
    arrays["K"][0, 0] += 1.0
    changed_corners, _ = cv2.projectPoints(
        arrays["checkerboard_object_points_m"],
        np.zeros(3),
        arrays["T_camera_checkerboard"][:3, 3],
        arrays["K"],
        arrays["D"],
    )
    arrays["checkerboard_corners_px"] = changed_corners.reshape(-1, 2)
    np.savez_compressed(tmp_path / "pose_002.npz", **arrays)
    with pytest.raises(CaptureSessionImportError, match="K differs"):
        load_capture_session(tmp_path)

    (tmp_path / "pose_002.npz").unlink()
    arrays = _capture_arrays("pose_002", 2)
    arrays["checkerboard_corners_px"] = arrays[
        "checkerboard_corners_px"
    ].copy()
    arrays["checkerboard_corners_px"][0, 0] += 2.0
    np.savez_compressed(tmp_path / "pose_002.npz", **arrays)
    with pytest.raises(CaptureSessionImportError, match="inconsistent"):
        load_capture_session(tmp_path)
