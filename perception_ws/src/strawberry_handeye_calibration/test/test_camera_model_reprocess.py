"""Tests for immutable, camera-model-aware reprocessing of capture NPZs."""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from strawberry_handeye_calibration.camera_model_reprocess import (
    CAMERA_MODEL_SCHEMA_VERSION,
    CameraModelReprocessError,
    ndarray_sha256,
    reprocess_capture_session,
)
from strawberry_handeye_calibration.camera_model_reprocess_cli import main as cli_main
from strawberry_handeye_calibration.npz_session import CAPTURE_SCHEMA_VERSION
from strawberry_handeye_calibration.schema import SCHEMA_VERSION, load_dataset


def _capture_arrays(
    sample_id: str,
    index: int,
    *,
    K: np.ndarray | None = None,
    D: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    width, height = 80, 60
    columns, rows = 4, 3
    square = 0.03
    if K is None:
        K = np.array(
            [[100.0, 0.0, 39.5], [0.0, 101.0, 29.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if D is None:
        D = np.zeros(5, dtype=np.float64)
    objects = np.zeros((columns * rows, 3), dtype=np.float64)
    objects[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objects[:, :2] *= square
    rotation_vector = np.array((0.03 * index, -0.02, 0.01), dtype=np.float64)
    translation = np.array((-0.045, -0.03, 0.65 + index * 0.01))
    corners, _ = cv2.projectPoints(objects, rotation_vector, translation, K, D)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    T_camera_board = np.eye(4)
    T_camera_board[:3, :3] = rotation
    T_camera_board[:3, 3] = translation
    projected, _ = cv2.projectPoints(objects, rotation_vector, translation, K, D)
    residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    T_base_link = np.eye(4)
    base_angle = np.deg2rad(index * 5.0)
    T_base_link[:3, :3] = (
        (np.cos(base_angle), -np.sin(base_angle), 0.0),
        (np.sin(base_angle), np.cos(base_angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    T_base_link[:3, 3] = (index * 0.005, 0.0, 0.3)
    seconds, nanoseconds = 100 + index, 123_000_000
    return {
        "schema_version": np.array(CAPTURE_SCHEMA_VERSION),
        "sample_id": np.array(sample_id),
        "timestamp_sec": np.array(seconds + nanoseconds * 1.0e-9),
        "scene_id": np.array("test_session"),
        "observation_id": np.array(f"obs-{index}"),
        "source_name": np.array("gemini2xl_TESTSERIAL"),
        "source_type": np.array(1, dtype=np.uint8),
        "stamp": np.array((seconds, nanoseconds), dtype=np.int64),
        "frame_id": np.array("camera_color_optical_frame"),
        "rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "depth_m": np.ones((height, width), dtype=np.float32),
        "K": K,
        "D": D,
        "valid_depth_fraction": np.array(1.0, dtype=np.float32),
        "board_squares": np.array((columns + 1, rows + 1), dtype=np.int32),
        "board_inner_corners": np.array((columns, rows), dtype=np.int32),
        "square_size_m": np.array(square),
        "checkerboard_corners_px": corners.reshape(-1, 2),
        "checkerboard_object_points_m": objects,
        "checkerboard_outer_corners_px": np.array(
            ((20.0, 15.0), (60.0, 15.0), (60.0, 45.0), (20.0, 45.0))
        ),
        "checkerboard_minimum_border_margin_px": np.array(10.0),
        "checkerboard_reprojection_rms_px": np.array(rms),
        "corner_count": np.array(columns * rows, dtype=np.int32),
        "reprojection_rms_px": np.array(rms),
        "T_camera_checkerboard": T_camera_board,
        "base_frame": np.array("base_link"),
        "link_frame": np.array("link7"),
        "T_base_link7": T_base_link,
        "tf_lookup_mode": np.array("observation_stamp"),
        "joint_names": np.array([f"joint{i}" for i in range(1, 8)]),
        "joint_positions": np.full(7, index * 0.04),
        "joint_selection_mode": np.array("nearest_observation_ros_stamp"),
        "joint_maximum_position_span_rad": np.array(0.0001),
        "motion_commands_sent": np.array(0, dtype=np.uint8),
    }


def _save_capture(
    directory: Path,
    sample_id: str,
    index: int,
    **kwargs: np.ndarray,
) -> Path:
    path = directory / f"{sample_id}.npz"
    np.savez_compressed(path, **_capture_arrays(sample_id, index, **kwargs))
    return path


def _model_document(
    *,
    K: np.ndarray | None = None,
    D: list[float] | None = None,
    source_name: str = "gemini2xl_TESTSERIAL",
    width: int = 80,
    height: int = 60,
) -> dict:
    if K is None:
        K = _capture_arrays("unused", 0)["K"]
    if D is None:
        D = [0.08, -0.03, 0.001, -0.002, 0.005]
    return {
        "schema_version": CAMERA_MODEL_SCHEMA_VERSION,
        "source_name": source_name,
        "image_size": {"width": width, "height": height},
        "K": K.tolist(),
        "D": D,
        "distortion_model": "plumb_bob",
        "provenance": {
            "method": "live_camera_info_and_offline_intrinsic_audit",
            "recorded_at": "2026-08-13T00:00:00+08:00",
        },
    }


def _save_model(path: Path, document: dict | None = None) -> Path:
    if document is None:
        document = _model_document()
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def test_reprocess_is_immutable_and_records_model_and_field_hashes(tmp_path) -> None:
    captures = (
        _save_capture(tmp_path, "pose_001", 1),
        _save_capture(tmp_path, "pose_002", 2),
    )
    model_path = _save_model(tmp_path / "camera_model.json")
    original_file_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in captures
    }

    dataset = reprocess_capture_session(
        tmp_path,
        model_path,
        sample_ids=("pose_001", "pose_002"),
    )

    assert dataset.to_dict()["schema_version"] == SCHEMA_VERSION
    assert [sample.sample_id for sample in dataset.samples] == ["pose_001", "pose_002"]
    audit = dataset.metadata["camera_model_reprocessing"]
    assert audit["camera_model_sha256"] == hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    assert not audit["K_policy"]["K_changed_beyond_tolerance"]
    assert not audit["safety"]["safe_for_robot_use"]
    assert audit["opencv"]["pnp_flag"] == "SOLVEPNP_IPPE"
    assert len(audit["sample_audit"]) == 2
    for path, item in zip(captures, audit["sample_audit"], strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_file_hashes[
            path.name
        ]
        assert item["source_npz_sha256"] == original_file_hashes[path.name]
        assert len(item["checkerboard_corners_sha256"]) == 64
        assert len(item["checkerboard_object_points_sha256"]) == 64
        with np.load(path, allow_pickle=False) as archive:
            assert item["checkerboard_corners_sha256"] == ndarray_sha256(
                archive["checkerboard_corners_px"]
            )
            assert item["checkerboard_object_points_sha256"] == ndarray_sha256(
                archive["checkerboard_object_points_m"]
            )
    assert np.allclose(
        dataset.samples[0].T_base_link7,
        _capture_arrays("pose_001", 1)["T_base_link7"],
    )


def test_default_rejects_k_change_but_explicit_mode_is_diagnostic(tmp_path) -> None:
    _save_capture(tmp_path, "pose_001", 1)
    changed_K = _capture_arrays("unused", 0)["K"].copy()
    changed_K[0, 0] += 0.01
    model_path = _save_model(
        tmp_path / "camera_model.json",
        _model_document(K=changed_K),
    )
    with pytest.raises(CameraModelReprocessError, match="K differs"):
        reprocess_capture_session(tmp_path, model_path)

    dataset = reprocess_capture_session(
        tmp_path,
        model_path,
        allow_k_change=True,
    )
    policy = dataset.metadata["camera_model_reprocessing"]["K_policy"]
    safety = dataset.metadata["camera_model_reprocessing"]["safety"]
    assert policy["K_changed_beyond_tolerance"]
    assert policy["diagnostic_only_due_to_K_change"]
    assert not safety["safe_for_robot_use"]
    assert "diagnostic" in safety["reason"]


@pytest.mark.parametrize(
    ("model_change", "message"),
    (
        ({"source_name": "gemini2xl_OTHER"}, "source_name/serial"),
        ({"image_size": {"width": 81, "height": 60}}, "image size"),
        ({"D": [0.0] * 5}, "non-zero"),
        ({"provenance": {}}, "non-empty"),
    ),
)
def test_model_source_size_nonzero_d_and_provenance_fail_closed(
    tmp_path,
    model_change,
    message,
) -> None:
    _save_capture(tmp_path, "pose_001", 1)
    document = _model_document()
    document.update(model_change)
    model_path = _save_model(tmp_path / "camera_model.json", document)
    with pytest.raises(CameraModelReprocessError, match=message):
        reprocess_capture_session(tmp_path, model_path)


def test_nonzero_original_d_and_changed_object_points_are_rejected(tmp_path) -> None:
    original_D = np.array((0.01, -0.005, 0.0, 0.0, 0.0))
    _save_capture(tmp_path, "pose_001", 1, D=original_D)
    model_path = _save_model(tmp_path / "camera_model.json")
    with pytest.raises(CameraModelReprocessError, match="original D=0"):
        reprocess_capture_session(tmp_path, model_path)

    (tmp_path / "pose_001.npz").unlink()
    arrays = _capture_arrays("pose_001", 1)
    arrays["checkerboard_object_points_m"] = arrays[
        "checkerboard_object_points_m"
    ].copy()
    arrays["checkerboard_object_points_m"][0, 0] += 0.001
    np.savez_compressed(tmp_path / "pose_001.npz", **arrays)
    with pytest.raises(CameraModelReprocessError, match="object points"):
        reprocess_capture_session(tmp_path, model_path)


def test_camera_model_rejects_duplicate_keys_and_unknown_fields(tmp_path) -> None:
    _save_capture(tmp_path, "pose_001", 1)
    model_path = tmp_path / "camera_model.json"
    text = json.dumps(_model_document())
    model_path.write_text(text[:-1] + ', "D": [1, 0, 0, 0, 0]}', encoding="utf-8")
    with pytest.raises(CameraModelReprocessError, match="duplicate JSON key 'D'"):
        reprocess_capture_session(tmp_path, model_path)

    document = _model_document()
    document["typo_DD"] = document["D"]
    _save_model(model_path, document)
    with pytest.raises(CameraModelReprocessError, match="unknown typo_DD"):
        reprocess_capture_session(tmp_path, model_path)


def test_cli_writes_loadable_dataset_and_cannot_overwrite_model(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _save_capture(tmp_path, "pose_001", 1)
    model_path = _save_model(tmp_path / "camera_model.json")
    output_path = tmp_path / "reprocessed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "handeye_reprocess_camera_model",
            "--input-directory",
            str(tmp_path),
            "--camera-model",
            str(model_path),
            "--output",
            str(output_path),
            "--sample-ids",
            "pose_001",
        ],
    )
    cli_main()
    assert len(load_dataset(output_path).samples) == 1
    printed = json.loads(capsys.readouterr().out)
    assert not printed["robot_contacted"]
    assert not printed["source_npz_modified"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "handeye_reprocess_camera_model",
            "--input-directory",
            str(tmp_path),
            "--camera-model",
            str(model_path),
            "--output",
            str(model_path),
        ],
    )
    with pytest.raises(SystemExit):
        cli_main()
