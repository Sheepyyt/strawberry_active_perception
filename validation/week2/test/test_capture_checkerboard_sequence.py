"""无需 ROS 或相机的多帧 Observation 采集工具单测。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest


WEEK2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK2_DIR))

import capture_checkerboard_sequence as capture  # noqa: E402


def _header(sec=10, nanosec=20, frame="camera_color_optical_frame"):
    return NS(stamp=NS(sec=sec, nanosec=nanosec), frame_id=frame)


def _message(observation_id="obs-1", sec=10, scene="checkerboard"):
    height, width = 2, 3
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.array([[0.4, np.inf, 0.5], [0.0, 0.6, 0.7]], dtype="<f4")
    header = _header(sec=sec)
    image_rgb = NS(
        header=header, height=height, width=width, step=width * 3,
        encoding="rgb8", is_bigendian=False, data=rgb.tobytes(),
    )
    image_depth = NS(
        header=header, height=height, width=width, step=width * 4,
        encoding="32FC1", is_bigendian=False, data=depth.tobytes(),
    )
    pose = NS(
        header=NS(frame_id="camera_session"),
        pose=NS(
            position=NS(x=0.0, y=0.0, z=0.0),
            orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    info = NS(
        header=header, height=height, width=width,
        k=[500.0, 0.0, 1.0, 0.0, 501.0, 0.5, 0.0, 0.0, 1.0],
    )
    return NS(
        header=header, scene_id=scene, observation_id=observation_id,
        source_name="gemini2xl", source_type=1, color=image_rgb, depth=image_depth,
        camera_info=info, camera_pose=pose, pose_valid=True,
        valid_depth_fraction=4 / 6, color_depth_skew_sec=0.001,
    )


def _frame(index: int) -> capture.DecodedFrame:
    message = _message(observation_id=f"obs-{index}", sec=10 + index)
    return capture.decode_observation(
        message, expected_scene="checkerboard", expected_observation_id=f"obs-{index}"
    )


def test_decode_canonical_rgb_depth_k_stamp_and_pose():
    frame = _frame(1)
    assert frame.rgb.shape == (2, 3, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth_m.dtype == np.float32
    assert np.isnan(frame.depth_m[0, 1])
    assert np.isnan(frame.depth_m[1, 0])
    assert frame.K[1, 1] == 501.0
    assert frame.stamp.tolist() == [11, 20]
    np.testing.assert_array_equal(frame.T_world_camera_optical, np.eye(4))


@pytest.mark.parametrize("field", ["scene", "id"])
def test_decode_rejects_response_identity_mismatch(field):
    message = _message()
    scene = "wrong" if field == "scene" else "checkerboard"
    observation_id = "wrong" if field == "id" else "obs-1"
    with pytest.raises(capture.CaptureError, match="不匹配"):
        capture.decode_observation(
            message, expected_scene=scene, expected_observation_id=observation_id
        )


def test_decode_rejects_wrong_encoding_or_dimensions():
    message = _message()
    message.depth.encoding = "16UC1"
    with pytest.raises(capture.CaptureError, match="32FC1"):
        capture.decode_observation(
            message, expected_scene="checkerboard", expected_observation_id="obs-1"
        )
    message = _message()
    message.camera_info.width = 4
    with pytest.raises(capture.CaptureError, match="CameraInfo 尺寸"):
        capture.decode_observation(
            message, expected_scene="checkerboard", expected_observation_id="obs-1"
        )


def test_save_compressed_npz_is_loadable_without_pickle(tmp_path):
    destination = tmp_path / "sequence.npz"
    capture.save_sequence([_frame(0), _frame(1)], destination)
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["rgb"].shape == (2, 2, 3, 3)
        assert archive["depth_m"].shape == (2, 2, 3)
        assert archive["K"].shape == (2, 3, 3)
        assert archive["stamp"].tolist() == [[10, 20], [11, 20]]
        assert archive["observation_id"].tolist() == ["obs-0", "obs-1"]
        assert archive["scene_id"].tolist() == ["checkerboard", "checkerboard"]
        assert archive["T_world_camera_optical"].shape == (2, 4, 4)
        assert all(value.dtype.kind != "O" for value in archive.values())


def test_invalid_sequence_leaves_no_success_file(tmp_path):
    destination = tmp_path / "must_not_exist.npz"
    duplicate = _frame(0)
    with pytest.raises(capture.CaptureError, match="重复"):
        capture.save_sequence([duplicate, duplicate], destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_existing_output_is_never_overwritten(tmp_path):
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"keep me")
    with pytest.raises(capture.CaptureError, match="拒绝"):
        capture.save_sequence([_frame(0)], destination)
    assert destination.read_bytes() == b"keep me"
