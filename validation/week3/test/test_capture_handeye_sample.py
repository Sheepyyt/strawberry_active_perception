"""无需 ROS graph、相机或机械臂的手眼现场采集器单测。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
import sys

import cv2
import numpy as np
import pytest


WEEK3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK3))

import capture_handeye_sample as capture  # noqa: E402


def _header(sec=100, nanosec=20, frame="camera_color_optical_frame"):
    return NS(stamp=NS(sec=sec, nanosec=nanosec), frame_id=frame)


def _observation(
    *, observation_id="obs-1", scene="handeye", width=6, height=4
):
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.full((height, width), 0.6, dtype="<f4")
    depth[0, 0] = np.nan
    header = _header()
    color_header = _header(nanosec=1_000_020)
    color = NS(
        header=color_header,
        width=width,
        height=height,
        step=width * 3,
        encoding="rgb8",
        is_bigendian=False,
        data=rgb.tobytes(),
    )
    depth_message = NS(
        header=header,
        width=width,
        height=height,
        step=width * 4,
        encoding="32FC1",
        is_bigendian=False,
        data=depth.tobytes(),
    )
    info = NS(
        header=header,
        width=width,
        height=height,
        k=[500.0, 0.0, 2.5, 0.0, 501.0, 1.5, 0.0, 0.0, 1.0],
        d=[0.0] * 5,
    )
    return NS(
        header=header,
        scene_id=scene,
        observation_id=observation_id,
        source_type=1,
        source_name="gemini2xl_serial_hash",
        color=color,
        depth=depth_message,
        camera_info=info,
        valid_depth_fraction=float(np.isfinite(depth).mean()),
        color_depth_skew_sec=0.001,
    )


def _joint_message(
    *, sec=100, nanosec=0, positions=None, velocities=None, names=None
):
    names = list(capture.JOINT_NAMES if names is None else names)
    positions = np.zeros(7) if positions is None else np.asarray(positions)
    velocities = np.zeros(7) if velocities is None else np.asarray(velocities)
    return NS(
        header=_header(sec=sec, nanosec=nanosec, frame=""),
        name=names,
        position=positions.tolist(),
        velocity=velocities.tolist(),
    )


def _stationary(count=7, *, movement=0.0, velocity=0.0):
    samples = []
    for index in range(count):
        positions = np.zeros(7)
        positions[2] = movement * index / max(1, count - 1)
        samples.append(
            capture.JointSample(
                stamp_ns=100_000_000_000 + index * 200_000_000,
                received_monotonic=index * 0.2,
                positions=positions,
                velocities=np.full(7, velocity),
            )
        )
    return samples


def _synthetic_board(*, crop_right=False, rotate=False):
    square = 36
    columns, rows = capture.BOARD_SQUARES
    board = np.empty((rows * square, columns * square), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            value = 0 if (row + column) % 2 == 0 else 255
            board[
                row * square : (row + 1) * square,
                column * square : (column + 1) * square,
            ] = value
    canvas = np.full((460, 620), 180, dtype=np.uint8)
    x0, y0 = (210 if crop_right else 70), 60
    x1 = min(canvas.shape[1], x0 + board.shape[1])
    canvas[y0 : y0 + board.shape[0], x0:x1] = board[:, : x1 - x0]
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    if rotate:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    return rgb


def _detection_and_observation():
    rgb = _synthetic_board()
    K = np.array([[500.0, 0.0, 310.0], [0.0, 500.0, 230.0], [0.0, 0.0, 1.0]])
    detection = capture.detect_checkerboard(
        rgb,
        K,
        np.zeros(5),
        minimum_border_margin_px=8.0,
        maximum_reprojection_rms_px=1.0,
    )
    observation = capture.DecodedObservation(
        rgb=rgb,
        depth_m=np.full(rgb.shape[:2], 0.6, dtype=np.float32),
        K=K,
        D=np.zeros(5),
        stamp=np.array([100, 20], dtype=np.int64),
        scene_id="handeye",
        observation_id="obs-1",
        optical_frame="camera_color_optical_frame",
        source_name="gemini2xl_serial_hash",
        source_type=1,
        valid_depth_fraction=1.0,
        color_depth_skew_sec=0.001,
    )
    stationary = capture.stationary_evidence(
        _stationary(),
        minimum_duration_sec=1.0,
        maximum_position_span_rad=0.002,
        maximum_absolute_velocity_rad_s=0.01,
        maximum_feedback_age_sec=0.25,
        now_monotonic=1.21,
    )
    selected = _stationary()[-1]
    return detection, observation, stationary, selected


def test_decode_canonical_observation_checks_stamp_k_and_payload():
    result = capture.decode_observation(
        _observation(), expected_scene="handeye", expected_observation_id="obs-1"
    )
    assert result.rgb.shape == (4, 6, 3)
    assert result.depth_m.dtype == np.float32
    assert np.isnan(result.depth_m[0, 0])
    assert result.K[1, 1] == 501.0
    assert result.stamp.tolist() == [100, 20]

    bad = _observation()
    bad.camera_info.header = _header(nanosec=21)
    with pytest.raises(capture.CaptureError, match="CameraInfo stamp"):
        capture.decode_observation(
            bad, expected_scene="handeye", expected_observation_id="obs-1"
        )

    empty_id = _observation(observation_id="")
    with pytest.raises(capture.CaptureError, match="不能为空"):
        capture.decode_observation(
            empty_id, expected_scene="handeye", expected_observation_id="obs-1"
        )


def test_joint_feedback_requires_complete_finite_positions_and_velocities():
    result = capture.ordered_joint_sample(_joint_message(), 1.0)
    assert result.positions.shape == (7,)
    assert result.velocities.shape == (7,)

    incomplete = _joint_message(names=capture.JOINT_NAMES[:-1])
    incomplete.position = [0.0] * 6
    incomplete.velocity = [0.0] * 6
    with pytest.raises(capture.CaptureError, match="joint7"):
        capture.ordered_joint_sample(incomplete, 1.0)
    missing_velocity = _joint_message()
    missing_velocity.velocity = []
    with pytest.raises(capture.CaptureError, match="velocity"):
        capture.ordered_joint_sample(missing_velocity, 1.0)


def test_stationary_gate_checks_window_age_position_span_and_velocity():
    evidence = capture.stationary_evidence(
        _stationary(),
        minimum_duration_sec=1.0,
        maximum_position_span_rad=0.002,
        maximum_absolute_velocity_rad_s=0.01,
        maximum_feedback_age_sec=0.25,
        now_monotonic=1.21,
    )
    assert evidence.maximum_position_span_rad == 0.0
    assert evidence.window_duration_sec == pytest.approx(1.2)

    with pytest.raises(capture.CaptureError, match="最大关节跨度"):
        capture.stationary_evidence(
            _stationary(movement=0.003),
            minimum_duration_sec=1.0,
            maximum_position_span_rad=0.002,
            maximum_absolute_velocity_rad_s=0.01,
            maximum_feedback_age_sec=0.25,
            now_monotonic=1.21,
        )
    with pytest.raises(capture.CaptureError, match="最大反馈速度"):
        capture.stationary_evidence(
            _stationary(velocity=0.02),
            minimum_duration_sec=1.0,
            maximum_position_span_rad=0.002,
            maximum_absolute_velocity_rad_s=0.01,
            maximum_feedback_age_sec=0.25,
            now_monotonic=1.21,
        )

    interrupted = _stationary(count=5)
    interrupted = [
        capture.JointSample(
            stamp_ns=sample.stamp_ns,
            received_monotonic=received,
            positions=sample.positions,
            velocities=sample.velocities,
        )
        for sample, received in zip(interrupted, (0.0, 0.1, 0.2, 10.0, 10.1))
    ]
    with pytest.raises(capture.CaptureError, match="反馈不连续"):
        capture.stationary_evidence(
            interrupted,
            minimum_duration_sec=2.0,
            maximum_position_span_rad=0.002,
            maximum_absolute_velocity_rad_s=0.01,
            maximum_feedback_age_sec=0.25,
            maximum_feedback_gap_sec=0.25,
            now_monotonic=10.15,
        )


def test_recent_window_and_nearest_stamp_are_explicit():
    samples = _stationary(count=12)
    recent = capture.recent_joint_samples(
        samples, now_monotonic=2.2, duration_sec=1.0
    )
    assert recent[0].received_monotonic == pytest.approx(1.2)
    selected, delta = capture.nearest_joint_sample(
        samples, 101_000_000_000, 0.11
    )
    assert selected.stamp_ns == 101_000_000_000
    assert delta == 0.0
    with pytest.raises(capture.CaptureError, match="超过门限"):
        capture.nearest_joint_sample(samples, 200_000_000_000, 0.20)
    assert capture.normalized_duration_fields(1.9999999999) == (2, 0)


def test_checkerboard_detection_requires_all_88_corners_and_full_outer_board():
    rgb = _synthetic_board()
    K = np.array([[500.0, 0.0, 310.0], [0.0, 500.0, 230.0], [0.0, 0.0, 1.0]])
    result = capture.detect_checkerboard(
        rgb,
        K,
        np.zeros(5),
        minimum_border_margin_px=8.0,
        maximum_reprojection_rms_px=1.0,
    )
    assert result.corners_px.shape == (88, 2)
    assert result.object_points_m.shape == (88, 3)
    assert result.minimum_border_margin_px > 8.0
    assert result.reprojection_rms_px < 1.0
    np.testing.assert_allclose(result.T_camera_checkerboard[3], [0, 0, 0, 1])
    assert np.linalg.det(result.T_camera_checkerboard[:3, :3]) == pytest.approx(1.0)

    with pytest.raises(capture.CaptureError, match="未完整检测|边缘"):
        capture.detect_checkerboard(
            _synthetic_board(crop_right=True),
            K,
            np.zeros(5),
            minimum_border_margin_px=8.0,
            maximum_reprojection_rms_px=1.0,
        )


def test_corner_orientation_uses_dark_physical_origin_after_image_rotation():
    K = np.array([[500.0, 0.0, 310.0], [0.0, 500.0, 230.0], [0.0, 0.0, 1.0]])
    first = capture.detect_checkerboard(
        _synthetic_board(), K, np.zeros(5),
        minimum_border_margin_px=8.0, maximum_reprojection_rms_px=1.0,
    )
    rotated = capture.detect_checkerboard(
        _synthetic_board(rotate=True), K, np.zeros(5),
        minimum_border_margin_px=8.0, maximum_reprojection_rms_px=1.0,
    )
    # 物理黑色原点随图像旋转到右下；不会被错误重命名为新的图像左上角。
    assert first.corners_px[0, 0] < first.corners_px[-1, 0]
    assert rotated.corners_px[0, 0] > rotated.corners_px[-1, 0]


def test_npz_is_atomic_non_object_and_has_core_transform_convention(tmp_path):
    detection, observation, stationary, selected = _detection_and_observation()
    arrays = capture.sample_arrays(
        sample_id="pose_001",
        observation=observation,
        detection=detection,
        stationary=stationary,
        T_base_link7=np.eye(4),
        tf_lookup_mode="observation_stamp",
        tf_stamp_ns=100_000_000_020,
        selected_joint=selected,
        joint_observation_delta_sec=0.0,
        joint_tf_delta_sec=0.0,
        joint_selection_mode="nearest_observation_ros_stamp",
    )
    path = capture.save_npz_atomic(arrays, tmp_path / "pose_001.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert all(value.dtype.kind != "O" for value in archive.values())
        assert archive["sample_id"].item() == "pose_001"
        assert archive["corner_count"].item() == 88
        assert archive["timestamp_sec"].dtype == np.float64
        assert archive["motion_commands_sent"].item() == 0
        assert archive["joint_maximum_feedback_gap_sec"].item() == pytest.approx(0.2)
        assert archive["joint_tf_delta_sec"].item() == 0.0
        assert archive["base_frame"].item() == "base_link"
        assert archive["link_frame"].item() == "link7"
        np.testing.assert_array_equal(archive["T_base_link7"], np.eye(4))
        np.testing.assert_allclose(archive["T_camera_checkerboard"][3], [0, 0, 0, 1])
    with pytest.raises(capture.CaptureError, match="拒绝覆盖"):
        capture.save_npz_atomic(arrays, path)


def test_duplicate_observation_stamp_pose_and_changed_k_are_rejected(tmp_path):
    detection, observation, stationary, selected = _detection_and_observation()
    arrays = capture.sample_arrays(
        sample_id="pose_001", observation=observation, detection=detection,
        stationary=stationary, T_base_link7=np.eye(4),
        tf_lookup_mode="latest_while_stationary", tf_stamp_ns=100,
        selected_joint=selected, joint_observation_delta_sec=0.0,
        joint_tf_delta_sec=0.0,
        joint_selection_mode="nearest_tf_stamp_while_stationary",
    )
    capture.save_npz_atomic(arrays, tmp_path / "pose_001.npz")
    with pytest.raises(capture.CaptureError, match="Observation ID"):
        capture.assert_unique_sample(
            tmp_path, sample_id="pose_002", observation_id="obs-1",
            observation_stamp_ns=101, positions=np.ones(7), K=observation.K,
            optical_frame=observation.optical_frame, source_name=observation.source_name,
            minimum_joint_separation_rad=0.03,
        )
    with pytest.raises(capture.CaptureError, match="近重复"):
        capture.assert_unique_sample(
            tmp_path, sample_id="pose_002", observation_id="obs-2",
            observation_stamp_ns=101, positions=np.full(7, 0.01), K=observation.K,
            optical_frame=observation.optical_frame, source_name=observation.source_name,
            minimum_joint_separation_rad=0.03,
        )
    changed = observation.K.copy()
    changed[0, 0] += 1.0
    with pytest.raises(capture.CaptureError, match="内参 K"):
        capture.assert_unique_sample(
            tmp_path, sample_id="pose_002", observation_id="obs-2",
            observation_stamp_ns=101, positions=np.ones(7), K=changed,
            optical_frame=observation.optical_frame, source_name=observation.source_name,
            minimum_joint_separation_rad=0.03,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("stamp", np.array([100], dtype=np.int64), "stamp 非法"),
        ("stamp", np.array([100, 1_000_000_000], dtype=np.int64), "stamp 非法"),
        ("joint_positions", np.zeros(6), "joint_positions 非法"),
        ("joint_positions", np.array([0, 0, 0, np.nan, 0, 0, 0]), "joint_positions 非法"),
    ),
)
def test_malformed_same_schema_npz_fails_closed(tmp_path, field, value, message):
    detection, observation, stationary, selected = _detection_and_observation()
    arrays = capture.sample_arrays(
        sample_id="pose_001", observation=observation, detection=detection,
        stationary=stationary, T_base_link7=np.eye(4),
        tf_lookup_mode="observation_stamp", tf_stamp_ns=100,
        selected_joint=selected, joint_observation_delta_sec=0.0,
        joint_tf_delta_sec=0.0,
        joint_selection_mode="nearest_observation_ros_stamp",
    )
    arrays[field] = value
    capture.save_npz_atomic(arrays, tmp_path / "pose_001.npz")
    with pytest.raises(capture.CaptureError, match=message):
        capture.assert_unique_sample(
            tmp_path, sample_id="pose_002", observation_id="obs-2",
            observation_stamp_ns=101, positions=np.ones(7), K=observation.K,
            optical_frame=observation.optical_frame, source_name=observation.source_name,
            minimum_joint_separation_rad=0.03,
        )


def test_atomic_save_never_replaces_racing_destination(tmp_path, monkeypatch):
    destination = tmp_path / "pose_001.npz"
    real_link = capture.os.link

    def create_competing_file_then_link(source, target):
        destination.write_bytes(b"other-process")
        return real_link(source, target)

    monkeypatch.setattr(capture.os, "link", create_competing_file_then_link)
    with pytest.raises(capture.CaptureError, match="拒绝覆盖"):
        capture.save_npz_atomic({"sample_id": np.array("pose_001")}, destination)
    assert destination.read_bytes() == b"other-process"
    assert not list(tmp_path.glob(".*.tmp"))


def test_source_contains_no_motion_client_or_publisher():
    source = (WEEK3 / "capture_handeye_sample.py").read_text(encoding="utf-8")
    forbidden = (
        "create_publisher(",
        "ActionClient(",
        "/control/",
        "MoveToPose",
        "FollowJointTrajectory",
        "enable_execution",
    )
    assert all(token not in source for token in forbidden)
    launch = (WEEK3 / "launch/read_only_nero_tf.launch.py").read_text(encoding="utf-8")
    assert "robot_state_publisher" in launch
    assert '("joint_states", "/feedback/joint_states")' in launch
    assert "/control/" not in launch
