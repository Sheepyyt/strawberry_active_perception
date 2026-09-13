"""Unit-level ROS wrapper tests with an injected, GPU-free NBV backend."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Callable

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
import numpy as np
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from rclpy.parameter import Parameter

from strawberry_gradient_nbv.core import NBVInputError
from strawberry_gradient_nbv.image_codec import ObservationDecodeError
from strawberry_gradient_nbv.ros_node import (
    GradientNBVNode,
    _configuration_mapping,
    _device_argument,
)
from strawberry_perception_interfaces.action import ComputeNextView
from strawberry_perception_interfaces.msg import NextView, Observation
from strawberry_perception_interfaces.srv import (
    ConfigureNBV,
    EvaluateViewCandidates,
    ResetNBVMap,
)


class _FakeBackend:
    """Small transactional backend whose map is one observable integer."""

    def __init__(self) -> None:
        self.configurations = []
        self.configure_error: Exception | None = None
        self.reset_error: Exception | None = None
        self.update_error: Exception | None = None
        self.bad_result = False
        self.state = 0
        self.update_calls = 0
        self.reset_calls = 0
        self.snapshot_calls = 0
        self.restore_calls = 0
        self.evaluate_calls = 0
        self.evaluate_error: Exception | None = None

    def configure(self, config) -> None:
        if self.configure_error is not None:
            raise self.configure_error
        self.configurations.append(config)
        self.state = 0

    def reset(self) -> bool:
        self.reset_calls += 1
        had_data = self.state != 0
        self.state = 0
        if self.reset_error is not None:
            raise self.reset_error
        return had_data

    def snapshot(self) -> int:
        self.snapshot_calls += 1
        return self.state

    def restore(self, snapshot: int) -> None:
        self.restore_calls += 1
        self.state = snapshot

    def update_and_plan(self, depth, mask, intrinsics, pose):
        assert depth.shape == mask.shape == (2, 2)
        assert intrinsics.shape == (3, 3)
        assert pose.shape == (4, 4)
        self.update_calls += 1
        self.state += 1
        if self.update_error is not None:
            raise self.update_error
        return SimpleNamespace(
            pose=np.eye(4, dtype=np.float64),
            gain=float("nan") if self.bad_result else 2.5,
            coverage=0.4,
            total_voxel_count=10,
            observed_voxel_count=4,
            occupied_voxel_count=2,
            unknown_voxel_count=6,
            optimization_iterations=3,
            compute_time_ms=7.5,
        )

    def evaluate_candidate_poses(
        self, reference_pose, candidate_poses, intrinsics, height, width
    ):
        assert reference_pose.shape == (4, 4)
        assert candidate_poses.ndim == 3 and candidate_poses.shape[1:] == (4, 4)
        assert intrinsics.shape == (3, 3)
        assert (height, width) == (2, 2)
        self.evaluate_calls += 1
        if self.evaluate_error is not None:
            raise self.evaluate_error
        return 1.25, np.arange(candidate_poses.shape[0], dtype=float) + 2.0

    def visualization_state(self):
        observed = np.asarray([[[self.state > 0]]], dtype=bool)
        return {
            "dimensions": np.array([1, 1, 1], dtype=np.int32),
            "origin_m": np.array([-0.5, -0.5, 0.0]),
            "voxel_size_m": np.asarray(1.0),
            "target_center_m": np.array([0.0, 0.0, 0.5]),
            "target_roi_size_m": np.array([1.0, 1.0, 1.0]),
            "observed": observed,
            "occupied": observed.copy(),
            "target": observed.copy(),
        }


class _GoalHandle:
    def __init__(self, scene_id: str, observation_id: str) -> None:
        self.request = ComputeNextView.Goal()
        self.request.scene_id = scene_id
        self.request.observation_id = observation_id
        self.is_cancel_requested = False
        self.feedback = []
        self.terminal_state = None

    def publish_feedback(self, message) -> None:
        self.feedback.append(deepcopy(message))

    def succeed(self) -> None:
        self.terminal_state = "succeeded"

    def abort(self) -> None:
        self.terminal_state = "aborted"

    def canceled(self) -> None:
        self.terminal_state = "canceled"


class _PublisherRecorder:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(deepcopy(message))


@pytest.fixture
def wrapper():
    if not rclpy.ok():
        rclpy.init()
    backend = _FakeBackend()
    node = GradientNBVNode(backend=backend)
    publisher = _PublisherRecorder()
    node._next_view_publisher = publisher
    try:
        yield node, backend, publisher
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _stamp(offset_nanoseconds: int = 0) -> Time:
    total = 10_000_000_000 + offset_nanoseconds
    return Time(sec=total // 1_000_000_000, nanosec=total % 1_000_000_000)


def _configuration(scene_id: str = "scene-a") -> ConfigureNBV.Request:
    request = ConfigureNBV.Request()
    request.scene_id = scene_id
    request.world_frame = "fixture_world"
    request.target_center.x = 0.0
    request.target_center.y = 0.0
    request.target_center.z = 0.8
    request.map_size.x = 0.3
    request.map_size.y = 0.3
    request.map_size.z = 0.3
    request.target_roi_size.x = 0.15
    request.target_roi_size.y = 0.15
    request.target_roi_size.z = 0.15
    request.observation_min.x = -1.0
    request.observation_min.y = -1.0
    request.observation_min.z = -1.0
    request.observation_max.x = 1.0
    request.observation_max.y = 1.0
    request.observation_max.z = 1.0
    request.voxel_size = 0.03
    request.depth_min = 0.2
    request.depth_max = 2.0
    request.samples_per_ray = 8
    request.optimization_steps = 3
    request.max_step = 0.1
    request.random_seed = 7
    return request


def _set_image(message, encoding: str, array: np.ndarray, stamp: Time) -> None:
    message.header.stamp = stamp
    message.header.frame_id = "camera_optical_frame"
    message.height = int(array.shape[0])
    message.width = int(array.shape[1])
    message.encoding = encoding
    message.is_bigendian = False
    message.step = int(array.strides[0])
    message.data = np.ascontiguousarray(array).tobytes()


def _observation(observation_id: str = "obs-1") -> Observation:
    message = Observation()
    message.header.stamp = _stamp()
    message.header.frame_id = "camera_optical_frame"
    message.scene_id = "scene-a"
    message.observation_id = observation_id
    message.source_type = Observation.SOURCE_SYNTHETIC
    message.source_name = "wrapper_fixture"
    color = np.array(
        [[[220, 20, 20], [1, 2, 3]], [[4, 5, 6], [7, 8, 9]]],
        dtype=np.uint8,
    )
    depth = np.array([[1.0, np.nan], [0.3, 2.5]], dtype=np.float32)
    mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    _set_image(message.color, "rgb8", color, _stamp(1_000_000))
    _set_image(message.depth, "32FC1", depth, _stamp())
    _set_image(message.target_mask, "mono8", mask, _stamp(1_000_000))
    message.camera_info.header.stamp = _stamp()
    message.camera_info.header.frame_id = "camera_optical_frame"
    message.camera_info.width = 2
    message.camera_info.height = 2
    message.camera_info.k = [100.0, 0.0, 0.5, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0]
    message.camera_info.d = [0.0] * 5
    message.camera_pose.header.stamp = _stamp()
    message.camera_pose.header.frame_id = "fixture_world"
    message.camera_pose.pose.orientation.w = 1.0
    message.pose_valid = True
    # Canonical validity is the finite-depth fraction.  The NBV session may
    # independently use a narrower [depth_min, depth_max] interval.
    message.valid_depth_fraction = 0.75
    message.color_depth_skew_sec = 0.001
    return message


def _configure(node: GradientNBVNode) -> ConfigureNBV.Response:
    response = node._on_configure(_configuration(), ConfigureNBV.Response())
    assert response.success
    return response


def _compute(node: GradientNBVNode, observation_id: str = "obs-1"):
    goal = _GoalHandle("scene-a", observation_id)
    return goal, node._execute_compute(goal)


def _candidate_request(observation_id: str = "obs-1"):
    request = EvaluateViewCandidates.Request()
    request.scene_id = "scene-a"
    request.observation_id = observation_id
    pose = PoseStamped()
    pose.header.frame_id = "fixture_world"
    pose.header.stamp = _stamp()
    pose.pose.orientation.w = 1.0
    request.candidate_poses = [pose]
    return request


def test_configuration_codes_are_specific_and_device_auto_is_supported() -> None:
    assert _device_argument("") is None
    assert _device_argument(" auto ") is None
    assert _device_argument("cuda:1") == "cuda:1"

    request = _configuration()
    request.scene_id = ""
    with pytest.raises(ValueError) as error:
        _configuration_mapping(request)
    assert error.value.code == ConfigureNBV.Response.INVALID_SCENE_ID

    request = _configuration()
    request.target_roi_size.x = 0.4
    with pytest.raises(ValueError) as error:
        _configuration_mapping(request)
    assert error.value.code == ConfigureNBV.Response.INVALID_TARGET_ROI

    request = _configuration()
    request.depth_max = request.depth_min
    with pytest.raises(ValueError) as error:
        _configuration_mapping(request)
    assert error.value.code == ConfigureNBV.Response.INVALID_DEPTH_RANGE

    request = _configuration()
    request.voxel_size = np.finfo(np.float32).tiny
    with pytest.raises(ValueError) as error:
        _configuration_mapping(request)
    assert error.value.code == ConfigureNBV.Response.INVALID_MAP_BOUNDS


def test_configure_is_atomic_and_reset_requires_exact_scene(wrapper) -> None:
    node, backend, _publisher = wrapper
    node._on_observation(_observation("pre-config"))
    response = _configure(node)
    assert response.code == ConfigureNBV.Response.SUCCESS
    assert backend.configurations[-1].scene_id == "scene-a"
    assert not node._observations

    sentinel = NextView()
    node._results[("scene-a", "old")] = sentinel
    node._on_observation(_observation("old"))
    backend.state = 3
    backend.configure_error = RuntimeError("allocation failed")
    failed = node._on_configure(_configuration("scene-b"), ConfigureNBV.Response())
    assert not failed.success
    assert failed.code == ConfigureNBV.Response.INTERNAL_ERROR
    assert node._configuration["scene_id"] == "scene-a"
    assert ("scene-a", "old") in node._results
    assert ("scene-a", "old") in node._observations
    assert backend.state == 3

    mismatch_request = ResetNBVMap.Request()
    mismatch_request.scene_id = "scene-b"
    mismatch = node._on_reset(mismatch_request, ResetNBVMap.Response())
    assert not mismatch.success
    assert mismatch.code == ResetNBVMap.Response.SCENE_MISMATCH
    assert backend.reset_calls == 0
    assert backend.state == 3

    backend.reset_error = RuntimeError("reset failed after mutation")
    failed_reset = node._on_reset(reset_request := ResetNBVMap.Request(), ResetNBVMap.Response())
    # Populate after construction to keep the generated service type explicit.
    # The empty request is rejected without entering the backend.
    assert failed_reset.code == ResetNBVMap.Response.INVALID_SCENE_ID
    reset_request.scene_id = "scene-a"
    failed_reset = node._on_reset(reset_request, ResetNBVMap.Response())
    assert not failed_reset.success
    assert failed_reset.code == ResetNBVMap.Response.INTERNAL_ERROR
    assert backend.state == 3
    assert ("scene-a", "old") in node._observations
    backend.reset_error = None

    reset = node._on_reset(reset_request, ResetNBVMap.Response())
    assert reset.success and reset.code == ResetNBVMap.Response.SUCCESS
    assert backend.state == 0
    assert not node._results
    assert not node._observations
    old_goal, old_result = _compute(node, "old")
    assert old_goal.terminal_state == "aborted"
    assert old_result.next_view.code == NextView.OBSERVATION_NOT_FOUND
    already_empty = node._on_reset(reset_request, ResetNBVMap.Response())
    assert already_empty.success
    assert already_empty.code == ResetNBVMap.Response.ALREADY_EMPTY


def test_observation_cache_is_bounded_and_duplicate_ids_are_first_wins(wrapper) -> None:
    node, _backend, _publisher = wrapper
    for index in range(33):
        node._on_observation(_observation(f"obs-{index}"))
    assert len(node._observations) == 32
    assert ("scene-a", "obs-0") not in node._observations

    duplicate = _observation("obs-32")
    duplicate.source_name = "mutated_duplicate"
    node._on_observation(duplicate)
    assert node._observations[("scene-a", "obs-32")].source_name == "wrapper_fixture"


def test_strict_observation_contract_rejects_each_invalid_field(wrapper) -> None:
    node, backend, _publisher = wrapper
    _configure(node)

    def set_zero_stamp(message: Observation) -> None:
        message.depth.header.stamp = Time()

    def set_large_color_skew(message: Observation) -> None:
        message.color.header.stamp = _stamp(6_000_000)
        message.color_depth_skew_sec = 0.006

    mutations: tuple[Callable[[Observation], None], ...] = (
        lambda message: setattr(message, "source_type", Observation.SOURCE_UNKNOWN),
        lambda message: setattr(message, "source_type", 99),
        lambda message: setattr(message, "source_name", ""),
        lambda message: setattr(message.header, "frame_id", ""),
        lambda message: setattr(message.color.header, "frame_id", "other_optical"),
        lambda message: setattr(message.camera_pose.header, "frame_id", "other_world"),
        lambda message: setattr(message.camera_info, "width", 3),
        lambda message: message.camera_info.k.__setitem__(0, 0.0),
        lambda message: setattr(message, "pose_valid", False),
        lambda message: setattr(message.camera_pose.pose.orientation, "w", 0.5),
        lambda message: setattr(message.target_mask, "data", bytes((255, 1, 0, 0))),
        lambda message: setattr(message.target_mask, "data", bytes(4)),
        lambda message: setattr(message.target_mask, "data", bytes((0, 0, 0, 255))),
        lambda message: setattr(message.header, "stamp", _stamp(1)),
        lambda message: setattr(message.camera_info.header, "stamp", _stamp(1)),
        set_zero_stamp,
        set_large_color_skew,
        lambda message: setattr(message.target_mask.header, "stamp", _stamp(6_000_000)),
        lambda message: setattr(message, "color_depth_skew_sec", 0.002),
        lambda message: setattr(message, "valid_depth_fraction", 0.25),
    )
    for mutate in mutations:
        message = _observation()
        mutate(message)
        with pytest.raises(ObservationDecodeError):
            node._decode_observation(message, "scene-a", "obs-1")
    assert backend.update_calls == 0
    assert backend.state == 0


def test_invalid_action_returns_structured_error_without_touching_map(wrapper) -> None:
    node, backend, publisher = wrapper
    _configure(node)
    message = _observation()
    message.valid_depth_fraction = 0.0
    node._on_observation(message)
    goal, result = _compute(node)
    assert goal.terminal_state == "aborted"
    assert not result.next_view.success
    assert result.next_view.code == NextView.INVALID_REQUEST
    assert "valid_depth_fraction" in result.next_view.reason
    assert backend.snapshot_calls == backend.update_calls == 0
    assert backend.state == 0
    assert publisher.messages[-1].code == NextView.INVALID_REQUEST


def test_action_success_is_idempotent_cached_and_publishes_all_phases(wrapper) -> None:
    node, backend, publisher = wrapper
    _configure(node)
    original = _observation()
    node._on_observation(original)

    first_goal, first = _compute(node)
    assert first_goal.terminal_state == "succeeded"
    assert first.next_view.success
    assert first.next_view.code == NextView.SUCCESS
    assert first.next_view.pose.header.frame_id == "fixture_world"
    assert first.next_view.gain == pytest.approx(2.5)
    assert backend.update_calls == 1 and backend.state == 1
    phases = [feedback.phase for feedback in first_goal.feedback]
    assert phases == [
        ComputeNextView.Feedback.PHASE_VALIDATING,
        ComputeNextView.Feedback.PHASE_UPDATING_MAP,
        ComputeNextView.Feedback.PHASE_OPTIMIZING,
        ComputeNextView.Feedback.PHASE_FINALIZING,
    ]

    second_goal, second = _compute(node)
    assert second_goal.terminal_state == "succeeded"
    assert second.next_view.success
    assert second.next_view.compute_time_ms == first.next_view.compute_time_ms
    assert backend.update_calls == 1 and backend.state == 1
    assert len(publisher.messages) == 2
    assert node._next_view_qos.history == HistoryPolicy.KEEP_LAST
    assert node._next_view_qos.depth == 1
    assert node._next_view_qos.reliability == ReliabilityPolicy.RELIABLE
    assert node._next_view_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert node.get_parameter("reset_service").value == "/strawberry/nbv/reset_map"
    assert (
        node.get_parameter("evaluate_candidates_service").value
        == "/strawberry/nbv/evaluate_candidates"
    )


def test_candidate_scoring_requires_processed_observation_and_is_read_only(wrapper) -> None:
    node, backend, _publisher = wrapper
    not_configured = node._on_evaluate_candidates(
        _candidate_request(), EvaluateViewCandidates.Response()
    )
    assert not_configured.code == EvaluateViewCandidates.Response.NOT_CONFIGURED

    _configure(node)
    node._on_observation(_observation())
    not_processed = node._on_evaluate_candidates(
        _candidate_request(), EvaluateViewCandidates.Response()
    )
    assert not_processed.code == EvaluateViewCandidates.Response.OBSERVATION_NOT_PROCESSED
    _goal, result = _compute(node)
    assert result.next_view.success
    state_before = backend.state
    snapshots_before = backend.snapshot_calls

    success = node._on_evaluate_candidates(
        _candidate_request(), EvaluateViewCandidates.Response()
    )
    assert success.success
    assert success.current_gain == pytest.approx(1.25)
    assert success.candidate_gains == pytest.approx([2.0])
    assert backend.evaluate_calls == 1
    assert backend.state == state_before
    assert backend.snapshot_calls == snapshots_before


def test_candidate_scoring_rejects_bad_pose_without_backend_call(wrapper) -> None:
    node, backend, _publisher = wrapper
    _configure(node)
    node._on_observation(_observation())
    _goal, result = _compute(node)
    assert result.next_view.success

    mutations = (
        lambda request: setattr(request, "scene_id", "wrong-scene"),
        lambda request: setattr(request.candidate_poses[0].header, "frame_id", "wrong"),
        lambda request: setattr(request.candidate_poses[0].header, "stamp", _stamp(1)),
        lambda request: setattr(request.candidate_poses[0].pose.orientation, "w", 0.5),
        lambda request: setattr(request.candidate_poses[0].pose.position, "x", 2.0),
    )
    for mutate in mutations:
        request = _candidate_request()
        mutate(request)
        response = node._on_evaluate_candidates(
            request, EvaluateViewCandidates.Response()
        )
        assert not response.success
    assert backend.evaluate_calls == 0


def test_enabled_map_snapshot_is_written_once_per_unique_update(wrapper, tmp_path) -> None:
    node, backend, _publisher = wrapper
    node.set_parameters(
        [Parameter("map_snapshot_directory", Parameter.Type.STRING, str(tmp_path))]
    )
    _configure(node)
    node._on_observation(_observation("map/step:1"))

    _first_goal, first = _compute(node, "map/step:1")
    assert first.next_view.success
    paths = list(tmp_path.rglob("*.npz"))
    assert len(paths) == 1
    assert "map_step_001" in paths[0].name

    _second_goal, second = _compute(node, "map/step:1")
    assert second.next_view.success
    assert backend.update_calls == 1
    assert list(tmp_path.rglob("*.npz")) == paths


def test_result_packaging_failure_rolls_back_and_does_not_mark_id_processed(wrapper) -> None:
    node, backend, _publisher = wrapper
    _configure(node)
    node._on_observation(_observation())
    backend.bad_result = True

    first_goal, first = _compute(node)
    assert first_goal.terminal_state == "aborted"
    assert first.next_view.code == NextView.INTERNAL_ERROR
    assert "non-finite" in first.next_view.reason
    assert backend.update_calls == 1
    assert backend.restore_calls == 1
    assert backend.state == 0
    assert not node._results

    _second_goal, second = _compute(node)
    assert second.next_view.code == NextView.INTERNAL_ERROR
    assert backend.update_calls == 2
    assert backend.restore_calls == 2
    assert backend.state == 0


def test_backend_failure_is_classified_and_transactionally_restored(wrapper) -> None:
    node, backend, _publisher = wrapper
    _configure(node)
    node._on_observation(_observation())
    backend.update_error = NBVInputError("valid depth rays do not intersect map")
    goal, result = _compute(node)
    assert goal.terminal_state == "aborted"
    assert result.next_view.code == NextView.MAP_UPDATE_FAILED
    assert "map update failed" in result.next_view.reason
    assert backend.update_calls == 1
    assert backend.restore_calls == 1
    assert backend.state == 0
    assert not node._results


def test_scene_reference_errors_do_not_start_a_transaction(wrapper) -> None:
    node, backend, _publisher = wrapper
    missing_goal, missing = _compute(node)
    assert missing_goal.terminal_state == "aborted"
    assert missing.next_view.code == NextView.NOT_CONFIGURED

    _configure(node)
    mismatch_goal = _GoalHandle("other-scene", "obs-1")
    mismatch = node._execute_compute(mismatch_goal)
    assert mismatch.next_view.code == NextView.SCENE_MISMATCH
    not_found_goal, not_found = _compute(node, "missing")
    assert not_found_goal.terminal_state == "aborted"
    assert not_found.next_view.code == NextView.OBSERVATION_NOT_FOUND
    assert backend.snapshot_calls == backend.update_calls == 0
