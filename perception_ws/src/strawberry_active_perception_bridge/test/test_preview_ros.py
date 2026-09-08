"""In-process ROS integration test for the read-only SolveIK bridge."""

from __future__ import annotations

import json
import threading
import time

import rclpy
import numpy as np
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from strawberry_active_perception_bridge.preview_node import NBVIKPreviewNode
from strawberry_active_perception_bridge.preview_node import _pose_message_to_matrix
from strawberry_nero_interfaces.msg import IKResult
from strawberry_nero_interfaces.srv import SolveIK
from strawberry_perception_interfaces.msg import NextView, Observation


def _state_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def test_failed_full_step_is_halved_without_motion_command() -> None:
    rclpy.init()
    bridge = NBVIKPreviewNode()
    fixture = Node("preview_bridge_test_fixture")
    requests = []
    diagnostic = []
    complete = threading.Event()

    def solve(request, response):
        requests.append(request)
        response.result.success = len(requests) >= 2
        response.result.code = (
            IKResult.SUCCESS if response.result.success else IKResult.UNREACHABLE
        )
        response.result.reason = "accepted" if response.result.success else "retry"
        response.result.solved_pose = request.target_pose
        response.result.solve_time_ms = 0.5
        return response

    fixture.create_service(SolveIK, "/strawberry_nero/solve_ik", solve)
    observation_publisher = fixture.create_publisher(
        Observation, "/strawberry/perception/observation", _state_qos()
    )
    view_publisher = fixture.create_publisher(
        NextView, "/strawberry/nbv/next_view", _state_qos()
    )

    def on_diagnostic(message: DiagnosticArray) -> None:
        diagnostic.extend(message.status)
        if message.status and message.status[0].name.endswith("ik_preview_success"):
            complete.set()

    fixture.create_subscription(
        DiagnosticArray,
        "/strawberry/active_perception/ik_preview",
        on_diagnostic,
        _state_qos(),
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge)
    executor.add_node(fixture)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if bridge._solve_client.service_is_ready():
                break
            time.sleep(0.05)
        observation = Observation()
        observation.scene_id = "fixture_scene"
        observation.observation_id = "view_000"
        observation.header.frame_id = "camera_sim_optical_frame"
        observation.camera_pose.header.frame_id = "base_link"
        observation.camera_pose.pose.orientation.w = 1.0
        observation.pose_valid = True
        observation_publisher.publish(observation)
        time.sleep(0.25)

        next_view = NextView()
        next_view.scene_id = observation.scene_id
        next_view.observation_id = observation.observation_id
        next_view.success = True
        next_view.code = NextView.SUCCESS
        next_view.pose.header.frame_id = "base_link"
        next_view.pose.pose.position.z = 0.08
        next_view.pose.pose.orientation.w = 1.0
        view_publisher.publish(next_view)

        assert complete.wait(timeout=8.0)
        assert len(requests) == 2
        assert all(request.controlled_frame == "link7" for request in requests)
        # Reapply the fixture mount: the second camera displacement is exactly
        # half of the first, even though link7 itself is offset and rotated.
        first_camera = _pose_message_to_matrix(requests[0].target_pose) @ bridge._mount
        second_camera = _pose_message_to_matrix(requests[1].target_pose) @ bridge._mount
        np.testing.assert_allclose(first_camera[:3, 3], (0.0, 0.0, 0.08))
        np.testing.assert_allclose(second_camera[:3, 3], (0.0, 0.0, 0.04))
        status = next(
            item for item in diagnostic if item.name.endswith("ik_preview_success")
        )
        values = {entry.key: entry.value for entry in status.values}
        assert values["outcome_code"] == '"SUCCESS"'
        assert values["attempt_count"] == "2"
        attempts = json.loads(values["attempts"])
        assert len(attempts) == 2
        assert attempts[0]["success"] is False
        assert attempts[1]["success"] is True
        assert values["motion_command_count"] == "0"
        assert values["move_action_client_created"] == "false"
        assert values["can_publisher_created"] == "false"
    finally:
        executor.shutdown(timeout_sec=2.0)
        bridge.destroy_node()
        fixture.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)


def test_all_unreachable_candidates_end_with_no_reachable_view() -> None:
    """Four rejected SolveIK calls must terminate safely without execution."""
    rclpy.init()
    bridge = NBVIKPreviewNode()
    fixture = Node("preview_bridge_all_unreachable_fixture")
    requests = []
    diagnostics = []
    complete = threading.Event()

    def solve(request, response):
        requests.append(request)
        response.result.success = False
        response.result.code = IKResult.UNREACHABLE
        response.result.reason = "fixture unreachable"
        return response

    fixture.create_service(SolveIK, "/strawberry_nero/solve_ik", solve)
    observation_publisher = fixture.create_publisher(
        Observation, "/strawberry/perception/observation", _state_qos()
    )
    view_publisher = fixture.create_publisher(
        NextView, "/strawberry/nbv/next_view", _state_qos()
    )

    def on_diagnostic(message: DiagnosticArray) -> None:
        diagnostics.extend(message.status)
        if message.status and message.status[0].name.endswith("no_reachable_view"):
            complete.set()

    fixture.create_subscription(
        DiagnosticArray,
        "/strawberry/active_perception/ik_preview",
        on_diagnostic,
        _state_qos(),
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge)
    executor.add_node(fixture)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not bridge._solve_client.service_is_ready():
            time.sleep(0.05)
        observation = Observation()
        observation.scene_id = "unreachable_scene"
        observation.observation_id = "unreachable_view"
        observation.header.frame_id = "camera_sim_optical_frame"
        observation.camera_pose.header.frame_id = "base_link"
        observation.camera_pose.pose.orientation.w = 1.0
        observation.pose_valid = True
        observation_publisher.publish(observation)
        time.sleep(0.25)

        next_view = NextView()
        next_view.scene_id = observation.scene_id
        next_view.observation_id = observation.observation_id
        next_view.success = True
        next_view.code = NextView.SUCCESS
        next_view.pose.header.frame_id = "base_link"
        next_view.pose.pose.position.x = 0.1
        next_view.pose.pose.orientation.w = 1.0
        view_publisher.publish(next_view)

        assert complete.wait(timeout=8.0)
        assert len(requests) == 4
        assert all(request.controlled_frame == "link7" for request in requests)
        status = next(
            item for item in diagnostics if item.name.endswith("no_reachable_view")
        )
        values = {entry.key: entry.value for entry in status.values}
        assert status.level == status.ERROR
        assert values["outcome_code"] == '"NO_REACHABLE_VIEW"'
        assert values["attempt_count"] == "4"
        assert values["motion_command_count"] == "0"
        assert values["move_action_client_created"] == "false"
        assert values["can_publisher_created"] == "false"
    finally:
        executor.shutdown(timeout_sec=2.0)
        bridge.destroy_node()
        fixture.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)


def test_observed_motion_command_fails_closed_before_solve_ik() -> None:
    """A real command-topic message must invalidate the preview, not be ignored."""
    rclpy.init()
    bridge = NBVIKPreviewNode()
    fixture = Node("preview_bridge_motion_monitor_fixture")
    solve_requests = []
    diagnostics = []
    complete = threading.Event()

    def solve(request, response):
        solve_requests.append(request)
        response.result.success = True
        return response

    fixture.create_service(SolveIK, "/strawberry_nero/solve_ik", solve)
    observation_publisher = fixture.create_publisher(
        Observation, "/strawberry/perception/observation", _state_qos()
    )
    view_publisher = fixture.create_publisher(
        NextView, "/strawberry/nbv/next_view", _state_qos()
    )
    command_publisher = fixture.create_publisher(
        JointState,
        "/control/move_j",
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
    )

    def on_diagnostic(message: DiagnosticArray) -> None:
        diagnostics.extend(message.status)
        if message.status and message.status[0].name.endswith(
            "motion_activity_observed"
        ):
            complete.set()

    fixture.create_subscription(
        DiagnosticArray,
        "/strawberry/active_perception/ik_preview",
        on_diagnostic,
        _state_qos(),
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge)
    executor.add_node(fixture)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (
            command_publisher.get_subscription_count() == 0
        ):
            time.sleep(0.05)
        assert command_publisher.get_subscription_count() >= 1
        command_publisher.publish(JointState())
        while time.monotonic() < deadline and bridge._motion_command_count == 0:
            time.sleep(0.05)
        assert bridge._motion_command_count == 1

        observation = Observation()
        observation.scene_id = "motion_seen_scene"
        observation.observation_id = "motion_seen_view"
        observation.header.frame_id = "camera_sim_optical_frame"
        observation.camera_pose.header.frame_id = "base_link"
        observation.camera_pose.pose.orientation.w = 1.0
        observation.pose_valid = True
        observation_publisher.publish(observation)
        time.sleep(0.25)

        next_view = NextView()
        next_view.scene_id = observation.scene_id
        next_view.observation_id = observation.observation_id
        next_view.success = True
        next_view.code = NextView.SUCCESS
        next_view.pose.header.frame_id = "base_link"
        next_view.pose.pose.orientation.w = 1.0
        view_publisher.publish(next_view)

        assert complete.wait(timeout=5.0)
        assert solve_requests == []
        status = next(
            item
            for item in diagnostics
            if item.name.endswith("motion_activity_observed")
        )
        values = {entry.key: entry.value for entry in status.values}
        assert values["motion_command_count"] == "1"
        assert values["motion_command_count_observed"] == "1"
    finally:
        executor.shutdown(timeout_sec=2.0)
        bridge.destroy_node()
        fixture.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)
