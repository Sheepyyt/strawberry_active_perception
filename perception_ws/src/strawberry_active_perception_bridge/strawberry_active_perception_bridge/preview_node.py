"""ROS 2 read-only bridge from Gradient-NBV output to Placo SolveIK.

The node deliberately owns no action client and no joint/CAN publisher.  It
converts each successful camera target to a link7 target, calls only the
existing SolveIK service, and step-halves an unreachable target three times.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import threading
from typing import Optional

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from strawberry_nero_interfaces.srv import SolveIK
from strawberry_perception_interfaces.msg import NextView, Observation

from .calibration_report import load_verified_handeye_report
from .transforms import (
    matrix_to_pose_components,
    pose_components_to_matrix,
    preview_camera_targets,
)


def _pose_message_to_matrix(message: PoseStamped) -> np.ndarray:
    position = message.pose.position
    orientation = message.pose.orientation
    return pose_components_to_matrix(
        (position.x, position.y, position.z),
        (orientation.x, orientation.y, orientation.z, orientation.w),
    )


def _matrix_to_pose_message(
    transform: np.ndarray,
    frame_id: str,
    stamp,
) -> PoseStamped:
    position, quaternion = matrix_to_pose_components(transform)
    result = PoseStamped()
    result.header.frame_id = frame_id
    result.header.stamp = stamp
    result.pose.position.x = float(position[0])
    result.pose.position.y = float(position[1])
    result.pose.position.z = float(position[2])
    result.pose.orientation.x = float(quaternion[0])
    result.pose.orientation.y = float(quaternion[1])
    result.pose.orientation.z = float(quaternion[2])
    result.pose.orientation.w = float(quaternion[3])
    return result


class NBVIKPreviewNode(Node):
    """Perform idempotent, motion-free IK checks for successful NBV poses."""

    def __init__(self) -> None:
        super().__init__("nbv_ik_preview")
        self._callbacks = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._observations: OrderedDict[tuple[str, str], Observation] = OrderedDict()
        self._processed: set[tuple[str, str]] = set()
        self._active_key: Optional[tuple[str, str]] = None
        self._attempts: tuple[tuple[float, np.ndarray], ...] = ()
        self._attempt_index = 0
        self._attempt_records: list[dict[str, object]] = []
        self._active_future = None
        self._attempt_stamp = None
        self._timeout_timer = None
        self._motion_command_count = 0

        self.declare_parameter(
            "observation_topic", "/strawberry/perception/observation"
        )
        self.declare_parameter("next_view_topic", "/strawberry/nbv/next_view")
        self.declare_parameter("solve_ik_service", "/strawberry_nero/solve_ik")
        self.declare_parameter(
            "diagnostics_topic", "/strawberry/active_perception/ik_preview"
        )
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("expected_camera_frame", "camera_sim_optical_frame")
        self.declare_parameter("calibration_source", "simulation_fixture")
        self.declare_parameter("handeye_report_path", "")
        self.declare_parameter("handeye_report_sha256", "")
        self.declare_parameter("minimum_calibration_samples", 30)
        self.declare_parameter("motion_command_topic", "/control/move_j")
        self.declare_parameter("preview_alphas", [1.0, 0.5, 0.25, 0.125])
        self.declare_parameter("service_timeout_sec", 2.0)
        self.declare_parameter(
            "link7_camera_transform",
            [
                0.0,
                0.0,
                1.0,
                0.10,
                -1.0,
                0.0,
                0.0,
                0.00,
                0.0,
                -1.0,
                0.0,
                0.00,
                0.0,
                0.0,
                0.0,
                1.00,
            ],
        )

        self._base_frame = str(self.get_parameter("base_frame").value).strip()
        self._camera_frame = str(
            self.get_parameter("expected_camera_frame").value
        ).strip()
        self._calibration_source = str(
            self.get_parameter("calibration_source").value
        ).strip()
        self._alphas = tuple(
            float(value) for value in self.get_parameter("preview_alphas").value
        )
        self._service_timeout = float(
            self.get_parameter("service_timeout_sec").value
        )
        self._report = None
        if self._calibration_source == "validated_report":
            if self._base_frame != "base_link":
                raise ValueError(
                    "validated_report mode requires base_frame='base_link'"
                )
            if self._camera_frame != "camera_color_optical_frame":
                raise ValueError(
                    "validated_report mode requires expected_camera_frame="
                    "'camera_color_optical_frame'"
                )
            self._report = load_verified_handeye_report(
                str(self.get_parameter("handeye_report_path").value),
                str(self.get_parameter("handeye_report_sha256").value),
                int(self.get_parameter("minimum_calibration_samples").value),
            )
            self._mount = self._report.transform_link7_camera_optical.copy()
        elif self._calibration_source == "simulation_fixture":
            if self._camera_frame == "camera_color_optical_frame":
                raise ValueError(
                    "simulation_fixture mode may not claim the real camera frame"
                )
            mount_values = np.asarray(
                self.get_parameter("link7_camera_transform").value,
                dtype=float,
            )
            if mount_values.shape != (16,):
                raise ValueError(
                    "link7_camera_transform must contain 16 row-major values"
                )
            self._mount = mount_values.reshape(4, 4)
        else:
            raise ValueError(
                "calibration_source must be 'simulation_fixture' or "
                "'validated_report'"
            )
        # Let the shared pure validator reject reflections or malformed poses.
        preview_camera_targets(np.eye(4), np.eye(4), self._mount, (1.0,))
        if not self._base_frame or not self._camera_frame:
            raise ValueError("base_frame and expected_camera_frame must be non-empty")
        if self._service_timeout <= 0.0:
            raise ValueError("service_timeout_sec must be positive")

        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Observation,
            str(self.get_parameter("observation_topic").value),
            self._on_observation,
            state_qos,
            callback_group=self._callbacks,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=32,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._motion_command_topic = str(
            self.get_parameter("motion_command_topic").value
        ).strip()
        if not self._motion_command_topic:
            raise ValueError("motion_command_topic must be non-empty")
        self.create_subscription(
            JointState,
            self._motion_command_topic,
            self._on_motion_command,
            command_qos,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            NextView,
            str(self.get_parameter("next_view_topic").value),
            self._on_next_view,
            state_qos,
            callback_group=self._callbacks,
        )
        self._diagnostics = self.create_publisher(
            DiagnosticArray,
            str(self.get_parameter("diagnostics_topic").value),
            state_qos,
        )
        self._solve_client = self.create_client(
            SolveIK,
            str(self.get_parameter("solve_ik_service").value),
            callback_group=self._callbacks,
        )
        if self._report is None:
            calibration_summary = "the mount is a simulation fixture"
        else:
            calibration_summary = (
                f"hand-eye report {self._report.session_id} is bound to "
                f"SHA256 {self._report.sha256}"
            )
        self.get_logger().warning(
            "NBV IK preview is motion-free: only SolveIK is available; "
            + calibration_summary
        )

    def _on_motion_command(self, _message: JointState) -> None:
        """Count command traffic independently; never publish or forward it."""
        with self._lock:
            self._motion_command_count += 1
            count = self._motion_command_count
        self.get_logger().error(
            f"Observed {count} message(s) on {self._motion_command_topic}; "
            "read-only preview results are invalid"
        )

    def _on_observation(self, message: Observation) -> None:
        key = (message.scene_id, message.observation_id)
        if not all(key):
            self._publish_status(
                DiagnosticStatus.WARN,
                "observation_rejected",
                "Observation is missing scene_id or observation_id",
            )
            return
        with self._lock:
            self._observations[key] = message
            self._observations.move_to_end(key)
            while len(self._observations) > 32:
                self._observations.popitem(last=False)

    def _on_next_view(self, message: NextView) -> None:
        key = (message.scene_id, message.observation_id)
        if not message.success:
            self._publish_status(
                DiagnosticStatus.WARN,
                "nbv_rejected",
                message.reason or f"NBV failed with code {message.code}",
                key,
                {"nbv_code": message.code},
            )
            return
        with self._lock:
            if self._motion_command_count:
                self._publish_status(
                    DiagnosticStatus.ERROR,
                    "motion_activity_observed",
                    "A motion command was observed after the preview monitor started",
                    key,
                )
                return
            if key in self._processed:
                self.get_logger().info(
                    f"Ignoring already previewed NextView {key[0]}/{key[1]}"
                )
                return
            if self._active_key is not None:
                self._publish_status(
                    DiagnosticStatus.WARN,
                    "preview_busy",
                    "Another read-only IK preview is active",
                    key,
                )
                return
            observation = self._observations.get(key)
            if observation is None:
                self._publish_status(
                    DiagnosticStatus.ERROR,
                    "observation_not_found",
                    "No matching canonical Observation is cached",
                    key,
                )
                return
            try:
                self._validate_frames(observation, message)
                current = _pose_message_to_matrix(observation.camera_pose)
                target = _pose_message_to_matrix(message.pose)
                self._attempts = preview_camera_targets(
                    current,
                    target,
                    self._mount,
                    self._alphas,
                )
            except ValueError as error:
                self._publish_status(
                    DiagnosticStatus.ERROR,
                    "invalid_pose",
                    str(error),
                    key,
                )
                return
            self._active_key = key
            self._attempt_index = 0
            self._attempt_records = []
        self._request_current_attempt(message.pose.header.stamp)

    def _validate_frames(self, observation: Observation, result: NextView) -> None:
        if not observation.pose_valid:
            raise ValueError("Observation camera pose is explicitly invalid")
        if observation.header.frame_id != self._camera_frame:
            raise ValueError(
                "Observation optical frame mismatch: "
                f"expected {self._camera_frame}, got {observation.header.frame_id}"
            )
        if observation.camera_pose.header.frame_id != self._base_frame:
            raise ValueError(
                "Observation world frame mismatch: "
                f"expected {self._base_frame}, got "
                f"{observation.camera_pose.header.frame_id}"
            )
        if result.pose.header.frame_id != self._base_frame:
            raise ValueError(
                "NextView world frame mismatch: "
                f"expected {self._base_frame}, got {result.pose.header.frame_id}"
            )
        if (
            self._calibration_source == "validated_report"
            and observation.source_type == Observation.SOURCE_SYNTHETIC
        ):
            raise ValueError(
                "validated_report mode rejects synthetic Observations; "
                "use the explicitly offline real-preview fixture instead"
            )

    def _request_current_attempt(self, stamp) -> None:
        with self._lock:
            if self._active_key is None or self._attempt_index >= len(self._attempts):
                return
            alpha, target = self._attempts[self._attempt_index]
            key = self._active_key
        if not self._solve_client.wait_for_service(
            timeout_sec=self._service_timeout
        ):
            self._finish_failure(
                "solve_ik_unavailable",
                "Placo SolveIK service is not available",
            )
            return
        request = SolveIK.Request()
        request.target_pose = _matrix_to_pose_message(
            target,
            self._base_frame,
            stamp,
        )
        request.controlled_frame = "link7"
        future = self._solve_client.call_async(request)
        with self._lock:
            self._active_future = future
            self._attempt_stamp = stamp
            timer = self.create_timer(
                self._service_timeout,
                self._on_attempt_timeout,
                callback_group=self._callbacks,
            )
            self._timeout_timer = timer
        future.add_done_callback(self._on_solve_done)
        self.get_logger().info(
            f"SolveIK preview {key[0]}/{key[1]} alpha={alpha:.3f}"
        )

    def _cancel_timeout_timer(self) -> None:
        timer = self._timeout_timer
        self._timeout_timer = None
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)

    def _on_attempt_timeout(self) -> None:
        with self._lock:
            future = self._active_future
            if future is None or future.done():
                return
            future.cancel()
        self._cancel_timeout_timer()
        self._finish_failure(
            "solve_ik_timeout",
            f"SolveIK did not respond within {self._service_timeout:.3f} s",
        )

    def _on_solve_done(self, future) -> None:
        self._cancel_timeout_timer()
        with self._lock:
            if future is not self._active_future or self._active_key is None:
                return
            self._active_future = None
            alpha = self._attempts[self._attempt_index][0]
            key = self._active_key
        if future.cancelled():
            return
        try:
            result = future.result().result
        except Exception as error:  # rclpy transports implementation exceptions.
            self._finish_failure("solve_ik_error", str(error))
            return
        # The request pose is the only meaningful stamp/frame for the next
        # retry.  Failed SolveIK results are allowed to leave solved_pose at its
        # default value, so never feed that default stamp back into the bridge.
        retry_stamp = self._attempt_stamp
        record = {
            "alpha": alpha,
            "success": bool(result.success),
            "code": int(result.code),
            "reason": result.reason,
            "position_error_m": float(result.position_error_m),
            "orientation_error_rad": float(result.orientation_error_rad),
            "solve_time_ms": float(result.solve_time_ms),
            "max_joint_delta_rad": float(result.max_joint_delta_rad),
        }
        with self._lock:
            self._attempt_records.append(record)
            motion_command_count = self._motion_command_count
        if motion_command_count:
            self._finish_failure(
                "motion_activity_observed",
                "Motion command traffic was observed during SolveIK preview",
            )
            return
        if result.success:
            fields = dict(record)
            fields["outcome_code"] = "SUCCESS"
            fields["attempt_count"] = len(self._attempt_records)
            # Preserve every rejected and accepted SolveIK result in the final
            # diagnostic.  This is the durable preview record; no joint target
            # is ever forwarded to an execution surface.
            fields["attempts"] = list(self._attempt_records)
            fields["solution_joint_positions"] = list(
                result.solution_joint_state.position
            )
            self._publish_status(
                DiagnosticStatus.OK,
                "ik_preview_success",
                "A motion-free Placo SolveIK candidate was accepted",
                key,
                fields,
            )
            self._finish_active(key)
            return
        with self._lock:
            self._attempt_index += 1
            more_attempts = self._attempt_index < len(self._attempts)
        if more_attempts:
            self._request_current_attempt(retry_stamp)
        else:
            self._finish_failure(
                "no_reachable_view",
                "Full and step-halved camera candidates all failed SolveIK",
            )

    def _finish_failure(self, name: str, reason: str) -> None:
        with self._lock:
            key = self._active_key
            attempts = list(self._attempt_records)
        fields = {
            "outcome_code": name.upper(),
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        self._publish_status(
            DiagnosticStatus.ERROR,
            name,
            reason,
            key,
            fields,
        )
        if key is not None:
            self._finish_active(key)

    def _finish_active(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._processed.add(key)
            self._active_key = None
            self._attempts = ()
            self._attempt_index = 0
            self._attempt_records = []
            self._active_future = None
            self._attempt_stamp = None

    def _publish_status(
        self,
        level: int,
        name: str,
        message: str,
        key: Optional[tuple[str, str]] = None,
        values: Optional[dict[str, object]] = None,
    ) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.level = level
        status.name = f"strawberry_active_perception_bridge/{name}"
        status.hardware_id = (
            "validated_handeye_no_motion"
            if self._report is not None
            else "simulation_only_no_motion"
        )
        status.message = message
        fields: dict[str, object] = {
            "controlled_frame": "link7",
            "camera_frame": self._camera_frame,
            "base_frame": self._base_frame,
            "calibration_source": self._calibration_source,
            "move_action_client_created": False,
            "can_publisher_created": False,
        }
        with self._lock:
            motion_command_count = self._motion_command_count
        fields["motion_command_topic"] = self._motion_command_topic
        fields["motion_command_count"] = motion_command_count
        fields["motion_command_count_observed"] = motion_command_count
        fields["motion_command_publisher_count"] = self.count_publishers(
            self._motion_command_topic
        )
        if self._report is not None:
            fields.update(
                {
                    "handeye_report_path": str(self._report.path),
                    "handeye_report_sha256": self._report.sha256,
                    "handeye_report_version": self._report.report_version,
                    "handeye_session_id": self._report.session_id,
                    "handeye_sample_count": self._report.sample_count,
                    "T_link7_camera_optical": self._mount.tolist(),
                }
            )
        if key is not None:
            fields["scene_id"] = key[0]
            fields["observation_id"] = key[1]
        fields.update(values or {})
        status.values = [
            KeyValue(key=str(field), value=json.dumps(value, ensure_ascii=False))
            for field, value in fields.items()
        ]
        array.status = [status]
        self._diagnostics.publish(array)
        log = self.get_logger().info
        if level == DiagnosticStatus.WARN:
            log = self.get_logger().warning
        elif level == DiagnosticStatus.ERROR:
            log = self.get_logger().error
        log(f"{name}: {message}")


def main(args=None) -> None:
    """Run callbacks concurrently so service responses cannot starve topics."""
    rclpy.init(args=args)
    node = NBVIKPreviewNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # A launch-level SIGINT may arrive again while entities are being
            # destroyed.  The process is already safe and owns no commands.
            pass


if __name__ == "__main__":
    main()
