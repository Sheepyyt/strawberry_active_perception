"""
ROS 2 integration for Placo IK and NERO joint-position execution.

This node deliberately has exactly one robot command output: ``control/move_j``.
Cartesian commands, ``move_js`` and MIT control are never used here.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Pose, PoseStamped, Transform
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformException, TransformListener

from agx_arm_msgs.msg import AgxArmStatus
from strawberry_nero_interfaces.action import MoveToPose, RecoverToSafe
from strawberry_nero_interfaces.msg import IKResult as IKResultMsg
from strawberry_nero_interfaces.srv import SolveIK

from .ik_core import IKConfig, IKErrorCode, IKResult, PlacoIKSolver
from .models import (
    JointLimitRecoveryConfig,
    JointLimitRecoveryPlan,
    NERO_JOINT_NAMES,
    READY_JOINT_POSITIONS,
)
from .ros_utils import (
    duration_to_seconds,
    joint_state_from_arrays,
    matrix_to_pose_stamped,
    ordered_joint_arrays,
    pose_error,
    pose_to_matrix,
    seconds_to_duration,
)
from .trajectory import (
    JointLimitRecoveryMonitor,
    TrajectoryConfig,
    TrajectoryGenerator,
    TrajectoryResult,
)


@dataclass(frozen=True)
class FeedbackSnapshot:
    """A coherent copy of the latest measured state."""

    positions: np.ndarray
    velocities: np.ndarray
    received_monotonic: float
    stamp: object
    velocities_valid: bool = True


@dataclass(frozen=True)
class TargetContext:
    """Transforms needed to solve for link7 and report the controlled frame."""

    target_tip_in_base: np.ndarray
    target_controlled_in_base: np.ndarray
    base_from_reference: np.ndarray
    tip_from_controlled: np.ndarray
    reference_frame: str
    controlled_frame: str


class MetricsLogger:
    """Append action metrics to CSV and maintain a compact JSON summary."""

    _FIELDS = (
        "run_id",
        "timestamp",
        "success",
        "code",
        "reason",
        "solve_time_ms",
        "motion_time_s",
        "position_error_m",
        "orientation_error_rad",
        "sigma_min",
        "condition_number",
        "max_joint_delta_rad",
    )

    def __init__(self, enabled: bool, output_dir: str, logger) -> None:
        self._enabled = enabled
        self._logger = logger
        self._lock = threading.Lock()
        self._directory = Path(os.path.expanduser(output_dir))
        self._csv_path = self._directory / "motion_metrics.csv"
        self._json_path = self._directory / "motion_summary.json"
        if self._enabled:
            try:
                self._directory.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                self._enabled = False
                self._logger.error(f"Cannot create metrics directory: {error}")

    def record(self, row: dict) -> None:
        """Persist one action outcome without ever failing the control action."""
        if not self._enabled:
            return
        complete = {field: row.get(field, "") for field in self._FIELDS}
        with self._lock:
            try:
                new_file = not self._csv_path.exists()
                with self._csv_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=self._FIELDS)
                    if new_file:
                        writer.writeheader()
                    writer.writerow(complete)
                self._write_summary()
            except OSError as error:
                self._logger.error(f"Cannot write metrics: {error}")

    def _write_summary(self) -> None:
        rows = []
        with self._csv_path.open("r", newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
        successes = sum(value.get("success") == "True" for value in rows)
        solve_times = sorted(
            float(value["solve_time_ms"])
            for value in rows
            if value.get("solve_time_ms") not in (None, "")
        )
        summary = {
            "samples": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else 0.0,
            "ik_time_p95_ms": _percentile(solve_times, 95.0),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        temporary = self._json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._json_path)


class NeroExecutor:
    """The only component allowed to send joint commands to the NERO driver."""

    def __init__(
        self,
        node: "NeroControlNode",
        command_publisher,
        emergency_client,
        joint_names: Sequence[str],
        simulation_mode: bool,
    ) -> None:
        self._node = node
        self._publisher = command_publisher
        self._emergency_client = emergency_client
        self._joint_names = tuple(joint_names)
        self._simulation_mode = simulation_mode
        self._last_command: Optional[np.ndarray] = None
        self._command_count = 0

    @property
    def last_command(self) -> Optional[np.ndarray]:
        return None if self._last_command is None else self._last_command.copy()

    @property
    def command_count(self) -> int:
        """Return the number of commands successfully handed to the backend."""
        return self._command_count

    def send(self, positions: Sequence[float], velocities=None) -> None:
        """Send one complete and finite joint1..joint7 command."""
        values = np.asarray(positions, dtype=float)
        if values.shape != (len(self._joint_names),) or not np.all(np.isfinite(values)):
            raise ValueError("refusing to send an incomplete or non-finite command")
        if self._simulation_mode:
            velocity_values = (
                np.zeros_like(values)
                if velocities is None
                else np.asarray(velocities, dtype=float)
            )
            self._node.set_simulated_state(values, velocity_values)
        else:
            message = joint_state_from_arrays(
                self._joint_names,
                values,
                self._node.get_clock().now().to_msg(),
            )
            self._publisher.publish(message)
        self._last_command = values.copy()
        self._command_count += 1

    def hold(self, measured_positions: Sequence[float], reason: str) -> None:
        """Request a soft position hold; this is not a physical emergency stop."""
        try:
            self.send(measured_positions, np.zeros(len(self._joint_names)))
        except ValueError as error:
            self._node.get_logger().error(f"Cannot send hold command: {error}")
            return
        self._node.get_logger().warning(f"Soft hold requested: {reason}")
        if self._simulation_mode:
            return
        if self._emergency_client.service_is_ready():
            self._emergency_client.call_async(Empty.Request())
        else:
            self._node.get_logger().warning(
                "Driver emergency_stop service is unavailable; move_j hold was sent"
            )

    def hold_if_commanded_since(
        self,
        measured_positions: Sequence[float],
        reason: str,
        command_count_before: int,
    ) -> bool:
        """Hold only if this action already emitted a trajectory command."""
        if self._command_count <= command_count_before:
            self._node.get_logger().warning(
                "Skipping soft hold because this action sent no motion command"
            )
            return False
        self.hold(measured_positions, reason)
        return True


class NeroControlNode(Node):
    """Placo IK, trajectory generation and guarded NERO execution."""

    def __init__(self) -> None:
        super().__init__("nero_control")
        self._callbacks = ReentrantCallbackGroup()
        self._state_lock = threading.RLock()
        self._solver_lock = threading.Lock()
        self._goal_lock = threading.Lock()
        self._goal_reserved = False
        self._first_motion_test_consumed = False
        self._feedback: Optional[FeedbackSnapshot] = None
        self._stationary_since: Optional[float] = None
        self._arm_status: Optional[AgxArmStatus] = None
        self._arm_status_monotonic = 0.0

        self._declare_parameters()
        self._load_parameters()
        self._validate_configuration()

        urdf_path = self._resolve_urdf_path(self._urdf_path)
        ik_config = IKConfig(
            position_weight=self._position_weight,
            orientation_weight=self._orientation_weight,
            posture_weight=self._posture_weight,
            regularization=self._regularization_weight,
            max_iterations=self._max_iterations,
            timeout_s=self._solve_timeout,
            position_tolerance_m=self._ik_position_tolerance,
            position_convergence_target_m=(
                self._ik_position_convergence_target
            ),
            orientation_tolerance_rad=self._ik_orientation_tolerance,
            position_deadband_m=self._position_deadband,
            orientation_deadband_rad=self._orientation_deadband,
            solver_dt_s=self._solver_dt,
            solver_velocity_limit_rad_s=min(self._trajectory_velocity_limits),
            max_joint_delta_rad=self._max_joint_delta,
            singular_sigma_min=self._singularity_sigma_min,
            singular_condition_max=self._singularity_condition_max,
        )
        sdk_limits = np.column_stack(
            (self._configured_lower_limits, self._configured_upper_limits)
        )
        self._solver = PlacoIKSolver(urdf_path, ik_config, sdk_joint_limits=sdk_limits)
        trajectory_config = TrajectoryConfig(
            frequency_hz=self._trajectory_rate,
            max_velocity_rad_s=min(self._trajectory_velocity_limits),
            max_acceleration_rad_s2=min(self._trajectory_acceleration_limits),
        )
        self._trajectory_generator = TrajectoryGenerator(
            trajectory_config,
            joint_limits=self._solver.safe_joint_limits,
        )
        safe_limits = self._solver.safe_joint_limits
        self._raw_joint_limits = np.column_stack((
            safe_limits[:, 0] - ik_config.joint_margin_rad,
            safe_limits[:, 1] + ik_config.joint_margin_rad,
        ))
        self._recovery_config = JointLimitRecoveryConfig(
            raw_interior_margin_rad=self._recovery_raw_margin,
            safe_interior_margin_rad=self._recovery_safe_margin,
            max_raw_start_violation_rad=self._recovery_max_raw_violation,
            max_safe_start_violation_rad=self._recovery_max_safe_violation,
            max_ingress_delta_rad=self._recovery_max_ingress_delta,
            max_total_delta_rad=self._recovery_max_total_delta,
        )
        recovery_trajectory_config = TrajectoryConfig(
            frequency_hz=self._trajectory_rate,
            max_velocity_rad_s=self._recovery_velocity_limit,
            max_acceleration_rad_s2=self._recovery_acceleration_limit,
        )
        self._recovery_generator = TrajectoryGenerator(
            recovery_trajectory_config,
            joint_limits=safe_limits,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        trajectory_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._command_publisher = self.create_publisher(
            JointState, self._command_topic, 10
        )
        self._trajectory_publisher = self.create_publisher(
            JointTrajectory, self._planned_trajectory_topic, trajectory_qos
        )
        self._target_publisher = self.create_publisher(
            PoseStamped, self._target_pose_topic, trajectory_qos
        )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray, self._diagnostics_topic, 10
        )
        self._sim_feedback_publisher = self.create_publisher(
            JointState, self._joint_feedback_topic, 10
        )

        self.create_subscription(
            JointState,
            self._joint_feedback_topic,
            self._joint_feedback_callback,
            20,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            AgxArmStatus,
            self._arm_status_topic,
            self._arm_status_callback,
            10,
            callback_group=self._callbacks,
        )
        emergency_client = self.create_client(
            Empty, self._emergency_stop_service, callback_group=self._callbacks
        )
        self._executor_backend = NeroExecutor(
            self,
            self._command_publisher,
            emergency_client,
            self._joint_names,
            self._simulation_mode,
        )

        self.create_service(
            SolveIK,
            self._solve_service_name,
            self._solve_service_callback,
            callback_group=self._callbacks,
        )
        self.create_service(
            SetBool,
            self._execution_enable_service_name,
            self._execution_enable_callback,
            callback_group=self._callbacks,
        )
        self._action_server = ActionServer(
            self,
            MoveToPose,
            self._action_name,
            execute_callback=self._execute_action,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callbacks,
        )
        self._recovery_action_server = ActionServer(
            self,
            RecoverToSafe,
            self._recovery_action_name,
            execute_callback=self._execute_recovery_action,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callbacks,
        )

        self._metrics = MetricsLogger(
            self._metrics_enabled, self._metrics_output_dir, self.get_logger()
        )
        self.create_timer(0.5, self._publish_diagnostics, callback_group=self._callbacks)
        if self._simulation_mode:
            self.set_simulated_state(
                np.asarray(self._ready_positions), np.zeros(len(self._joint_names))
            )
            self.create_timer(
                1.0 / self._trajectory_rate,
                self._publish_simulated_feedback,
                callback_group=self._callbacks,
            )

        ready_metrics = self._solver.configuration_metrics(self._ready_positions)
        self.get_logger().info(
            "NERO Placo controller ready: mode=%s, execution=%s, "
            "ready sigma_min=%.4f, condition=%.2f"
            % (
                "sim" if self._simulation_mode else "real",
                self._execution_enabled,
                ready_metrics.sigma_min,
                ready_metrics.condition_number,
            )
        )
        if not self._simulation_mode:
            self.get_logger().warning(
                "No environment collision checking is active. The software hold is "
                "not a physical emergency stop. Keep the hardware E-stop reachable."
            )

    def _declare_parameters(self) -> None:
        defaults = {
            "urdf_path": "",
            "base_frame": "base_link",
            "model_tip_frame": "link7",
            "default_controlled_frame": "link7",
            "camera_transform_valid": False,
            "joint_names": list(NERO_JOINT_NAMES),
            "ready_joint_positions": list(READY_JOINT_POSITIONS),
            "joint_lower_limits": [-2.70526, -1.74, -2.75, -1.01, -2.75, -0.73, -1.5707963],
            "joint_upper_limits": [2.70526, 1.74, 2.75, 2.14, 2.75, 0.95, 1.5707963],
            "position_weight": 1.0,
            "orientation_weight": 0.3,
            "posture_weight": 0.001,
            "regularization_weight": 1.0e-6,
            "solver_dt": 0.02,
            "max_iterations": 200,
            "solve_timeout_sec": 0.020,
            "ik_position_tolerance_m": 0.002,
            "ik_position_convergence_target_m": 0.001,
            "ik_orientation_tolerance_rad": math.radians(2.0),
            "singularity_sigma_min": 0.03,
            "singularity_condition_max": 100.0,
            "position_deadband_m": 0.001,
            "orientation_deadband_rad": math.radians(0.5),
            "max_joint_delta_rad": 0.35,
            "precision_max_joint_delta_rad": 0.12,
            "trajectory_velocity_limits": [0.30] * 7,
            "trajectory_acceleration_limits": [0.50] * 7,
            "trajectory_rate_hz": 50.0,
            "allow_limit_recovery_execution": False,
            "verified_driver_speed_percent": 0,
            "recovery_raw_interior_margin_rad": 0.005,
            "recovery_safe_interior_margin_rad": 0.010,
            "recovery_max_raw_start_violation_rad": 0.060,
            "recovery_max_safe_start_violation_rad": 0.120,
            "recovery_max_ingress_delta_rad": 0.080,
            "recovery_max_total_delta_rad": 0.120,
            "recovery_velocity_limit_rad_s": 0.050,
            "recovery_acceleration_limit_rad_s2": 0.100,
            "recovery_stationary_velocity_rad_s": 0.020,
            "recovery_stationary_duration_sec": 0.50,
            "recovery_sigma_min": 0.10,
            "recovery_condition_max": 20.0,
            "recovery_phase_a_pause_error_rad": 0.030,
            "recovery_phase_a_abort_error_rad": 0.060,
            "recovery_tracking_pause_error_rad": 0.030,
            "recovery_tracking_abort_error_rad": 0.060,
            "recovery_tracking_abort_duration_sec": 0.30,
            "recovery_final_joint_tolerance_rad": 0.003,
            "recovery_final_safe_margin_rad": 0.005,
            "recovery_fixed_joint_tolerance_rad": 0.003,
            "recovery_progress_tolerance_rad": 0.001,
            "recovery_plan_start_tolerance_rad": 0.001,
            "recovery_timeout_max_sec": 15.0,
            "tracking_pause_error_rad": 0.08,
            "tracking_abort_error_rad": 0.15,
            "tracking_abort_duration_sec": 0.5,
            "feedback_timeout_sec": 0.2,
            "settle_joint_tolerance_rad": 0.02,
            "settle_velocity_tolerance_rad_s": 0.02,
            "settle_duration_sec": 0.3,
            "final_position_tolerance_m": 0.010,
            "final_orientation_tolerance_rad": math.radians(5.0),
            "default_action_timeout_sec": 30.0,
            "first_motion_test_mode": False,
            "precision_test_mode": False,
            "simulation_mode": True,
            "execution_enabled_on_start": True,
            "require_arm_status": False,
            "tf_timeout_sec": 0.2,
            "joint_feedback_topic": "feedback/joint_states",
            "arm_status_topic": "feedback/arm_status",
            "command_topic": "control/move_j",
            "planned_trajectory_topic": "planned_joint_trajectory",
            "diagnostics_topic": "diagnostics",
            "target_pose_topic": "target_pose",
            "action_name": "move_to_pose",
            "recovery_action_name": "recover_to_safe",
            "solve_service_name": "solve_ik",
            "execution_enable_service_name": "enable_execution",
            "emergency_stop_service": "emergency_stop",
            "metrics_log_enabled": True,
            "metrics_output_dir": "~/.ros/strawberry_nero_metrics",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        def value(name):
            return self.get_parameter(name).value

        self._urdf_path = str(value("urdf_path"))
        self._base_frame = str(value("base_frame"))
        self._model_tip_frame = str(value("model_tip_frame"))
        self._default_controlled_frame = str(value("default_controlled_frame"))
        self._camera_transform_valid = bool(value("camera_transform_valid"))
        self._joint_names = tuple(str(item) for item in value("joint_names"))
        self._ready_positions = tuple(float(item) for item in value("ready_joint_positions"))
        self._configured_lower_limits = np.asarray(value("joint_lower_limits"), dtype=float)
        self._configured_upper_limits = np.asarray(value("joint_upper_limits"), dtype=float)
        self._position_weight = float(value("position_weight"))
        self._orientation_weight = float(value("orientation_weight"))
        self._posture_weight = float(value("posture_weight"))
        self._regularization_weight = float(value("regularization_weight"))
        self._solver_dt = float(value("solver_dt"))
        self._max_iterations = int(value("max_iterations"))
        self._solve_timeout = float(value("solve_timeout_sec"))
        self._ik_position_tolerance = float(value("ik_position_tolerance_m"))
        self._ik_position_convergence_target = float(
            value("ik_position_convergence_target_m")
        )
        self._ik_orientation_tolerance = float(value("ik_orientation_tolerance_rad"))
        self._singularity_sigma_min = float(value("singularity_sigma_min"))
        self._singularity_condition_max = float(value("singularity_condition_max"))
        self._position_deadband = float(value("position_deadband_m"))
        self._orientation_deadband = float(value("orientation_deadband_rad"))
        self._max_joint_delta = float(value("max_joint_delta_rad"))
        self._precision_max_joint_delta = float(
            value("precision_max_joint_delta_rad")
        )
        self._trajectory_velocity_limits = tuple(
            float(item) for item in value("trajectory_velocity_limits")
        )
        self._trajectory_acceleration_limits = tuple(
            float(item) for item in value("trajectory_acceleration_limits")
        )
        self._trajectory_rate = float(value("trajectory_rate_hz"))
        self._allow_recovery_execution = bool(
            value("allow_limit_recovery_execution")
        )
        self._verified_driver_speed_percent = int(
            value("verified_driver_speed_percent")
        )
        self._recovery_raw_margin = float(
            value("recovery_raw_interior_margin_rad")
        )
        self._recovery_safe_margin = float(
            value("recovery_safe_interior_margin_rad")
        )
        self._recovery_max_raw_violation = float(
            value("recovery_max_raw_start_violation_rad")
        )
        self._recovery_max_safe_violation = float(
            value("recovery_max_safe_start_violation_rad")
        )
        self._recovery_max_ingress_delta = float(
            value("recovery_max_ingress_delta_rad")
        )
        self._recovery_max_total_delta = float(
            value("recovery_max_total_delta_rad")
        )
        self._recovery_velocity_limit = float(
            value("recovery_velocity_limit_rad_s")
        )
        self._recovery_acceleration_limit = float(
            value("recovery_acceleration_limit_rad_s2")
        )
        self._recovery_stationary_velocity = float(
            value("recovery_stationary_velocity_rad_s")
        )
        self._recovery_stationary_duration = float(
            value("recovery_stationary_duration_sec")
        )
        self._recovery_sigma_min = float(value("recovery_sigma_min"))
        self._recovery_condition_max = float(value("recovery_condition_max"))
        self._recovery_phase_a_pause_error = float(
            value("recovery_phase_a_pause_error_rad")
        )
        self._recovery_phase_a_abort_error = float(
            value("recovery_phase_a_abort_error_rad")
        )
        self._recovery_tracking_pause_error = float(
            value("recovery_tracking_pause_error_rad")
        )
        self._recovery_tracking_abort_error = float(
            value("recovery_tracking_abort_error_rad")
        )
        self._recovery_tracking_abort_duration = float(
            value("recovery_tracking_abort_duration_sec")
        )
        self._recovery_final_joint_tolerance = float(
            value("recovery_final_joint_tolerance_rad")
        )
        self._recovery_final_safe_margin = float(
            value("recovery_final_safe_margin_rad")
        )
        self._recovery_fixed_joint_tolerance = float(
            value("recovery_fixed_joint_tolerance_rad")
        )
        self._recovery_progress_tolerance = float(
            value("recovery_progress_tolerance_rad")
        )
        self._recovery_plan_start_tolerance = float(
            value("recovery_plan_start_tolerance_rad")
        )
        self._recovery_timeout_max = float(
            value("recovery_timeout_max_sec")
        )
        self._tracking_pause_error = float(value("tracking_pause_error_rad"))
        self._tracking_abort_error = float(value("tracking_abort_error_rad"))
        self._tracking_abort_duration = float(value("tracking_abort_duration_sec"))
        self._feedback_timeout = float(value("feedback_timeout_sec"))
        self._settle_joint_tolerance = float(value("settle_joint_tolerance_rad"))
        self._settle_velocity_tolerance = float(value("settle_velocity_tolerance_rad_s"))
        self._settle_duration = float(value("settle_duration_sec"))
        self._final_position_tolerance = float(value("final_position_tolerance_m"))
        self._final_orientation_tolerance = float(value("final_orientation_tolerance_rad"))
        self._default_action_timeout = float(value("default_action_timeout_sec"))
        self._first_motion_test_mode = bool(value("first_motion_test_mode"))
        self._precision_test_mode = bool(value("precision_test_mode"))
        self._simulation_mode = bool(value("simulation_mode"))
        self._execution_enabled = bool(value("execution_enabled_on_start"))
        self._require_arm_status = bool(value("require_arm_status"))
        self._tf_timeout = float(value("tf_timeout_sec"))
        self._joint_feedback_topic = str(value("joint_feedback_topic"))
        self._arm_status_topic = str(value("arm_status_topic"))
        self._command_topic = str(value("command_topic"))
        self._planned_trajectory_topic = str(value("planned_trajectory_topic"))
        self._diagnostics_topic = str(value("diagnostics_topic"))
        self._target_pose_topic = str(value("target_pose_topic"))
        self._action_name = str(value("action_name"))
        self._recovery_action_name = str(value("recovery_action_name"))
        self._solve_service_name = str(value("solve_service_name"))
        self._execution_enable_service_name = str(value("execution_enable_service_name"))
        self._emergency_stop_service = str(value("emergency_stop_service"))
        self._metrics_enabled = bool(value("metrics_log_enabled"))
        self._metrics_output_dir = str(value("metrics_output_dir"))

        # A 15 mm first motion is smaller than the normal research acceptance
        # envelope. Tighten its gates so a stationary or barely moving arm
        # cannot be reported as successful.
        if self._first_motion_test_mode or self._precision_test_mode:
            self._tracking_pause_error = min(self._tracking_pause_error, 0.015)
            self._tracking_abort_error = min(self._tracking_abort_error, 0.030)
            self._tracking_abort_duration = min(
                self._tracking_abort_duration, 0.20
            )
            self._settle_joint_tolerance = min(
                self._settle_joint_tolerance, 0.005
            )
            self._final_position_tolerance = min(
                self._final_position_tolerance, 0.003
            )
            self._final_orientation_tolerance = min(
                self._final_orientation_tolerance, math.radians(2.0)
            )

    def _validate_configuration(self) -> None:
        if self._joint_names != NERO_JOINT_NAMES:
            raise ValueError("joint_names must be exactly joint1 through joint7")
        vector_lengths = (
            len(self._ready_positions),
            len(self._configured_lower_limits),
            len(self._configured_upper_limits),
            len(self._trajectory_velocity_limits),
            len(self._trajectory_acceleration_limits),
        )
        if any(length != 7 for length in vector_lengths):
            raise ValueError("all joint parameter arrays must contain seven values")
        if np.any(self._configured_lower_limits >= self._configured_upper_limits):
            raise ValueError("each lower joint limit must be below its upper limit")
        if min(self._trajectory_velocity_limits) <= 0.0:
            raise ValueError("trajectory velocity limits must be positive")
        if min(self._trajectory_acceleration_limits) <= 0.0:
            raise ValueError("trajectory acceleration limits must be positive")
        if self._trajectory_rate <= 0.0 or self._feedback_timeout <= 0.0:
            raise ValueError("trajectory rate and feedback timeout must be positive")
        if not (
            0.0 < self._precision_max_joint_delta <= self._max_joint_delta
        ):
            raise ValueError(
                "precision joint delta must be positive and no larger than "
                "the general IK joint delta"
            )
        recovery_values = (
            self._recovery_raw_margin,
            self._recovery_safe_margin,
            self._recovery_max_raw_violation,
            self._recovery_max_safe_violation,
            self._recovery_max_ingress_delta,
            self._recovery_max_total_delta,
            self._recovery_velocity_limit,
            self._recovery_acceleration_limit,
            self._recovery_stationary_velocity,
            self._recovery_stationary_duration,
            self._recovery_sigma_min,
            self._recovery_condition_max,
            self._recovery_phase_a_pause_error,
            self._recovery_phase_a_abort_error,
            self._recovery_tracking_pause_error,
            self._recovery_tracking_abort_error,
            self._recovery_tracking_abort_duration,
            self._recovery_final_joint_tolerance,
            self._recovery_final_safe_margin,
            self._recovery_fixed_joint_tolerance,
            self._recovery_progress_tolerance,
            self._recovery_plan_start_tolerance,
            self._recovery_timeout_max,
        )
        if not all(np.isfinite(item) and item > 0.0 for item in recovery_values):
            raise ValueError("joint-limit recovery parameters must be positive")
        if self._recovery_phase_a_pause_error >= self._recovery_phase_a_abort_error:
            raise ValueError("phase-A pause error must be below its abort error")
        if self._recovery_tracking_pause_error >= self._recovery_tracking_abort_error:
            raise ValueError("recovery pause error must be below its abort error")
        if self._recovery_final_safe_margin >= self._recovery_safe_margin:
            raise ValueError(
                "final recovery margin must be below the planned safe margin"
            )
        if self._recovery_velocity_limit > min(self._trajectory_velocity_limits):
            raise ValueError("recovery velocity cannot exceed ordinary limits")
        if self._recovery_acceleration_limit > min(
            self._trajectory_acceleration_limits
        ):
            raise ValueError("recovery acceleration cannot exceed ordinary limits")
        if (
            self._allow_recovery_execution
            and self._verified_driver_speed_percent != 10
        ):
            raise ValueError(
                "limit recovery execution requires verified driver speed 10%"
            )
        if (
            self._first_motion_test_mode
            and self._verified_driver_speed_percent != 10
        ):
            raise ValueError("first motion test requires verified driver speed 10%")
        if (
            self._precision_test_mode
            and self._verified_driver_speed_percent != 10
        ):
            raise ValueError("precision test requires verified driver speed 10%")
        if self._first_motion_test_mode and self._precision_test_mode:
            raise ValueError(
                "first_motion_test_mode and precision_test_mode are mutually exclusive"
            )
        if (self._first_motion_test_mode or self._precision_test_mode) and (
            self._base_frame != "base_link"
            or self._model_tip_frame != "link7"
            or self._default_controlled_frame != "link7"
        ):
            raise ValueError(
                "supervised real-arm tests require base_link and link7 frames"
            )
        forbidden = ("move_p", "move_l", "move_c", "move_js", "move_mit")
        if any(name in self._command_topic for name in forbidden):
            raise ValueError("command_topic must be the ordinary move_j interface")
        if not self._command_topic.rstrip("/").endswith("move_j"):
            raise ValueError("command_topic must end in move_j")

    @staticmethod
    def _resolve_urdf_path(configured_path: str) -> str:
        if configured_path:
            path = Path(os.path.expanduser(configured_path)).resolve()
        else:
            share = Path(get_package_share_directory("agx_arm_description"))
            candidates = (
                share
                / "agx_arm_urdf"
                / "nero"
                / "urdf"
                / "nero_description.urdf",
                share / "urdf" / "nero_description.urdf",
                share / "urdf" / "nero" / "nero_description.urdf",
            )
            path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
        if not path.is_file():
            raise FileNotFoundError(f"NERO URDF not found: {path}")
        return str(path)

    def _joint_feedback_callback(self, message: JointState) -> None:
        try:
            positions, velocities = ordered_joint_arrays(message, self._joint_names)
        except ValueError as error:
            self.get_logger().warning(f"Ignoring invalid joint feedback: {error}")
            return
        stamp = message.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
        velocity_by_name = dict(zip(message.name, message.velocity))
        velocities_valid = all(
            name in velocity_by_name
            and np.isfinite(float(velocity_by_name[name]))
            for name in self._joint_names
        )
        received = time.monotonic()
        snapshot = FeedbackSnapshot(
            positions,
            velocities,
            received,
            stamp,
            velocities_valid,
        )
        with self._state_lock:
            self._feedback = snapshot
            if (
                velocities_valid
                and np.max(np.abs(velocities))
                <= self._recovery_stationary_velocity
            ):
                if self._stationary_since is None:
                    self._stationary_since = received
            else:
                self._stationary_since = None

    def _arm_status_callback(self, message: AgxArmStatus) -> None:
        with self._state_lock:
            self._arm_status = copy.deepcopy(message)
            self._arm_status_monotonic = time.monotonic()

    def set_simulated_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Update the deterministic kinematic simulation backend."""
        now = self.get_clock().now().to_msg()
        received = time.monotonic()
        velocity_values = np.asarray(velocities, dtype=float).copy()
        with self._state_lock:
            self._feedback = FeedbackSnapshot(
                np.asarray(positions, dtype=float).copy(),
                velocity_values,
                received,
                now,
                True,
            )
            if np.max(np.abs(velocity_values)) <= self._recovery_stationary_velocity:
                if self._stationary_since is None:
                    self._stationary_since = received
            else:
                self._stationary_since = None

    def _publish_simulated_feedback(self) -> None:
        snapshot, _, _ = self._snapshot(require_fresh=False)
        if snapshot is None:
            return
        message = joint_state_from_arrays(
            self._joint_names,
            snapshot.positions,
            self.get_clock().now().to_msg(),
            velocities=snapshot.velocities,
        )
        self._sim_feedback_publisher.publish(message)

    def _snapshot(
        self, *, require_fresh: bool = True
    ) -> tuple[Optional[FeedbackSnapshot], int, str]:
        with self._state_lock:
            snapshot = self._feedback
        if snapshot is None:
            return None, IKResultMsg.FEEDBACK_STALE, "尚未收到完整的 7 关节反馈"
        age = time.monotonic() - snapshot.received_monotonic
        if require_fresh and age > self._feedback_timeout:
            return (
                snapshot,
                IKResultMsg.FEEDBACK_STALE,
                f"关节反馈已过期：{age:.3f}s > {self._feedback_timeout:.3f}s",
            )
        return snapshot, IKResultMsg.SUCCESS, ""

    def _driver_health(self) -> tuple[bool, str]:
        if self._simulation_mode or not self._require_arm_status:
            return True, ""
        with self._state_lock:
            status = copy.deepcopy(self._arm_status)
            age = time.monotonic() - self._arm_status_monotonic
        if status is None or age > self._feedback_timeout:
            return False, "机械臂状态反馈缺失或过期"
        if status.arm_status != 0:
            return False, f"机械臂报告故障状态 arm_status={status.arm_status}"
        if status.ctrl_mode != 1:
            return False, f"机械臂不在 CAN 控制模式 ctrl_mode={status.ctrl_mode}"
        if len(status.joint_angle_limit) != 7:
            return False, "关节限位状态不是完整 7 位"
        if len(status.communication_status_joint) != 7:
            return False, "关节通信状态不是完整 7 位"
        limited = [index + 1 for index, flag in enumerate(status.joint_angle_limit) if flag]
        communication = [
            index + 1
            for index, flag in enumerate(status.communication_status_joint)
            if flag
        ]
        if limited:
            return False, f"关节限位报警：{limited}"
        if communication:
            return False, f"关节通信报警：{communication}"
        return True, ""

    def _execution_enable_callback(self, request, response):
        with self._goal_lock:
            active = self._goal_reserved
        if request.data and active:
            response.success = False
            response.message = "已有动作执行中，不能改变执行门"
            return response
        self._execution_enabled = bool(request.data)
        response.success = True
        state = "开启" if self._execution_enabled else "关闭"
        response.message = f"本节点执行门已{state}；此操作不会使能电机"
        self.get_logger().warning(response.message)
        return response

    def _lookup_transform_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        if target_frame == source_frame:
            return np.eye(4)
        transform = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=RclpyDuration(seconds=self._tf_timeout),
        )
        return _transform_to_matrix(transform.transform)

    def _prepare_target(
        self, target: PoseStamped, controlled_frame: str
    ) -> TargetContext:
        reference_frame = target.header.frame_id.strip()
        if not reference_frame:
            raise ValueError("target_pose.header.frame_id 不能为空")
        target_controlled_in_reference = pose_to_matrix(target.pose)
        base_from_reference = self._lookup_transform_matrix(
            self._base_frame, reference_frame
        )
        target_controlled_in_base = (
            base_from_reference @ target_controlled_in_reference
        )
        selected_frame = controlled_frame.strip() or self._default_controlled_frame
        if selected_frame == self._model_tip_frame:
            tip_from_controlled = np.eye(4)
        else:
            if not self._camera_transform_valid:
                raise PermissionError(
                    "非 link7 控制被锁定：尚未确认毫米级 link7→camera 外参"
                )
            tip_from_controlled = self._lookup_transform_matrix(
                self._model_tip_frame, selected_frame
            )
        target_tip_in_base = target_controlled_in_base @ np.linalg.inv(
            tip_from_controlled
        )
        return TargetContext(
            target_tip_in_base,
            target_controlled_in_base,
            base_from_reference,
            tip_from_controlled,
            reference_frame,
            selected_frame,
        )

    def _solve(
        self,
        target: PoseStamped,
        controlled_frame: str,
        posture_reference_message: Optional[JointState] = None,
    ) -> tuple[IKResult, Optional[TargetContext], Optional[FeedbackSnapshot]]:
        snapshot, code, reason = self._snapshot()
        if snapshot is None or code != IKResultMsg.SUCCESS:
            return _failure_ik_result(code, reason), None, snapshot
        try:
            context = self._prepare_target(target, controlled_frame)
        except TransformException as error:
            return (
                _failure_ik_result(IKResultMsg.TF_UNAVAILABLE, str(error)),
                None,
                snapshot,
            )
        except PermissionError as error:
            return (
                _failure_ik_result(IKResultMsg.TF_UNAVAILABLE, str(error)),
                None,
                snapshot,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            return (
                _failure_ik_result(IKResultMsg.INVALID_TARGET, str(error)),
                None,
                snapshot,
            )

        posture_reference = None
        if posture_reference_message is not None and (
            posture_reference_message.name
            or posture_reference_message.position
        ):
            try:
                posture_reference, _ = ordered_joint_arrays(
                    posture_reference_message,
                    self._joint_names,
                )
            except ValueError as error:
                return (
                    _failure_ik_result(
                        IKResultMsg.INVALID_TARGET,
                        f"姿态偏好无效：{error}",
                    ),
                    context,
                    snapshot,
                )

        current_tip = self._solver.forward_kinematics(snapshot.positions)
        current_controlled = current_tip @ context.tip_from_controlled
        position_delta, orientation_delta = pose_error(
            context.target_controlled_in_base, current_controlled
        )
        posture_change_needed = bool(
            posture_reference is not None
            and np.max(np.abs(posture_reference - snapshot.positions))
            > self._position_deadband
        )
        if (
            position_delta < self._position_deadband
            and orientation_delta < self._orientation_deadband
            and not posture_change_needed
        ):
            metrics = self._solver.configuration_metrics(snapshot.positions)
            return (
                IKResult(
                    True,
                    IKErrorCode.ALREADY_AT_TARGET,
                    "目标变化处于 1 mm / 0.5° 死区内，不发送运动命令",
                    tuple(snapshot.positions),
                    position_delta,
                    orientation_delta,
                    0.0,
                    0,
                    metrics.sigma_min,
                    metrics.condition_number,
                    0.0,
                ),
                context,
                snapshot,
            )
        with self._solver_lock:
            result = self._solver.solve(
                context.target_tip_in_base,
                snapshot.positions,
                controlled_frame=self._model_tip_frame,
                posture_reference_joints=posture_reference,
            )
        return result, context, snapshot

    def _ik_message(
        self,
        result: IKResult,
        context: Optional[TargetContext],
        snapshot: Optional[FeedbackSnapshot],
    ) -> IKResultMsg:
        message = IKResultMsg()
        message.success = bool(result.success)
        message.code = int(result.error_code)
        message.reason = result.message
        positions = np.asarray(result.joint_positions, dtype=float)
        if positions.shape != (7,) or not np.all(np.isfinite(positions)):
            positions = (
                snapshot.positions.copy()
                if snapshot is not None
                else np.zeros(len(self._joint_names))
            )
        stamp = self.get_clock().now().to_msg()
        message.solution_joint_state = joint_state_from_arrays(
            self._joint_names, positions, stamp
        )
        try:
            tip_pose = self._solver.forward_kinematics(positions)
            if context is None:
                solved = tip_pose
                frame = self._base_frame
            else:
                controlled_pose = tip_pose @ context.tip_from_controlled
                solved = np.linalg.inv(context.base_from_reference) @ controlled_pose
                frame = context.reference_frame
            message.solved_pose = matrix_to_pose_stamped(solved, frame, stamp)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            message.solved_pose.header.frame_id = self._base_frame
            message.solved_pose.header.stamp = stamp
            message.solved_pose.pose.orientation.w = 1.0
        message.position_error_m = float(result.position_error_m)
        message.orientation_error_rad = float(result.orientation_error_rad)
        message.solve_time_ms = float(result.solve_time_ms)
        message.sigma_min = float(result.sigma_min)
        message.condition_number = float(result.condition_number)
        message.max_joint_delta_rad = float(result.max_joint_delta_rad)
        return message

    def _solve_service_callback(self, request, response):
        try:
            result, context, snapshot = self._solve(
                request.target_pose,
                request.controlled_frame,
                getattr(request, "posture_reference", None),
            )
            response.result = self._ik_message(result, context, snapshot)
            if context is not None:
                self._target_publisher.publish(
                    matrix_to_pose_stamped(
                        context.target_controlled_in_base,
                        self._base_frame,
                        self.get_clock().now().to_msg(),
                    )
                )
        except Exception as error:  # Keep service failures structured.
            self.get_logger().exception(f"Unexpected IK service error: {error}")
            response.result = self._ik_message(
                _failure_ik_result(IKResultMsg.INTERNAL_ERROR, str(error)),
                None,
                None,
            )
        return response

    def _goal_callback(self, goal_request) -> GoalResponse:
        with self._goal_lock:
            if self._goal_reserved:
                self.get_logger().warning("Rejecting goal: another motion is active")
                return GoalResponse.REJECT
            self._goal_reserved = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel_callback(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _first_motion_request_error(
        self,
        result: IKResult,
        context: Optional[TargetContext],
        snapshot: Optional[FeedbackSnapshot],
    ) -> Optional[tuple[int, str]]:
        """Lock first-motion mode to one strict link7-local +X 15 mm goal."""
        if not self._first_motion_test_mode:
            return None
        if self._first_motion_test_consumed:
            return (
                IKResultMsg.DRIVER_FAULT,
                "首次 15mm 真机测试已经使用；禁止在同一启动中重复执行",
            )
        if self._verified_driver_speed_percent != 10:
            return IKResultMsg.DRIVER_FAULT, "首次真机测试要求驱动速度为 10%"
        if context is None or snapshot is None:
            return IKResultMsg.INVALID_TARGET, "首次真机测试缺少目标或真实反馈"
        if context.reference_frame != "base_link":
            return IKResultMsg.INVALID_TARGET, "首次真机目标必须使用 base_link"
        if context.controlled_frame != "link7":
            return IKResultMsg.INVALID_TARGET, "首次真机目标必须控制 link7"
        if int(result.error_code) != IKResultMsg.SUCCESS:
            return (
                IKResultMsg.INVALID_TARGET,
                "首次真机测试只接受非零的 Placo SUCCESS 目标",
            )

        current_tip = self._solver.forward_kinematics(snapshot.positions)
        expected_target = current_tip.copy()
        expected_target[:3, 3] += current_tip[:3, 0] * 0.015
        target_position_error, target_orientation_error = pose_error(
            expected_target, context.target_tip_in_base
        )
        if target_position_error > 0.001:
            return (
                IKResultMsg.INVALID_TARGET,
                "首次真机目标不是当前 link7 +X 15mm："
                f"偏差 {target_position_error * 1000.0:.3f}mm",
            )
        if target_orientation_error > math.radians(0.5):
            return (
                IKResultMsg.INVALID_TARGET,
                "首次真机目标姿态相对当前 link7 发生了旋转",
            )
        strict_values = (
            result.position_error_m,
            result.orientation_error_rad,
            result.max_joint_delta_rad,
            result.sigma_min,
            result.condition_number,
        )
        if not all(np.isfinite(value) for value in strict_values):
            return IKResultMsg.INVALID_TARGET, "首次真机 IK 诊断包含无效数值"
        if result.position_error_m > 0.003:
            return IKResultMsg.UNREACHABLE, "首次真机 IK 位置残差超过 3mm"
        if result.orientation_error_rad > math.radians(2.0):
            return IKResultMsg.UNREACHABLE, "首次真机 IK 姿态残差超过 2°"
        if result.max_joint_delta_rad > 0.10:
            return IKResultMsg.JOINT_DELTA_TOO_LARGE, "首次真机关节变化超过 0.10rad"
        if result.sigma_min < 0.10 or result.condition_number > 20.0:
            return IKResultMsg.NEAR_SINGULARITY, "首次真机 IK 过于接近奇异点"
        if not snapshot.velocities_valid:
            return IKResultMsg.FEEDBACK_STALE, "首次真机测试缺少完整关节速度"
        max_speed = float(np.max(np.abs(snapshot.velocities)))
        if max_speed > self._settle_velocity_tolerance:
            return (
                IKResultMsg.DRIVER_FAULT,
                f"首次真机测试开始前仍在运动：{max_speed:.4f}rad/s",
            )
        with self._state_lock:
            stationary_since = self._stationary_since
        stationary_time = (
            0.0
            if stationary_since is None
            else time.monotonic() - stationary_since
        )
        if stationary_time < self._recovery_stationary_duration:
            return (
                IKResultMsg.DRIVER_FAULT,
                "首次真机测试要求机械臂连续静止满 "
                f"{self._recovery_stationary_duration:.2f}s",
            )
        return None

    def _precision_motion_request_error(
        self,
        result: IKResult,
        context: Optional[TargetContext],
        snapshot: Optional[FeedbackSnapshot],
    ) -> Optional[tuple[int, str]]:
        """Apply strict repeatability-test gates without fixing one direction."""
        if not self._precision_test_mode:
            return None
        if self._verified_driver_speed_percent != 10:
            return IKResultMsg.DRIVER_FAULT, "精度测试要求驱动速度为 10%"
        if context is None or snapshot is None:
            return IKResultMsg.INVALID_TARGET, "精度测试缺少目标或真实反馈"
        if (
            context.reference_frame != "base_link"
            or context.controlled_frame != "link7"
        ):
            return (
                IKResultMsg.INVALID_TARGET,
                "精度测试只接受 base_link 中的 link7 目标",
            )
        if int(result.error_code) != IKResultMsg.SUCCESS:
            return IKResultMsg.INVALID_TARGET, "精度测试只接受非零 Placo 目标"
        strict_values = (
            result.position_error_m,
            result.orientation_error_rad,
            result.max_joint_delta_rad,
            result.sigma_min,
            result.condition_number,
        )
        if not all(np.isfinite(value) for value in strict_values):
            return IKResultMsg.INVALID_TARGET, "精度测试 IK 诊断包含无效数值"
        if result.position_error_m > 0.003:
            return IKResultMsg.UNREACHABLE, "精度测试 IK 位置残差超过 3mm"
        if result.orientation_error_rad > math.radians(2.0):
            return IKResultMsg.UNREACHABLE, "精度测试 IK 姿态残差超过 2°"
        if result.max_joint_delta_rad > self._precision_max_joint_delta:
            return (
                IKResultMsg.JOINT_DELTA_TOO_LARGE,
                "精度测试单段关节变化超过 "
                f"{self._precision_max_joint_delta:.2f}rad",
            )
        if result.sigma_min < 0.10 or result.condition_number > 20.0:
            return IKResultMsg.NEAR_SINGULARITY, "精度测试 IK 过于接近奇异点"
        if not snapshot.velocities_valid:
            return IKResultMsg.FEEDBACK_STALE, "精度测试缺少完整关节速度"
        max_speed = float(np.max(np.abs(snapshot.velocities)))
        if max_speed > self._settle_velocity_tolerance:
            return (
                IKResultMsg.DRIVER_FAULT,
                f"精度测试开始前仍在运动：{max_speed:.4f}rad/s",
            )
        return None

    def _execute_action(self, goal_handle):
        started = time.monotonic()
        command_count_before = self._executor_backend.command_count
        run_id = str(uuid.uuid4())
        ik_message = IKResultMsg()
        context = None
        final_code = IKResultMsg.INTERNAL_ERROR
        final_reason = "动作未完成"
        success = False
        try:
            self._publish_action_feedback(
                goal_handle, MoveToPose.Feedback.STAGE_VALIDATING, "检查目标和真实反馈", started
            )
            self._publish_action_feedback(
                goal_handle, MoveToPose.Feedback.STAGE_SOLVING_IK, "使用真实关节角初始化 Placo", started
            )
            result, context, snapshot = self._solve(
                goal_handle.request.target_pose,
                goal_handle.request.controlled_frame,
                getattr(goal_handle.request, "posture_reference", None),
            )
            ik_message = self._ik_message(result, context, snapshot)
            if context is not None:
                self._target_publisher.publish(
                    matrix_to_pose_stamped(
                        context.target_controlled_in_base,
                        self._base_frame,
                        self.get_clock().now().to_msg(),
                    )
                )
            if not result.success:
                final_code, final_reason = int(result.error_code), result.message
                goal_handle.abort()
                return self._action_result(ik_message, context, final_code, final_reason)
            first_motion_error = self._first_motion_request_error(
                result, context, snapshot
            )
            if first_motion_error is not None:
                final_code, final_reason = first_motion_error
                goal_handle.abort()
                return self._action_result(
                    ik_message, context, final_code, final_reason
                )
            precision_error = self._precision_motion_request_error(
                result, context, snapshot
            )
            if precision_error is not None:
                final_code, final_reason = precision_error
                goal_handle.abort()
                return self._action_result(
                    ik_message, context, final_code, final_reason
                )
            if int(result.error_code) == IKResultMsg.ALREADY_AT_TARGET:
                final_code, final_reason, success = (
                    IKResultMsg.ALREADY_AT_TARGET,
                    result.message,
                    True,
                )
                goal_handle.succeed()
                return self._action_result(ik_message, context, final_code, final_reason)
            if goal_handle.is_cancel_requested:
                final_code, final_reason = IKResultMsg.CANCELED, "动作在规划前被取消"
                self._soft_hold(final_reason, command_count_before)
                goal_handle.canceled()
                return self._action_result(ik_message, context, final_code, final_reason)
            if not self._execution_enabled:
                final_code = IKResultMsg.DRIVER_FAULT
                final_reason = "执行门关闭：IK 已完成，但没有向机械臂发送命令"
                goal_handle.abort()
                return self._action_result(ik_message, context, final_code, final_reason)
            healthy, health_reason = self._driver_health()
            if not healthy:
                final_code, final_reason = IKResultMsg.DRIVER_FAULT, health_reason
                goal_handle.abort()
                return self._action_result(ik_message, context, final_code, final_reason)

            self._publish_action_feedback(
                goal_handle,
                MoveToPose.Feedback.STAGE_GENERATING_TRAJECTORY,
                "生成零起止速度和加速度的五次轨迹",
                started,
            )
            trajectory = self._trajectory_generator.generate(
                snapshot.positions, result.joint_positions
            )
            if not trajectory.success:
                final_code = IKResultMsg.TRAJECTORY_LIMIT_VIOLATION
                final_reason = trajectory.message
                goal_handle.abort()
                return self._action_result(ik_message, context, final_code, final_reason)
            self._trajectory_publisher.publish(self._trajectory_message(trajectory))
            timeout = duration_to_seconds(goal_handle.request.timeout)
            if timeout <= 0.0:
                timeout = self._default_action_timeout
            if self._first_motion_test_mode:
                self._first_motion_test_consumed = True
            execution_error = self._run_trajectory(
                goal_handle,
                trajectory,
                context,
                started,
                timeout,
                require_valid_velocities=(
                    self._first_motion_test_mode or self._precision_test_mode
                ),
            )
            if execution_error is not None:
                final_code, final_reason = execution_error
                self._soft_hold(final_reason, command_count_before)
                if final_code == IKResultMsg.CANCELED:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
                return self._action_result(ik_message, context, final_code, final_reason)

            final_tracking_error = self._final_pose_tracking_error(context)
            if final_tracking_error is not None:
                final_code, final_reason = final_tracking_error
                self._soft_hold(final_reason, command_count_before)
                goal_handle.abort()
                return self._action_result(
                    ik_message,
                    context,
                    final_code,
                    final_reason,
                )

            final_code, final_reason, success = IKResultMsg.SUCCESS, "目标已稳定到达", True
            goal_handle.succeed()
            return self._action_result(ik_message, context, final_code, final_reason)
        except Exception as error:
            final_code, final_reason = IKResultMsg.INTERNAL_ERROR, str(error)
            self.get_logger().exception(f"Unexpected action error: {error}")
            self._soft_hold(final_reason, command_count_before)
            goal_handle.abort()
            return self._action_result(ik_message, context, final_code, final_reason)
        finally:
            if self._first_motion_test_mode and self._first_motion_test_consumed:
                self._execution_enabled = False
            with self._goal_lock:
                self._goal_reserved = False
            _, position_error_value, orientation_error_value = (
                self._measured_result_pose(context)
            )
            self._metrics.record(
                {
                    "run_id": run_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "success": success,
                    "code": final_code,
                    "reason": final_reason,
                    "solve_time_ms": ik_message.solve_time_ms,
                    "motion_time_s": time.monotonic() - started,
                    "position_error_m": position_error_value,
                    "orientation_error_rad": orientation_error_value,
                    "sigma_min": ik_message.sigma_min,
                    "condition_number": ik_message.condition_number,
                    "max_joint_delta_rad": ik_message.max_joint_delta_rad,
                }
            )

    def _execute_recovery_action(self, goal_handle):
        """Preview or execute the strictly inward two-stage limit recovery."""
        started = time.monotonic()
        command_count_before = self._executor_backend.command_count
        plan: Optional[JointLimitRecoveryPlan] = None
        min_sigma = 0.0
        max_condition = math.inf
        final_code = RecoverToSafe.Result.INTERNAL_ERROR
        final_reason = "安全区恢复未完成"
        try:
            self._publish_action_feedback(
                goal_handle,
                RecoverToSafe.Feedback.STAGE_VALIDATING,
                "读取真实关节并检查专用恢复边界",
                started,
                feedback_type=RecoverToSafe.Feedback,
            )
            snapshot, code, reason = self._snapshot()
            if snapshot is None or code != IKResultMsg.SUCCESS:
                final_code = RecoverToSafe.Result.FEEDBACK_STALE
                final_reason = reason
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            requested_timeout = duration_to_seconds(goal_handle.request.timeout)
            if requested_timeout <= 0.0:
                timeout = min(
                    self._default_action_timeout,
                    self._recovery_timeout_max,
                )
            elif requested_timeout > self._recovery_timeout_max:
                final_code = RecoverToSafe.Result.INVALID_STATE
                final_reason = (
                    f"恢复超时上限为 {self._recovery_timeout_max:.1f}s，"
                    f"请求值 {requested_timeout:.1f}s 被拒绝"
                )
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )
            else:
                timeout = requested_timeout

            plan = self._recovery_generator.generate_limit_recovery(
                snapshot.positions,
                self._raw_joint_limits,
                self._recovery_config,
            )
            if not plan.success:
                final_code = int(plan.error_code)
                final_reason = plan.message
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            if plan.already_safe:
                metrics = self._solver.configuration_metrics(
                    plan.start_positions
                )
                min_sigma = metrics.sigma_min
                max_condition = metrics.condition_number
                final_code = RecoverToSafe.Result.ALREADY_SAFE
                final_reason = "所有关节已经位于保守安全范围内，没有发送命令"
                goal_handle.succeed()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            pattern_error = self._recovery_pattern_error(plan)
            if pattern_error is not None:
                final_code, final_reason = pattern_error
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            min_sigma, max_condition = self._recovery_path_metrics(plan)
            if (
                min_sigma < self._recovery_sigma_min
                or max_condition > self._recovery_condition_max
            ):
                final_code = RecoverToSafe.Result.NEAR_SINGULARITY
                final_reason = (
                    "恢复路径接近奇异：sigma_min="
                    f"{min_sigma:.4f}, condition={max_condition:.2f}"
                )
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            phase_a_trajectory = plan.phase_a_trajectory
            phase_b_trajectory = plan.phase_b_trajectory
            if phase_b_trajectory is None or not phase_b_trajectory.success:
                final_code = RecoverToSafe.Result.TRAJECTORY_LIMIT_VIOLATION
                final_reason = "恢复规划缺少可执行的第二阶段轨迹"
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )
            self._trajectory_publisher.publish(
                self._recovery_trajectory_message(plan)
            )
            self.get_logger().warning(
                "Joint-limit recovery plan: joints=%s, ingress=%s, target=%s, "
                "max_delta=%.4frad, sigma_min=%.4f, condition=%.2f"
                % (
                    plan.recovering_joint_indices,
                    np.round(plan.ingress_positions, 6).tolist(),
                    np.round(plan.target_positions, 6).tolist(),
                    plan.max_joint_delta_rad,
                    min_sigma,
                    max_condition,
                )
            )

            if goal_handle.is_cancel_requested:
                final_code = RecoverToSafe.Result.CANCELED
                final_reason = "恢复动作在发送任何命令前被取消"
                goal_handle.canceled()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )
            if not goal_handle.request.execute:
                self._publish_action_feedback(
                    goal_handle,
                    RecoverToSafe.Feedback.STAGE_PREVIEW_READY,
                    "预览完成：execute=false，未发送任何机械臂命令",
                    started,
                    reference_positions=plan.target_positions,
                    feedback_type=RecoverToSafe.Feedback,
                )
                final_code = RecoverToSafe.Result.PREVIEW_READY
                final_reason = "安全区恢复预览完成；没有发送任何真机命令"
                goal_handle.succeed()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            if time.monotonic() - started > timeout:
                final_code = RecoverToSafe.Result.TIMEOUT
                final_reason = "恢复预览与检查超过动作超时"
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            execution_error = self._recovery_pre_execution_error(plan)
            if execution_error is not None:
                final_code, final_reason = execution_error
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                )

            progress_monitor = JointLimitRecoveryMonitor(
                plan,
                self._solver.safe_joint_limits,
                fixed_joint_tolerance_rad=(
                    self._recovery_fixed_joint_tolerance
                ),
                progress_tolerance_rad=self._recovery_progress_tolerance,
            )
            state_validator = self._recovery_state_validator(progress_monitor)
            if phase_a_trajectory is not None:
                phase_a_error = self._run_recovery_ingress(
                    goal_handle,
                    plan,
                    started,
                    timeout,
                    state_validator,
                )
                if phase_a_error is not None:
                    final_code, final_reason = phase_a_error
                    self._soft_hold(final_reason, command_count_before)
                    if final_code == RecoverToSafe.Result.CANCELED:
                        goal_handle.canceled()
                    else:
                        goal_handle.abort()
                    return self._recovery_result(
                        plan,
                        final_code,
                        final_reason,
                        started,
                        min_sigma,
                        max_condition,
                        motion_started=(
                            self._executor_backend.command_count
                            > command_count_before
                        ),
                    )

            phase_b_error = self._run_trajectory(
                goal_handle,
                phase_b_trajectory,
                None,
                started,
                timeout,
                feedback_type=RecoverToSafe.Feedback,
                tracking_pause_error=self._recovery_tracking_pause_error,
                tracking_abort_error=self._recovery_tracking_abort_error,
                tracking_abort_duration=self._recovery_tracking_abort_duration,
                settle_joint_tolerance=self._recovery_final_joint_tolerance,
                settle_velocity_tolerance=self._recovery_stationary_velocity,
                settle_duration=self._recovery_stationary_duration,
                state_validator=state_validator,
                require_valid_velocities=True,
            )
            if phase_b_error is not None:
                final_code, final_reason = phase_b_error
                self._soft_hold(final_reason, command_count_before)
                if final_code == RecoverToSafe.Result.CANCELED:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                    motion_started=(
                        self._executor_backend.command_count
                        > command_count_before
                    ),
                )

            final_error = self._validate_recovery_result(plan)
            if final_error is not None:
                final_code, final_reason = final_error
                self._soft_hold(final_reason, command_count_before)
                goal_handle.abort()
                return self._recovery_result(
                    plan,
                    final_code,
                    final_reason,
                    started,
                    min_sigma,
                    max_condition,
                    motion_started=(
                        self._executor_backend.command_count
                        > command_count_before
                    ),
                )

            final_code = RecoverToSafe.Result.SUCCESS
            final_reason = "越界关节已单调进入保守安全范围并稳定"
            goal_handle.succeed()
            return self._recovery_result(
                plan,
                final_code,
                final_reason,
                started,
                min_sigma,
                max_condition,
                motion_started=(
                    self._executor_backend.command_count
                    > command_count_before
                ),
            )
        except Exception as error:
            final_code = RecoverToSafe.Result.INTERNAL_ERROR
            final_reason = str(error)
            self.get_logger().exception(
                f"Unexpected joint-limit recovery error: {error}"
            )
            self._soft_hold(final_reason, command_count_before)
            goal_handle.abort()
            return self._recovery_result(
                plan,
                final_code,
                final_reason,
                started,
                min_sigma,
                max_condition,
                motion_started=(
                    self._executor_backend.command_count
                    > command_count_before
                ),
            )
        finally:
            with self._goal_lock:
                self._goal_reserved = False

    def _recovery_pattern_error(
        self, plan: JointLimitRecoveryPlan
    ) -> Optional[tuple[int, str]]:
        """Limit the first hardware version to the audited J2/J4 directions."""
        allowed = {2, 4}
        recovering = set(plan.recovering_joint_indices)
        if not recovering.issubset(allowed):
            return (
                RecoverToSafe.Result.JOINT_LIMIT_VIOLATION,
                "首次真机恢复只允许关节 2 和关节 4；其他越界关节需人工检查",
            )
        start = np.asarray(plan.start_positions, dtype=float)
        limits = self._solver.safe_joint_limits
        if 2 in recovering and start[1] >= limits[1, 0]:
            return (
                RecoverToSafe.Result.DISCONTINUOUS_SOLUTION,
                "关节 2 不是已审计的下限侧越界方向",
            )
        if 4 in recovering and start[3] <= limits[3, 1]:
            return (
                RecoverToSafe.Result.DISCONTINUOUS_SOLUTION,
                "关节 4 不是已审计的上限侧越界方向",
            )
        return None

    def _recovery_pre_execution_error(
        self, plan: JointLimitRecoveryPlan
    ) -> Optional[tuple[int, str]]:
        """Check the independent gates that preview mode intentionally skips."""
        if not self._allow_recovery_execution:
            return (
                RecoverToSafe.Result.DRIVER_FAULT,
                "专用恢复执行锁关闭；当前只允许 execute=false 预览",
            )
        if not self._execution_enabled:
            return (
                RecoverToSafe.Result.DRIVER_FAULT,
                "本节点执行门关闭，没有向机械臂发送命令",
            )
        if self._verified_driver_speed_percent != 10:
            return (
                RecoverToSafe.Result.DRIVER_FAULT,
                "未验证原厂驱动 speed_percent=10，拒绝恢复",
            )
        snapshot, code, reason = self._snapshot()
        if snapshot is None or code != IKResultMsg.SUCCESS:
            return RecoverToSafe.Result.FEEDBACK_STALE, reason
        if not snapshot.velocities_valid:
            return (
                RecoverToSafe.Result.FEEDBACK_STALE,
                "恢复执行要求完整、有限的 7 关节速度反馈",
            )
        planned_start = np.asarray(plan.start_positions, dtype=float)
        start_error = float(
            np.max(np.abs(snapshot.positions - planned_start))
        )
        if start_error > self._recovery_plan_start_tolerance:
            return (
                RecoverToSafe.Result.DISCONTINUOUS_SOLUTION,
                "规划后真实起点已变化："
                f"{start_error:.4f}rad > "
                f"{self._recovery_plan_start_tolerance:.4f}rad；请重新预览",
            )
        with self._state_lock:
            stationary_since = self._stationary_since
        stationary_time = (
            0.0
            if stationary_since is None
            else time.monotonic() - stationary_since
        )
        if stationary_time < self._recovery_stationary_duration:
            return (
                RecoverToSafe.Result.DRIVER_FAULT,
                "真实关节尚未以低速稳定满 "
                f"{self._recovery_stationary_duration:.2f}s",
            )
        healthy, health_reason = self._driver_health()
        if not healthy:
            return RecoverToSafe.Result.DRIVER_FAULT, health_reason
        return None

    def _recovery_path_metrics(
        self, plan: JointLimitRecoveryPlan
    ) -> tuple[float, float]:
        """Evaluate singularity bounds at every planned recovery sample."""
        samples = [plan.start_positions]
        if plan.phase_a_trajectory is not None:
            samples.extend(
                point.positions for point in plan.phase_a_trajectory.points[1:]
            )
        else:
            samples.append(plan.ingress_positions)
        if plan.phase_b_trajectory is not None:
            samples.extend(
                point.positions for point in plan.phase_b_trajectory.points[1:]
            )
        metrics = [
            self._solver.configuration_metrics(positions)
            for positions in samples
        ]
        return (
            min(item.sigma_min for item in metrics),
            max(item.condition_number for item in metrics),
        )

    @staticmethod
    def _recovery_state_validator(
        monitor: JointLimitRecoveryMonitor,
    ) -> Callable[[FeedbackSnapshot], Optional[tuple[int, str]]]:
        """Adapt the pure progress monitor to the shared execution loop."""
        def validate(snapshot: FeedbackSnapshot) -> Optional[tuple[int, str]]:
            reason = monitor.validate(snapshot.positions)
            if reason is None:
                return None
            return RecoverToSafe.Result.DISCONTINUOUS_SOLUTION, reason

        return validate

    def _validate_recovery_result(
        self, plan: JointLimitRecoveryPlan
    ) -> Optional[tuple[int, str]]:
        """Require tight tracking and a measured margin inside safe limits."""
        snapshot, code, reason = self._snapshot()
        if snapshot is None or code != IKResultMsg.SUCCESS:
            return RecoverToSafe.Result.FEEDBACK_STALE, reason
        if not snapshot.velocities_valid:
            return (
                RecoverToSafe.Result.FEEDBACK_STALE,
                "最终验收缺少完整、有限的关节速度反馈",
            )
        healthy, health_reason = self._driver_health()
        if not healthy:
            return RecoverToSafe.Result.DRIVER_FAULT, health_reason
        final_speed = float(np.max(np.abs(snapshot.velocities)))
        if final_speed > self._recovery_stationary_velocity:
            return (
                RecoverToSafe.Result.TRACKING_ERROR,
                f"最终关节速度 {final_speed:.4f}rad/s 尚未稳定",
            )
        target = np.asarray(plan.target_positions, dtype=float)
        target_error = float(np.max(np.abs(target - snapshot.positions)))
        if target_error > self._recovery_final_joint_tolerance:
            return (
                RecoverToSafe.Result.TRACKING_ERROR,
                f"恢复终点最大关节误差 {target_error:.4f}rad 超限",
            )

        recovering = np.asarray(plan.recovering_joint_indices, dtype=int) - 1
        safe_limits = self._solver.safe_joint_limits
        outside_safe = (
            (snapshot.positions < safe_limits[:, 0])
            | (snapshot.positions > safe_limits[:, 1])
        )
        if np.any(outside_safe):
            joints = [int(index + 1) for index in np.flatnonzero(outside_safe)]
            return (
                RecoverToSafe.Result.JOINT_LIMIT_VIOLATION,
                f"最终仍有真实关节位于保守安全范围外：{joints}",
            )
        lower = safe_limits[recovering, 0] + self._recovery_final_safe_margin
        upper = safe_limits[recovering, 1] - self._recovery_final_safe_margin
        measured = snapshot.positions[recovering]
        if np.any(measured < lower) or np.any(measured > upper):
            return (
                RecoverToSafe.Result.JOINT_LIMIT_VIOLATION,
                "恢复后的真实关节没有进入要求的安全内侧余量",
            )

        fixed = np.asarray(
            [
                index
                for index in range(7)
                if index + 1 not in plan.recovering_joint_indices
            ],
            dtype=int,
        )
        if fixed.size:
            start = np.asarray(plan.start_positions, dtype=float)
            fixed_error = float(
                np.max(np.abs(snapshot.positions[fixed] - start[fixed]))
            )
            if fixed_error > self._recovery_fixed_joint_tolerance:
                return (
                    RecoverToSafe.Result.TRACKING_ERROR,
                    "原本安全的关节发生了非预期移动："
                    f"{fixed_error:.4f}rad",
                )
        metrics = self._solver.configuration_metrics(snapshot.positions)
        if (
            metrics.sigma_min < self._recovery_sigma_min
            or metrics.condition_number > self._recovery_condition_max
        ):
            return (
                RecoverToSafe.Result.NEAR_SINGULARITY,
                "最终真实构型接近奇异："
                f"sigma_min={metrics.sigma_min:.4f}, "
                f"condition={metrics.condition_number:.2f}",
            )
        return None

    def _recovery_result(
        self,
        plan: Optional[JointLimitRecoveryPlan],
        code: int,
        reason: str,
        started: float,
        min_sigma: float,
        max_condition: float,
        motion_started: bool = False,
    ) -> RecoverToSafe.Result:
        """Build one complete action result without pretending recovery is IK."""
        output = RecoverToSafe.Result()
        output.success = code in (
            RecoverToSafe.Result.SUCCESS,
            RecoverToSafe.Result.ALREADY_SAFE,
            RecoverToSafe.Result.PREVIEW_READY,
        )
        output.code = int(code)
        output.reason = reason
        output.executed = bool(
            code == RecoverToSafe.Result.SUCCESS or motion_started
        )
        stamp = self.get_clock().now().to_msg()
        fallback = np.zeros(7, dtype=float)
        snapshot, _, _ = self._snapshot(require_fresh=False)
        achieved = fallback if snapshot is None else snapshot.positions
        velocities = None if snapshot is None else snapshot.velocities
        safe_limits = self._solver.safe_joint_limits
        output.robot_is_safe = bool(
            snapshot is not None
            and np.all(achieved >= safe_limits[:, 0])
            and np.all(achieved <= safe_limits[:, 1])
        )
        if plan is None:
            start = ingress = target = achieved
            phase_a_duration = 0.0
            phase_b_duration = 0.0
        else:
            start = plan.start_positions or tuple(achieved)
            ingress = plan.ingress_positions or start
            target = plan.target_positions or ingress
            phase_a_duration = (
                0.0
                if plan.phase_a_trajectory is None
                else plan.phase_a_trajectory.duration_s
            )
            phase_b_duration = (
                0.0
                if plan.phase_b_trajectory is None
                else plan.phase_b_trajectory.duration_s
            )
            output.max_raw_violation_rad = plan.max_raw_violation_rad
            output.max_safe_violation_rad = plan.max_safe_violation_rad
            output.max_joint_delta_rad = plan.max_joint_delta_rad
        output.start_joint_state = joint_state_from_arrays(
            self._joint_names, start, stamp
        )
        output.ingress_joint_state = joint_state_from_arrays(
            self._joint_names, ingress, stamp
        )
        output.target_joint_state = joint_state_from_arrays(
            self._joint_names, target, stamp
        )
        output.achieved_joint_state = joint_state_from_arrays(
            self._joint_names,
            achieved,
            stamp,
            velocities=velocities,
        )
        output.min_sigma_min = float(min_sigma)
        output.max_condition_number = float(max_condition)
        output.phase_a_duration = seconds_to_duration(phase_a_duration)
        output.phase_b_duration = seconds_to_duration(phase_b_duration)
        output.elapsed = seconds_to_duration(time.monotonic() - started)
        return output

    def _run_recovery_ingress(
        self,
        goal_handle,
        plan: JointLimitRecoveryPlan,
        started: float,
        timeout: float,
        state_validator: Callable[
            [FeedbackSnapshot], Optional[tuple[int, str]]
        ],
    ) -> Optional[tuple[int, str]]:
        """
        Enter raw limits with one legal move_j target and measured monitoring.

        A measured startup angle can lie just outside the NERO firmware's raw
        command envelope. Sending the first nominal trajectory sample would
        therefore be clamped by the firmware and make the robot jump ahead of
        the software timeline. The first command is instead the audited ingress
        target, which is already inside the raw envelope. NERO performs its
        normal 10-percent-speed joint interpolation while this loop verifies
        fresh feedback, strictly inward progress and fixed-joint stability.
        """
        target = np.asarray(plan.ingress_positions, dtype=float)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            return (
                RecoverToSafe.Result.TRAJECTORY_LIMIT_VIOLATION,
                "阶段 A 入口目标不是完整有限的 7 关节角",
            )
        if np.any(target < self._raw_joint_limits[:, 0]) or np.any(
            target > self._raw_joint_limits[:, 1]
        ):
            return (
                RecoverToSafe.Result.JOINT_LIMIT_VIOLATION,
                "阶段 A 入口目标不在原始关节范围内",
            )

        snapshot, code, reason = self._snapshot()
        if snapshot is None or code != IKResultMsg.SUCCESS:
            return RecoverToSafe.Result.FEEDBACK_STALE, reason
        state_error = state_validator(snapshot)
        if state_error is not None:
            return state_error
        initial_error = float(np.max(np.abs(target - snapshot.positions)))
        if initial_error > self._recovery_max_ingress_delta + 1.0e-12:
            return (
                RecoverToSafe.Result.JOINT_DELTA_TOO_LARGE,
                "阶段 A 真实起点到入口目标跨度过大："
                f"{initial_error:.4f}rad",
            )
        if goal_handle.is_cancel_requested:
            return RecoverToSafe.Result.CANCELED, "阶段 A 在发送命令前被取消"
        if not self._execution_enabled:
            return RecoverToSafe.Result.DRIVER_FAULT, "执行门在阶段 A 前被关闭"
        healthy, health_reason = self._driver_health()
        if not healthy:
            return RecoverToSafe.Result.DRIVER_FAULT, health_reason

        # This is the first and only phase-A command. It is inside the raw
        # envelope, so neither pyAgxArm nor the firmware needs to clamp it.
        self._executor_backend.send(target, np.zeros(7, dtype=float))
        best_error = initial_error
        last_progress = time.monotonic()
        settled_since: Optional[float] = None
        progress_timeout = max(
            1.0,
            3.0 * self._recovery_tracking_abort_duration,
        )

        while True:
            now = time.monotonic()
            if goal_handle.is_cancel_requested:
                return RecoverToSafe.Result.CANCELED, "用户在阶段 A 取消动作"
            if not self._execution_enabled:
                return (
                    RecoverToSafe.Result.DRIVER_FAULT,
                    "执行门在阶段 A 中被关闭",
                )
            if now - started > timeout:
                return (
                    RecoverToSafe.Result.TIMEOUT,
                    f"阶段 A 超过动作超时 {timeout:.2f}s",
                )
            snapshot, code, reason = self._snapshot()
            if snapshot is None or code != IKResultMsg.SUCCESS:
                return RecoverToSafe.Result.FEEDBACK_STALE, reason
            healthy, health_reason = self._driver_health()
            if not healthy:
                return RecoverToSafe.Result.DRIVER_FAULT, health_reason
            if not snapshot.velocities_valid:
                return (
                    RecoverToSafe.Result.FEEDBACK_STALE,
                    "阶段 A 缺少完整、有限的关节速度反馈",
                )
            state_error = state_validator(snapshot)
            if state_error is not None:
                return state_error

            target_error = float(
                np.max(np.abs(target - snapshot.positions))
            )
            if target_error < best_error - self._recovery_progress_tolerance:
                best_error = target_error
                last_progress = now
            elif (
                target_error > self._recovery_final_joint_tolerance
                and now - last_progress > progress_timeout
            ):
                return (
                    RecoverToSafe.Result.TRACKING_ERROR,
                    "阶段 A 连续没有向入口目标取得进展："
                    f"当前误差 {target_error:.4f}rad",
                )

            max_speed = float(np.max(np.abs(snapshot.velocities)))
            if (
                target_error <= self._recovery_final_joint_tolerance
                and max_speed <= self._recovery_stationary_velocity
            ):
                if settled_since is None:
                    settled_since = now
                if now - settled_since >= self._recovery_stationary_duration:
                    return None
            else:
                settled_since = None

            self._publish_action_feedback(
                goal_handle,
                RecoverToSafe.Feedback.STAGE_ENTERING_RAW_LIMITS,
                "阶段 A：NERO 以 10% 速度进入原始关节范围",
                started,
                reference_positions=target,
                feedback_type=RecoverToSafe.Feedback,
            )
            time.sleep(0.02)

    def _run_trajectory(
        self,
        goal_handle,
        trajectory: TrajectoryResult,
        context: Optional[TargetContext],
        started: float,
        timeout: float,
        *,
        feedback_type=MoveToPose.Feedback,
        executing_stage: Optional[int] = None,
        tracking_pause_error: Optional[float] = None,
        tracking_abort_error: Optional[float] = None,
        tracking_abort_duration: Optional[float] = None,
        settle_joint_tolerance: Optional[float] = None,
        settle_velocity_tolerance: Optional[float] = None,
        settle_duration: Optional[float] = None,
        executing_message: str = "正在跟随平滑关节轨迹",
        state_validator: Optional[
            Callable[[FeedbackSnapshot], Optional[tuple[int, str]]]
        ] = None,
        require_valid_velocities: bool = False,
    ) -> Optional[tuple[int, str]]:
        period = 1.0 / self._trajectory_rate
        active_executing_stage = (
            feedback_type.STAGE_EXECUTING
            if executing_stage is None
            else executing_stage
        )
        active_pause_error = (
            self._tracking_pause_error
            if tracking_pause_error is None
            else tracking_pause_error
        )
        active_abort_error = (
            self._tracking_abort_error
            if tracking_abort_error is None
            else tracking_abort_error
        )
        active_abort_duration = (
            self._tracking_abort_duration
            if tracking_abort_duration is None
            else tracking_abort_duration
        )
        active_joint_tolerance = (
            self._settle_joint_tolerance
            if settle_joint_tolerance is None
            else settle_joint_tolerance
        )
        active_velocity_tolerance = (
            self._settle_velocity_tolerance
            if settle_velocity_tolerance is None
            else settle_velocity_tolerance
        )
        active_settle_duration = (
            self._settle_duration
            if settle_duration is None
            else settle_duration
        )
        pause_started: Optional[float] = None
        last_send = time.monotonic() - period
        points = trajectory.points
        if not points:
            return IKResultMsg.TRAJECTORY_LIMIT_VIOLATION, "轨迹不包含任何采样点"
        for point in points:
            while True:
                now = time.monotonic()
                if goal_handle.is_cancel_requested:
                    return IKResultMsg.CANCELED, "用户取消动作"
                if not self._execution_enabled:
                    return IKResultMsg.DRIVER_FAULT, "执行门在运动中被关闭"
                if now - started > timeout:
                    return IKResultMsg.TIMEOUT, f"动作超过超时 {timeout:.2f}s"
                snapshot, code, reason = self._snapshot()
                if snapshot is None or code != IKResultMsg.SUCCESS:
                    return IKResultMsg.FEEDBACK_STALE, reason
                healthy, health_reason = self._driver_health()
                if not healthy:
                    return IKResultMsg.DRIVER_FAULT, health_reason
                if require_valid_velocities and not snapshot.velocities_valid:
                    return (
                        IKResultMsg.FEEDBACK_STALE,
                        "执行期间缺少完整、有限的关节速度反馈",
                    )
                if state_validator is not None:
                    state_error = state_validator(snapshot)
                    if state_error is not None:
                        return state_error
                candidate_error = float(
                    np.max(np.abs(np.asarray(point.positions) - snapshot.positions))
                )
                last_command = self._executor_backend.last_command
                command_error = (
                    0.0
                    if last_command is None
                    else float(np.max(np.abs(last_command - snapshot.positions)))
                )
                if command_error > active_abort_error:
                    return (
                        IKResultMsg.TRACKING_ERROR,
                        f"实际关节落后指令 {command_error:.3f}rad，超过 "
                        f"{active_abort_error:.3f}rad",
                    )
                if candidate_error > active_pause_error:
                    if pause_started is None:
                        pause_started = now
                    if now - pause_started > active_abort_duration:
                        return (
                            IKResultMsg.TRACKING_ERROR,
                            f"规划领先实际值 {candidate_error:.3f}rad 持续超过 "
                            f"{active_abort_duration:.2f}s",
                        )
                    self._publish_action_feedback(
                        goal_handle,
                        active_executing_stage,
                        "反馈落后，暂停推进轨迹",
                        started,
                        context=context,
                        reference_positions=point.positions,
                        feedback_type=feedback_type,
                    )
                    time.sleep(min(period, 0.02))
                    continue
                pause_started = None
                remaining = period - (now - last_send)
                if remaining > 0.0:
                    time.sleep(min(remaining, 0.02))
                    continue
                self._executor_backend.send(point.positions, point.velocities)
                last_send = time.monotonic()
                self._publish_action_feedback(
                    goal_handle,
                    active_executing_stage,
                    executing_message,
                    started,
                    context=context,
                    reference_positions=point.positions,
                    feedback_type=feedback_type,
                )
                break

        settled_since: Optional[float] = None
        target_joints = np.asarray(points[-1].positions)
        while True:
            now = time.monotonic()
            if goal_handle.is_cancel_requested:
                return IKResultMsg.CANCELED, "用户在稳定检查阶段取消动作"
            if not self._execution_enabled:
                return IKResultMsg.DRIVER_FAULT, "执行门在稳定检查阶段被关闭"
            if now - started > timeout:
                return IKResultMsg.TIMEOUT, f"稳定检查超过动作超时 {timeout:.2f}s"
            snapshot, code, reason = self._snapshot()
            if snapshot is None or code != IKResultMsg.SUCCESS:
                return IKResultMsg.FEEDBACK_STALE, reason
            healthy, health_reason = self._driver_health()
            if not healthy:
                return IKResultMsg.DRIVER_FAULT, health_reason
            if require_valid_velocities and not snapshot.velocities_valid:
                return (
                    IKResultMsg.FEEDBACK_STALE,
                    "稳定检查期间缺少完整、有限的关节速度反馈",
                )
            if state_validator is not None:
                state_error = state_validator(snapshot)
                if state_error is not None:
                    return state_error
            joint_error = float(np.max(np.abs(target_joints - snapshot.positions)))
            joint_speed = float(np.max(np.abs(snapshot.velocities)))
            last_command = self._executor_backend.last_command
            command_reference = (
                target_joints
                if last_command is None
                else np.asarray(last_command, dtype=float)
            )
            command_error = float(
                np.max(np.abs(command_reference - snapshot.positions))
            )
            if command_error > active_abort_error:
                return (
                    IKResultMsg.TRACKING_ERROR,
                    f"稳定阶段实际关节落后指令 {command_error:.3f}rad，超过 "
                    f"{active_abort_error:.3f}rad",
                )
            if (
                joint_error <= active_joint_tolerance
                and joint_speed <= active_velocity_tolerance
            ):
                if settled_since is None:
                    settled_since = now
                if now - settled_since >= active_settle_duration:
                    break
            else:
                settled_since = None
            self._publish_action_feedback(
                goal_handle,
                feedback_type.STAGE_HOLDING,
                "等待真实关节稳定",
                started,
                context=context,
                reference_positions=target_joints,
                feedback_type=feedback_type,
            )
            time.sleep(min(period, 0.02))
        return None

    def _soft_hold(self, reason: str, command_count_before: int) -> None:
        snapshot, _, _ = self._snapshot(require_fresh=False)
        if snapshot is not None:
            self._executor_backend.hold_if_commanded_since(
                snapshot.positions,
                reason,
                command_count_before,
            )

    def _trajectory_message(self, trajectory: TrajectoryResult) -> JointTrajectory:
        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = list(self._joint_names)
        for point in trajectory.points:
            output = JointTrajectoryPoint()
            output.positions = list(point.positions)
            output.velocities = list(point.velocities)
            output.accelerations = list(point.accelerations)
            output.time_from_start = seconds_to_duration(point.time_from_start_s)
            message.points.append(output)
        return message

    def _recovery_trajectory_message(
        self, plan: JointLimitRecoveryPlan
    ) -> JointTrajectory:
        """Publish both smooth recovery phases as one auditable trajectory."""
        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = list(self._joint_names)

        phase_a = plan.phase_a_trajectory
        phase_b = plan.phase_b_trajectory
        phase_a_points = () if phase_a is None else phase_a.points
        phase_b_points = () if phase_b is None else phase_b.points
        entries = [(point, 0.0) for point in phase_a_points]
        phase_b_offset = 0.0 if phase_a is None else phase_a.duration_s
        phase_b_start = 0 if phase_a is None else 1
        entries.extend(
            (point, phase_b_offset)
            for point in phase_b_points[phase_b_start:]
        )
        for point, offset in entries:
            output = JointTrajectoryPoint()
            output.positions = list(point.positions)
            output.velocities = list(point.velocities)
            output.accelerations = list(point.accelerations)
            output.time_from_start = seconds_to_duration(
                offset + point.time_from_start_s
            )
            message.points.append(output)
        return message

    def _publish_action_feedback(
        self,
        goal_handle,
        stage: int,
        message: str,
        started: float,
        *,
        context: Optional[TargetContext] = None,
        reference_positions: Optional[Sequence[float]] = None,
        feedback_type=MoveToPose.Feedback,
    ) -> None:
        feedback = feedback_type()
        feedback.stage = stage
        feedback.status_message = message
        feedback.elapsed = seconds_to_duration(time.monotonic() - started)
        snapshot, _, _ = self._snapshot(require_fresh=False)
        if snapshot is not None:
            feedback.current_joint_state = joint_state_from_arrays(
                self._joint_names,
                snapshot.positions,
                snapshot.stamp,
                velocities=snapshot.velocities,
            )
            if reference_positions is not None:
                feedback.max_joint_error_rad = float(
                    np.max(
                        np.abs(
                            np.asarray(reference_positions, dtype=float)
                            - snapshot.positions
                        )
                    )
                )
            if context is not None:
                current = self._solver.forward_kinematics(snapshot.positions)
                current_controlled = current @ context.tip_from_controlled
                (
                    feedback.current_position_error_m,
                    feedback.current_orientation_error_rad,
                ) = pose_error(context.target_controlled_in_base, current_controlled)
        goal_handle.publish_feedback(feedback)

    def _final_pose_tracking_error(
        self, context: TargetContext
    ) -> Optional[tuple[int, str]]:
        """Validate the measured Cartesian result after joint settling."""
        _, position_error_value, orientation_error_value = (
            self._measured_result_pose(context)
        )
        if position_error_value > self._final_position_tolerance:
            return (
                IKResultMsg.TRACKING_ERROR,
                f"最终位置误差 {position_error_value * 1000.0:.2f}mm 超限",
            )
        if orientation_error_value > self._final_orientation_tolerance:
            return (
                IKResultMsg.TRACKING_ERROR,
                f"最终姿态误差 {math.degrees(orientation_error_value):.2f}° 超限",
            )
        return None

    def _measured_result_pose(
        self, context: Optional[TargetContext]
    ) -> tuple[PoseStamped, float, float]:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._base_frame
        pose.pose.orientation.w = 1.0
        snapshot, _, _ = self._snapshot(require_fresh=False)
        if snapshot is None or context is None:
            return pose, math.inf, math.inf
        try:
            current_tip = self._solver.forward_kinematics(snapshot.positions)
            current_controlled = current_tip @ context.tip_from_controlled
            reported = (
                np.linalg.inv(context.base_from_reference) @ current_controlled
            )
            pose = matrix_to_pose_stamped(
                reported,
                context.reference_frame,
                self.get_clock().now().to_msg(),
            )
            errors = pose_error(
                context.target_controlled_in_base, current_controlled
            )
            return pose, errors[0], errors[1]
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return pose, math.inf, math.inf

    def _action_result(
        self,
        ik_message: IKResultMsg,
        context: Optional[TargetContext],
        code: int,
        reason: str,
    ) -> MoveToPose.Result:
        output = MoveToPose.Result()
        output.ik_result = copy.deepcopy(ik_message)
        output.ik_result.code = int(code)
        output.ik_result.success = code in (
            IKResultMsg.SUCCESS,
            IKResultMsg.ALREADY_AT_TARGET,
        )
        output.ik_result.reason = reason
        (
            output.achieved_pose,
            output.final_position_error_m,
            output.final_orientation_error_rad,
        ) = self._measured_result_pose(context)
        return output

    def _publish_diagnostics(self) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = f"{self.get_fully_qualified_name()}: safety"
        status.hardware_id = "nero"
        snapshot, code, reason = self._snapshot()
        healthy, health_reason = self._driver_health()
        if snapshot is None or code != IKResultMsg.SUCCESS:
            status.level = DiagnosticStatus.ERROR
            status.message = reason
        elif not healthy:
            status.level = DiagnosticStatus.ERROR
            status.message = health_reason
        elif not self._execution_enabled:
            status.level = DiagnosticStatus.WARN
            status.message = "IK 可用，执行门关闭"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "反馈正常"
        status.values = [
            KeyValue(key="mode", value="sim" if self._simulation_mode else "real"),
            KeyValue(key="execution_enabled", value=str(self._execution_enabled)),
            KeyValue(
                key="precision_test_mode",
                value=str(self._precision_test_mode),
            ),
            KeyValue(key="feedback_fresh", value=str(code == IKResultMsg.SUCCESS)),
            KeyValue(key="environment_collision_checking", value="false"),
            KeyValue(key="command_interface", value=self._command_topic),
        ]
        array.status.append(status)
        self._diagnostics_publisher.publish(array)


def _transform_to_matrix(transform: Transform) -> np.ndarray:
    pose = Pose()
    pose.position.x = transform.translation.x
    pose.position.y = transform.translation.y
    pose.position.z = transform.translation.z
    pose.orientation = transform.rotation
    return pose_to_matrix(pose)


def _failure_ik_result(code: int, reason: str) -> IKResult:
    return IKResult(
        False,
        IKErrorCode(code),
        reason,
        (),
        math.inf,
        math.inf,
        0.0,
        0,
        0.0,
        math.inf,
        math.inf,
    )


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NeroControlNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
