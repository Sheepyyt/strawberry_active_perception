"""Guarded operator tool for the first NERO recovery and Placo real-arm move."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable, Optional, Sequence

import numpy as np

from action_msgs.msg import GoalStatus
from agx_arm_msgs.msg import AgxArmStatus
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
import rclpy
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from strawberry_nero_interfaces.action import MoveToPose, RecoverToSafe
from strawberry_nero_interfaces.msg import IKResult as IKResultMsg
from strawberry_nero_interfaces.srv import SolveIK

from .axis_suite import (
    AXIS_DIRECTIONS,
    AXIS_MAX_JOINT_DELTA_RAD,
    AXIS_MAX_RETURN_JOINT_ERROR_RAD,
    AxisReportWriter,
    AxisSuitePlan,
    case_to_row,
    plan_axis_case,
    plan_axis_suite,
    print_axis_suite,
)
from .ik_core import PlacoIKSolver
from .models import NERO_JOINT_NAMES, READY_JOINT_POSITIONS
from .ros_utils import matrix_to_pose_stamped, pose_error
from .trajectory import TrajectoryGenerator


RECOVERY_ACTION = "/strawberry_nero/recover_to_safe"
MOVE_ACTION = "/strawberry_nero/move_to_pose"
SOLVE_SERVICE = "/strawberry_nero/solve_ik"
JOINT_TOPIC = "/feedback/joint_states"
ARM_STATUS_TOPIC = "/feedback/arm_status"
MOVE_J_TOPIC = "/control/move_j"
FORBIDDEN_COMMAND_TOPICS = (
    "/control/joint_states",
    "/control/move_p",
    "/control/move_l",
    "/control/move_c",
    "/control/move_js",
    "/control/move_mit",
)

SMOKE_DISTANCE_M = 0.015
ROUNDTRIP_DISTANCE_M = 0.010
FEEDBACK_MAX_AGE_S = 0.20
STATIONARY_SPEED_RAD_S = 0.020
# The server always initializes Placo from its own newest feedback.  This
# client-side comparison only detects movement between preview and Action
# submission, so allow two milliradians for encoder settling/quantization.
PLAN_START_TOLERANCE_RAD = 0.002
SMOKE_MAX_JOINT_DELTA_RAD = 0.10
SMOKE_MIN_SIGMA = 0.10
SMOKE_MAX_CONDITION = 20.0
SMOKE_MAX_IK_POSITION_ERROR_M = 0.003
SMOKE_MAX_IK_ORIENTATION_ERROR_RAD = np.radians(2.0)
SMOKE_MAX_FINAL_POSITION_ERROR_M = 0.003
SMOKE_MAX_FINAL_ORIENTATION_ERROR_RAD = np.radians(2.0)
ROUNDTRIP_MAX_JOINT_DELTA_RAD = 0.08
ROUNDTRIP_MAX_RETURN_JOINT_ERROR_RAD = 0.02
CENTER_READY_SEGMENTS = 10
CENTER_READY_MAX_TOTAL_JOINT_DELTA_RAD = 0.50
CENTER_READY_MAX_SEGMENT_JOINT_DELTA_RAD = 0.06
CENTER_READY_FINAL_JOINT_TOLERANCE_RAD = 0.05
AXIS_SUITE_READY_JOINT_TOLERANCE_RAD = 0.05
MOTION_START_POSITION_DELTA_RAD = 0.00035

POST_MOTION_SAFETY_NOTICE = (
    "\n【重要：不要在机械臂悬空时直接失能】\n"
    "本工具只关闭两个软件控制门，不会失能电机。NERO 在失能或断电后可能因重力直接下落；"
    "请保持人员和物品远离机械臂下方。若现场只能人工托住，必须由两人配合："
    "一人先在避开关节夹点的位置托稳，另一人确认后才执行失能；本程序绝不会自动失能。"
)


class SmokeTestError(RuntimeError):
    """A structured refusal that must stop the supervised smoke test."""


@dataclass(frozen=True)
class MeasuredState:
    """One fresh, complete seven-joint feedback sample."""

    positions: np.ndarray
    velocities: np.ndarray
    received_monotonic: float


@dataclass(frozen=True)
class RoundTripPlan:
    """Frozen start and outbound targets for one supervised 10 mm test."""

    start_state: MeasuredState
    start_pose: object
    outbound_pose: object


@dataclass(frozen=True)
class CenterReadyPlan:
    """Ten small Placo targets leading toward the user-verified ready pose."""

    start_state: MeasuredState
    target_poses: tuple[object, ...]
    predicted_joint_positions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class AxisSuiteRealPlan:
    """Initial real feedback and six ROS-integrated direction previews."""

    start_state: MeasuredState
    suite_plan: AxisSuitePlan
    preview_results: tuple[IKResultMsg, ...]


@dataclass
class _MotionWatch:
    """Mutable joint-feedback watch active during one Action request."""

    baseline_positions: np.ndarray
    target_submitted_monotonic: float
    motion_started_monotonic: Optional[float] = None


@dataclass(frozen=True)
class ExecutionObservation:
    """Client-side end-to-end timing for one completed MoveToPose request."""

    response_latency_s: Optional[float]
    total_duration_s: float
    motion_start_detected: bool
    position_threshold_rad: float = MOTION_START_POSITION_DELTA_RAD


def nero_urdf_path() -> Path:
    """Locate the installed NERO URDF used by the controller."""
    share = Path(get_package_share_directory("agx_arm_description"))
    return share / "agx_arm_urdf" / "nero" / "urdf" / "nero_description.urdf"


def local_x_target(
    solver: PlacoIKSolver,
    measured_joints: Sequence[float],
    distance_m: float,
) -> np.ndarray:
    """Freeze a link7-local X displacement as one base-link target pose."""
    distance = float(distance_m)
    if not np.isfinite(distance) or not np.isclose(
        distance, SMOKE_DISTANCE_M, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("smoke displacement must be exactly link7 +X 15 mm")
    current = solver.forward_kinematics(measured_joints, "link7")
    target = current.copy()
    target[:3, 3] += current[:3, 0] * distance
    return target


def roundtrip_local_x_target(
    solver: PlacoIKSolver,
    measured_joints: Sequence[float],
    distance_m: float,
) -> np.ndarray:
    """Build the fixed link7-local +X 10 mm repeatability target."""
    distance = float(distance_m)
    if not np.isfinite(distance) or not np.isclose(
        distance, ROUNDTRIP_DISTANCE_M, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("roundtrip displacement must be exactly link7 +X 10 mm")
    current = solver.forward_kinematics(measured_joints, "link7")
    target = current.copy()
    target[:3, 3] += current[:3, 0] * distance
    return target


def recovery_result_allows_placo(result: RecoverToSafe.Result) -> bool:
    """Return true only after physical recovery or an already-safe reading."""
    executed_success = (
        result.code == RecoverToSafe.Result.SUCCESS
        and result.executed
        and result.robot_is_safe
    )
    already_safe = (
        result.code == RecoverToSafe.Result.ALREADY_SAFE
        and not result.executed
        and result.robot_is_safe
    )
    return bool(result.success and (executed_success or already_safe))


def validate_smoke_ik(
    result: IKResultMsg,
    max_joint_delta_rad: float = SMOKE_MAX_JOINT_DELTA_RAD,
) -> None:
    """Apply tighter first-motion gates to a normal Placo service result."""
    if not result.success or result.code != IKResultMsg.SUCCESS:
        raise SmokeTestError(
            f"Placo 预览被拒绝：code={result.code}, {result.reason}"
        )
    diagnostic_values = np.asarray(
        [
            result.position_error_m,
            result.orientation_error_rad,
            result.solve_time_ms,
            result.sigma_min,
            result.condition_number,
            result.max_joint_delta_rad,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(diagnostic_values)):
        raise SmokeTestError("Placo 诊断包含 NaN 或 infinity")
    if np.any(diagnostic_values[[0, 1, 2, 4, 5]] < 0.0):
        raise SmokeTestError("Placo 诊断包含不可能的负值")
    if tuple(result.solution_joint_state.name) != NERO_JOINT_NAMES:
        raise SmokeTestError("Placo 解的关节名称/顺序不是 joint1 到 joint7")
    solution = np.asarray(result.solution_joint_state.position, dtype=float)
    if solution.shape != (7,) or not np.all(np.isfinite(solution)):
        raise SmokeTestError("Placo 解不是完整有限的 7 关节角")
    if result.position_error_m > SMOKE_MAX_IK_POSITION_ERROR_M:
        raise SmokeTestError(
            "Placo 位置残差过大："
            f"{result.position_error_m * 1000.0:.3f} mm"
        )
    if result.orientation_error_rad > SMOKE_MAX_IK_ORIENTATION_ERROR_RAD:
        raise SmokeTestError(
            "Placo 姿态残差过大："
            f"{np.degrees(result.orientation_error_rad):.3f}°"
        )
    if result.max_joint_delta_rad > max_joint_delta_rad:
        raise SmokeTestError(
            f"关节变化 {result.max_joint_delta_rad:.4f} rad 超过 "
            f"{max_joint_delta_rad:.3f} rad 上限"
        )
    if result.sigma_min < SMOKE_MIN_SIGMA:
        raise SmokeTestError(
            f"sigma_min={result.sigma_min:.4f}，过于接近奇异点"
        )
    if result.condition_number > SMOKE_MAX_CONDITION:
        raise SmokeTestError(
            f"condition={result.condition_number:.2f}，过于接近奇异点"
        )


def _put_ik_in_report(row: dict, phase: str, result: IKResultMsg) -> None:
    """Copy one ROS Placo result into an axis-suite report row."""
    row[f"{phase}_solve_time_ms"] = float(result.solve_time_ms)
    row[f"{phase}_position_error_m"] = float(result.position_error_m)
    row[f"{phase}_orientation_error_rad"] = float(
        result.orientation_error_rad
    )
    row[f"{phase}_max_joint_delta_rad"] = float(
        result.max_joint_delta_rad
    )
    row[f"{phase}_sigma_min"] = float(result.sigma_min)
    row[f"{phase}_condition_number"] = float(
        result.condition_number
    )


def real_axis_preview_rows(plan: AxisSuiteRealPlan) -> list[dict]:
    """Build report rows from the direct and ROS-service prechecks."""
    rows = []
    for index, (case, result) in enumerate(zip(
        plan.suite_plan.cases,
        plan.preview_results,
    )):
        row = case_to_row(index, case, plan.start_state.positions)
        _put_ik_in_report(row, "outbound", result)
        row["accepted"] = True
        row["reason"] = "真机预览通过，尚未执行"
        rows.append(row)
    return rows


def require_confirmation(
    expected: str,
    input_function: Callable[[str], str] = input,
) -> None:
    """Require one exact, case-sensitive operator confirmation token."""
    if input_function is input and not sys.stdin.isatty():
        raise SmokeTestError("真机执行确认必须在交互式终端中输入")
    answer = input_function(
        "确认工作区仍为空，观察人员仍在工作区外守住控制箱断电位置；"
        f"输入 {expected} 继续："
    ).strip()
    if answer != expected:
        raise SmokeTestError("确认词不匹配；没有发送运动请求")


class NeroRealSmokeTest(Node):
    """Preview and request the two explicitly authorized first-arm motions."""

    def __init__(self, node_name: str = "nero_real_smoke_test") -> None:
        super().__init__(node_name)
        self._joint_state: Optional[MeasuredState] = None
        self._arm_status: Optional[AgxArmStatus] = None
        self._arm_status_monotonic = 0.0
        self._last_feedback_stage: Optional[tuple[str, int, str]] = None
        self._active_goal_handle = None
        self._active_result_future = None
        self._motion_watch: Optional[_MotionWatch] = None
        self._last_execution_observation: Optional[
            ExecutionObservation
        ] = None

        self.create_subscription(
            JointState,
            JOINT_TOPIC,
            self._joint_callback,
            20,
        )
        self.create_subscription(
            AgxArmStatus,
            ARM_STATUS_TOPIC,
            self._arm_status_callback,
            10,
        )
        self._recovery_client = ActionClient(
            self, RecoverToSafe, RECOVERY_ACTION
        )
        self._move_client = ActionClient(self, MoveToPose, MOVE_ACTION)
        self._solve_client = self.create_client(SolveIK, SOLVE_SERVICE)
        self._parameter_client = self.create_client(
            GetParameters, "/nero_control/get_parameters"
        )
        self._driver_gate_client = self.create_client(
            SetBool, "/control_enable"
        )
        self._controller_gate_client = self.create_client(
            SetBool, "/strawberry_nero/enable_execution"
        )
        self._solver = PlacoIKSolver(nero_urdf_path())

    def _joint_callback(self, message: JointState) -> None:
        """Keep only complete finite joint1-through-joint7 feedback."""
        if len(message.name) != len(set(message.name)):
            return
        positions = dict(zip(message.name, message.position))
        if len(message.velocity) != len(message.name):
            return
        velocities = dict(zip(message.name, message.velocity))
        if any(name not in positions or name not in velocities
               for name in NERO_JOINT_NAMES):
            return
        position_array = np.asarray(
            [positions[name] for name in NERO_JOINT_NAMES], dtype=float
        )
        velocity_array = np.asarray(
            [velocities[name] for name in NERO_JOINT_NAMES], dtype=float
        )
        if not (
            np.all(np.isfinite(position_array))
            and np.all(np.isfinite(velocity_array))
        ):
            return
        received = time.monotonic()
        self._joint_state = MeasuredState(
            positions=position_array,
            velocities=velocity_array,
            received_monotonic=received,
        )
        watch = self._motion_watch
        if watch is not None and watch.motion_started_monotonic is None:
            delta = float(np.max(np.abs(
                position_array - watch.baseline_positions
            )))
            if delta >= MOTION_START_POSITION_DELTA_RAD:
                watch.motion_started_monotonic = received

    @property
    def last_execution_observation(self) -> Optional[ExecutionObservation]:
        """Return timing from the most recent MoveToPose requested here."""
        return self._last_execution_observation

    def _arm_status_callback(self, message: AgxArmStatus) -> None:
        """Store the newest driver health message and local receive time."""
        self._arm_status = message
        self._arm_status_monotonic = time.monotonic()

    def _spin_until(self, predicate, timeout_s: float, description: str) -> None:
        """Spin this node until a predicate is true or fail with context."""
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise SmokeTestError(f"等待{description}超时（{timeout_s:.1f} s）")

    def _wait_future(self, future, timeout_s: float, description: str):
        """Wait for one ROS future while continuing to process feedback."""
        self._spin_until(future.done, timeout_s, description)
        error = future.exception()
        if error is not None:
            raise SmokeTestError(f"{description}失败：{error}")
        return future.result()

    def measured_state(self, timeout_s: float = 3.0) -> MeasuredState:
        """Return a newly received complete joint sample."""
        requested_at = time.monotonic()
        self._spin_until(
            lambda: (
                self._joint_state is not None
                and self._joint_state.received_monotonic >= requested_at
            ),
            timeout_s,
            "完整关节反馈",
        )
        assert self._joint_state is not None
        return self._joint_state

    def _verify_controller_mode(
        self,
        require_recovery_unlock: bool,
        precision_test: Optional[bool],
    ) -> None:
        """Verify launch-time locks before any physical action request."""
        if not self._parameter_client.wait_for_service(timeout_sec=5.0):
            raise SmokeTestError("无法读取 nero_control 安全参数")
        names = [
            "simulation_mode",
            "first_motion_test_mode",
            "precision_test_mode",
            "verified_driver_speed_percent",
            "allow_limit_recovery_execution",
        ]
        request = GetParameters.Request()
        request.names = names
        response = self._wait_future(
            self._parameter_client.call_async(request),
            5.0,
            "读取 nero_control 安全参数",
        )
        if len(response.values) != len(names):
            raise SmokeTestError("nero_control 安全参数返回不完整")
        values = dict(zip(names, response.values))
        if (
            values["simulation_mode"].type != ParameterType.PARAMETER_BOOL
            or values["simulation_mode"].bool_value
        ):
            raise SmokeTestError("当前不是 real 控制模式")
        first_mode = values["first_motion_test_mode"]
        precision_mode = values["precision_test_mode"]
        if (
            first_mode.type != ParameterType.PARAMETER_BOOL
            or precision_mode.type != ParameterType.PARAMETER_BOOL
        ):
            raise SmokeTestError("真机测试模式参数类型错误")
        if precision_test is None:
            if first_mode.bool_value == precision_mode.bool_value:
                raise SmokeTestError(
                    "安全区恢复要求开启且只开启一个监督测试模式"
                )
        elif precision_test:
            if first_mode.bool_value or not precision_mode.bool_value:
                raise SmokeTestError(
                    "往返测试要求 first_motion_test_mode=false 且 "
                    "precision_test_mode=true"
                )
        elif not first_mode.bool_value or precision_mode.bool_value:
            raise SmokeTestError(
                "首次测试要求 first_motion_test_mode=true 且 "
                "precision_test_mode=false"
            )
        speed = values["verified_driver_speed_percent"]
        if (
            speed.type != ParameterType.PARAMETER_INTEGER
            or speed.integer_value != 10
        ):
            raise SmokeTestError("驱动速度没有锁定为 10%")
        recovery = values["allow_limit_recovery_execution"]
        if require_recovery_unlock and (
            recovery.type != ParameterType.PARAMETER_BOOL
            or not recovery.bool_value
        ):
            raise SmokeTestError("安全区恢复专用锁没有打开")

    def _validate_command_publishers(self) -> None:
        """Reject every command source except this controller's move_j."""
        publishers = self.get_publishers_info_by_topic(MOVE_J_TOPIC)
        if len(publishers) != 1:
            raise SmokeTestError(
                f"{MOVE_J_TOPIC} 必须恰有一个发布者，当前为 {len(publishers)}"
            )
        publisher = publishers[0]
        if (
            publisher.node_name != "nero_control"
            or publisher.node_namespace != "/"
        ):
            raise SmokeTestError(
                "move_j 发布者必须恰好是 /nero_control，当前为 "
                f"{publisher.node_namespace}/{publisher.node_name}"
            )
        for topic in FORBIDDEN_COMMAND_TOPICS:
            count = len(self.get_publishers_info_by_topic(topic))
            if count != 0:
                raise SmokeTestError(
                    f"{topic} 存在 {count} 个发布者；禁止打开原厂控制门"
                )

    def _wait_stationary_state(self, duration_s: float = 0.50) -> MeasuredState:
        """Require a continuous low-speed window, not one zero-speed sample."""
        deadline = time.monotonic() + 4.0
        stationary_since: Optional[float] = None
        latest: Optional[MeasuredState] = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            latest = self._joint_state
            now = time.monotonic()
            if (
                latest is None
                or now - latest.received_monotonic > FEEDBACK_MAX_AGE_S
            ):
                stationary_since = None
                continue
            speed = float(np.max(np.abs(latest.velocities)))
            if speed > STATIONARY_SPEED_RAD_S:
                stationary_since = None
                continue
            if stationary_since is None:
                stationary_since = now
            if now - stationary_since >= duration_s:
                return latest
        raise SmokeTestError("机械臂未连续静止满 0.5 s")

    def _set_gate(self, client, service_name: str, enabled: bool) -> None:
        """Set one command gate and require an explicit successful response."""
        if not client.wait_for_service(timeout_sec=3.0):
            raise SmokeTestError(f"{service_name} 服务不可用")
        request = SetBool.Request()
        request.data = enabled
        response = self._wait_future(
            client.call_async(request),
            3.0,
            f"设置 {service_name}={str(enabled).lower()}",
        )
        if not response.success:
            raise SmokeTestError(
                f"{service_name} 拒绝请求：{response.message}"
            )

    @contextmanager
    def _temporary_command_gates(self):
        """Open both software gates only around one authorized Action."""
        driver_open = False
        controller_open = False
        try:
            self._set_gate(self._driver_gate_client, "/control_enable", True)
            driver_open = True
            self._set_gate(
                self._controller_gate_client,
                "/strawberry_nero/enable_execution",
                True,
            )
            controller_open = True
            yield
        finally:
            if controller_open:
                try:
                    self._set_gate(
                        self._controller_gate_client,
                        "/strawberry_nero/enable_execution",
                        False,
                    )
                except SmokeTestError as error:
                    print(f"警告：{error}", file=sys.stderr)
            if driver_open:
                try:
                    self._set_gate(
                        self._driver_gate_client, "/control_enable", False
                    )
                except SmokeTestError as error:
                    print(f"警告：{error}", file=sys.stderr)

    def _execution_preflight(
        self,
        *,
        require_recovery_unlock: bool = False,
        precision_test: Optional[bool] = False,
    ) -> MeasuredState:
        """Check fresh measured health and sole command ownership."""
        self._verify_controller_mode(
            require_recovery_unlock,
            precision_test,
        )
        self._validate_command_publishers()
        state = self.measured_state()
        requested_at = time.monotonic()
        self._spin_until(
            lambda: self._arm_status_monotonic >= requested_at,
            3.0,
            "机械臂状态反馈",
        )
        status = self._arm_status
        if status is None:
            raise SmokeTestError("没有机械臂状态反馈")
        now = time.monotonic()
        if now - state.received_monotonic > FEEDBACK_MAX_AGE_S:
            raise SmokeTestError("关节反馈超过 0.2 s，拒绝执行")
        if now - self._arm_status_monotonic > FEEDBACK_MAX_AGE_S:
            raise SmokeTestError("状态反馈超过 0.2 s，拒绝执行")
        if status.ctrl_mode != 1:
            raise SmokeTestError(
                f"ctrl_mode={status.ctrl_mode}，不是 CAN 控制模式"
            )
        if status.arm_status != 0:
            raise SmokeTestError(
                f"arm_status={status.arm_status}，机械臂尚未正常使能"
            )
        if len(status.joint_angle_limit) != 7:
            raise SmokeTestError("关节限位状态不是完整 7 位")
        if len(status.communication_status_joint) != 7:
            raise SmokeTestError("关节通信状态不是完整 7 位")
        if any(status.joint_angle_limit):
            raise SmokeTestError("驱动报告关节限位异常")
        if any(status.communication_status_joint):
            raise SmokeTestError("驱动报告关节通信异常")
        return self._wait_stationary_state()

    def _feedback_callback(self, label: str):
        """Build a concise action feedback printer."""
        def callback(wrapper) -> None:
            feedback = wrapper.feedback
            key = (label, int(feedback.stage), str(feedback.status_message))
            if key == self._last_feedback_stage:
                return
            self._last_feedback_stage = key
            print(f"[{label}] {feedback.status_message}")

        return callback

    def _send_action_goal(
        self,
        client: ActionClient,
        goal,
        label: str,
        timeout_s: float,
    ):
        """Send one action goal and synchronously cancel it on interruption."""
        if not client.wait_for_server(timeout_sec=5.0):
            raise SmokeTestError(f"{label} Action 不可用")
        self._last_feedback_stage = None
        send_future = client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback(label),
        )
        goal_handle = self._wait_future(
            send_future, 5.0, f"{label}目标接收"
        )
        if not goal_handle.accepted:
            raise SmokeTestError(f"{label}目标被 Action Server 拒绝")
        self._active_goal_handle = goal_handle
        result_future = None
        try:
            result_future = goal_handle.get_result_async()
            self._active_result_future = result_future
            wrapped_result = self._wait_future(
                result_future, timeout_s, f"{label}执行结果"
            )
        except BaseException:
            self._cancel_goal_and_wait(
                goal_handle, result_future, label
            )
            raise
        finally:
            if self._active_goal_handle is goal_handle:
                self._active_goal_handle = None
                self._active_result_future = None
        return wrapped_result

    def _cancel_goal_and_wait(
        self,
        goal_handle,
        result_future,
        label: str,
    ) -> bool:
        """Request cancellation and wait briefly for an explicit terminal state."""
        if goal_handle is None or not rclpy.ok():
            return False
        try:
            response = self._wait_future(
                goal_handle.cancel_goal_async(),
                2.0,
                f"{label}取消响应",
            )
            if not response.goals_canceling:
                print(
                    f"警告：{label} Action Server 没有接受取消请求",
                    file=sys.stderr,
                )
                return False
            if result_future is None:
                print(f"{label}：服务器已接受取消请求")
                return True
            wrapped = self._wait_future(
                result_future,
                3.0,
                f"{label}取消后的终态",
            )
            terminal_states = (
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED,
                GoalStatus.STATUS_SUCCEEDED,
            )
            if wrapped.status not in terminal_states:
                print(
                    f"警告：{label} 取消后状态仍为 {wrapped.status}",
                    file=sys.stderr,
                )
                return False
            print(f"{label}：取消后服务器终态={wrapped.status}")
            return True
        except BaseException as error:
            print(f"警告：{label} 取消确认失败：{error}", file=sys.stderr)
            return False

    @staticmethod
    def _print_recovery_result(result: RecoverToSafe.Result) -> None:
        """Print the fields an operator must verify before continuing."""
        print("\n=== 安全区恢复结果 ===")
        print(f"code={result.code}, success={result.success}")
        print(
            f"executed={result.executed}, "
            f"robot_is_safe={result.robot_is_safe}"
        )
        print(f"说明：{result.reason}")
        print(
            "起点："
            + np.array2string(
                np.asarray(result.start_joint_state.position), precision=6
            )
        )
        print(
            "阶段 A："
            + np.array2string(
                np.asarray(result.ingress_joint_state.position), precision=6
            )
        )
        print(
            "阶段 B："
            + np.array2string(
                np.asarray(result.target_joint_state.position), precision=6
            )
        )
        print(
            f"最大关节变化={result.max_joint_delta_rad:.6f} rad, "
            f"sigma_min={result.min_sigma_min:.4f}, "
            f"condition={result.max_condition_number:.2f}"
        )

    def preview_recovery(self) -> RecoverToSafe.Result:
        """Request a guaranteed no-command recovery preview."""
        goal = RecoverToSafe.Goal()
        goal.execute = False
        goal.timeout.sec = 10
        wrapped = self._send_action_goal(
            self._recovery_client, goal, "恢复预览", 12.0
        )
        self._print_recovery_result(wrapped.result)
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            raise SmokeTestError(
                f"恢复预览 Action 状态异常：{wrapped.status}"
            )
        if wrapped.result.code not in (
            RecoverToSafe.Result.PREVIEW_READY,
            RecoverToSafe.Result.ALREADY_SAFE,
        ):
            raise SmokeTestError("恢复预览未通过，禁止执行")
        preview_ready = (
            wrapped.result.code == RecoverToSafe.Result.PREVIEW_READY
            and wrapped.result.success
            and not wrapped.result.executed
            and not wrapped.result.robot_is_safe
        )
        already_safe = (
            wrapped.result.code == RecoverToSafe.Result.ALREADY_SAFE
            and wrapped.result.success
            and not wrapped.result.executed
            and wrapped.result.robot_is_safe
        )
        if not (preview_ready or already_safe):
            raise SmokeTestError("恢复预览字段相互矛盾，禁止执行")
        return wrapped.result

    def execute_recovery(self) -> RecoverToSafe.Result:
        """Request the controller's inward-only physical recovery."""
        self._execution_preflight(
            require_recovery_unlock=True,
            precision_test=None,
        )
        goal = RecoverToSafe.Goal()
        goal.execute = True
        goal.timeout.sec = 15
        with self._temporary_command_gates():
            wrapped = self._send_action_goal(
                self._recovery_client, goal, "安全区恢复", 20.0
            )
        self._print_recovery_result(wrapped.result)
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            raise SmokeTestError(
                f"安全区恢复 Action 状态异常：{wrapped.status}"
            )
        if not recovery_result_allows_placo(wrapped.result):
            raise SmokeTestError("机械臂没有被确认处于安全范围")
        return wrapped.result

    @staticmethod
    def _print_ik_result(result: IKResultMsg, distance_m: float) -> None:
        """Print the Placo evidence for one fixed local-X target."""
        direction = "+X" if distance_m > 0.0 else "-X"
        distance_mm = abs(distance_m) * 1000.0
        print(f"\n=== Placo {distance_mm:.0f} mm 预览 ===")
        print(
            "目标：从最新真实姿态沿 link7 "
            f"{direction} 移动 {distance_mm:.0f} mm"
        )
        print(f"code={result.code}, success={result.success}")
        print(f"说明：{result.reason}")
        print(f"IK 耗时={result.solve_time_ms:.3f} ms")
        print(
            f"残差={result.position_error_m * 1000.0:.3f} mm / "
            f"{np.degrees(result.orientation_error_rad):.4f}°"
        )
        print(
            f"最大关节变化={result.max_joint_delta_rad:.6f} rad, "
            f"sigma_min={result.sigma_min:.4f}, "
            f"condition={result.condition_number:.2f}"
        )
        print(
            "Placo 目标关节角："
            + np.array2string(
                np.asarray(result.solution_joint_state.position), precision=6
            )
        )

    def preview_placo(self, distance_m: float):
        """Create a live relative target and call the no-motion IK service."""
        state = self.measured_state()
        safe_limits = self._solver.safe_joint_limits
        if np.any(state.positions < safe_limits[:, 0]) or np.any(
            state.positions > safe_limits[:, 1]
        ):
            raise SmokeTestError(
                "真实关节仍在保守安全范围外；必须先完成 recover"
            )
        target_matrix = local_x_target(
            self._solver, state.positions, distance_m
        )
        request = SolveIK.Request()
        request.target_pose = matrix_to_pose_stamped(
            target_matrix,
            "base_link",
            self.get_clock().now().to_msg(),
        )
        request.controlled_frame = "link7"
        if not self._solve_client.wait_for_service(timeout_sec=5.0):
            raise SmokeTestError("SolveIK 服务不可用")
        response = self._wait_future(
            self._solve_client.call_async(request),
            5.0,
            "Placo SolveIK 预览",
        )
        self._print_ik_result(response.result, distance_m)
        validate_smoke_ik(response.result)
        return request.target_pose, response.result, state

    def execute_placo(
        self,
        target_pose,
        planned_state: MeasuredState,
        *,
        precision_test: bool = False,
        label: str = "Placo 真机运动",
        posture_reference: Optional[Sequence[float]] = None,
    ):
        """Execute one already-previewed fixed Cartesian target."""
        state = self._execution_preflight(precision_test=precision_test)
        start_drift = float(
            np.max(np.abs(state.positions - planned_state.positions))
        )
        if start_drift > PLAN_START_TOLERANCE_RAD:
            raise SmokeTestError(
                f"确认期间关节漂移 {start_drift:.6f} rad；请重新预览"
            )
        goal = MoveToPose.Goal()
        goal.target_pose = target_pose
        goal.controlled_frame = "link7"
        if posture_reference is not None:
            goal.posture_reference.name = list(NERO_JOINT_NAMES)
            goal.posture_reference.position = [
                float(value) for value in posture_reference
            ]
        goal.timeout.sec = 15
        watch = None
        completed_monotonic = None
        self._last_execution_observation = None
        try:
            with self._temporary_command_gates():
                watch = _MotionWatch(
                    baseline_positions=state.positions.copy(),
                    target_submitted_monotonic=time.monotonic(),
                )
                self._motion_watch = watch
                wrapped = self._send_action_goal(
                    self._move_client, goal, label, 20.0
                )
                completed_monotonic = time.monotonic()
        finally:
            if watch is not None:
                finished = (
                    time.monotonic()
                    if completed_monotonic is None
                    else completed_monotonic
                )
                response_latency = None
                if watch.motion_started_monotonic is not None:
                    response_latency = (
                        watch.motion_started_monotonic
                        - watch.target_submitted_monotonic
                    )
                self._last_execution_observation = ExecutionObservation(
                    response_latency_s=response_latency,
                    total_duration_s=(
                        finished - watch.target_submitted_monotonic
                    ),
                    motion_start_detected=(response_latency is not None),
                )
            self._motion_watch = None
        result = wrapped.result
        print("\n=== Placo 真机结果 ===")
        print(
            f"Action status={wrapped.status}, code={result.ik_result.code}, "
            f"success={result.ik_result.success}"
        )
        print(f"说明：{result.ik_result.reason}")
        print(
            f"最终实测误差={result.final_position_error_m * 1000.0:.3f} mm / "
            f"{np.degrees(result.final_orientation_error_rad):.4f}°"
        )
        observation = self._last_execution_observation
        if observation is not None and observation.motion_start_detected:
            print(
                "目标提交到编码器开始变化="
                f"{observation.response_latency_s * 1000.0:.1f} ms "
                f"(阈值 {observation.position_threshold_rad:.5f} rad)"
            )
        else:
            print("目标提交到编码器开始变化：本次未可靠检测")
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            raise SmokeTestError("MoveToPose 没有成功完成")
        if (
            not result.ik_result.success
            or result.ik_result.code != IKResultMsg.SUCCESS
        ):
            raise SmokeTestError("MoveToPose 返回失败的 IK/执行结果")
        final_errors = np.asarray(
            [
                result.final_position_error_m,
                result.final_orientation_error_rad,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(final_errors)) or np.any(final_errors < 0.0):
            raise SmokeTestError("MoveToPose 最终误差无效")
        if result.final_position_error_m > SMOKE_MAX_FINAL_POSITION_ERROR_M:
            raise SmokeTestError("运动完成，但 3 mm smoke 精度验收未通过")
        if (
            result.final_orientation_error_rad
            > SMOKE_MAX_FINAL_ORIENTATION_ERROR_RAD
        ):
            raise SmokeTestError("运动完成，但 2° smoke 姿态验收未通过")
        return result

    def _solve_pose_preview(
        self,
        target_pose,
        label: str,
        posture_reference: Optional[Sequence[float]] = None,
    ) -> IKResultMsg:
        """Call SolveIK for one pose without opening either command gate."""
        request = SolveIK.Request()
        request.target_pose = target_pose
        request.controlled_frame = "link7"
        if posture_reference is not None:
            request.posture_reference.name = list(NERO_JOINT_NAMES)
            request.posture_reference.position = [
                float(value) for value in posture_reference
            ]
        if not self._solve_client.wait_for_service(timeout_sec=5.0):
            raise SmokeTestError("SolveIK 服务不可用")
        response = self._wait_future(
            self._solve_client.call_async(request),
            5.0,
            label,
        )
        return response.result

    def preview_roundtrip(self) -> RoundTripPlan:
        """Preview both legs of a fixed 10 mm out-and-back test."""
        state = self.measured_state()
        safe_limits = self._solver.safe_joint_limits
        if np.any(state.positions < safe_limits[:, 0]) or np.any(
            state.positions > safe_limits[:, 1]
        ):
            raise SmokeTestError("真实关节仍在保守安全范围外；必须先恢复")
        start_matrix = self._solver.forward_kinematics(
            state.positions, "link7"
        )
        outbound_matrix = roundtrip_local_x_target(
            self._solver,
            state.positions,
            ROUNDTRIP_DISTANCE_M,
        )
        stamp = self.get_clock().now().to_msg()
        start_pose = matrix_to_pose_stamped(
            start_matrix, "base_link", stamp
        )
        outbound_pose = matrix_to_pose_stamped(
            outbound_matrix, "base_link", stamp
        )
        outbound = self._solve_pose_preview(
            outbound_pose,
            "Placo 10 mm 去程预览",
        )
        print("\n=== Placo 10 mm 往返预览：去程 ===")
        self._print_ik_result(outbound, ROUNDTRIP_DISTANCE_M)
        validate_smoke_ik(outbound)
        if outbound.max_joint_delta_rad > ROUNDTRIP_MAX_JOINT_DELTA_RAD:
            raise SmokeTestError("往返去程关节变化超过 0.08 rad")

        predicted_return = self._solver.solve(
            start_matrix,
            outbound.solution_joint_state.position,
            "link7",
        )
        if not predicted_return.success or int(predicted_return.error_code) != 0:
            raise SmokeTestError(
                f"往返回程离线预检失败：{predicted_return.message}"
            )
        predicted_repeat_error = float(np.max(np.abs(
            np.asarray(predicted_return.joint_positions) - state.positions
        )))
        print("\n=== Placo 10 mm 往返预览：预计回程 ===")
        print(
            f"残差={predicted_return.position_error_m * 1000.0:.3f} mm / "
            f"{np.degrees(predicted_return.orientation_error_rad):.4f}°"
        )
        print(
            f"最大关节变化={predicted_return.max_joint_delta_rad:.6f} rad, "
            f"预计回到起点关节误差={predicted_repeat_error:.6f} rad"
        )
        if (
            predicted_return.max_joint_delta_rad
            > ROUNDTRIP_MAX_JOINT_DELTA_RAD
            or predicted_repeat_error > ROUNDTRIP_MAX_RETURN_JOINT_ERROR_RAD
        ):
            raise SmokeTestError("预计回程不连续或不能稳定回到起点附近")
        return RoundTripPlan(state, start_pose, outbound_pose)

    def execute_roundtrip(self, plan: RoundTripPlan) -> None:
        """Execute an outbound move, re-solve from feedback, then return."""
        self.execute_placo(
            plan.outbound_pose,
            plan.start_state,
            precision_test=True,
            label="Placo 往返去程",
        )
        return_state = self.measured_state()
        return_result = self._solve_pose_preview(
            plan.start_pose,
            "Placo 10 mm 回程重新求解",
        )
        print("\n=== Placo 10 mm 往返：回程实时预览 ===")
        self._print_ik_result(return_result, -ROUNDTRIP_DISTANCE_M)
        validate_smoke_ik(return_result)
        if return_result.max_joint_delta_rad > ROUNDTRIP_MAX_JOINT_DELTA_RAD:
            raise SmokeTestError("实时回程关节变化超过 0.08 rad")
        self.execute_placo(
            plan.start_pose,
            return_state,
            precision_test=True,
            label="Placo 往返回程",
        )
        final_state = self.measured_state()
        joint_repeat_error = float(np.max(np.abs(
            final_state.positions - plan.start_state.positions
        )))
        print(
            "\n往返结束：相对测试起点的最大关节误差="
            f"{joint_repeat_error:.6f} rad"
        )
        if joint_repeat_error > ROUNDTRIP_MAX_RETURN_JOINT_ERROR_RAD:
            raise SmokeTestError("末端已回程，但关节重复误差超过 0.02 rad")

    def preview_center_ready(self) -> CenterReadyPlan:
        """Plan ten small Placo steps toward the verified ready pose."""
        state = self.measured_state()
        safe_limits = self._solver.safe_joint_limits
        if np.any(state.positions < safe_limits[:, 0]) or np.any(
            state.positions > safe_limits[:, 1]
        ):
            raise SmokeTestError("真实关节仍在保守安全范围外；必须先恢复")
        ready = np.asarray(READY_JOINT_POSITIONS, dtype=float)
        total_delta = float(np.max(np.abs(ready - state.positions)))
        print("\n=== Placo 分段进入 ready 姿态预览 ===")
        print(f"当前到 ready 的最大关节差={total_delta:.6f} rad")
        if total_delta <= CENTER_READY_FINAL_JOINT_TOLERANCE_RAD:
            print("已经位于 ready 邻域，不需要发送运动请求")
            return CenterReadyPlan(state, (), ())
        if total_delta > CENTER_READY_MAX_TOTAL_JOINT_DELTA_RAD:
            raise SmokeTestError(
                f"当前到 ready 的跨度 {total_delta:.4f} rad 超过 "
                "0.50 rad 专用上限"
            )
        trajectory = TrajectoryGenerator(joint_limits=safe_limits)
        predicted = state.positions.copy()
        target_poses = []
        predicted_positions = []
        stamp = self.get_clock().now().to_msg()
        print("段  残差(mm)  最大Δq(rad)  sigma_min  condition  轨迹(s)")
        for index in range(1, CENTER_READY_SEGMENTS + 1):
            fraction = index / CENTER_READY_SEGMENTS
            reference = state.positions + (
                ready - state.positions
            ) * fraction
            target_transform = self._solver.forward_kinematics(
                reference,
                "link7",
            )
            result = self._solver.solve(
                target_transform,
                predicted,
                "link7",
                posture_reference_joints=reference,
            )
            if not result.success:
                raise SmokeTestError(
                    f"ready 第 {index} 段 Placo 被拒绝：{result.message}"
                )
            if (
                result.max_joint_delta_rad
                > CENTER_READY_MAX_SEGMENT_JOINT_DELTA_RAD
            ):
                raise SmokeTestError(
                    f"ready 第 {index} 段关节变化 "
                    f"{result.max_joint_delta_rad:.6f} rad 超过 0.06 rad"
                )
            if result.sigma_min < 0.10 or result.condition_number > 20.0:
                raise SmokeTestError(
                    f"ready 第 {index} 段过于接近奇异点"
                )
            trajectory_result = trajectory.generate(
                predicted,
                result.joint_positions,
            )
            if not trajectory_result.success:
                raise SmokeTestError(
                    f"ready 第 {index} 段轨迹被拒绝："
                    f"{trajectory_result.message}"
                )
            print(
                f"{index:>2}/{CENTER_READY_SEGMENTS}  "
                f"{result.position_error_m * 1000.0:>8.3f}  "
                f"{result.max_joint_delta_rad:>12.6f}  "
                f"{result.sigma_min:>9.4f}  "
                f"{result.condition_number:>9.2f}  "
                f"{trajectory_result.duration_s:>7.3f}"
            )
            target_poses.append(matrix_to_pose_stamped(
                target_transform,
                "base_link",
                stamp,
            ))
            predicted = np.asarray(result.joint_positions, dtype=float)
            predicted_positions.append(tuple(float(value) for value in predicted))

        final_joint_error = float(np.max(np.abs(predicted - ready)))
        if final_joint_error > CENTER_READY_FINAL_JOINT_TOLERANCE_RAD:
            raise SmokeTestError(
                "分段路径末端没有进入 ready 关节邻域："
                f"{final_joint_error:.6f} rad"
            )
        print(
            "预计最终相对 ready 最大关节差="
            f"{final_joint_error:.6f} rad"
        )
        return CenterReadyPlan(
            state,
            tuple(target_poses),
            tuple(predicted_positions),
        )

    def execute_center_ready(self, plan: CenterReadyPlan) -> None:
        """Execute the prechecked ready path with live IK before every leg."""
        if not plan.target_poses:
            print("机械臂已经位于 ready 邻域；没有发送运动请求。")
            return
        for index, target_pose in enumerate(plan.target_poses, start=1):
            state = self.measured_state()
            posture_reference = plan.predicted_joint_positions[index - 1]
            result = self._solve_pose_preview(
                target_pose,
                f"ready 第 {index}/{len(plan.target_poses)} 段实时重算",
                posture_reference=posture_reference,
            )
            self._print_axis_ik_result(
                f"ready 第 {index}/{len(plan.target_poses)} 段",
                result,
            )
            self._validate_axis_ik(result, f"ready 第 {index} 段")
            if (
                result.max_joint_delta_rad
                > CENTER_READY_MAX_SEGMENT_JOINT_DELTA_RAD
            ):
                raise SmokeTestError(
                    f"ready 第 {index} 段实时关节变化超过 0.06 rad"
                )

            self.execute_placo(
                target_pose,
                state,
                precision_test=True,
                label=f"Placo ready 第 {index}/{len(plan.target_poses)} 段",
                posture_reference=posture_reference,
            )

        final_state = self.measured_state()
        ready = np.asarray(READY_JOINT_POSITIONS, dtype=float)
        joint_error = float(np.max(np.abs(final_state.positions - ready)))
        actual_pose = self._solver.forward_kinematics(
            final_state.positions,
            "link7",
        )
        ready_pose = self._solver.forward_kinematics(ready, "link7")
        position_error, orientation_error = pose_error(
            ready_pose,
            actual_pose,
        )
        print("\n=== ready 中心化结果 ===")
        print(f"最大关节差={joint_error:.6f} rad")
        print(
            f"末端位姿差={position_error * 1000.0:.3f} mm / "
            f"{np.degrees(orientation_error):.4f}°"
        )
        if joint_error > CENTER_READY_FINAL_JOINT_TOLERANCE_RAD:
            raise SmokeTestError("最终没有进入 ready 的 0.05 rad 关节邻域")
        if (
            position_error > SMOKE_MAX_FINAL_POSITION_ERROR_M
            or orientation_error > SMOKE_MAX_FINAL_ORIENTATION_ERROR_RAD
        ):
            raise SmokeTestError("最终末端位姿没有进入 ready 精度范围")

    @staticmethod
    def _print_axis_ik_result(label: str, result: IKResultMsg) -> None:
        """Print the ROS SolveIK evidence for one fixed axis target."""
        print(f"\n--- {label} ---")
        print(f"code={result.code}, success={result.success}")
        print(f"说明：{result.reason}")
        print(
            f"残差={result.position_error_m * 1000.0:.3f} mm / "
            f"{np.degrees(result.orientation_error_rad):.4f}°，"
            f"最大关节变化={result.max_joint_delta_rad:.6f} rad"
        )
        print(
            f"IK 耗时={result.solve_time_ms:.3f} ms，"
            f"sigma_min={result.sigma_min:.4f}，"
            f"condition={result.condition_number:.2f}"
        )

    @staticmethod
    def _validate_axis_ik(result: IKResultMsg, label: str) -> None:
        """Apply the precision-mode limits to one axis-suite leg."""
        try:
            validate_smoke_ik(result)
        except SmokeTestError as error:
            raise SmokeTestError(f"{label}：{error}") from error
        if result.max_joint_delta_rad > AXIS_MAX_JOINT_DELTA_RAD:
            raise SmokeTestError(
                f"{label}：关节变化超过 0.08 rad"
            )

    def preview_axis_suite(self) -> AxisSuiteRealPlan:
        """Preview the locked direction profile without opening gates."""
        state = self.measured_state()
        safe_limits = self._solver.safe_joint_limits
        if np.any(state.positions < safe_limits[:, 0]) or np.any(
            state.positions > safe_limits[:, 1]
        ):
            raise SmokeTestError("真实关节仍在保守安全范围外；必须先恢复")
        ready_error = float(np.max(np.abs(
            state.positions - np.asarray(READY_JOINT_POSITIONS, dtype=float)
        )))
        if ready_error > AXIS_SUITE_READY_JOINT_TOLERANCE_RAD:
            raise SmokeTestError(
                "当前虽在安全范围，但不在六方向测试的 ready 中心邻域；"
                f"最大关节差={ready_error:.6f} rad。请先运行 center-ready 预览。"
            )

        trajectory = TrajectoryGenerator(joint_limits=safe_limits)
        suite = plan_axis_suite(self._solver, trajectory, state.positions)
        print_axis_suite(suite)
        if not suite.passed:
            raise SmokeTestError("六方向离线预检未全部通过，禁止真机执行")

        stamp = self.get_clock().now().to_msg()
        preview_results = []
        print("\n=== 六方向 ROS 2 SolveIK 预览（仍不运动） ===")
        for case in suite.cases:
            distance_mm = case.direction.distance_m * 1000.0
            target_pose = matrix_to_pose_stamped(
                case.target_transform,
                "base_link",
                stamp,
            )
            result = self._solve_pose_preview(
                target_pose,
                f"{case.direction.label} {distance_mm:.0f} mm ROS 预览",
            )
            self._print_axis_ik_result(
                f"{case.direction.label} {distance_mm:.0f} mm 去程",
                result,
            )
            self._validate_axis_ik(
                result,
                f"{case.direction.label} 去程预览",
            )
            direct_solution = np.asarray(
                case.outbound_ik.joint_positions,
                dtype=float,
            )
            service_solution = np.asarray(
                result.solution_joint_state.position,
                dtype=float,
            )
            solution_difference = float(np.max(np.abs(
                service_solution - direct_solution
            )))
            if solution_difference > AXIS_MAX_RETURN_JOINT_ERROR_RAD:
                raise SmokeTestError(
                    f"{case.direction.label}：直接 Placo 与 ROS 2 解相差 "
                    f"{solution_difference:.6f} rad"
                )
            preview_results.append(result)

        return AxisSuiteRealPlan(
            start_state=state,
            suite_plan=suite,
            preview_results=tuple(preview_results),
        )

    @staticmethod
    def save_axis_preview(
        plan: AxisSuiteRealPlan,
        output_directory: Path,
    ) -> tuple[AxisReportWriter, list[dict]]:
        """Save the no-motion real preview before optional execution."""
        writer = AxisReportWriter(output_directory, "real")
        rows = real_axis_preview_rows(plan)
        writer.write(
            rows,
            anchor_joints=plan.start_state.positions,
            completed=False,
            passed=False,
            message="六方向真机预览通过，尚未执行",
        )
        print(f"\nCSV：{writer.csv_path}")
        print(f"JSON：{writer.json_path}")
        return writer, rows

    def execute_axis_suite(
        self,
        plan: AxisSuiteRealPlan,
        writer: AxisReportWriter,
        rows: list[dict],
    ) -> None:
        """Execute six pairs, each frozen from its latest real anchor."""
        if len(rows) != len(AXIS_DIRECTIONS):
            raise SmokeTestError("六方向报告行数不完整")
        trajectory = TrajectoryGenerator(
            joint_limits=self._solver.safe_joint_limits
        )
        for index, preview_case in enumerate(plan.suite_plan.cases):
            row = rows[index]
            direction = preview_case.direction
            label = direction.label
            distance_mm = direction.distance_m * 1000.0
            try:
                outbound_state = self.measured_state()
                pair_anchor_transform = self._solver.forward_kinematics(
                    outbound_state.positions,
                    "link7",
                )
                live_case = plan_axis_case(
                    self._solver,
                    trajectory,
                    outbound_state.positions,
                    direction,
                    pair_anchor_transform,
                )
                if not live_case.accepted:
                    raise SmokeTestError(
                        f"{label} 组开始时的直接 Placo 预检失败："
                        f"{live_case.reason}"
                    )
                row.clear()
                row.update(case_to_row(
                    index,
                    live_case,
                    outbound_state.positions,
                ))
                stamp = self.get_clock().now().to_msg()
                target_pose = matrix_to_pose_stamped(
                    live_case.target_transform,
                    "base_link",
                    stamp,
                )
                pair_anchor_pose = matrix_to_pose_stamped(
                    pair_anchor_transform,
                    "base_link",
                    stamp,
                )

                outbound_preview = self._solve_pose_preview(
                    target_pose,
                    f"{label} {distance_mm:.0f} mm 去程实时重算",
                )
                self._print_axis_ik_result(
                    f"{label} {distance_mm:.0f} mm 去程实时重算",
                    outbound_preview,
                )
                self._validate_axis_ik(
                    outbound_preview,
                    f"{label} 去程实时重算",
                )
                outbound_result = self.execute_placo(
                    target_pose,
                    outbound_state,
                    precision_test=True,
                    label=f"六方向 {label} 去程",
                )
                _put_ik_in_report(
                    row,
                    "outbound",
                    outbound_result.ik_result,
                )
                row["outbound_final_position_error_m"] = float(
                    outbound_result.final_position_error_m
                )
                row["outbound_final_orientation_error_rad"] = float(
                    outbound_result.final_orientation_error_rad
                )

                return_state = self.measured_state()
                return_preview = self._solve_pose_preview(
                    pair_anchor_pose,
                    f"{label} {distance_mm:.0f} mm 回程实时重算",
                )
                self._print_axis_ik_result(
                    f"{label} {distance_mm:.0f} mm 回程实时重算",
                    return_preview,
                )
                self._validate_axis_ik(
                    return_preview,
                    f"{label} 回程实时重算",
                )
                return_result = self.execute_placo(
                    pair_anchor_pose,
                    return_state,
                    precision_test=True,
                    label=f"六方向 {label} 回程",
                )
                _put_ik_in_report(row, "return", return_result.ik_result)
                row["return_final_position_error_m"] = float(
                    return_result.final_position_error_m
                )
                row["return_final_orientation_error_rad"] = float(
                    return_result.final_orientation_error_rad
                )

                final_state = self.measured_state()
                return_joint_error = float(np.max(np.abs(
                    final_state.positions - outbound_state.positions
                )))
                row["return_joint_error_rad"] = return_joint_error
                if return_joint_error > AXIS_MAX_RETURN_JOINT_ERROR_RAD:
                    raise SmokeTestError(
                        f"{label} 回程关节误差 {return_joint_error:.6f} rad "
                        "超过 0.02 rad"
                    )
                row["accepted"] = True
                row["reason"] = "真机去程和回程均通过"
                print(
                    f"\n{label} {distance_mm:.0f} mm 方向完成："
                    "回到本组锚点关节误差="
                    f"{return_joint_error:.6f} rad"
                )
            except BaseException as error:
                row["accepted"] = False
                detail = str(error).strip() or type(error).__name__
                row["reason"] = f"真机执行停止：{detail}"
                writer.write(
                    rows,
                    anchor_joints=plan.start_state.positions,
                    completed=False,
                    passed=False,
                    message=f"在 {label} 方向停止：{detail}",
                )
                print(
                    f"\n已保存中途报告：{writer.json_path}",
                    file=sys.stderr,
                )
                raise

            writer.write(
                rows,
                anchor_joints=plan.start_state.positions,
                completed=False,
                passed=False,
                message=f"已完成 {index + 1}/{len(AXIS_DIRECTIONS)} 个方向",
            )

        writer.write(
            rows,
            anchor_joints=plan.start_state.positions,
            completed=True,
            passed=True,
            message="六方向分级位移真机往返全部通过",
        )
        print(f"\n最终 CSV：{writer.csv_path}")
        print(f"最终 JSON：{writer.json_path}")

    def cancel_active_goal(self) -> None:
        """Cancel an active goal; the physical E-stop remains primary."""
        goal_handle = self._active_goal_handle
        result_future = self._active_result_future
        if goal_handle is None:
            return
        self._cancel_goal_and_wait(goal_handle, result_future, "当前动作")
        if self._active_goal_handle is goal_handle:
            self._active_goal_handle = None
            self._active_result_future = None


def _argument_parser() -> argparse.ArgumentParser:
    """Create the intentionally small first-hardware-test CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute guarded NERO recovery, fixed 15 mm Placo "
            "smoke, 10 mm round-trip, ready centering, and locked "
            "six-direction tests. "
            "This tool never enables or disables motors by itself."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    recover = subparsers.add_parser(
        "recover", help="preview or execute inward-only safe-limit recovery"
    )
    recover.add_argument(
        "--execute",
        action="store_true",
        help="request physical recovery after typing RECOVER",
    )
    placo = subparsers.add_parser(
        "placo", help="preview or execute the fixed link7-local 15 mm move"
    )
    placo.add_argument(
        "--execute",
        action="store_true",
        help="request physical MoveToPose after typing PLACO",
    )
    roundtrip = subparsers.add_parser(
        "roundtrip",
        help="preview or execute one fixed link7-local +X 10 mm round trip",
    )
    roundtrip.add_argument(
        "--execute",
        action="store_true",
        help="request the supervised out-and-back test after typing ROUNDTRIP",
    )
    center_ready = subparsers.add_parser(
        "center-ready",
        help="preview or execute ten Placo steps to the verified ready pose",
    )
    center_ready.add_argument(
        "--execute",
        action="store_true",
        help="request the ten guarded legs after typing CENTER_READY",
    )
    axis_suite = subparsers.add_parser(
        "axis-suite",
        help="preview or execute locked 15 mm pairs on six axes",
    )
    axis_suite.add_argument(
        "--execute",
        action="store_true",
        help="request all twelve fixed legs after typing AXIS_SUITE",
    )
    axis_suite.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path.home()
            / ".ros"
            / "strawberry_nero_control"
            / "axis_suite"
        ),
        help="directory for the CSV and JSON validation report",
    )
    return parser


def run_command(
    node: NeroRealSmokeTest,
    arguments,
    input_function: Callable[[str], str] = input,
) -> int:
    """Run preview first; executions temporarily open only software gates."""
    if arguments.command == "recover":
        preview = node.preview_recovery()
        if not arguments.execute:
            print("\n预览完成：没有发送任何真机运动请求。")
            return 0
        if recovery_result_allows_placo(preview):
            print("机械臂已经位于安全范围，不需要再次恢复。")
            return 0
        require_confirmation("RECOVER", input_function)
        node.execute_recovery()
        print("\n安全区恢复通过；现在可以进行下一项 Placo 无运动预览。")
        print(POST_MOTION_SAFETY_NOTICE)
        return 0

    safe_preview = node.preview_recovery()
    if not (
        safe_preview.success
        and safe_preview.code == RecoverToSafe.Result.ALREADY_SAFE
        and not safe_preview.executed
        and safe_preview.robot_is_safe
    ):
        raise SmokeTestError(
            "恢复预览没有确认 ALREADY_SAFE；禁止进入 Placo 阶段"
        )
    if arguments.command == "placo":
        distance_m = SMOKE_DISTANCE_M
        target_pose, _, planned_state = node.preview_placo(distance_m)
        if not arguments.execute:
            print("\n预览完成：Placo 已求解，但没有发送真机运动请求。")
            return 0
        require_confirmation("PLACO", input_function)
        node.execute_placo(target_pose, planned_state)
        print("\n15 mm Placo 真机 smoke test 通过。")
        print(POST_MOTION_SAFETY_NOTICE)
        return 0

    if arguments.command == "roundtrip":
        plan = node.preview_roundtrip()
        if not arguments.execute:
            print("\n往返预览完成：去程和预计回程均通过，未发送运动请求。")
            return 0
        require_confirmation("ROUNDTRIP", input_function)
        node.execute_roundtrip(plan)
        print("\n10 mm Placo 真机往返测试通过。")
        print(POST_MOTION_SAFETY_NOTICE)
        return 0

    if arguments.command == "center-ready":
        plan = node.preview_center_ready()
        if not arguments.execute:
            print("\nready 分段预览完成：没有发送真机运动请求。")
            return 0
        if not plan.target_poses:
            print("\n机械臂已经位于 ready 邻域，无需中心化运动。")
            return 0
        require_confirmation("CENTER_READY", input_function)
        node.execute_center_ready(plan)
        print("\nPlaco 分段进入 ready 姿态测试通过。")
        print(POST_MOTION_SAFETY_NOTICE)
        return 0

    plan = node.preview_axis_suite()
    writer, rows = node.save_axis_preview(plan, arguments.output_dir)
    if not arguments.execute:
        print(
            "\n六方向预览完成：12 段均已预检，但没有发送真机运动请求。"
        )
        return 0
    require_confirmation("AXIS_SUITE", input_function)
    node.execute_axis_suite(plan, writer, rows)
    print("\n六方向分级位移 Placo 真机往返测试通过。")
    print(POST_MOTION_SAFETY_NOTICE)
    return 0


def main(args: Optional[Sequence[str]] = None) -> int:
    """Console entry point with safe cleanup and non-zero refusal codes."""
    raw_args = list(sys.argv if args is None else [sys.argv[0], *args])
    cli_args = remove_ros_args(args=raw_args)[1:]
    arguments = _argument_parser().parse_args(cli_args)
    rclpy.init(
        args=raw_args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    try:
        node = NeroRealSmokeTest()
        return run_command(node, arguments)
    except (SmokeTestError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n操作已停止或被拒绝：{error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print(
            "\n操作已取消；若机械臂正在运动，请观察其保持状态，"
            "异常时由观察人员在工作区外切断控制箱电源（机械臂可能下落）。",
            file=sys.stderr,
        )
        return 130
    finally:
        if node is not None:
            node.cancel_active_goal()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
