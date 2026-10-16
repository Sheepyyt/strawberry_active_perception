#!/usr/bin/env python3
"""Small typed ROS probes used by demo_runtime; no motion-goal publisher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def cancel_motion_goals(node, cancel_goal, spin_until_complete, close_gates, report):
    """Cancel ordinary and startup-recovery goals, then close both gates."""
    failures = []
    report["cancel_actions"] = {}
    report["goals_canceling"] = 0
    try:
        for action in ("move_to_pose", "recover_to_safe"):
            client = None
            result = report["cancel_actions"][action] = {}
            try:
                name = f"/strawberry_nero/{action}/_action/cancel_goal"
                client = node.create_client(cancel_goal, name)
                if not client.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError("取消服务未就绪，不能确认正常停止；保留驱动")
                # Zero UUID + zero stamp cancels every goal of this action.
                future = client.call_async(cancel_goal.Request())
                spin_until_complete(node, future, timeout_sec=5.0)
                if not future.done() or future.result() is None:
                    raise RuntimeError("取消动作未返回；不能据此认为机械臂静止")
                response = future.result()
                result.update(return_code=int(response.return_code),
                              goals_canceling=len(response.goals_canceling))
                report["goals_canceling"] += result["goals_canceling"]
                if action == "move_to_pose":
                    report["cancel_return_code"] = result["return_code"]
                if result["return_code"] != 0:
                    raise RuntimeError(f"取消请求未成功，返回码 {result['return_code']}")
            except Exception as error:
                result["error"] = str(error)
                failures.append(f"{action}: {error}")
            finally:
                if client is not None:
                    node.destroy_client(client)
    finally:
        try:
            close_gates()
        except Exception as error:
            failures.append(str(error))
    if failures:
        raise RuntimeError("; ".join(failures))


def close_driver_gate(call, set_bool, report):
    """Close only the driver's gate after the owned controller has exited."""
    response = call(set_bool, "/control_enable", set_bool.Request(data=False), 5.0)
    report["/control_enable"] = response.message
    report["driver_gate_false_ack_count"] = report.get("driver_gate_false_ack_count", 0) + 1


def disable_driver(call, set_bool, report):
    """Disable through the surviving driver, requiring both gate-close ACKs."""
    close_driver_gate(call, set_bool, report)
    try:
        response = call(set_bool, "/enable_agx_arm", set_bool.Request(data=False), 12.0)
        report["message"] = response.message
        report["driver_disable_acknowledged"] = True
    finally:
        close_driver_gate(call, set_bool, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["ready", "close-gates", "close-driver-gate",
                                             "enable", "disable", "disable-driver", "halt", "joints"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scene", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import SetBool

    rclpy.init()
    node = Node("strawberry_demo_probe")
    report = {"operation": args.operation, "success": False, "time_ns": time.time_ns()}

    def call(service_type, name, request, timeout=10.0):
        client = node.create_client(service_type, name)
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"服务未就绪: {name}")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            if not future.done() or future.result() is None:
                raise RuntimeError(f"服务无响应: {name}")
            response = future.result()
            if not response.success:
                raise RuntimeError(f"{name}: {getattr(response, 'message', getattr(response, 'reason', 'failed'))}")
            return response
        finally:
            node.destroy_client(client)

    def close_gates():
        failures = []
        for name in ("/control_enable", "/strawberry_nero/enable_execution"):
            try:
                response = call(SetBool, name, SetBool.Request(data=False), 5.0)
                report[name] = response.message
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError("; ".join(failures))

    try:
        if args.operation == "ready":
            wanted = {"/control_enable", "/strawberry_nero/enable_execution", "/enable_agx_arm",
                      "/strawberry/perception/capture_observation", "/strawberry/nbv/configure",
                      "/strawberry_nero/solve_ik"}
            wanted_nodes = {"strawberry_learned_mask", "handeye_read_only_robot_state_publisher"}
            deadline = time.monotonic() + args.timeout
            missing = wanted
            missing_nodes = wanted_nodes
            while time.monotonic() < deadline:
                missing = wanted - {name for name, _ in node.get_service_names_and_types()}
                names = {name for name, _ in node.get_node_names_and_namespaces()}
                missing_nodes = wanted_nodes - names
                if not missing and not missing_nodes:
                    break
                rclpy.spin_once(node, timeout_sec=0.2)
            else:
                raise RuntimeError(f"启动超时，缺少服务 {sorted(missing)}，缺少节点 {sorted(missing_nodes)}")
            report["services"] = sorted(wanted)
            report["required_nodes"] = sorted(wanted_nodes)
            close_gates()
        elif args.operation == "halt":
            from action_msgs.srv import CancelGoal
            from sensor_msgs.msg import JointState
            from rclpy.qos import qos_profile_sensor_data
            import numpy as np

            cancel_motion_goals(node, CancelGoal, rclpy.spin_until_future_complete,
                                close_gates, report)
            if report["goals_canceling"]:
                report["note"] = "取消正在执行的动作会触发现有控制器的电子阻尼停止并锁门，机械臂可能缓慢下降，请现场托护"
                print(report["note"], flush=True)
            samples = []
            from strawberry_active_perception_bridge.real_nbv_contract import ordered_joint_positions

            def sample(message):
                try:
                    joints = ordered_joint_positions(list(message.name), list(message.position), label="stop verification")
                    if np.isfinite(joints).all():
                        samples.append((time.monotonic(), joints))
                except (ValueError, TypeError):
                    pass

            node.create_subscription(JointState, "/feedback/joint_states",
                                     sample,
                                     qos_profile_sensor_data)
            deadline = time.monotonic() + args.timeout
            started = time.monotonic()
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.1)
                recent = [(stamp, joints) for stamp, joints in samples if stamp >= time.monotonic() - 1.5 and len(joints) == 7]
                if len(recent) >= 5 and recent[-1][0] - recent[0][0] >= 1.0 and time.monotonic() - recent[-1][0] <= .3:
                    span = np.ptp(np.asarray([joints for _, joints in recent]), axis=0)
                    if float(span.max()) <= .003:
                        report["stationary_position_span_rad"] = span.tolist()
                        report["stationary_window_sec"] = recent[-1][0] - recent[0][0]
                        report["stationarity_verified"] = True
                        break
                # A disabled v1.11 cold start has no joint push. This is not a
                # stationary proof; allow stopping perception while retaining
                # the driver, and leave disable to its checked service.
                if not samples and time.monotonic() - started > 4.0 and report["goals_canceling"] == 0:
                    report["stationarity_verified"] = False
                    report["note"] = "没有关节反馈、没有待取消目标；保留驱动和电机状态，未证明机械臂静止"
                    break
            else:
                raise RuntimeError("关节位置序列未证明静止；保留驱动，停止后续操作")
        elif args.operation == "close-gates":
            close_gates()
        elif args.operation == "close-driver-gate":
            close_driver_gate(call, SetBool, report)
        elif args.operation == "disable-driver":
            disable_driver(call, SetBool, report)
        elif args.operation in ("enable", "disable"):
            close_gates()
            response = call(SetBool, "/enable_agx_arm", SetBool.Request(data=args.operation == "enable"), 12.0)
            report["message"] = response.message
            close_gates()
        elif args.operation == "joints":
            from sensor_msgs.msg import JointState
            from rclpy.qos import qos_profile_sensor_data
            received = []
            node.create_subscription(JointState, "/feedback/joint_states", received.append, qos_profile_sensor_data)
            deadline = time.monotonic() + args.timeout
            while not received and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if not received or len(received[-1].position) != 7:
                raise RuntimeError("未收到新的七关节反馈；失能的 v1.11 固件可能不推送反馈")
            report["joint_names"] = list(received[-1].name)
            report["joint_positions_rad"] = list(received[-1].position)
        report["success"] = True
    except Exception as error:
        report["error"] = str(error)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
