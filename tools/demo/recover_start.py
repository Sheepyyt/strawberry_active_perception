#!/usr/bin/env python3
"""Preview and, when authorized, perform the controller's inward-only recovery.

No joint target is supplied by this script. RecoverToSafe derives the minimum
inward correction from fresh feedback; it never goes to a fixed ready pose.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def abort_requested(args) -> bool:
    path = getattr(args, "abort_file", None)
    if path is None:
        return False
    try:
        return path.stat().st_mtime_ns >= args.abort_not_before_ns
    except FileNotFoundError:
        return False


def recover(node, args, result_type, serialize) -> dict:
    """Reuse existing guarded Actions; persist each result before continuing."""
    if args.execute and not args.workspace_clear:
        raise RuntimeError("执行关节向内恢复需要 --execute --workspace-clear")
    if abort_requested(args):
        raise RuntimeError("本轮已请求停止，取消关节恢复")
    preview = node.preview_recovery()
    write_json(args.output_dir / "preview.json", serialize(preview))
    if preview.code == result_type.ALREADY_SAFE:
        if not (preview.success and preview.robot_is_safe and not preview.executed):
            raise RuntimeError("ALREADY_SAFE 字段不一致，禁止继续")
        return {"status": "already_safe", "executed": False, "robot_is_safe": True}
    if not (preview.code == result_type.PREVIEW_READY and preview.success
            and not preview.executed and not preview.robot_is_safe):
        raise RuntimeError("恢复预览未通过，禁止执行")
    if not args.execute:
        return {"status": "preview_ready", "executed": False, "robot_is_safe": False}
    if abort_requested(args):
        raise RuntimeError("本轮已请求停止，取消关节恢复")
    execution = node.execute_recovery()
    write_json(args.output_dir / "execution.json", serialize(execution))
    if not (execution.success and execution.robot_is_safe and (
            (execution.code == result_type.SUCCESS and execution.executed)
            or (execution.code == result_type.ALREADY_SAFE and not execution.executed))):
        raise RuntimeError("恢复执行未确认关节已进入安全范围")
    post = node.preview_recovery()
    write_json(args.output_dir / "post_preview.json", serialize(post))
    if not (post.code == result_type.ALREADY_SAFE and post.success
            and post.robot_is_safe and not post.executed):
        raise RuntimeError("恢复后新反馈仍未确认 ALREADY_SAFE，禁止继续 NBV")
    return {"status": "recovered", "executed": bool(execution.executed), "robot_is_safe": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workspace-clear", action="store_true")
    parser.add_argument("--abort-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--abort-not-before-ns", type=int, default=time.time_ns(), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute and not args.workspace_clear:
        parser.error("执行需要已确认现场清空并使用 --workspace-clear")

    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from strawberry_nero_control.real_smoke_test import NeroRealSmokeTest
    from strawberry_nero_interfaces.action import RecoverToSafe

    # Keep ROS alive during Python interruption so the established client can
    # cancel an active Action and wait for its explicit terminal result.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupt)
    report = {"schema": "strawberry_demo_start_recovery/v1", "success": False,
              "created_ns": time.time_ns(), "execute_requested": args.execute,
              "workspace_clearance_confirmed": args.workspace_clear}
    node = None
    watcher_stop = threading.Event()

    def watch_abort():
        while not watcher_stop.wait(.1):
            if abort_requested(args):
                os.kill(os.getpid(), signal.SIGINT)
                return

    try:
        if args.abort_file:
            threading.Thread(target=watch_abort, daemon=True).start()
        node = NeroRealSmokeTest("strawberry_demo_start_recovery")
        report.update(recover(node, args, RecoverToSafe.Result, message_to_ordereddict))
        report["success"] = True
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error) or type(error).__name__)
    finally:
        watcher_stop.set()
        if node is not None:
            node.cancel_active_goal()
            if args.execute:
                gate_errors = []
                for client, name in ((node._controller_gate_client, "/strawberry_nero/enable_execution"),
                                     (node._driver_gate_client, "/control_enable")):
                    try:
                        node._set_gate(client, name, False)
                    except Exception as error:
                        gate_errors.append(str(error))
                if gate_errors:
                    report.update(success=False, gate_errors=gate_errors)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
