#!/usr/bin/env python3
"""Supervised one-terminal demo orchestration; the supervisor owns all motion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

import yaml

from tools.demo.result_report import build_overview, render_index, render_text

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "results/demo/.runtime"
STATE = RUNTIME / "state.json"
ABORT = RUNTIME / "abort_requested"
INVOCATION_START_NS = time.time_ns()
PYTHON = "/usr/bin/python3"
SUPERVISOR_CONFIG = ROOT / "perception_ws/src/strawberry_active_perception_bridge/config/real_nbv_supervisor_large_step.yaml"
MASK_CONFIG = ROOT / "perception_ws/src/strawberry_learned_mask/config/yolo11m_strawberry.yaml"
MASK_OVERLAY = ROOT / "perception_ws/src/strawberry_learned_mask/config/real_nbv_use_learned_mask.yaml"
NBV_CONFIG = ROOT / "perception_ws/src/strawberry_gradient_nbv/config/real_nbv.yaml"
ROBOT_PROFILE = ROOT / "nero_ws/src/strawberry_nero_control/config/large_nbv_experiment.yaml"
REQUIRED_SERVICES = ("robot", "tf", "camera", "mask", "nbv")
TF_NODE = "/handeye_read_only_robot_state_publisher"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def profile_max_step() -> float:
    """Use the planner profile's search radius, without a separate demo cap."""
    config = yaml.safe_load(SUPERVISOR_CONFIG.read_text())
    value = float(config["real_nbv_supervisor"]["ros__parameters"]["max_step_m"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("算法配置 max_step_m 必须是有限正数")
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"services": {}}


def update_state(**changes) -> dict:
    state = read_state()
    state.update(changes)
    write_json(STATE, state)
    return state


def pid_identity(pid: int) -> str | None:
    try:
        # /proc/PID/stat's command may contain spaces; field 22 is starttime.
        content = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if content[0] == "Z" else content[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def alive(record: dict | None) -> bool:
    return bool(record and record.get("start_ticks") and record.get("boot_id", BOOT_ID) == BOOT_ID
                and pid_identity(record["pid"]) == record["start_ticks"])


def check_abort() -> None:
    try:
        if ABORT.stat().st_mtime_ns >= INVOCATION_START_NS:
            raise RuntimeError("操作者已请求停止，本轮不再继续")
    except FileNotFoundError:
        pass


def group_members(record: dict | None) -> dict[int, str]:
    """Track children even if ros2 launch exits before its child processes."""
    if not record or not record.get("start_ticks"):
        return {}
    if record.get("boot_id", BOOT_ID) != BOOT_ID:
        return {}
    leader_identity = pid_identity(record["pid"])
    if leader_identity is not None and leader_identity != record["start_ticks"]:
        return {}
    members = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == record["pid"] and int(fields[3]) == record["pid"] and int(fields[19]) >= int(record["start_ticks"]):
                members[int(path.name)] = fields[19]
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            pass
    return members


def owned_processes(state: dict, name: str) -> bool:
    """Ownership survives the launcher's exit; health still requires alive()."""
    return bool(group_members(state["services"].get(name)))


def controller_present(state: dict) -> bool:
    if owned_processes(state, "controller"):
        return True
    for pid in group_members(state["services"].get("robot")):
        try:
            arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(os.fsdecode(arg)).name == "nero_control_node" for arg in arguments if arg):
            return True
    return False


def robot_identity(record: dict | None) -> dict | None:
    if not record or not record.get("start_ticks"):
        return None
    return {"pid": record["pid"], "start_ticks": record["start_ticks"],
            "boot_id": record.get("boot_id", BOOT_ID)}


def disable_confirmed(state: dict) -> bool:
    receipt = state.get("shutdown") or {}
    identity = robot_identity(state["services"].get("robot"))
    return bool(identity and receipt.get("phase") == "disable_confirmed"
                and receipt.get("robot") == identity and identity["boot_id"] == BOOT_ID
                and state.get("motor_enable_requested") is False)


def finish_robot_shutdown() -> None:
    """Resume process cleanup only after a durable, identity-bound disable ack."""
    state = read_state()
    if not disable_confirmed(state):
        raise RuntimeError("没有与本轮驱动绑定的失能确认，不能强制清理机械臂进程")
    print("已确认电机失能，正在退出本轮控制器和驱动；驱动退出最多等待 12 秒。", flush=True)
    stop_service("controller", allow_kill=True)
    stop_service("robot", allow_kill=True)
    remaining = [name for name in ("controller", "robot") if owned_processes(read_state(), name)]
    if remaining:
        raise RuntimeError(f"失能后的进程仍未退出: {remaining}")
    receipt = dict(state["shutdown"], phase="complete", completed_at_ns=time.time_ns())
    update_state(shutdown=receipt, motor_enable_requested=False)


def reconcile_shutdown() -> None:
    state = read_state()
    if disable_confirmed(state):
        print("检测到上次已失能但退出尚未完成，继续清理本脚本所属进程。", flush=True)
        finish_robot_shutdown()


def halt_owned_robot(state: dict, output: Path) -> None:
    if owned_processes(state, "robot"):
        operation = "halt" if controller_present(state) else "close-driver-gate"
        probe(operation, output=output, timeout=15)


def interrupt_supervisor() -> None:
    record = read_state()["services"].get("supervisor")
    if group_members(record):
        os.killpg(record["pid"], signal.SIGINT)


def service_command(name: str, directory: Path) -> list[str]:
    if name == "robot":
        return ["ros2", "launch", "strawberry_nero_control", "real.launch.py",
                f"profile_config_file:={ROBOT_PROFILE}", "can_port:=can0", "speed_percent:=10",
                "startup_enable:=false", "precision_test_mode:=true", "first_motion_test_mode:=false",
                "allow_limit_recovery_execution:=true", "launch_rviz:=false"]
    if name == "camera":
        return ["ros2", "launch", "strawberry_observation", "gemini2xl_observation.launch.py"]
    if name == "tf":
        return [PYTHON, str(ROOT / "tools/calibration/read_only_nero_tf.launch.py")]
    if name == "mask":
        return [str(ROOT / ".venv-mask/bin/python"), "-m", "strawberry_learned_mask.ros_node",
                "--ros-args", "--params-file", str(MASK_CONFIG), "-p", f"audit_directory:={directory / 'mask_audit'}"]
    if name == "nbv":
        return [str(ROOT / ".venv-nbv/bin/python"), "-m", "strawberry_gradient_nbv.ros_node",
                "--ros-args", "--params-file", str(NBV_CONFIG), "-p", f"map_snapshot_directory:={directory / 'maps'}"]
    raise ValueError(name)


def spawn(name: str, command: list[str], directory: Path) -> dict:
    check_abort()
    state = read_state()
    if owned_processes(state, name):
        raise RuntimeError(f"{name} 已运行，请勿重复启动")
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{name}.log"
    with log.open("ab", buffering=0) as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    record = {"pid": process.pid, "start_ticks": pid_identity(process.pid), "boot_id": BOOT_ID, "command": command,
              "log": str(log), "started_at": datetime.now(timezone.utc).isoformat()}
    state = read_state()
    state["services"][name] = record
    if name == "robot":
        state.update(shutdown=None, motor_enable_requested=False)
    write_json(STATE, state)
    print(f"启动 {name}，日志: {log.relative_to(ROOT)}", flush=True)
    return record


def stop_service(name: str, *, allow_kill=False) -> None:
    state = read_state()
    record = state["services"].get(name)
    if group_members(record):
        # launch handles TERM; Python ROS nodes handle INT and flush evidence.
        sig = signal.SIGTERM if name in ("robot", "camera", "tf") else signal.SIGINT
        try:
            os.killpg(record["pid"], sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 12
        while group_members(record) and time.monotonic() < deadline:
            time.sleep(.1)
        if group_members(record) and allow_kill:
            # Allowed only after this invocation positively confirmed disable.
            print(f"{name} 尚有已确认失能后的残留进程，清理本轮进程组。", flush=True)
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 3
            while group_members(record) and time.monotonic() < deadline:
                time.sleep(.1)
        if group_members(record):
            raise RuntimeError(f"{name} 未正常退出，保留进程记录；检查 {record['log']}")
    state = read_state()
    state["services"].pop(name, None)
    write_json(STATE, state)


def command(command: list[str], *, log: Path | None = None, timeout: float = 60, check=True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout)
    if check and result.returncode:
        raise RuntimeError(result.stdout[-5000:] or f"命令失败: {command[0]}")
    return result


def probe(operation: str, *, output: Path | None = None, timeout=30) -> None:
    args = [PYTHON, str(ROOT / "tools/demo/ros_probe.py"), operation, "--timeout", str(timeout)]
    if output:
        args += ["--output", str(output)]
    result = command(args, timeout=timeout + 25)
    print(result.stdout.strip(), flush=True)


def can_ready() -> bool:
    result = command(["ip", "-details", "link", "show", "can0"], check=False)
    return result.returncode == 0 and "UP" in result.stdout.partition(">")[0] and "bitrate 1000000" in result.stdout and "can state ERROR-ACTIVE" in result.stdout


def prepare(args) -> None:
    reconcile_shutdown()
    state = read_state()
    orphaned = [name for name, record in state["services"].items()
                if group_members(record) and not alive(record)]
    if orphaned:
        raise RuntimeError(f"本脚本上轮启动器已退出，但仍有所属子进程 {orphaned}；"
                           "这不是其他终端的节点。先承托整臂并执行 shutdown --arm-supported 完成收尾")
    running = {name for name, record in state["services"].items() if alive(record)}
    if "controller" in state["services"] and "controller" not in running:
        raise RuntimeError("独立控制器已退出；先 status 检查，不能沿用缺少控制器的驱动")
    if running and not ("robot" in running and running <= {"robot", "controller"}):
        if all(alive(state["services"].get(name)) for name in ("robot", "camera", "mask", "nbv")):
            # Upgrade an existing demo session without restarting the driver or
            # camera: older launch_rviz=false sessions omitted the TF publisher.
            directory = Path(state["startup_dir"])
            if "tf" not in running:
                nodes = command(["ros2", "node", "list", "--no-daemon"], timeout=12).stdout.splitlines()
                if TF_NODE in nodes:
                    raise RuntimeError("发现其他终端的只读 TF 发布者；先检查现有进程，避免重复发布")
                spawn("tf", service_command("tf", directory), directory / "logs")
            probe("ready", output=directory / "readiness.json", timeout=60)
            print("服务已经启动。继续运行 ./demo_operator.sh check 或 run --workspace-clear")
            return
        raise RuntimeError("存在上轮服务：先 status 检查；正常停止后可重新 prepare，机器人驱动将继续保留")
    nodes = command(["ros2", "node", "list", "--no-daemon"], timeout=12).stdout
    allowed_nodes = {"/agx_arm_ctrl_single_node", "/nero_control"} if "robot" in running else set()
    occupied = [name for name in ("/agx_arm_ctrl_single_node", "/nero_control", "/camera/camera", "/strawberry_learned_mask", "/gradient_nbv", TF_NODE) if name in nodes.splitlines() and name not in allowed_nodes]
    if occupied:
        raise RuntimeError(f"发现由其他终端启动的节点 {occupied}。请先正常停止旧进程，避免重复控制同一硬件")
    if not can_ready():
        print("激活 can0（只启用通信，不使能电机；sudo 可能要求系统密码）", flush=True)
        subprocess.run(["sudo", "bash", str(ROOT / "nero_ws/src/agx_arm_ros/scripts/can_activate.sh"), "can0", "1000000"], cwd=ROOT, check=True)
    if not can_ready():
        raise RuntimeError("can0 未达到 UP / ERROR-ACTIVE / 1 Mbit/s")
    directory = new_directory("startup")
    update_state(startup_dir=str(directory), motor_enable_requested=state.get("motor_enable_requested", False) if "robot" in running else False, abort_requested=False)
    for name in REQUIRED_SERVICES:
        if name not in running:
            spawn(name, service_command(name, directory), directory / "logs")
    probe("ready", output=directory / "readiness.json", timeout=60)
    try:
        check_camera(directory / "initialization", camera_only=True)
    finally:
        render(directory)
    print(f"准备结果目录: {directory}")
    print("准备完成：电机未自动使能，两道运动门已关闭。\n下一步: ./demo_operator.sh run --workspace-clear")


def new_directory(prefix: str = "run") -> Path:
    identifier = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    directory = ROOT / "results/demo" / f"{prefix}_{identifier}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def check_services() -> None:
    state = read_state()
    failed = [name for name in REQUIRED_SERVICES if not alive(state["services"].get(name))]
    if "controller" in state["services"] and not alive(state["services"]["controller"]):
        failed.append("controller")
    if failed:
        raise RuntimeError(f"服务未运行 {failed}，请先 prepare / status 检查日志")


def check_camera(directory: Path, *, camera_only: bool) -> None:
    args = [PYTHON, str(ROOT / "tools/demo/initialization.py"), "--output-dir", str(directory), "--timeout", "30"]
    if camera_only:
        args.append("--camera-only")
    result = command(args, log=directory.parent / "initialization.log", timeout=70)
    print(result.stdout[-5000:].strip(), flush=True)


def enable(args, directory: Path | None = None) -> None:
    if not args.workspace_clear:
        raise RuntimeError("使能前需确认整臂扫掠区清空且有人现场看护，然后使用 --workspace-clear")
    check_services()
    check_abort()
    directory = directory or Path(read_state()["startup_dir"])
    # A failed enable response may still have changed motor state. Invalidate
    # any old shutdown receipt before sending the enable request.
    update_state(shutdown=None, motor_enable_requested=True)
    probe("enable", output=directory / "enable.json")
    update_state(motor_enable_requested=True)
    check_abort()
    probe("joints", output=directory / "initial_joints.json", timeout=8)
    print("电机已请求使能，两道运动门保持关闭；没有发送移动/回零命令。")


def recover_initial_joints(args, directory: Path) -> None:
    """Use existing inward-only recovery before the Cartesian NBV preflight."""
    check_abort()
    if not args.workspace_clear:
        raise RuntimeError("执行关节向内恢复需要已确认现场清空")
    print("检查当前关节是否需要向内恢复；已在安全范围时保持原位。", flush=True)
    invocation = [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/demo/recover_start.py"),
                  "--output-dir", str(directory / "recovery"), "--execute", "--workspace-clear",
                  "--abort-file", str(ABORT), "--abort-not-before-ns", str(INVOCATION_START_NS)]
    result = command(invocation, log=directory / "recovery.log", timeout=90)
    print(result.stdout[-5000:].strip(), flush=True)
    check_abort()


def supervisor_command(directory: Path, args, execute: bool) -> list[str]:
    parameters = {
        "scene_id": directory.name, "execute": execute, "max_step_m": args.max_step,
        "coverage_target": args.coverage, "max_motion_steps": args.steps,
        "handeye_report_path": str(ROOT / "config/calibration/handeye_report.json"),
        "output_path": str(directory / "preview.json"),
        "execution_output_path": str(directory / "execution.json"),
    }
    if execute:
        parameters.update(execution_plan_path=str(directory / "preview.json"),
                          execution_plan_sha256=hashlib.sha256((directory / "preview.json").read_bytes()).hexdigest(),
                          operator_workspace_clearance_confirmed=True,
                          execution_authorization_token=f"EXECUTE_REAL_NBV_LARGE_SESSION_{args.steps}",
                          authorization_receipt_path=str(directory / "authorization_receipt.json"))
    result = ["ros2", "run", "strawberry_active_perception_bridge", "real_nbv_supervisor", "--ros-args",
              "--params-file", str(SUPERVISOR_CONFIG), "--params-file", str(MASK_OVERLAY)]
    for key, value in parameters.items():
        result += ["-p", f"{key}:={str(value).lower() if isinstance(value, bool) else value}"]
    return result


def record_configuration(directory: Path, args) -> None:
    config_dir = directory / "config"
    config_dir.mkdir()
    configs = {}
    for source in (SUPERVISOR_CONFIG, MASK_CONFIG, MASK_OVERLAY, NBV_CONFIG, ROBOT_PROFILE,
                   ROOT / "config/calibration/handeye_report.json"):
        data = source.read_bytes()
        (config_dir / source.name).write_bytes(data)
        configs[str(source.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    write_json(directory / "run.json", {"schema": "strawberry_demo_run/v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "scene_id": directory.name, "max_step_m": args.max_step, "coverage_target": args.coverage,
               "max_motion_steps": args.steps, "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
               "config_sha256": configs, "git_commit": command(["git", "rev-parse", "HEAD"]).stdout.strip(),
               "git_status": command(["git", "status", "--short"]).stdout.splitlines(),
               "services": read_state()["services"]})


def wait_supervisor(record: dict) -> None:
    last_notice = 0.0
    deadline = time.monotonic() + 1200
    while alive(record):
        check_abort()
        if time.monotonic() > deadline:
            raise RuntimeError("监督器超过 20 分钟总时限，请检查本轮日志")
        if time.monotonic() - last_notice > 20:
            print(f"监督器正在运行，实时日志: {record['log']}", flush=True)
            last_notice = time.monotonic()
        time.sleep(.25)


def run_session(args, *, execute: bool) -> None:
    check_services()
    if execute and not args.workspace_clear:
        raise RuntimeError("运行前确认整臂扫掠区清空且有人现场看护，使用 run --workspace-clear")
    directory = new_directory()
    update_state(current_run=str(directory), abort_requested=False)
    print(f"本轮独立结果目录: {directory}", flush=True)
    record_configuration(directory, args)
    try:
        if execute:
            enable(args, directory)
            recover_initial_joints(args, directory)
        check_abort()
        check_camera(directory / "initialization", camera_only=False)
        check_abort()
        probe("close-gates", output=directory / "gates_before_preview.json")
        # Repoint snapshots before Configure; each run also uses a unique scene.
        configured = command(["ros2", "param", "set", "/gradient_nbv", "map_snapshot_directory", str(directory / "maps")], log=directory / "map_directory.log")
        if "Set parameter successful" not in configured.stdout:
            raise RuntimeError(f"不能设置本轮地图目录: {configured.stdout}")
        recorder = [PYTHON, str(ROOT / "tools/recording/observation_recorder.py"), "--output-dir", str(directory / "observations"),
                    "--scene-id", directory.name, "--timeout-sec", "3600"]
        spawn("recorder", recorder, directory / "logs")
        time.sleep(.8)
        record = spawn("supervisor", supervisor_command(directory, args, False), directory / "logs/preview")
        wait_supervisor(record)
        stop_service("supervisor")
        check_abort()
        preview = json.loads((directory / "preview.json").read_text())
        if preview.get("status") != "passed_preview_only":
            raise RuntimeError(f"只读预演未通过: {preview.get('reason', preview.get('status'))}")
        print("本轮预演通过；目标中心/ROI 来自本次新观测。", flush=True)
        if execute:
            check_abort()
            record = spawn("supervisor", supervisor_command(directory, args, True), directory / "logs/execution")
            wait_supervisor(record)
            stop_service("supervisor")
            execution = json.loads((directory / "execution.json").read_text())
            print(f"执行结果: {execution.get('status')}\n{execution.get('reason', '')}", flush=True)
            if not str(execution.get("status", "")).startswith("executed_"):
                raise RuntimeError(f"监督器停止: {execution.get('reason', execution.get('status'))}")
    except BaseException as error:
        write_json(directory / "failure.json", {"error": str(error), "type": type(error).__name__, "time_ns": time.time_ns()})
        raise
    finally:
        cleanup_errors = []
        try:
            if alive(read_state()["services"].get("supervisor")):
                interrupt_supervisor()
                try:
                    probe("halt", output=directory / "halt.json", timeout=15)
                finally:
                    stop_service("supervisor")
            probe("close-gates", output=directory / "gates_after_run.json")
        except Exception as error:
            cleanup_errors.append(str(error))
        try:
            stop_service("recorder")
        except Exception as error:
            cleanup_errors.append(str(error))
        if cleanup_errors:
            write_json(directory / "cleanup_errors.json", {"errors": cleanup_errors})
            print("停止验证未完整通过，保留机器人驱动: " + "; ".join(cleanup_errors), file=sys.stderr)
        source_audit = Path(read_state()["startup_dir"]) / "mask_audit"
        for path in source_audit.glob(f"{directory.name}*.json"):
            audit = json.loads(path.read_text())
            if str(audit.get("scene_id", "")).startswith(directory.name + "_") or audit.get("scene_id") == directory.name:
                destination = directory / "mask_audit" / path.name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(path, destination)
        render(directory)
        if cleanup_errors:
            raise RuntimeError("本轮停止验证未完整通过；查看 cleanup_errors.json，机械臂驱动已保留")
    print("正常暂停: ./demo_operator.sh stop（保持电机使能）")


def execution_map_snapshots(document: dict, snapshots: list[Path]) -> list[Path]:
    """Select only observations named by this execution, never preview maps."""
    import numpy as np

    wanted = [document.get("corrected_observation", {}).get("observation_id")]
    wanted += [step.get("post_map_update", {}).get("observation_id") for step in document.get("motion_steps", [])]
    wanted = [identity for identity in wanted if identity]
    matched = {}
    for path in snapshots:
        with np.load(path, allow_pickle=False) as snapshot:
            identity = str(snapshot["observation_id"].item())
            scene = str(snapshot["scene_id"].item())
        if identity in wanted and scene == document.get("scene_id"):
            if identity in matched:
                raise RuntimeError(f"执行观测有重复地图快照，拒绝混用: {identity}")
            matched[identity] = path
    return [matched[identity] for identity in wanted if identity in matched]


def render(directory: Path) -> None:
    if not directory.is_dir():
        raise RuntimeError(f"结果目录不存在: {directory}")
    env_python = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(ROOT / "perception_ws/src/strawberry_gradient_nbv") + (":" + env_python if env_python else "")
    outputs = []

    def run_renderer(cmd, log, timeout):
        try:
            result = command(cmd, log=log, timeout=timeout, check=False)
            outputs.append({"command": cmd, "returncode": result.returncode})
        except Exception as error:
            outputs.append({"command": cmd, "error": str(error)})

    execution = directory / "execution.json"
    snapshots = sorted((directory / "maps").glob("**/*.npz"))
    if execution.exists():
        document = json.loads(execution.read_text())
        # Exact scene/observation IDs also handle execution failing before its
        # own Configure: old preview maps must not become execution evidence.
        try:
            execution_snapshots = execution_map_snapshots(document, snapshots)
        except Exception as error:
            execution_snapshots = []
            outputs.append({"map_selection_error": str(error)})
        commands = []
        if any("coverage" in step and "post_map_update" in step for step in document.get("motion_steps", [])):
            commands.append([PYTHON, str(ROOT / "tools/visualization/session_visualization.py"), str(execution),
                             "--compact", "--output-dir", str(directory / "figures/session"),
                             "--snapshots", *map(str, execution_snapshots)])
        if execution_snapshots:
            commands.append([PYTHON, str(ROOT / "tools/visualization/voxel_cloud_visualization.py"), *map(str, execution_snapshots),
                             "--compact", "--output-dir", str(directory / "figures/voxel")])
        for index, cmd in enumerate(commands):
            run_renderer(cmd, directory / f"logs/render_{index}.log", 240)
    else:
        document = json.loads((directory / "preview.json").read_text()) if (directory / "preview.json").exists() else {}
    observations = sorted((directory / "observations").glob("*.npz"))
    if execution.exists():
        try:
            observations = execution_map_snapshots(document, observations)
        except Exception as error:
            observations = []
            outputs.append({"observation_selection_error": str(error)})
    if observations:
        cmd = [PYTHON, str(ROOT / "tools/visualization/observation_visualization.py"), *map(str, observations),
               "--compact", "--output-dir", str(directory / "figures/observations")]
        run_renderer(cmd, directory / "logs/render_observations.log", 180)
    write_json(directory / "render.json", {"commands": outputs})
    failure = json.loads((directory / "failure.json").read_text()) if (directory / "failure.json").exists() else {}
    initial_path = directory / "initialization/report.json"
    initialization = json.loads(initial_path.read_text()) if initial_path.exists() else {}
    cleanup_path = directory / "cleanup_errors.json"
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else {}
    summary = {"status": "cleanup_unverified" if cleanup else document.get("status", "failed" if failure else "initialization_only"),
               "reason": "; ".join(cleanup.get("errors", [])) or document.get("reason") or failure.get("error") or initialization.get("reason"),
               "supervisor_status": document.get("status"),
               "initialization_passed": initialization.get("passed"), "initialization": initialization,
               "render_errors": [item for item in outputs if item.get("returncode", 0) != 0 or any("error" in key for key in item)],
               "session_progress": document.get("session_progress"), "result_directory": str(directory)}
    motion_steps = document.get("motion_steps", [])
    summary.update(actual_motion_goal_count=document.get("motion_goal_count", 0),
                   completed_motion_actions=sum(step.get("execution", {}).get("action_status") == 4 for step in motion_steps),
                   map_updated_motion_count=sum("coverage" in step and "post_map_update" in step for step in motion_steps))
    if summary["completed_motion_actions"] > summary["map_updated_motion_count"]:
        summary["figure_scope_note"] = "部分运动到位后未取得有效观测。覆盖率及观测轨迹图仅包含已完成地图更新的步骤，不代表全部实际运动。"
    write_json(directory / "summary.json", summary)
    write_json(directory / "overview.json", build_overview(directory, summary, document))
    links = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".html", ".png", ".gif", ".json") and path.name not in ("index.html", "manifest.json"):
            relative = path.relative_to(directory).as_posix()
            links.append(f'<li><a href="{html.escape(relative, quote=True)}">{html.escape(relative)}</a></li>')
    page = render_index(directory, summary, document, links)
    (directory / "index.html").write_text(page, encoding="utf-8")
    explanation = render_text(directory, summary, document)
    (directory / "结果说明.txt").write_text(explanation, encoding="utf-8")
    files = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.rglob("*"))
             if path.is_file() and path.name != "manifest.json" and path.suffix != ".log"}
    write_json(directory / "manifest.json", {"schema": "strawberry_demo_manifest/v1", "sha256": files})
    # Rebuilding a historical report or preparing another startup must never
    # replace the latest execution's familiar entry point.
    current_run = read_state().get("current_run")
    if current_run and directory.resolve() == Path(current_run).resolve():
        latest = ROOT / "results/index.html"
        relative = os.path.relpath(directory / "index.html", latest.parent)
        href = html.escape(relative, quote=True)
        latest.write_text(page.replace('<head>', f'<head><base href="{href}">', 1), encoding="utf-8")
        (ROOT / "results/结果说明.txt").write_text(explanation, encoding="utf-8")
    print("\n" + explanation, flush=True)
    print(f"中文说明文件: {directory / '结果说明.txt'}", flush=True)
    print(f"用浏览器看图: ./demo_operator.sh open {directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory}", flush=True)


def result_directory(requested: Path | None) -> Path:
    state = read_state()
    directory = requested or Path(state.get("current_run") or state.get("startup_dir") or "")
    if str(directory) == "." or not directory.is_dir():
        raise RuntimeError("尚无可读结果；先运行演示，或指定已有结果目录")
    return directory.resolve()


def show_result(directory: Path, *, in_browser: bool = False) -> None:
    """Read local results; never initialize ROS or touch the robot lifecycle."""
    explanation = directory / "结果说明.txt"
    page = directory / "index.html"
    if not explanation.is_file() or not page.is_file():
        render(directory)
    else:
        print(explanation.read_text(encoding="utf-8"), flush=True)
    if not in_browser:
        print("看图请运行: ./demo_operator.sh open", flush=True)
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print(f"当前终端没有图形桌面。中文结论已显示，也可直接阅读 {explanation}。", flush=True)
        return
    # Explicitly use a browser instead of the editor associated with .html.
    browser = next((path for name in ("microsoft-edge", "microsoft-edge-stable", "google-chrome", "chromium", "firefox")
                    if (path := shutil.which(name))), None)
    if browser is None:
        raise RuntimeError(f"未找到浏览器；中文说明已保存到 {explanation}")
    log_path = RUNTIME / "browser.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([browser, page.as_uri()], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    try:
        code = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        code = None
    if code not in (None, 0):
        raise RuntimeError(f"浏览器启动失败（退出码 {code}）；中文结论见 {explanation}，启动详情见 {log_path}")
    print("已请求浏览器打开本轮图文报告。以后运行 ./demo_operator.sh open 即可。", flush=True)


def acquire_operation_lock():
    """Exclude new operations while stop owns its independent entry lock."""
    with (RUNTIME / "stop.lock").open("a") as stop_lock:
        try:
            fcntl.flock(stop_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("停止流程正在进行；等待 stop/shutdown 完成后再开始新操作") from error
        operation_lock = (RUNTIME / "operation.lock").open("a")
        try:
            fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            operation_lock.close()
            raise RuntimeError("已有 prepare/run 正在执行；status 和 stop 可从另一终端使用") from error
        return operation_lock


def stop(args, *, shutdown=False) -> None:
    if shutdown and not args.arm_supported:
        raise RuntimeError("shutdown 会失能，机械臂可能下落；先可靠托住/支撑整臂，再使用 --arm-supported")
    # Hold this before publishing abort, until every stop/disable action ends.
    # A running operation released its shared entry lock already, so stop can
    # interrupt it without waiting for operation.lock first.
    with (RUNTIME / "stop.lock").open("a") as stop_lock:
        try:
            fcntl.flock(stop_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("另一个停止流程正在进行，请等待其完成") from error
        stop_locked(args, shutdown=shutdown)


def stop_locked(args, *, shutdown=False) -> None:
    # A separate sentinel cannot be lost in concurrent state.json updates.
    ABORT.touch()
    with (RUNTIME / "operation.lock").open("a") as operation_lock:
        interrupt_supervisor()
        state = read_state()
        if not disable_confirmed(state):
            halt_owned_robot(state, Path(state.get("startup_dir", RUNTIME)) / "stop.json")
        deadline = time.monotonic() + 240
        announced = 0.0
        while True:
            try:
                fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError("运行进程尚未完成停止与结果保存；保留机器人，请稍后再次 stop")
                if time.monotonic() - announced > 15:
                    print("已请求取消，等待正在运行的流程退出并保存结果……", flush=True)
                    announced = time.monotonic()
                time.sleep(.2)
        state = read_state()
        robot_present = owned_processes(state, "robot")
        if not disable_confirmed(state):
            halt_owned_robot(state, Path(state.get("startup_dir", RUNTIME)) / "stop_final.json")
        for name in ("supervisor", "recorder", "nbv", "mask", "camera", "tf"):
            stop_service(name)
        if shutdown:
            if disable_confirmed(read_state()):
                finish_robot_shutdown()
            elif robot_present:
                disable_path = Path(state["startup_dir"]) / "disable.json"
                # The controller may have exited with its launch parent. The
                # driver's checked disable service remains the authority.
                probe("disable-driver", output=disable_path)
                update_state(motor_enable_requested=False, shutdown={
                    "phase": "disable_confirmed", "robot": robot_identity(state["services"]["robot"]),
                    "confirmed_at_ns": time.time_ns(), "disable_report": str(disable_path),
                })
                finish_robot_shutdown()
            elif owned_processes(state, "controller") or state.get("motor_enable_requested"):
                raise RuntimeError("驱动已退出，无法确认电机失能；保留独立控制器，请现场检查")
            else:
                stop_service("controller")
                stop_service("robot")
            remaining = [name for name in read_state()["services"] if owned_processes(read_state(), name)]
            if remaining:
                raise RuntimeError(f"仍有本脚本所属进程未退出，收尾未完成: {remaining}")
            print("失能收尾完成，本脚本所属进程已全部退出。可关闭电源；再次演示先运行 prepare。")
        else:
            print("感知服务已停止；机械臂驱动和电机状态保留。断电前必须先支撑整臂，再 shutdown --arm-supported。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="草莓主动感知：一终端启动、现场闭环、独立结果和支撑后失能")
    sub = parser.add_subparsers(dest="operation")
    for name in ("prepare", "status", "stop", "plan"):
        sub.add_parser(name)
    sub.add_parser("help")
    check = sub.add_parser("check", help="只采集新图像和 YOLO 结果，不使能")
    check.add_argument("--with-joints", action="store_true")
    motor = sub.add_parser("enable", help="只使能并保持运动门关闭，不回零")
    motor.add_argument("--workspace-clear", action="store_true")
    for name in ("preview", "run"):
        child = sub.add_parser(name, help="新建独立目录并预演" if name == "preview" else "新预演通过后自动执行并保存结果")
        child.add_argument("--workspace-clear", action="store_true")
        child.add_argument("--max-step", type=float, default=None, metavar="METRES",
                           help="默认沿用算法配置的 max_step_m（当前 0.10 m）；可选此参数缩小搜索半径，不额外限制到 25 mm")
        child.add_argument("--steps", type=int, default=8)
        child.add_argument("--coverage", type=float, default=.8)
    report = sub.add_parser("report", help="只重画图；默认最近一轮")
    report.add_argument("directory", nargs="?", type=Path)
    for name, description in (("result", "直接显示中文结果，不运行硬件"), ("open", "显示中文结果并用浏览器打开图文报告")):
        child = sub.add_parser(name, help=description)
        child.add_argument("directory", nargs="?", type=Path)
    analysis = sub.add_parser("analyze", help="离线分析已有运行并生成中文 Word / PDF 周报",
                              description="仅读取保存的观测、地图和候选记录，生成 Word、PDF、三张分析图与统计表；不启动硬件。")
    analysis.add_argument("directory", nargs="?", type=Path,
                          help="已有运行目录；省略时读取最近一轮。输出到 results/weekly/运行编号/")
    off = sub.add_parser("shutdown", help="可靠支撑整臂后失能并结束服务")
    off.add_argument("--arm-supported", action="store_true")
    args = parser.parse_args(argv)
    if args.operation in (None, "help"):
        parser.print_help()
        print("\n常用顺序：prepare → run --workspace-clear → stop → shutdown --arm-supported\n未使能时人工大致瞄准草莓即可；每轮自动重新计算目标位置，不回固定起点。")
        return 0
    if hasattr(args, "max_step"):
        try:
            native_max_step = profile_max_step()
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            parser.error(f"无法读取算法搜索半径配置：{error}")
        if args.max_step is None:
            args.max_step = native_max_step
        if not (0 < args.max_step <= native_max_step and 1 <= args.steps <= 8 and 0 < args.coverage <= 1):
            parser.error(f"0 < max-step ≤ 算法配置的 {native_max_step:g} m，1 ≤ steps ≤ 8，0 < coverage ≤ 1")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = None
    if args.operation not in ("status", "stop", "shutdown", "report", "result", "open", "analyze", "plan"):
        try:
            lock = acquire_operation_lock()
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 2
    try:
        if args.operation == "prepare":
            prepare(args)
        elif args.operation == "plan":
            print(json.dumps({name: service_command(name, ROOT / "results/demo/EXAMPLE") for name in REQUIRED_SERVICES}, indent=2))
        elif args.operation == "status":
            state = read_state()
            print(json.dumps({**state, "services": {name: {**record, "running": alive(record),
                  "owned_processes": group_members(record),
                  "orphaned_children": bool(group_members(record)) and not alive(record)}
                  for name, record in state["services"].items()}}, ensure_ascii=False, indent=2))
        elif args.operation == "check":
            check_services()
            directory = new_directory("check")
            try:
                check_camera(directory / "initialization", camera_only=not args.with_joints)
            finally:
                render(directory)
        elif args.operation == "enable":
            enable(args)
        elif args.operation in ("preview", "run"):
            run_session(args, execute=args.operation == "run")
        elif args.operation == "report":
            render(result_directory(args.directory))
        elif args.operation in ("result", "open"):
            show_result(result_directory(args.directory), in_browser=args.operation == "open")
        elif args.operation == "analyze":
            from tools.analysis.weekly_report import generate_weekly

            directory = result_directory(args.directory)
            generate_weekly(directory, ROOT / "results/weekly" / directory.name)
        elif args.operation in ("stop", "shutdown"):
            stop(args, shutdown=args.operation == "shutdown")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if args.operation in ("report", "result", "open", "analyze"):
            print(f"结果查看未完成：{error}", file=sys.stderr)
            return 2
        print(f"未继续：{error}\n已保存的日志/结果不会删除。使用 status 检查；普通暂停用 stop，支撑后才 shutdown。", file=sys.stderr)
        return 2
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
