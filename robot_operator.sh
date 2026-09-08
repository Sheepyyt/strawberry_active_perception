#!/usr/bin/env bash

# 面向现场操作者的 NERO 状态与软件停止入口。
# 本脚本目前故意不提供“使能”或“运动”命令。

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_PID_FILE="/tmp/strawberry_robot_readonly_${UID}.pid"

source /opt/ros/jazzy/setup.bash

if [[ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]] ||
  [[ ! -f "$PROJECT_ROOT/nero_ws/install/setup.bash" ]]; then
  echo "机械臂程序还没有构建，或项目 Python 环境不存在。"
  exit 2
fi

source "$PROJECT_ROOT/.venv/bin/activate"
source "$PROJECT_ROOT/nero_ws/install/setup.bash"
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77

print_help() {
  cat <<'EOF'
用法：
  ./robot_operator.sh can             只读检查 CAN 通信状态
  ./robot_operator.sh start-readonly  启动未使能、双控制门关闭的只读驱动
  ./robot_operator.sh status          检查节点、控制门和关节反馈
  ./robot_operator.sh pose            读取七关节角并计算当前 link7 位姿
  ./robot_operator.sh e-stop          关闭两道控制门并请求电子阻尼急停
  ./robot_operator.sh stop            正常停止本脚本启动的只读驱动

如果 CAN 尚未激活，请先执行：
  cd nero_ws/src/agx_arm_ros
  sudo bash scripts/can_activate.sh can0 1000000

电子阻尼急停不是机械锁止：抬起的机械臂可能缓慢下降。
直接失能或断电可能使机械臂立即下落。本脚本当前没有任何使能或运动子命令。
EOF
}

can_is_ready() {
  ip link show can0 >/dev/null 2>&1 || return 1
  local details
  details="$(ip -details link show can0 2>/dev/null)"
  grep -qE '<[^>]*UP[^>]*>' <<<"$details" || return 1
  grep -q 'bitrate 1000000' <<<"$details" || return 1
  grep -q 'can state ERROR-ACTIVE' <<<"$details" || return 1
}

print_can_status() {
  if ! ip link show can0 >/dev/null 2>&1; then
    echo "CAN 通信：没有找到 can0。"
    return 1
  fi
  local details
  details="$(ip -details -statistics link show can0 2>/dev/null)"
  if can_is_ready; then
    echo "CAN 通信：已激活（UP、ERROR-ACTIVE、1000000 bit/s）。"
    return 0
  fi
  echo "CAN 通信：尚未达到可读状态。"
  echo "需要看到 UP、ERROR-ACTIVE 和 bitrate 1000000；当前摘要："
  grep -E '^[0-9]+: can0:|can state|bitrate|bus-errors|RX:|TX:' <<<"$details" || true
  return 1
}

read_running_robot_pid() {
  [[ -f "$ROBOT_PID_FILE" ]] || return 1
  local candidate
  candidate="$(<"$ROBOT_PID_FILE")"
  [[ "$candidate" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$candidate" 2>/dev/null || return 1
  [[ -r "/proc/$candidate/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' <"/proc/$candidate/cmdline")"
  [[ "$command_line" == *"ros2 launch strawberry_nero_control real.launch.py"* ]] ||
    return 1
  printf '%s\n' "$candidate"
}

force_software_gates_closed() {
  local driver_ready=false
  local controller_ready=false
  for _ in {1..100}; do
    if ros2 service list 2>/dev/null | grep -qx '/control_enable'; then
      driver_ready=true
    fi
    if ros2 service list 2>/dev/null |
      grep -qx '/strawberry_nero/enable_execution'; then
      controller_ready=true
    fi
    if [[ "$driver_ready" == true && "$controller_ready" == true ]]; then
      break
    fi
    sleep 0.1
  done
  if [[ "$driver_ready" != true || "$controller_ready" != true ]]; then
    echo "只读驱动启动失败：10 秒内没有看到两道软件控制门。"
    return 1
  fi

  local driver_reply
  local controller_reply
  driver_reply="$(timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool \
    '{data: false}' 2>&1)" || return 1
  controller_reply="$(timeout 5 ros2 service call \
    /strawberry_nero/enable_execution std_srvs/srv/SetBool \
    '{data: false}' 2>&1)" || return 1
  grep -q 'success=True' <<<"$driver_reply" || return 1
  grep -q 'success=True' <<<"$controller_reply" || return 1
  echo "软件安全门：原厂控制门=关闭，Placo 执行门=关闭。"
}

case "${1:-help}" in
  can)
    print_can_status
    ;;
  start-readonly)
    if running_pid="$(read_running_robot_pid)"; then
      echo "只读机械臂驱动已经运行（进程 $running_pid）。"
      exit 0
    fi
    rm -f "$ROBOT_PID_FILE"
    if ! print_can_status; then
      echo
      echo "请先按照上方提示，用 sudo 激活 CAN；这不会使能机械臂。"
      exit 3
    fi
    echo "正在启动只读驱动：电机不使能、不修改 SDK 当前速度设定、两道运动门关闭。"
    echo "注意：NERO 固件 1.11 在未使能时可能不推送关节反馈。"
    ros2 launch strawberry_nero_control real.launch.py \
      can_port:=can0 \
      speed_percent:=0 \
      allow_limit_recovery_execution:=false \
      first_motion_test_mode:=false \
      precision_test_mode:=false \
      startup_enable:=false \
      launch_rviz:=false &
    robot_pid=$!
    printf '%s\n' "$robot_pid" >"$ROBOT_PID_FILE"
    cleanup_robot() {
      if kill -0 "$robot_pid" 2>/dev/null; then
        kill -TERM "$robot_pid" 2>/dev/null || true
        wait "$robot_pid" 2>/dev/null || true
      fi
      if [[ -f "$ROBOT_PID_FILE" ]] &&
        [[ "$(<"$ROBOT_PID_FILE")" == "$robot_pid" ]]; then
        rm -f "$ROBOT_PID_FILE"
      fi
    }
    trap cleanup_robot INT TERM EXIT
    if ! force_software_gates_closed; then
      echo "无法证明两道运动门都已关闭，正在停止驱动。"
      exit 4
    fi
    echo "只读驱动已就绪。保持本终端打开；另开终端运行 status 或 pose。"
    wait "$robot_pid"
    ;;
  status)
    print_can_status || true
    if ros2 node list 2>/dev/null | grep -qx '/agx_arm_ctrl_single_node'; then
      echo "原厂驱动：正在运行。"
    else
      echo "原厂驱动：没有运行。"
      exit 1
    fi
    if ros2 node list 2>/dev/null | grep -qx '/nero_control'; then
      echo "Placo 控制器：正在运行。"
    else
      echo "Placo 控制器：没有运行。"
      exit 1
    fi
    if timeout 4 ros2 topic echo /feedback/joint_states \
      sensor_msgs/msg/JointState --once >/dev/null 2>&1; then
      echo "关节反馈：已收到完整消息；可以继续读取当前位姿。"
    else
      echo "关节反馈：4 秒内没有收到。机械臂失能时可能不持续反馈。"
      exit 5
    fi
    ;;
  pose)
    echo "正在读取一组新的七关节反馈并计算 link7；不会发送运动命令。"
    exec ros2 run strawberry_nero_control nero_pose_demo current
    ;;
  e-stop)
    echo "正在立即请求 NERO 电子阻尼急停，然后补关两道软件控制门……"
    echo "注意：机械臂可能缓慢下降；这不是机械锁止，也不会主动失能。"
    stop_ok=false
    stop_reply="$(timeout 2 ros2 service call /electronic_emergency_stop \
      std_srvs/srv/Trigger '{}' 2>&1)" || true
    if grep -q 'success=True' <<<"$stop_reply"; then
      stop_ok=true
      echo "电子阻尼急停帧已交给 CAN 驱动；控制箱没有独立回执，仍需目视确认。"
    else
      echo "电子阻尼急停未能确认发送；请立即使用现场断电手段。"
    fi
    if ros2 service list 2>/dev/null |
      grep -qx '/strawberry_nero/enable_execution'; then
      timeout 3 ros2 service call /strawberry_nero/enable_execution \
        std_srvs/srv/SetBool '{data: false}' >/dev/null 2>&1 || true
    fi
    if ros2 service list 2>/dev/null | grep -qx '/control_enable'; then
      timeout 3 ros2 service call /control_enable \
        std_srvs/srv/SetBool '{data: false}' >/dev/null 2>&1 || true
    fi
    echo "两道软件控制门已请求关闭；急停锁会拒绝重新开门，直到驱动重启。"
    if [[ "$stop_ok" != true ]]; then
      exit 7
    fi
    echo "请目视确认机械臂状态；通信失效时只能使用现场断电手段。"
    ;;
  stop)
    if ! running_pid="$(read_running_robot_pid)"; then
      rm -f "$ROBOT_PID_FILE"
      echo "没有找到由本脚本启动的只读驱动；它可能已经停止。"
      exit 0
    fi
    echo "正在停止只读机械臂驱动进程 $running_pid……"
    kill -TERM "$running_pid"
    for _ in {1..100}; do
      if ! kill -0 "$running_pid" 2>/dev/null; then
        rm -f "$ROBOT_PID_FILE"
        echo "只读机械臂驱动已停止。"
        exit 0
      fi
      sleep 0.1
    done
    echo "驱动没有在 10 秒内退出。请回到 start-readonly 终端按 Ctrl+C。"
    exit 1
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    echo "未知命令：$1"
    print_help
    exit 2
    ;;
esac
