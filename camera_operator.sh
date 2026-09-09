#!/usr/bin/env bash

# 面向现场操作者的 Gemini 2 XL 简明入口。

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_PID_FILE="/tmp/strawberry_camera_operator_${UID}.pid"

# Keep the camera on the same isolated ROS graph as the NERO controller.  A
# caller can still override either value explicitly before invoking the script.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

source /opt/ros/jazzy/setup.bash

if [[ ! -f "$PROJECT_ROOT/camera_ws/install/setup.bash" ]]; then
  echo "相机程序还没有构建。请先阅读 CAMERA_GUIDE_CN.md 的“首次构建”部分。"
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/perception_ws/install/setup.bash" ]]; then
  echo "感知程序还没有构建。请先阅读 CAMERA_GUIDE_CN.md 的“首次构建”部分。"
  exit 2
fi

source "$PROJECT_ROOT/camera_ws/install/setup.bash"
source "$PROJECT_ROOT/perception_ws/install/setup.bash"

print_usb_status() {
  local device_path=""
  local candidate=""
  for candidate in /sys/bus/usb/devices/*; do
    if [[ -f "$candidate/idVendor" && -f "$candidate/idProduct" ]] &&
      [[ "$(<"$candidate/idVendor")" == "2bc5" ]] &&
      [[ "$(<"$candidate/idProduct")" == "0671" ]]; then
      device_path="$candidate"
      break
    fi
  done

  if [[ -z "$device_path" ]]; then
    echo "USB 连接：没有找到 Gemini 2 XL。请检查相机是否插好。"
    return 1
  fi

  local speed
  speed="$(<"$device_path/speed")"
  if awk "BEGIN {exit !($speed >= 5000)}"; then
    echo "USB 连接：高速模式，${speed}M（达到 USB 3 要求）。"
  else
    echo "USB 连接：${speed}M（目前是 USB 2 速度；当前 10 帧/秒测试仍可继续）。"
  fi
}

print_help() {
  cat <<'EOF'
用法：
  ./camera_operator.sh usb       检查相机插口实际速度
  ./camera_operator.sh start     启动相机；保持这个终端不要关闭
  ./camera_operator.sh stop      正常停止由本脚本启动的相机
  ./camera_operator.sh view      打开实时彩色画面
  ./camera_operator.sh status    检查相机程序和深度数据是否正在工作
  ./camera_operator.sh test-strawberry  检查真实草莓能否通过 HSV 规则生成 mask
  ./camera_operator.sh test-red         上一命令的兼容别名

停止相机：运行 ./camera_operator.sh stop，或回到执行 start 的终端按 Ctrl+C 一次。
EOF
}

read_running_camera_pid() {
  [[ -f "$CAMERA_PID_FILE" ]] || return 1
  local candidate
  candidate="$(<"$CAMERA_PID_FILE")"
  [[ "$candidate" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$candidate" 2>/dev/null || return 1
  [[ -r "/proc/$candidate/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' <"/proc/$candidate/cmdline")"
  [[ "$command_line" == *"ros2 launch strawberry_observation gemini2xl_observation.launch.py"* ]] ||
    return 1
  printf '%s\n' "$candidate"
}

case "${1:-help}" in
  usb)
    print_usb_status
    ;;
  start)
    if running_pid="$(read_running_camera_pid)"; then
      echo "相机已经由本脚本启动（进程 $running_pid），不需要重复启动。"
      exit 0
    fi
    rm -f "$CAMERA_PID_FILE"
    print_usb_status || true
    echo "正在启动相机。看到 'Device Orbbec Gemini 2 XL connected' 后表示成功。"
    echo "不要关闭这个终端；停止时按 Ctrl+C。"
    ros2 launch strawberry_observation gemini2xl_observation.launch.py &
    camera_pid=$!
    printf '%s\n' "$camera_pid" >"$CAMERA_PID_FILE"
    cleanup_camera() {
      if kill -0 "$camera_pid" 2>/dev/null; then
        # ros2 launch is started as a background job, so its inherited SIGINT
        # disposition may be ignored. SIGTERM is handled cleanly by launch.
        kill -TERM "$camera_pid" 2>/dev/null || true
        wait "$camera_pid" 2>/dev/null || true
      fi
      if [[ -f "$CAMERA_PID_FILE" ]] &&
        [[ "$(<"$CAMERA_PID_FILE")" == "$camera_pid" ]]; then
        rm -f "$CAMERA_PID_FILE"
      fi
    }
    trap cleanup_camera INT TERM EXIT
    wait "$camera_pid"
    ;;
  stop)
    if ! running_pid="$(read_running_camera_pid)"; then
      rm -f "$CAMERA_PID_FILE"
      echo "没有找到由本脚本启动的相机进程；相机可能已经停止。"
      exit 0
    fi
    echo "正在正常停止相机进程 $running_pid……"
    kill -TERM "$running_pid"
    for _ in {1..100}; do
      if ! kill -0 "$running_pid" 2>/dev/null; then
        rm -f "$CAMERA_PID_FILE"
        echo "相机程序已停止。"
        exit 0
      fi
      sleep 0.1
    done
    echo "相机没有在 10 秒内退出。请回到 start 终端按 Ctrl+C，并把输出发给我。"
    exit 1
    ;;
  view)
    echo "正在打开实时彩色画面。若画面不出现，请先在另一个终端运行 start。"
    exec ros2 run rqt_image_view rqt_image_view /camera/color/image_raw
    ;;
  status)
    print_usb_status || true
    if ros2 node list 2>/dev/null | grep -qx '/camera/camera'; then
      echo "相机程序：正在运行。"
    else
      echo "相机程序：没有运行。请在另一个终端执行 ./camera_operator.sh start"
      exit 1
    fi
    if timeout 5 ros2 topic echo --once /camera/depth/image_raw \
      sensor_msgs/msg/Image --field header >/dev/null 2>&1; then
      echo "深度数据：已实际收到一帧，工作正常。"
    else
      echo "深度数据：5 秒内没有收到，请把终端输出发给我检查。"
      exit 1
    fi
    ;;
  test-strawberry|test-red)
    if ! ros2 service list 2>/dev/null |
      grep -qx '/strawberry/perception/capture_observation'; then
      echo "相机程序没有运行。请先在另一个终端执行 ./camera_operator.sh start"
      exit 1
    fi
    echo "正在用 HSV 颜色规则检查真实草莓，请保持相机和草莓静止几秒……"
    set +e
    result="$(timeout 10 ros2 service call \
      /strawberry/perception/capture_observation \
      strawberry_perception_interfaces/srv/CaptureObservation \
      "{scene_id: manual_red_test, not_before: {sec: 0, nanosec: 0}, timeout: {sec: 5, nanosec: 0}, discard_frames: 3, require_color: true, require_mask: true, require_pose: true}" \
      2>&1)"
    call_status=$?
    set -e
    if [[ $call_status -ne 0 ]]; then
      echo "测试命令未完成。请把下面内容发给我："
      echo "$result"
      exit 1
    fi
    if grep -q 'success=True' <<<"$result"; then
      echo "HSV 草莓候选 mask：成功。系统已经生成一份可供 NBV 使用的真实观测。"
    elif grep -q 'code=15' <<<"$result"; then
      echo "HSV 草莓候选 mask：尚未识别到足够清楚的草莓。"
      echo "请让草莓更靠近画面中央、适当靠近相机，并避免强反光，再试一次。"
      exit 3
    else
      echo "HSV 草莓候选 mask：未通过。请把下面内容发给我："
      echo "$result"
      exit 1
    fi
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
