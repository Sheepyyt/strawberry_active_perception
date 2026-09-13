#!/usr/bin/env bash

# Plain-language operator entry for the hash-pinned YOLO11 strawberry mask.

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PID_FILE="/tmp/strawberry_learned_mask_${UID}.pid"
CHECKPOINT="$PROJECT_ROOT/artifacts/models/yolo11m_strawberry_best.pt"
EXPECTED_SHA="7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357"
AUDIT_DIRECTORY="$PROJECT_ROOT/artifacts/learned_mask/runtime"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

source /opt/ros/jazzy/setup.bash
source "$PROJECT_ROOT/perception_ws/install/setup.bash"

require_runtime() {
  if [[ ! -x "$PROJECT_ROOT/.venv-mask/bin/python" ]]; then
    echo "学习式 mask 环境还没有安装。请先看 validation/week9/README.md。"
    exit 2
  fi
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "没有找到模型：$CHECKPOINT"
    echo "请把 best.pt 复制到上述位置。"
    exit 2
  fi
  local actual_sha
  actual_sha="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
  if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
    echo "模型 SHA 不匹配，拒绝加载。"
    echo "期望：$EXPECTED_SHA"
    echo "实际：$actual_sha"
    exit 2
  fi
}

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local candidate
  candidate="$(<"$PID_FILE")"
  [[ "$candidate" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$candidate" 2>/dev/null || return 1
  [[ -r "/proc/$candidate/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$candidate/cmdline" |
    grep -q "strawberry_learned_mask.ros_node" || return 1
  printf '%s\n' "$candidate"
}

print_help() {
  cat <<'EOF'
用法：
  ./mask_operator.sh start   启动 YOLO11 草莓 mask；保持终端不要关闭
  ./mask_operator.sh status  检查节点和输出 topic
  ./mask_operator.sh test    拍一帧但不运动，检查学习式 mask 是否通过
  ./mask_operator.sh stop    停止学习式 mask 节点

这个脚本没有机械臂控制接口。test 只拍照和分割，不会运动。
EOF
}

case "${1:-help}" in
  start)
    require_runtime
    if pid="$(running_pid)"; then
      echo "YOLO11 mask 已经运行（进程 $pid）。"
      exit 0
    fi
    rm -f "$PID_FILE"
    echo "正在加载 YOLO11m 分割模型，第一次启动可能需要几秒……"
    "$PROJECT_ROOT/.venv-mask/bin/python" \
      -m strawberry_learned_mask.ros_node --ros-args \
      --params-file \
      "$PROJECT_ROOT/perception_ws/install/strawberry_learned_mask/share/strawberry_learned_mask/config/yolo11m_strawberry.yaml" &
    mask_pid=$!
    printf '%s\n' "$mask_pid" >"$PID_FILE"
    cleanup() {
      if kill -0 "$mask_pid" 2>/dev/null; then
        kill -TERM "$mask_pid" 2>/dev/null || true
        wait "$mask_pid" 2>/dev/null || true
      fi
      if [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$mask_pid" ]]; then
        rm -f "$PID_FILE"
      fi
    }
    trap cleanup INT TERM EXIT
    wait "$mask_pid"
    ;;
  stop)
    if ! pid="$(running_pid)"; then
      rm -f "$PID_FILE"
      echo "YOLO11 mask 没有运行。"
      exit 0
    fi
    kill -TERM "$pid"
    echo "已请求停止 YOLO11 mask 进程 $pid。"
    ;;
  status)
    require_runtime
    if ! ros2 node list 2>/dev/null | grep -qx '/strawberry_learned_mask'; then
      echo "YOLO11 mask：没有运行。请执行 ./mask_operator.sh start"
      exit 1
    fi
    echo "YOLO11 mask：节点正在运行。"
    ros2 topic info /strawberry/perception/learned_observation |
      sed -n '1,4p'
    ;;
  test)
    require_runtime
    if ! ros2 node list 2>/dev/null | grep -qx '/strawberry_learned_mask'; then
      echo "请先在另一个终端执行 ./mask_operator.sh start"
      exit 1
    fi
    scene="manual_yolo11_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$AUDIT_DIRECTORY"
    echo "正在拍一帧并用 YOLO11 分割；不会发送机械臂命令……"
    ros2 service call \
      /strawberry/perception/capture_observation \
      strawberry_perception_interfaces/srv/CaptureObservation \
      "{scene_id: $scene, not_before: {sec: 0, nanosec: 0}, timeout: {sec: 5, nanosec: 0}, discard_frames: 3, require_color: true, require_mask: false, require_pose: false}" \
      >/tmp/strawberry_learned_mask_test_call.txt
    report=""
    for _ in {1..100}; do
      report="$(find "$AUDIT_DIRECTORY" -maxdepth 1 -type f -name "${scene}_*.json" -print -quit)"
      [[ -n "$report" ]] && break
      sleep 0.1
    done
    if [[ -z "$report" ]]; then
      echo "10 秒内没有收到 YOLO11 结果。"
      exit 1
    fi
    if jq -e '.accepted_for_nbv == true' "$report" >/dev/null; then
      echo "YOLO11 草莓 mask：通过。"
      jq '{confidence_threshold, accepted_instances, mask_pixels, valid_depth_pixels, backend}' "$report"
    else
      echo "YOLO11 草莓 mask：未通过；机械臂不会运动。"
      jq '{accepted_instances, rejected_instances, mask_pixels, valid_depth_pixels}' "$report"
      exit 3
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
