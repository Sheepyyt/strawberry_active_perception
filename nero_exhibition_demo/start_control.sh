#!/usr/bin/env bash
set -eo pipefail

demo_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "${demo_directory}/.." && pwd)"

usage() {
    echo "用法："
    echo "  $0 sim"
    echo "  $0 real [can_port] [usb_address]"
    echo "示例：$0 real can0"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi
mode="$1"
shift

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    echo "找不到 /opt/ros/jazzy/setup.bash" >&2
    exit 2
fi
if [[ ! -f "${project_directory}/.venv/bin/activate" ]]; then
    echo "找不到项目虚拟环境：${project_directory}/.venv" >&2
    exit 2
fi
if [[ ! -f "${project_directory}/nero_ws/install/setup.bash" ]]; then
    echo "找不到 nero_ws/install/setup.bash，请先按项目 README 构建工作区" >&2
    exit 2
fi

source /opt/ros/jazzy/setup.bash
source "${project_directory}/.venv/bin/activate"
source "${project_directory}/nero_ws/install/setup.bash"
set -u

unset ROS_LOCALHOST_ONLY || true
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77
export PYTHONDONTWRITEBYTECODE=1

if ros2 node list 2>/dev/null | grep -qx '/nero_control'; then
    echo "检测到 /nero_control 已经运行。请直接在另一个终端运行 run_demo.sh。" >&2
    exit 2
fi

if [[ "${mode}" == "sim" ]]; then
    if [[ $# -ne 0 ]]; then
        usage
        exit 2
    fi
    echo "启动无 CAN 的 MeshCat 演示环境……"
    exec ros2 launch strawberry_nero_control sim.launch.py \
        viewer_use_collision_meshes:=true
fi

if [[ "${mode}" != "real" || $# -gt 2 ]]; then
    usage
    exit 2
fi

can_port="${1:-can0}"
usb_address="${2:-}"
if [[ ! "${can_port}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "CAN 端口名称无效：${can_port}" >&2
    exit 2
fi

echo
echo "即将配置 ${can_port} 并使能真机。"
echo "先清空机械臂周围和下方，固定底座，并安排观察员守在控制箱外侧。"
read -r -p "确认控制箱已上电且现场已清空，输入 START：" answer
if [[ "${answer}" != "START" ]]; then
    echo "确认取消；没有配置 CAN，也没有启动驱动。" >&2
    exit 2
fi

sudo bash "${project_directory}/nero_ws/src/agx_arm_ros/scripts/can_activate.sh" \
    "${can_port}" 1000000 "${usb_address}"

can_details="$(ip -details link show "${can_port}")"
echo "${can_details}"
if ! grep -q 'bitrate 1000000' <<<"${can_details}"; then
    echo "${can_port} bitrate 不是 1000000，拒绝启动驱动。" >&2
    exit 2
fi
if ! grep -q 'ERROR-ACTIVE' <<<"${can_details}"; then
    echo "${can_port} 不是 ERROR-ACTIVE，拒绝启动驱动。" >&2
    exit 2
fi
if ! grep -qE '<([^,>]+,)*UP(,[^,>]+)*>' <<<"${can_details}"; then
    echo "${can_port} 没有 UP，拒绝启动驱动。" >&2
    exit 2
fi
if ! grep -qE '<([^,>]+,)*LOWER_UP(,[^,>]+)*>' <<<"${can_details}"; then
    echo "${can_port} 没有 LOWER_UP，拒绝启动驱动。" >&2
    exit 2
fi

echo
echo "CAN 检查通过，启动 NERO 驱动与 Placo 控制器（速度 10%）。"
echo "保持本终端运行；展示结束后先确认机械臂回到 ready，再按 Ctrl+C。"
exec ros2 launch strawberry_nero_control real.launch.py \
    can_port:="${can_port}" \
    profile_config_file:="${demo_directory}/config/nero_exhibition.yaml" \
    speed_percent:=10 \
    allow_limit_recovery_execution:=true \
    first_motion_test_mode:=false \
    precision_test_mode:=false \
    startup_enable:=true
