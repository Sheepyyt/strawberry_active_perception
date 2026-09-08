#!/usr/bin/env bash
set -eo pipefail

demo_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "${demo_directory}/.." && pwd)"

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

exec python "${demo_directory}/show_demo.py" "$@"
