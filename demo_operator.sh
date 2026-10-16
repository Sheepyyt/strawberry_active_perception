#!/usr/bin/env bash
# One terminal entry for the current NERO / Gemini / YOLO11 demonstration.
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
case "${1:-help}" in
  help|-h|--help|report|result|open|analyze|plan) ;;
  *)
    source /opt/ros/jazzy/setup.bash
    source "$PROJECT_ROOT/camera_ws/install/setup.bash"
    source "$PROJECT_ROOT/nero_ws/install/setup.bash"
    source "$PROJECT_ROOT/perception_ws/install/setup.bash"
    ;;
esac
unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=77
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec /usr/bin/python3 -m tools.demo.demo_runtime "$@"
