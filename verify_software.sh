#!/usr/bin/env bash
# Offline-only regression suite. This script never opens CAN or a camera.
set -eo pipefail

sap_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${sap_directory}"

if [[ ! -x .venv/bin/python || ! -x .venv-nbv/bin/python || \
      ! -x .venv-mask/bin/python ]]; then
    echo "缺少 .venv、.venv-nbv 或 .venv-mask；请先按 README 创建环境。" >&2
    exit 2
fi

if ! git -C nero_ws/src/agx_arm_ros apply --unidiff-zero --reverse --check \
    ../../../vendor_patches/agx_arm_ros/nero_v111_safety.patch >/dev/null 2>&1; then
    echo "AGX 安全补丁尚未应用；请先运行：" >&2
    echo "  ./vendor_patches/agx_arm_ros/apply_checked.sh" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
source perception_ws/install/setup.bash
set -u

echo "[1/9] NERO 主控制测试"
.venv/bin/python -m pytest -q nero_ws/src/strawberry_nero_control/test

echo "[2/9] AGX 安全服务测试"
PYTHONPATH="${sap_directory}/nero_ws/src/agx_arm_ros/src/agx_arm_ctrl:${PYTHONPATH:-}" \
  .venv/bin/python -m pytest -q \
    nero_ws/src/agx_arm_ros/src/agx_arm_ctrl/test/test_safety_services.py

export PYTHONPATH="${sap_directory}/perception_ws/src/strawberry_gradient_nbv:${PYTHONPATH:-}"

echo "[3/9] Gradient-NBV（含地图可视化）"
.venv-nbv/bin/python -m pytest -q \
    perception_ws/src/strawberry_gradient_nbv/test

echo "[4/9] 手眼、桥接和闭环监督器"
/usr/bin/python3 -m pytest -q perception_ws/src/strawberry_handeye_calibration/test
/usr/bin/python3 -m pytest -q \
    perception_ws/src/strawberry_active_perception_bridge/test \
    tests/nbv tests/demo

echo "[5/9] 相机/采集验证工具"
/usr/bin/python3 -m pytest -q tests/camera tests/calibration

echo "[6/9] 可达候选可视化与大步 IK 预检"
/usr/bin/python3 -m pytest -q \
    tests/visualization/test_reachable_candidate_visualization.py
PYTHONPATH="${sap_directory}/nero_ws/src/strawberry_nero_control:${PYTHONPATH:-}" \
  .venv/bin/python tools/nbv/large_step_ik_preflight.py \
  --output /tmp/strawberry_large_step_ik_preflight.json >/dev/null

echo "[7/9] 闭环证据记录、展示图与离线周报分析测试"
PYTHONPATH="${sap_directory}/perception_ws/src/strawberry_gradient_nbv:${PYTHONPATH:-}" \
  /usr/bin/python3 -m pytest -q \
    tests/visualization/test_visualization.py \
    tests/visualization/test_voxel_cloud_visualization.py tests/analysis

echo "[8/9] YOLO11 草莓 mask 接口与安全边界"
PYTHONPATH="${sap_directory}/perception_ws/src/strawberry_learned_mask:${PYTHONPATH:-}" \
  .venv-mask/bin/python -m pytest -q \
    perception_ws/src/strawberry_learned_mask/test
.venv-mask/bin/python -m flake8 \
  --config perception_ws/src/strawberry_learned_mask/setup.cfg \
  tools/mask/capture_learned_snapshot.py

echo "[9/9] ROS 接口与 C++ 相机适配器的已构建测试"
(
    cd perception_ws
    ../.venv-nbv/bin/python -m colcon test \
        --packages-select strawberry_perception_interfaces strawberry_observation
    ../.venv-nbv/bin/python -m colcon test-result --verbose
)

echo "全部离线软件测试通过；本脚本未访问相机、CAN 或机械臂。"
