#!/usr/bin/env bash
# Offline-only regression suite. This script never opens CAN or a camera.
set -eo pipefail

sap_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${sap_directory}"

if [[ ! -x .venv/bin/python || ! -x .venv-nbv/bin/python ]]; then
    echo "缺少 .venv 或 .venv-nbv；请先按 README 创建两个环境。" >&2
    exit 2
fi

if ! git -C nero_ws/src/agx_arm_ros apply --unidiff-zero --reverse --check \
    ../../../vendor_patches/agx_arm_ros/nero_v111_safety.patch >/dev/null 2>&1; then
    echo "AGX 安全补丁尚未应用；请先运行：" >&2
    echo "  ./vendor_patches/agx_arm_ros/apply_checked.sh" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
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

echo "[3/9] 展示程序隔离测试"
./nero_exhibition_demo/check_demo.sh

export PYTHONPATH="${sap_directory}/perception_ws/src/strawberry_gradient_nbv:${PYTHONPATH:-}"

echo "[4/9] Gradient-NBV（含地图可视化）"
.venv-nbv/bin/python -m pytest -q \
    perception_ws/src/strawberry_gradient_nbv/test

echo "[5/9] 手眼、桥接和闭环监督器"
/usr/bin/python3 -m pytest -q perception_ws/src/strawberry_handeye_calibration/test
/usr/bin/python3 -m pytest -q \
    perception_ws/src/strawberry_active_perception_bridge/test \
    validation/week4/test

echo "[6/9] 相机/采集验证工具"
/usr/bin/python3 -m pytest -q validation/week2/test validation/week3/test

echo "[7/9] 学习式 mask 离线边界与大步 IK 预检"
/usr/bin/python3 -m pytest -q \
    validation/week6/test_mask_model_evaluator.py \
    validation/week6/test_reachable_candidate_visualization.py
PYTHONPATH="${sap_directory}/nero_ws/src/strawberry_nero_control:${PYTHONPATH:-}" \
  .venv/bin/python validation/week6/large_step_ik_preflight.py \
  --output /tmp/strawberry_large_step_ik_preflight.json >/dev/null

echo "[8/9] 长闭环证据记录和展示图生成测试"
PYTHONPATH="${sap_directory}/perception_ws/src/strawberry_gradient_nbv:${PYTHONPATH:-}" \
  /usr/bin/python3 -m pytest -q validation/week7/test

echo "[9/9] ROS 接口与 C++ 相机适配器的已构建测试"
(
    cd perception_ws
    ../.venv-nbv/bin/python -m colcon test \
        --packages-select strawberry_perception_interfaces strawberry_observation
    ../.venv-nbv/bin/python -m colcon test-result --verbose
)

echo "全部离线软件测试通过；本脚本未访问相机、CAN 或机械臂。"
