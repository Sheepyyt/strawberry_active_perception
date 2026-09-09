# 单臂主动感知草莓（NERO + Gemini 2 XL + Gradient-NBV）

一句话说明：相机先拍草莓，程序把看到和没看到的三维空间记在体素地图里，Gradient-NBV
选择下一观察位置，Placo 把它换成机械臂关节角，安全监督器执行小步运动后再拍照并继续更新
同一张地图。整个计算和控制链不依赖 MoveIt。

## 当前进度

已经完成并保存证据的主链路是：

```text
Gemini RGB-D → mono8 目标 mask → 统一 Observation
       → 同一张体素地图持续更新 → Gradient-NBV
       → 正式手眼外参 → Placo SolveIK → NERO 小步运动 → 再观察
```

截至 2026-09-09：

- Placo 单臂 IK、轨迹、ROS 2 服务/Action 和真机安全门已完成；主控制测试 `105 passed`。
- Gemini 2 XL 在 `640×400@10 Hz` 下已通过 3 次冷启动和 30 分钟稳定性测试。当前线缆
  即使协商为 USB 2/480M，也足以继续这一低带宽实验。
- MoveIt-free Gradient-NBV 已通过合成五视角验收；ROI 有效射线覆盖率从 `20.69%` 增至
  `56.31%`。现在会为每次地图更新保存可重放的体素快照。
- 30 个真实姿态完成了 eye-in-hand 手眼标定；正式报告的 20/20 个留出切分通过，报告已
  放入版本库并由 SHA-256 锁定。
- 红色目标真实三步闭环使用同一张地图、只配置 1 次，覆盖率
  `24.68% → 40.72% → 44.75% → 46.37%`，总增量 `21.69` 个百分点；3 个高层运动均完成，
  每步结束后两道执行门关闭。
- 把测试物换成真实草莓后，用当前 HSV 红色规则 mask 连续完成 2 步，覆盖率
  `23.94% → 35.21% → 41.65%`。第三个建议位移不超过 1 mm；当前代码会把这种情况解释为
  “提前收敛”，但修复后尚未重新做真机复测。

这里的 `coverage` 是“目标 ROI 中有多少体素被有效深度射线碰到”，用于比较地图是否持续
获得新信息；它不是草莓真实表面覆盖率，也不能直接解释成“看清了百分之多少草莓”。

详细证据：

- [相机、接口与 Gradient-NBV 验收](validation/week2/README.md)
- [正式手眼标定结果](validation/week3/HAND_EYE_RESULT_CN.md)
- [红色目标三步真实闭环](validation/week4/README.md)
- [真实草莓 HSV-mask 多步闭环](validation/week5/README.md)
- [固定版本、证据 SHA 与离线测试清单](validation/REPRODUCIBILITY_MANIFEST.json)

## 先直观看懂体素地图

下面是同一合成场景连续 5 次观察后的俯视、正视和侧视图。灰色是还没看过，蓝色是深度
射线已经经过，深灰是测到的表面，红色是 mask 支持的目标，绿色是相机路径，橙圈是下一
建议位置。底部曲线显示地图覆盖率逐次增加。

![Gradient-NBV 五视角体素地图](validation/week2/artifacts/g2_nbv_map_final.png)

[打开动态 GIF 查看五次更新过程](validation/week2/artifacts/g2_nbv_map_progress.gif)

离线复现这张图（不会连接相机或机械臂）：

```bash
cd /home/yyt/strawberry_active_perception
PYTHONPATH=perception_ws/src/strawberry_gradient_nbv \
  .venv-nbv/bin/python -m strawberry_gradient_nbv.map_visualization \
  demo --device cpu --output-dir /tmp/nbv-map-demo
xdg-open /tmp/nbv-map-demo/nbv_map_final.png
```

真实 NBV 配置会把每步完整体素快照写入 `artifacts/nbv_map_snapshots/`。运行结束后可执行：

```bash
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
.venv-nbv/bin/python -m strawberry_gradient_nbv.map_visualization \
  render --output-dir /tmp/real-nbv-map artifacts/nbv_map_snapshots/场景名/generation_001/*.npz
```

旧的真实闭环只保存了覆盖率和体素数量，没有保存完整体素数组，所以能核验曲线和统计，
不能事后无损恢复成三维图；从本版本开始的新实验可以。

## 为什么当前最多三步，而不是达到目标才停

“三步”是第一轮真机验证的安全上限，不是 NBV 算法的最终停止目标。它用来先证明：第二、
第三步确实使用前面累积的同一张地图，并且连续开门、运动、关门不会失控。

当前监督器已经会在两种情况下提前停：单步 coverage 增量低于 `0.5` 个百分点，或算法提出
的下一位移不超过 `1 mm`。最终版本还应加入可配置的目标停止策略，同时保留最大步数/时间
作为兜底：

1. coverage 达到经过实验标定的目标；或连续若干视角几乎不再增长；
2. 预期信息增益连续若干次低于阈值；
3. 下一动作进入位移死区；
4. 达到最大步数、最大累计运动或任何安全错误。

不能现在随意写一个“80% 就完成”，因为当前 coverage 是 ROI 射线覆盖，不是真实草莓表面
真值。下一轮应先用多组回放数据画出“观察次数—coverage—增益”曲线，再固定阈值；届时
步数只是上限，达到目标会更早停止。

## “真正的草莓分割”是什么

是 mask。mask 是一张与彩色图同尺寸的黑白图：草莓像素为 255（白），其他像素为 0
（黑）。当前程序通过 HSV 颜色阈值找红色，因此镜头前放真实草莓可以跑通几何闭环，但它
不能理解“草莓”这个类别，红杯子、红纸或偏色光照都可能误判。

可以直接使用 [Meta 官方 SAM 3](https://github.com/facebookresearch/sam3) / SAM 3.1：用文本
提示 `strawberry` 先做零样本分割，不必先自行训练。接入方式是新增一个 mask provider，
把 SAM 输出转成现有 `mono8` mask；后面的 Observation、NBV、手眼、IK 和运动代码都复用。

但“无需训练”不等于“无需验证”。SAM 可能选中多个草莓、漏掉叶片或把包装图案当目标，
所以先做离线质量门：置信度、目标实例选择、至少 200 个有效 mask 深度像素、时间稳定性和
人工抽查。SAM 3.1 模型约 848M 参数，官方环境要求 Python 3.12、较新的 PyTorch/CUDA，
应放在独立 `.venv-sam3` 中按需运行，先测本机 8 GB 显存的速度与显存，不与 NBV 环境硬混。

需要操作者提供的不是训练代码，而是：

1. 在 Hugging Face 申请并接受官方 checkpoint 权限，然后在本机登录（不要把 token 提交）；
2. 保留 10～30 组不同距离、角度、遮挡和光照的真实草莓图用于验证；
3. 演示可以不标注，但要客观汇报准确性时，人工检查或标注其中一小批作为真值。

## 从新电脑复现

### 1. 克隆与系统依赖

```bash
git clone --recurse-submodules \
  https://github.com/Sheepyyt/strawberry_active_perception.git
cd strawberry_active_perception
git submodule update --init --recursive
sudo apt install libgoogle-glog-dev python3-venv python3-pip python3-opencv
```

NERO v1.11 的启动、电子阻尼急停和 `move_home` 控制门是固定上游 commit 的本地安全补丁：

```bash
./vendor_patches/agx_arm_ros/apply_checked.sh
```

脚本同时校验子模块 commit 和补丁 SHA；版本不符会拒绝修改，不会静默套到另一版驱动。

### 2. 两个互相隔离的 Python 环境

```bash
python3 -m venv --system-site-packages --prompt sap-core .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
deactivate

python3 -m venv --system-site-packages --prompt sap-nbv .venv-nbv
source .venv-nbv/bin/activate
python -m pip install -r perception_ws/src/strawberry_gradient_nbv/requirements-nbv.txt
deactivate
```

`.venv` 服务 Placo；`.venv-nbv` 服务 PyTorch/Gradient-NBV。相机适配器使用系统 C++ OpenCV，
从而避开 NumPy 2 与系统 `cv_bridge` 的 ABI 冲突。

### 3. 构建三个工作区

相机驱动先按其[中文说明](camera_ws/src/OrbbecSDK_ROS2/README_CN.MD)安装 udev 规则，然后：

```bash
source /opt/ros/jazzy/setup.bash

cd camera_ws
colcon build --symlink-install \
  --packages-select orbbec_camera_msgs orbbec_description orbbec_camera
cd ..

source .venv/bin/activate
cd nero_ws
python -m colcon build --symlink-install \
  --packages-select agx_arm_msgs agx_arm_description agx_arm_ctrl \
    strawberry_nero_interfaces strawberry_nero_control
cd ..
deactivate

source camera_ws/install/setup.bash
source nero_ws/install/setup.bash
source .venv-nbv/bin/activate
python -m colcon --log-base perception_ws/log build \
  --symlink-install --base-paths perception_ws/src \
  --build-base perception_ws/build --install-base perception_ws/install
deactivate
```

`nero_ws/colcon_defaults.yaml` 会忽略 MoveIt 包。

### 4. 一键离线回归

```bash
./verify_software.sh
```

这个脚本只运行测试，不打开相机、不激活 CAN、不使机械臂运动。正式外参报告位于
`validation/week3/artifacts/stability_pose001_030_factory_raw_D.json`，SHA-256 为
`31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f`。

## 每次硬件实验前

README 不保存“相机在线、机械臂已使能”等很快过期的状态；只相信本次命令输出：

```bash
./camera_operator.sh usb
./camera_operator.sh status
./camera_operator.sh test-strawberry
./robot_operator.sh can
./robot_operator.sh status
./robot_operator.sh pose
```

通俗的相机操作见 [CAMERA_GUIDE_CN.md](CAMERA_GUIDE_CN.md)，机械臂说明见
[NERO 控制 README](nero_ws/src/strawberry_nero_control/README.md)。真实运动仍必须清空整臂
扫掠区、检查线缆、安排观察员，并生成新的只读 preview/完整 SHA；历史授权和历史 JSON
不能重复用于新实验。本项目尚无环境碰撞模型，不能无人值守运行。

## 目录

```text
camera_ws/                    Orbbec ROS 2 驱动子模块与构建空间
nero_ws/                      NERO 驱动、Placo IK、轨迹与安全控制
perception_ws/src/
  strawberry_perception_interfaces/    统一消息、Service、Action
  strawberry_observation/              Gemini 适配、HSV mask、按需采集
  strawberry_gradient_nbv/             纯核心、回放、地图快照与可视化
  strawberry_handeye_calibration/      手眼标定与稳定性验证
  strawberry_active_perception_bridge/ NBV→手眼→IK→受监督运动
validation/week1..week5/      可提交的小型数据、报告和真实执行证据
vendor_patches/agx_arm_ros/   受版本/SHA 保护的 NERO v1.11 安全补丁
nero_exhibition_demo/         与科研参数隔离的展示程序
artifacts/                    大型 rosbag/现场原始数据（本机保留，不提交）
```

下一项最小开发任务是：接入 SAM 3.1 mask provider 并做离线 A/B 验证；随后把三步上限推广为
“目标/收敛条件决定何时停，最大步数和累计运动只负责兜底”的有限状态循环。双臂、采摘和
无人值守连续运动暂不进入本阶段。
