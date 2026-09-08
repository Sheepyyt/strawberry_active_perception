# 双臂主动感知草莓项目空间

当前已实现 NERO 单臂的 Placo 末端位姿 IK、ROS 2 接口、平滑关节轨迹、安全拒绝和真机
执行。系统不使用 MoveIt，也不调用 NERO 原厂笛卡尔 IK。Gemini 2 XL 的统一 RGB-D
Observation、脱离 MoveIt 的 Gradient-NBV 核心和真实闭环监督器位于
`perception_ws`。30 个实机姿态的手眼标定已经正式通过离线数值门禁并接入真实执行链路。
2026-08-14 已完成一次“真实相机拍摄 → Gradient-NBV → 正式手眼变换 → Placo SolveIK →
机械臂小步运动 → 再拍摄并更新地图”的闭环；程序只执行了一个目标，随后自动关闭两道
软件执行门，没有执行第二步。双臂协同和 VAMP 尚未开始。

## 下载

机械臂驱动和相机驱动是 Git 子模块：

```bash
git clone --recurse-submodules \
  https://github.com/Sheepyyt/strawberry_active_perception.git
cd strawberry_active_perception
git submodule update --init --recursive
```

GitHub 网页会把子模块显示成可点击的提交链接，而不是复制第三方仓库的全部文件，这是正常
现象。

## 创建环境并构建

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash

python3 -m venv --system-site-packages --prompt sap-core .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cd nero_ws
python -m colcon build --symlink-install \
  --packages-select strawberry_nero_interfaces strawberry_nero_control
source install/setup.bash
```

工作区的 `colcon_defaults.yaml` 会忽略 `agx_arm_moveit`。必须使用 `python -m colcon`，让
生成的 ROS 2 Python 程序使用安装了 Placo 的 `.venv`。

感知侧使用独立环境，避免 NumPy 2/PyTorch 与系统 OpenCV/cv_bridge 的 ABI 混用：

```bash
cd /home/yyt/strawberry_active_perception
python3 -m venv --system-site-packages --prompt sap-nbv .venv-nbv
source .venv-nbv/bin/activate
python -m pip install -r perception_ws/src/strawberry_gradient_nbv/requirements-nbv.txt

source /opt/ros/jazzy/setup.bash
source camera_ws/install/setup.bash
source nero_ws/install/setup.bash
python -m colcon --log-base perception_ws/log build \
  --symlink-install --base-paths perception_ws/src \
  --build-base perception_ws/build --install-base perception_ws/install \
  --event-handlers console_cohesion+
```

相机适配器是 C++ 节点，使用系统 ROS/OpenCV；NBV wrapper 使用 `.venv-nbv`，且不导入
`cv_bridge`。当前 Gemini 仍协商为 USB 2.0/480M，但已经在 `640x400@10 Hz` 的固定配置下
通过 30 分钟稳定性验收，因此不会阻塞后续低带宽实验。USB 3 SuperSpeed 是提高分辨率、
帧率或增加额外数据流时的推荐升级；若机械臂运动时线缆弯折导致断流，再将它升级为阻塞项。

## 当前结论（Week 2 + Week 3 + Week 4）

- G1 统一接口、G2 独立 Gradient-NBV 和 G4 无运动 Placo 仿真预检已经通过；
- G0 已通过：`libgoogle-glog-dev` 已安装，系统依赖检查和相机三包 clean build 均成功；
- G2 五视角 ROI coverage 从 20.6864% 增至 56.3128%，默认 GPU 验收约 0.4 s；
- Gemini 三次冷启动、30 分钟低带宽 transport、rosbag 回放和真实红色目标单视角 NBV
  已通过；360×270 mm 标定板的尺度、平面一致性和 20 帧四边彩深实体边缘对齐均已通过，
  G3 已在“静止相机、当前 640×400@10 Hz 配置”的范围内完成；
- 厂商的 `image_undistorted` 实际没有消除畸变：同一时间戳的原图和所谓去畸变图逐字节
  相同。现在由项目自己的 adapter 使用相机原始 `K/D` 真正校正 RGB；硬件已经对齐到彩色
  网格的 HW-D2C depth 保持原样，不做第二次 remap。独立 20 帧实机几何复核已经通过；
- 30 个同步机械臂姿态已按正确相机模型重新计算。正式手眼标定 20/20 个独立留出切分
  全部通过；最差留出 P95 为 `6.10 mm / 0.52°`，不同切分外参最大差异为
  `3.386 mm / 0.513°`。这说明外参已通过离线数值验收，不等于已经允许机械臂执行 NBV；
- 正式外参已进入真实 Placo `SolveIK` 的只读计算链。当前位姿和相机光学 +X 方向 5 mm
  候选均为 `passed`；5 mm 完整候选 `alpha=1` 首次求解成功，最大关节变化为 `0.012 rad`，
  末端位置/姿态求解误差约 `0.994 mm / 0.000577 rad`。两个独立运动命令观察计数均为 0，
  因此这是“算出可达解”，不是“机械臂已经走了 5 mm”；
- 在上述只读验证之后，真实闭环监督器又完成了一次正式小步运动。SHA 冻结目标对应的
  相机变化约为 `1.890 mm / 7.831°`；只发送了 1 个高层 `MoveToPose` 目标。动作后精确 TF
  测得相机目标误差为 `0.421 mm / 0.0115°`，红色目标仍清楚可见；ROI 地图 coverage 从
  `30.568%` 增至 `43.581%`，增加 `13.013` 个百分点，已观察体素增加 `105554`；
- 控制器为这 1 个平滑动作发送了 44 个约 50 Hz 的关节轨迹采样点，这不是 44 次运动。
  动作结束后控制器门和驱动门分别获得两次关闭回执，独立监视器继续观察 2 分钟，没有
  新命令、第二动作、CAN/限位/机械臂错误。完整证据见
  [`validation/week4/README.md`](validation/week4/README.md) 和
  [`real_nbv_post_motion_summary.json`](validation/week4/artifacts/real_nbv_post_motion_summary.json)；
- 面向现场操作者的启动、看图和红色物体测试步骤见
  [`CAMERA_GUIDE_CN.md`](CAMERA_GUIDE_CN.md)；完整门禁、实测数字和复现命令见
  [`validation/week2/README.md`](validation/week2/README.md)。手眼结果、矩阵、证据 SHA 和
  当前安全边界见 [`HAND_EYE_RESULT_CN.md`](validation/week3/HAND_EYE_RESULT_CN.md)。

README 不记录“相机在线、机械臂已使能”之类会过期的现场状态。每次实验都必须重新运行：

```bash
./camera_operator.sh usb
./camera_operator.sh status
./camera_operator.sh test-red
./robot_operator.sh can
./robot_operator.sh status
./robot_operator.sh pose
```

命令的当次输出才代表真实状态，历史 JSON 不能替代上电后的检查。早期两份只读证据见
[`current`](validation/week3/artifacts/real_handeye_current_solveik_preview.json)（SHA-256
`5b2c3d6096acc3caa12e49acfc0ddb0a9be4ca6bc481f81a1615a8a51f946413`）和
[`optical +X 5 mm`](validation/week3/artifacts/real_handeye_small_nbv_solveik_preview.json)
（SHA-256 `5ca282f883f2b0affe9b7719c6554843ab45c401216046e3eb3dadc0447f421e`）。

## 目录

```text
strawberry_active_perception/
├── camera_ws/src/OrbbecSDK_ROS2/        相机驱动子模块
├── perception_ws/
│   └── src/
│       ├── strawberry_perception_interfaces/  统一 Observation/NBV 接口
│       ├── strawberry_observation/            Gemini 按需采集适配器
│       ├── strawberry_gradient_nbv/           独立 Gradient-NBV 核心与回放
│       └── strawberry_active_perception_bridge/  Placo SolveIK 只读预检
├── nero_ws/
│   ├── src/agx_arm_ros/                 NERO 官方驱动子模块
│   ├── src/strawberry_nero_interfaces/  ROS 2 消息、服务和 Action 定义
│   ├── src/strawberry_nero_control/     Placo IK、轨迹、安全控制和 Demo
│   └── colcon_defaults.yaml             构建时忽略 MoveIt
├── validation/week1/                    保留的测试 CSV、JSON 和数据摘要
├── validation/week2/                    相机/NBV/预检工具、门禁与小型证据
├── validation/week3/                    手眼采集、标定结果与只读验证说明
├── validation/week4/                    第一次真实 NBV 小步闭环、监督器证据与说明
├── nero_exhibition_demo/                与科研参数隔离的大幅参观展示
├── vendor_patches/agx_arm_ros/           固定版本的 NERO v1.11 安全补丁
├── requirements.txt                     Python 运行依赖
└── README.md
```

`.venv/`、`nero_ws/build/`、`nero_ws/install/`、日志和 Python 缓存均为本机生成内容，不提交
Git。删除 `build/install` 后，重新执行上面的构建命令即可恢复。

## 两个 Strawberry 包为什么分开

- `strawberry_nero_interfaces` 只定义其他节点怎样请求 IK、运动和安全恢复，以及返回哪些字段；
- `strawberry_nero_control` 实现 Placo 求解、连续性检查、五次轨迹和 NERO 执行。

接口独立后，未来 Gradient-NBV 只需依赖稳定的小接口包，不必依赖控制程序内部实现。

`nero_ws/src/strawberry_nero_control/strawberry_nero_control/` 这个内层同名目录是 Python 源码
模块；外层目录是 ROS 2 Python 工程。它们不是两份重复代码，而是 `ament_python` 的标准
结构。

## 复现 Demo 和查看结果

- 从上电、CAN、恢复、ready、目标预览、真机执行、回程到安全失能的完整教程：
  [`nero_ws/src/strawberry_nero_control/README.md`](nero_ws/src/strawberry_nero_control/README.md)
- 测试过程、数据表和原始结果说明：
  [`validation/week1/README.md`](validation/week1/README.md)
- Gemini、统一接口、Gradient-NBV 和 SolveIK 预检的门禁与采集工具：
  [`validation/week2/README.md`](validation/week2/README.md)
- 手眼标定的通俗结论、正式矩阵、证据文件和下一步边界：
  [`validation/week3/HAND_EYE_RESULT_CN.md`](validation/week3/HAND_EYE_RESULT_CN.md)
- 第一次真实 Gradient-NBV 小步闭环、失败保护过程和动作后地图证据：
  [`validation/week4/README.md`](validation/week4/README.md)
