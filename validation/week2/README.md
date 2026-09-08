# Week 2：Gradient-NBV 独立运行与真实相机接口验收

本目录保存可重复的 Week 2 验收命令、只读采集工具和 G0–G4 机器可读证据。初始基线、
G2、G3 与 G4 artifact 分开保存；其中的 `status` 只对各自明确写出的范围有效，不能把
仿真预检解释成真机运动安全，也不能把相机 transport 子项解释成完整几何验收。

初始事实见 [`artifacts/baseline_initial.json`](artifacts/baseline_initial.json)：软件测试共
105 项通过，当时相机工作区仍缺 `libgoogle-glog-dev`。该依赖已于 2026-08-13 安装；联合
`rosdep check` 已无缺项，相机三包也已重新 clean build 成功。Gemini 2 XL 当前仍只协商到
**480 Mbit/s**。480M 是 USB 2.0 High-Speed，不是 SuperSpeed；当前只可把实测的
`640×400 @ 10 Hz` 低带宽 **transport 子项**写为条件通过，不能写成 USB 3 或高带宽已
验证。真实红色目标的静止单视角 NBV 已于 2026-08-13 跑通；360×270 mm、每格 30 mm
标定板的尺度和板面一致性也已通过。平放板的弱边缘被正确判为不可测；随后使用竖直悬空
标定板采集 20 帧，四条实体边缘均通过覆盖率与 ≤3 px 对齐门槛。G3 已在静止相机和当前
`640×400 @ 10 Hz` 配置范围内通过。相机采集时已经安装在未使能的机械臂末端，但没有同步
记录关节位姿，因此这仍不代表手眼标定或移动相机多视角已经完成。当前 USB 2 配置可继续
用于低带宽实验；USB 3 是高分辨率、高帧率或额外数据流的推荐升级，不是低速移动的先决条件。

## 工具

### USB 只读审计

```bash
cd /home/yyt/strawberry_active_perception
/usr/bin/python3 validation/week2/check_usb.py \
  --output validation/week2/artifacts/usb_latest.json
```

脚本只执行 `lsusb`、`lsusb -t` 并读取 `/sys/bus/usb/devices`。它记录协商速度、USB 规范
版本和 `bcdDevice`；后者明确不冒充相机固件版本。序列号默认不写入，只保留短哈希；确有
双相机设备绑定需求时才添加 `--include-serial`，并避免提交包含原始序列号的 artifact。

### RGB-D 指标采集

采集器使用系统 `rclpy` 读取 `sensor_msgs/Image`，不依赖 `cv_bridge`。transport callback
只记录轻量 metadata；深度像素默认每 30 帧抽样一次，由有界后台 worker 解码，避免逐帧
Python 像素分析反过来阻塞被测数据流。它离线执行一对一最近时间配对，并统计：帧率、按
预期帧率估算的丢帧、时间戳单调性、gap P99、彩深 skew P99、尺寸/encoding/frame/K
稳定性，以及 `16UC1` 或 `32FC1` 的抽样有效深度比例。

默认验收参数与 G3 一致：最大配对 skew `5 ms`，有效深度范围 `0.20–2.50 m`；可通过 CLI
覆盖以做诊断，但覆盖后的结果不能直接与默认门限混写。Orbbec 原始 `16UC1` 默认按
`0.001 m/unit` 解释，若设备实测尺度不同必须显式传 `--depth-scale-m-per-unit`。

```bash
source /opt/ros/jazzy/setup.bash
source /home/yyt/strawberry_active_perception/camera_ws/install/setup.bash

/usr/bin/python3 validation/week2/collect_camera_metrics.py \
  --duration-sec 60 \
  --expected-fps 10 \
  --depth-analysis-every-n-frames 30 \
  --output validation/week2/artifacts/camera_60s.json
```

若任何必需流缺失，脚本仍写出机器可读 JSON，但以退出码 2 和
`status: insufficient_data`/`error` 结束。

即使不在 callback 解码像素，`rclpy` 对大图的反序列化和 Python callback 调度仍可能令某
一路订阅饥饿。因此实时 collector 的 transport 计数只作诊断，不能单独把低计数判成设备
掉帧。正式长时间验收应同时用 C++ `ros2 bag record` 记录原始三路，再用离线顺序读取的
header stamp、topic count 和 K 作为传输权威；collector 主要补充抽样深度质量。

## 相机启动前置

先检查而不是假定依赖已满足：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
rosdep check --from-paths camera_ws/src/OrbbecSDK_ROS2 --ignore-src
```

初始 baseline 曾缺少 `apt:libgoogle-glog-dev`；该依赖现已安装，G0 的依赖检查和相机三包
clean build 均已通过。通过项目 adapter 的受控 launch 启动固定低带宽配置：

```bash
cd /home/yyt/strawberry_active_perception/camera_ws
colcon build --symlink-install \
  --packages-select orbbec_camera_msgs orbbec_description orbbec_camera
source install/setup.bash

source /home/yyt/strawberry_active_perception/perception_ws/install/setup.bash
ros2 launch strawberry_observation gemini2xl_observation.launch.py
```

该项目 launch 已将 Gemini 参数限定在节点作用域内，并使用已验证的
`time_domain=global`。不要绕过它直接运行 vendor launch，也不要添加已证明对当前驱动无效的
`use_hardware_time` 参数。

另开终端确认实际话题名称；若使用了不同 namespace，只覆盖采集器的三个 topic 参数，不要
修改统计逻辑：

```bash
ros2 topic list
ros2 topic info /camera/color/image_raw --verbose
ros2 topic info /camera/depth/image_raw --verbose
ros2 topic info /camera/depth/camera_info --verbose
```

## G3 实测流程

### 1. 三次冷启动

每次必须完整停止上面的 launch，确认进程退出，再重新 launch；随后分别执行以下命令。只
重启采集脚本不算冷启动。

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source camera_ws/install/setup.bash

/usr/bin/python3 validation/week2/collect_camera_metrics.py --duration-sec 60 \
  --output validation/week2/artifacts/cold_start_1.json
/usr/bin/python3 validation/week2/collect_camera_metrics.py --duration-sec 60 \
  --output validation/week2/artifacts/cold_start_2.json
/usr/bin/python3 validation/week2/collect_camera_metrics.py --duration-sec 60 \
  --output validation/week2/artifacts/cold_start_3.json
```

三份 JSON 都必须来自不同的 driver launch 生命周期，并记录实际从 launch 到首帧的人工
观察；3/3 成功才进入 30 分钟测试。

### 2. 30 分钟稳定性

```bash
/usr/bin/ros2 bag record -o artifacts/week2/g3_30min_raw \
  /camera/color/image_raw /camera/depth/image_raw /camera/depth/camera_info

/usr/bin/python3 validation/week2/collect_camera_metrics.py \
  --duration-sec 1800 --expected-fps 10 \
  --output validation/week2/artifacts/stability_30min.json
```

两条命令需并行运行。结束后先 `ros2 bag info` 检查 recorder 侧计数，再离线顺序解析完整
bag。门限中的 rate、drop、gap 均以消息 `header.stamp` 计算；bag receive gap 反映主机写盘/
调度，只作诊断，不替代 sensor-stamp 门限。5 ms 内配对统计之外的 unmatched endpoint
也不自动等价于源流掉帧，应与两路各自的连续时间戳统计分开解释。

### 3. 标定板几何/尺度

本次实测标定板为 12×9 格，黑白格区域 360×270 mm，每格 30 mm。已保存的 canonical
Observation 可用下面命令复算。原始 NPZ 在被忽略的 `artifacts/` 中；小型结果 JSON 和
对齐诊断图提交到 validation 目录：

```bash
/usr/bin/python3 validation/week2/analyze_checkerboard_geometry.py \
  --input artifacts/week2/g3_checkerboard_observation.npz \
  --board-width-mm 360 --board-height-mm 270 \
  --output validation/week2/artifacts/g3_checkerboard_geometry.json \
  --overlay validation/week2/artifacts/g3_checkerboard_alignment.png
```

该帧的板面 ROI 深度有效率 100%，平面残差 P90 为 0.780 mm；方格中值 30.239 mm，合并
误差 0.796%，横向误差 0.454%，纵向误差 1.586%。RGB PnP 与深度平面的板中心差 2.225 mm，
法向差 0.450°。因此该工作距离下的 metric scale 与平面一致性通过。

黑白格内部边界只有颜色变化，没有真实前后距离差，不能用于深度边缘验收。平放板与桌面
的实体高度差中值仅 2.165 mm，且当前只有一帧，所以报告将 edge 明确写为 `not_testable`，
不会用经过筛选的 1.67 px 诊断中值冒充通过证据。正式边缘测试需要：板与背景至少相差
20 mm（推荐垫高 50～100 mm）、四边均有有效背景深度、采集 10～30 组同步帧；每边绝对
偏差中值需 ≤3 px、全局 P95 ≤5 px，并检查每边沿线覆盖率，不能只保留最好测的片段。

正式多帧采集和复算命令：

```bash
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
/usr/bin/python3 validation/week2/capture_checkerboard_sequence.py \
  --scene checkerboard_suspended_edge_v3 --count 20 --interval 0.1 \
  --output artifacts/week2/g3_checkerboard_suspended_sequence.npz

/usr/bin/python3 validation/week2/analyze_checkerboard_sequence.py \
  --input artifacts/week2/g3_checkerboard_suspended_sequence.npz \
  --output validation/week2/artifacts/g3_checkerboard_multiframe_geometry.json \
  --overlay validation/week2/artifacts/g3_checkerboard_multiframe_alignment.png
```

最终四边可靠 profile 覆盖率为 100%、81.4%、75.9%、79.5%，沿边跨度均 ≥83.7%；四边
绝对错位中值为 1.65、0.81、2.00、0.13 px，P95 为 2.10、1.08、2.88、0.53 px。全局
P95 为 2.79 px，172 个可靠 profile 全部 ≤3 px。四边背景深度差中值均 ≥54 mm，门禁通过。
竖板存在稳定的毫米级弯曲/空间残差，因此严格平面残差仍引用第一次平放板的通过证据，未
用竖板覆盖或美化该结果。

### 4. rosbag 记录与回放

```bash
ros2 bag record \
  -o validation/week2/artifacts/gemini_rgbd \
  /camera/color/image_raw /camera/color/camera_info \
  /camera/depth/image_raw /camera/depth/camera_info \
  /tf /tf_static

ros2 bag info validation/week2/artifacts/gemini_rgbd
ros2 bag play validation/week2/artifacts/gemini_rgbd

/usr/bin/python3 validation/week2/collect_camera_metrics.py \
  --duration-sec 60 \
  --output validation/week2/artifacts/replay_60s.json
```

回放时不得启动真实相机 driver；检查 replay adapter 和真实 adapter 是否生成相同 canonical
字段、单位和 frame 语义。大型 bag 默认不提交 Git，只保留命令、元数据和汇总 JSON。

## G1/G2/G4 一键复现入口

生成确定性五视角 fixture、检查 NPZ，并按公共 Topic 合同回放：

```bash
cd /home/yyt/strawberry_active_perception
source .venv-nbv/bin/activate
export PYTHONPATH="$PWD/perception_ws/src/strawberry_gradient_nbv:$PYTHONPATH"
python -m strawberry_gradient_nbv.replay generate multiview /tmp/strawberry_multiview.npz
python -m strawberry_gradient_nbv.replay inspect /tmp/strawberry_multiview.npz

source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
ros2 run strawberry_gradient_nbv gradient_nbv_ros_replay /tmp/strawberry_multiview.npz
```

重跑 G2 数值/性能/异常输入验收并同时生成覆盖曲线：

```bash
source /home/yyt/strawberry_active_perception/.venv-nbv/bin/activate
cd /home/yyt/strawberry_active_perception
python validation/week2/run_nbv_acceptance.py \
  --output /tmp/g2_nbv_acceptance.json \
  --coverage-csv /tmp/g2_nbv_acceptance_coverage.csv
```

G4 需要两个终端。终端一只启动仿真、TF、NBV 和预检 bridge，不启动厂商 driver：

```bash
source /opt/ros/jazzy/setup.bash
source /home/yyt/strawberry_active_perception/nero_ws/install/setup.bash
source /home/yyt/strawberry_active_perception/perception_ws/install/setup.bash
ros2 launch strawberry_active_perception_bridge simulation_preview.launch.py
```

终端二发送一条 synthetic Observation，重复请求同一 NBV ID，并记录实时执行门、命令 Topic
与 ROS graph 证据：

```bash
source /opt/ros/jazzy/setup.bash
source /home/yyt/strawberry_active_perception/nero_ws/install/setup.bash
source /home/yyt/strawberry_active_perception/perception_ws/install/setup.bash
ros2 run strawberry_active_perception_bridge gradient_placo_pipeline_fixture \
  --ros-args -p output_path:=/tmp/g4_gradient_to_placo_pipeline.json
```

该入口不会调用 `MoveToPose`，也不会把 SolveIK 结果发布为轨迹。当前完整回归摘要见
[`artifacts/final_test_report.json`](artifacts/final_test_report.json)。

## G0–G4 门禁

编号严格对应已批准的整体计划：

| 门禁 | 内容与通过条件 | 当前可陈述状态 |
|---|---|---|
| G0 | Placo baseline 测试通过；相机 `rosdep check` 无缺项且 Orbbec 三包（msgs/description/camera）clean build 成功。 | **Passed**：105 项软件测试通过；依赖已补齐，相机三包于 2026-08-13 重新 clean build 成功。 |
| G1 | 统一接口可生成，真实/offline/synthetic/replay adapter 使用同一 `Observation` 合同，字段/单位/frame/time 和结构化错误均由测试锁定。 | **Passed**：rosidl/C++/Python 接口、Gemini adapter、fixture 与 NPZ/Topic replay 使用同一 wire contract；四来源 CDR 往返等价性与错误路径均有回归测试。REAL 交叉测试覆盖 adapter 规范化后的 wire 边界，实机传输另由 G3 证明。 |
| G2 | Gradient-NBV 无 MoveIt/ABB/ROS1 运行依赖；固定 seed 数值可重复，多视角覆盖增加；NaN、坏内参、错 scene、重复 ID 等输入被结构化拒绝。 | **Passed**：覆盖率 `20.6864% → 56.3128%`，后四帧 4/4 增长；320×200、128 samples、10 次优化为约 0.4 s、GPU peak allocated 1331.36 MiB。见 `artifacts/g2_nbv_acceptance.json` 与 coverage CSV。 |
| G3 | 真实相机完成 3 次冷启动、30 分钟稳定性、已知尺度/对齐和 rosbag 回放；四项全部通过。 | **Passed（静止相机、当前低带宽配置）**：transport、replay、真实红色目标单视角 NBV、标定板尺度/平面一致性及 20 帧四边实体边缘对齐均通过。见 `artifacts/g3_camera_runtime_summary.json`、`artifacts/g3_real_single_view_nbv.json`、`artifacts/g3_checkerboard_geometry.json` 与 `artifacts/g3_checkerboard_multiframe_geometry.json`。 |
| G4 | NBV camera pose 经 bridge 转为 `link7` 目标，只调用 Placo `SolveIK`；候选回退可成功，且不调用 Move Action、不发布 CAN 命令。 | **Passed（仿真预检范围）**：完整 synthetic Observation→NBV→SolveIK 成功；控制器实时诊断确认 `sim`/`execution=false`，独立订阅 `/control/move_j` 实测 0 条命令，节点图无厂商 driver。见 `artifacts/g4_gradient_to_placo_pipeline.json`；仍不代表碰撞检查、手眼标定或真机运动安全。 |

G3 的单项判据固定如下：

- 三次冷启动均在相同 `640×400 @ 10 Hz` 配置取得 color、registered depth、depth
  `CameraInfo`，无 driver 崩溃或设备重连。
- 每份 60 秒和 30 分钟 artifact 中：color/depth 时间戳严格单调，尺寸、encoding、frame ID
  和 K 均稳定；两流 `sensor_stamp_rate_hz >= 9.0`，估算丢帧率 `<= 1%`，gap P99
  `<= 150 ms`，绝对 skew P99 `<= 5 ms`。
- 标定板平面 ROI 有效深度比例 `>= 0.80`，平面残差 P90 `<= 2 mm`，水平和纵向方格尺寸
  误差分别 `<= 2%`；彩色 PnP 与深度平面的中心差不超过 `max(20 mm, 2%)`。
- 彩深实体边缘需有至少 20 mm 的物理前后距离差并采 10～30 帧；每边偏差中值 `<=3 px`、
  全局 P95 `<=5 px`，每边必须有足够且分布合理的有效样本。
- rosbag 可被重新播放并通过同一统计/接口校验，不依赖真实 Orbbec topic 的硬编码实现细节。

480M 的 transport 结论必须限定为“当前低带宽配置条件通过”。本轮 G3 已有独立的彩深
实体边缘证据，但不能外推到更高带宽配置。若 transport 任一项失败，应优先更换 SuperSpeed
端口/线缆或降低带宽并重新跑完整 transport 流程，而不是放宽统计门限。

## 工具单测

纯单测不需要相机、CAN、`cv_bridge` 或运行中的 ROS graph：

```bash
cd /home/yyt/strawberry_active_perception
/usr/bin/python3 -m pytest -q validation/week2/test
```

它覆盖时间配对、百分位、丢帧估算、时间戳异常、带 padding 的 `16UC1`、含 NaN 的
`32FC1`、CameraInfo K 稳定性、后台抽样 worker 不在 transport callback 做逐像素处理、
sysfs 设备发现以及 480M 条件分类。
