# Week 3：手眼标定采集与结果

30 个实机姿态已经采集完成，并在修正相机畸变模型后通过正式手眼标定门禁。最终矩阵、
20/20 留出验证、独立 20 帧固定板复核、两次已通过的真实只读 SolveIK 预检、证据 SHA、
当前运行状态和下一步操作边界，请先看
[`HAND_EYE_RESULT_CN.md`](HAND_EYE_RESULT_CN.md)。本页后续内容保留采集器的数据约定和
复现方法；它不是让现场人员现在继续采更多姿态的指令。

本目录提供 NERO 眼在手上（eye-in-hand）标定的**单姿态只读采集器**。每次执行只采当前
已经静止的一个姿态；程序不会移动机械臂，也不会创建运动 publisher、Action client、控制
服务或使能接口。需要改变姿态时，应由现场人员在本程序之外按已批准的机械臂操作流程完成，
停稳后再运行下一次采集。

## 固定数据约定

- 标定板：`12×9` 个格子，即 `11×8=88` 个内角；格边长 `30 mm`。
- 标定板物理坐标原点：程序把两个对角候选端中、相邻外格为黑色的一端定义为原点，
  用颜色约定消除普通对称棋盘的 180° 方向歧义。这里的“原点”不等于图像左上角；
  标定板在画面中可以旋转。打印图案需保持正常、清晰的黑白交替。
- `T_base_link7`：把 `link7` 中的点映射到 `base_link`，即
  `p_base = T_base_link7 @ p_link7`。
- `T_camera_checkerboard`：把棋盘坐标中的点映射到相机光学坐标，由 `solvePnP` 直接得到，
  即 `p_camera = T_camera_checkerboard @ p_checkerboard`。
- 相机光学轴遵循 ROS optical frame：`+X` 向右、`+Y` 向下、`+Z` 向前。
- 每个姿态保存一个压缩 NPZ；所有数组均为数值或 Unicode dtype，使用
  `numpy.load(path, allow_pickle=False)` 即可读取，不含 Python object/pickle。

## 相机模型为什么要修正

原来的 30 个 NPZ 保留了完整 RGB、角点和机械臂位姿，但当时把原始 RGB 错标成了
“畸变已经为零”。现场检查发现，Gemini 厂商节点发布的 `image_undistorted` 与同一时间戳
的 `image_raw` 逐字节完全相同；源码中用于校正的畸变参数在调用校正函数前已被清零，
所以该 topic 实际是 no-op（没有做任何事）。

项目现在不再依赖这个 topic。Observation adapter 读取原始彩色图及真实的非零畸变参数，
在 adapter 内用 OpenCV 真正 rectification，再发布 `D=0` 的统一 RGB 图。Gemini 的
HW-D2C 深度本来已经在目标彩色像素网格上，所以深度保持原样，绝不再做一次 remap。
对旧 30 姿态只离线重算棋盘 PnP；原始 NPZ 未被覆盖。正式结果见上面的中文结果文档。

## 运行前置

采集器不会启动相机或机械臂硬件节点。运行前应由现场流程确认已有：

1. `/feedback/joint_states`：完整且新鲜的 `joint1` 到 `joint7` 位置和速度。
2. `/strawberry/perception/capture_observation` 服务。
3. `/strawberry/perception/observation` canonical Observation topic。
4. Gemini 彩色画面中完整可见的标定板；四周建议至少留 8 px，实际应尽量留得更多。

使用系统 ROS Python 和 apt OpenCV，不激活项目 NumPy 2.x 虚拟环境：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
source perception_ws/install/setup.bash
/usr/bin/python3 -c 'import rclpy, cv2, tf2_ros; print(cv2.__version__)'
```

### 只读 TF

采集器优先向 tf2 查询曝光时刻的 `base_link→link7`。当前 real launch 如果没有
`robot_state_publisher`，可另开终端运行本目录的安全 launch：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
/usr/bin/python3 validation/week3/launch/read_only_nero_tf.launch.py
```

该 launch 只启动 `robot_state_publisher`，将其 `joint_states` 输入重映射到只读反馈
`/feedback/joint_states`；它不启动驱动、不创建控制 topic，也不发送运动命令。若 TF 无法取得，
采集会 fail closed，不会用单位矩阵或猜测值代替。由于相机 global time 与机械臂时间域可能
不同，曝光时刻 TF 查询失败时，只允许在整个采集窗已经通过严格静止门的条件下使用 latest
TF，并在 NPZ 的 `tf_lookup_mode` 中明确记录。

## 每个姿态采一次

标定数据建议放在全局 Git 忽略的 `artifacts/week3/handeye_session_001/`：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
source perception_ws/install/setup.bash

/usr/bin/python3 validation/week3/capture_handeye_sample.py \
  --sample-id pose_001 \
  --scene handeye_session_001 \
  --output-directory artifacts/week3/handeye_session_001
```

一次命令只会生成 `pose_001.npz`。现场人员按批准流程改变姿态并停稳后，使用新的
`--sample-id pose_002` 再次执行。工具没有 `--count` 或自动运动模式。

查看全部参数：

```bash
/usr/bin/python3 validation/week3/capture_handeye_sample.py --help
```

## 默认拒绝条件

任一条件不满足都不会留下成功文件：

- 最近关节反馈超过 `0.25 s`，相邻反馈接收间隙超过 `0.25 s`，少于 5 条，或不含完整、
  有限的 7 关节位置和速度。
- Capture 前后静止证据窗口不足；默认窗口内任一关节跨度超过 `0.002 rad`，或反馈速度
  绝对值超过 `0.01 rad/s`。
- Capture 服务 response 与 topic 的 `scene_id`、`observation_id` 或深度曝光 stamp 不一致。
- Observation 不是真实来源，或 RGB/depth/CameraInfo 的尺寸、frame、stamp、K、D、编码、
  有效深度比例或 5 ms skew 契约不一致。
- 未完整检测到 88 个内角、不能根据黑色原点格消除方向歧义、外推后的完整 12×9 板越界/
  离边界不足 8 px，或 PnP 重投影 RMS 超过 `1 px`。
- tf2 无法获得 `T_base_link7`。
- TF stamp 与本次连续静止窗口内最近 JointState 相差超过 `0.20 s`（防止使用缓存旧 TF）。
- 输出文件已存在；或同目录已有相同 sample ID、Observation ID、曝光时间戳、不同 K/frame/
  相机来源，以及最大关节差小于 `0.03 rad` 的近重复姿态。

这些门限可以通过 CLI 收紧或在有充分证据时显式覆盖，但每次 NPZ 都保留实际静止证据和
选择模式。不要为了“凑够样本”放宽门限或重复采相同姿态。

## NPZ 关键字段

- 身份：`schema_version`、`sample_id`、`scene_id`、`observation_id`、`timestamp_sec`、`stamp`。
- 原始证据：`rgb`、`depth_m`、`K`、`D`、相机 frame/source/skew/有效深度比例。
- 棋盘：88 个 `checkerboard_corners_px`、物点、完整外边界、重投影 RMS、
  `T_camera_checkerboard`。
- 机械臂：`joint_positions`、完整 joint window、stamp、最大跨度/速度/反馈接收间隙、
  关节—曝光时间选择模式和差值，以及关节—TF stamp 配对差值。
- TF：`T_base_link7`、`tf_lookup_mode`、`tf_stamp_ns`。
- 安全审计：`motion_commands_sent=0`。

所有旋转矩阵保存前都会验证正交性和 `det=+1`，齐次矩阵末行必须严格为
`[0, 0, 0, 1]`。

## 纯单测

测试不需要 ROS graph、相机或机械臂：

```bash
cd /home/yyt/strawberry_active_perception
/usr/bin/python3 -m pytest -q validation/week3/test
```

覆盖无 `cv_bridge` 解码、stamp/K 合同、完整 JointState、静止窗口、棋盘完整性与方向、
PnP/刚体矩阵、原子无 object NPZ、重复样本拒绝，以及源码不含运动 client/publisher。

## 正式外参的“只计算、不运动”预检（已完成）

当前正式报告为：

```text
artifacts/week3/handeye_session_001/stability_pose001_030_factory_raw_D.json
SHA256 31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f
```

这里的 SHA256 可以理解为报告的“文件指纹”。只要报告内容哪怕改了一个字符，指纹就会变化，
程序会直接拒绝启动。因此，实际使用的外参一定来自这份已经通过 30 个姿态、20/20 拆分检验的
报告，不会因为手工抄写矩阵而悄悄用错。

这一步只问 Placo：“这个姿态有没有 IK 解？”程序不会把解发送给机械臂。它具备三层保护：

1. `real_preview.launch.py` 只启动一个 bridge，不启动相机、机械臂驱动、控制器或任何静态 TF。
2. bridge 只有 `/strawberry_nero/solve_ik` Service client，没有 MoveToPose Action client，也没有
   关节/CAN publisher。
3. bridge 和测试工具分别订阅 `/control/move_j` 做实时计数；任何一方看到运动命令，结果都判失败。

运行前，现有系统需要已经提供：

- `/strawberry_nero/solve_ik`；
- `base_link <- link7` 的只读 TF；
- `/strawberry_nero/diagnostics`，并且其中 `execution_enabled=false`。

第一个终端只启动正式外参 bridge：

```bash
cd /home/yyt/strawberry_active_perception
export ROS_DOMAIN_ID=77
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
source perception_ws/install/setup.bash
ros2 launch strawberry_active_perception_bridge real_preview.launch.py
```

第二个终端先检查“当前姿态”。这个目标与机械臂当前 `link7` 姿态相同，是最保守的只读检查：

```bash
cd /home/yyt/strawberry_active_perception
export ROS_DOMAIN_ID=77
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
source perception_ws/install/setup.bash
ros2 run strawberry_active_perception_bridge real_handeye_preview \
  --ros-args \
  --params-file perception_ws/install/strawberry_active_perception_bridge/share/strawberry_active_perception_bridge/config/real_preview.yaml \
  -p preview_mode:=current \
  -p output_path:=/home/yyt/strawberry_active_perception/artifacts/week3/handeye_session_001/real_current_pose_ik_preview.json
```

如果当前姿态报告通过，再运行一个“向相机画面右侧平移 5 mm、仍看向原目标”的候选。注意，
这里仍然只算 IK，不会真的移动 5 mm：

```bash
ros2 run strawberry_active_perception_bridge real_handeye_preview \
  --ros-args \
  --params-file perception_ws/install/strawberry_active_perception_bridge/share/strawberry_active_perception_bridge/config/real_preview.yaml \
  -p preview_mode:=small_nbv \
  -p output_path:=/home/yyt/strawberry_active_perception/artifacts/week3/handeye_session_001/real_small_nbv_ik_preview.json
```

输出 JSON 会保存：正式报告路径和指纹、当前 `T_base_link7`、由正式外参得到的当前/目标相机
姿态、每次完整或缩短候选的 SolveIK 结果、控制器安全诊断，以及两个独立的实际运动命令计数。
只有 IK 成功、报告/外参完全一致、控制器执行门关闭、两个计数都为 0，最终状态才是 `passed`。

### 本次实跑结果

上述两项已经在真实机械臂反馈和正式外参下实际运行，结果均为 `passed`：

- `current`：对当前相机/`link7` 位姿做一致性检查，通过；
- `small_nbv`：目标只在计算中沿相机光学 `+X` 移动 5 mm；完整步长 `alpha=1` 第一次
  SolveIK 就成功，没有进入 `0.5/0.25/0.125` 缩短候选；最大关节解变化 `0.012 rad`，
  位置误差 `0.993657 mm`，姿态误差 `0.000576818 rad`。

安全证据也全部符合预期：bridge 内部计数和外部独立 topic 观察计数均为 0；控制器
`execution_enabled=false`；驱动 `control_enabled=false`；检查时关节速度为 0。5 mm 只是
送给 SolveIK 的数学目标，解没有发送给机械臂，机械臂没有发生运动。

- [`current` 只读结果](artifacts/real_handeye_current_solveik_preview.json)，SHA-256：
  `5b2c3d6096acc3caa12e49acfc0ddb0a9be4ca6bc481f81a1615a8a51f946413`
- [`optical +X 5 mm` 只读结果](artifacts/real_handeye_small_nbv_solveik_preview.json)，SHA-256：
  `5ca282f883f2b0affe9b7719c6554843ab45c401216046e3eb3dadc0447f421e`

这一步只证明“正式外参能进入真实 Placo 计算链，而且附近 5 mm 候选有解”。它不是一次
真实 NBV 运动。下一步必须另立真实闭环小步运动的安全门，完成现场空间、线缆、速度、控制
权限和停止方式检查并获得明确确认后，才可以让机械臂真正走一个小步。
