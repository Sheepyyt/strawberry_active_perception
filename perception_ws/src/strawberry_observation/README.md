# strawberry_observation

ROS 2 Jazzy 的 Gemini 2 XL 按需采集适配器。它把相机的对齐 RGB-D
流转换为 `strawberry_perception_interfaces/Observation`，使真实相机、回放与
Gradient-NBV 共享同一份数据契约。该包不依赖 MoveIt，也不控制机械臂。

## 当前边界

- 用 `message_filters::ApproximateTime` 同步原始彩色图、对齐深度图、原始
  彩色 `CameraInfo` 和注册深度 `CameraInfo`。适配器用彩色 K/D 和同一 K
  调用 OpenCV `undistort`，因此 canonical RGB、mask 和硬件 D2C 深度落在
  同一零畸变像素网格上，不依赖厂商的 `image_undistorted` 分支。
- 四路输入必须尺寸和光学 frame 一致；两份 K 必须一致；各 CameraInfo 的
  时间戳必须等于对应图像，且四个时间戳均要求 `sec >= 0`、
  `nanosec < 1e9`。相机模型只接受 `rational_polynomial` + 8 个有限 D 或
  `plumb_bob` + 5 个有限 D；明确拒绝 `equidistant`。注册深度 D 还必须全零，
  否则整组帧会被拒绝而不会发布错误 Observation。
- 接受 `16UC1` 或 `32FC1` 深度，输出单位为米的 `32FC1`；0、非有限值和
  配置深度范围外的像素统一为 NaN。
- 彩色图统一为 `rgb8`。
- 用 HSV 红色双区间、5x5 开运算、5x5 闭运算提取草莓候选，只保留最大
  连通域；默认最少 200 px。
- 服务只返回 `observation_id` 和深度曝光时间。完整 Observation 发布到
  Reliable + TransientLocal topic，避免把大图复制进服务响应。
- 当前姿态为 `camera_session -> camera optical` 的单位变换，且
  `pose_valid=true`。这只表示相机在一次固定采集会话中不动，不是手眼标定；
  上机械臂或融合跨相机位姿前必须替换为曝光时刻的真实变换。

## 构建

先确保 Orbbec 驱动已在 `camera_ws` 构建。驱动的 rosdep 当前还要求
`libgoogle-glog-dev`，建议先补齐系统依赖。然后构建感知工作区：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source camera_ws/install/setup.bash
python -m colcon --log-base perception_ws/log build \
  --base-paths perception_ws/src \
  --build-base perception_ws/build \
  --install-base perception_ws/install \
  --packages-up-to strawberry_observation
source perception_ws/install/setup.bash
```

本机项目的 Python 虚拟环境使用 NumPy 2.x，而 Jazzy 的 apt OpenCV/cv_bridge
使用 NumPy 1.x ABI。该节点是 C++，运行时无需激活项目 `.venv`。

## 启动相机与适配器

优先使用 USB 3.x；但当前实机链路只协商到 **480 Mbit/s USB 2.0 High-Speed**，不是
SuperSpeed。该链路只对下面固定的 `640x400@10 Hz` 低带宽 profile 完成了条件 transport
验收，不能外推到更高分辨率、帧率或额外数据流。当前已知设备序列号是
`AYML241003A`：

```bash
ros2 launch strawberry_observation gemini2xl_observation.launch.py \
  serial_number:=AYML241003A
```

包装 launch 固定以下最低相机配置：640x400@10 Hz 彩色/深度、硬件 D2C、
帧同步和 depth scale 开启，IR、IMU 与两类厂商点云关闭。厂商
`gemini2XL.launch.py` 没有声明 `time_domain`，因此包装 launch 在相机 include
作用域内用 `SetParameter` 明确设置 `time_domain=global`。包装 launch 不启用
厂商 `enable_color_undistortion`：该路径会在重映射前清零同一份畸变参数，且
没有 raw color 订阅时解码分支也不会可靠启动，所以去畸变由 adapter 完成。

适配器默认输入：

- `/camera/color/image_raw`
- `/camera/depth/image_raw`
- `/camera/color/camera_info`
- `/camera/depth/camera_info`

默认输出与服务：

- Observation：`/strawberry/perception/observation`
- Capture 服务：`/strawberry/perception/capture_observation`

所有输入/输出 topic、深度范围、深度尺度、同步容差、HSV 阈值、形态学核、
最小 mask 面积和固定姿态开关都在
`config/gemini2xl_observation.yaml` 参数化。
回放原始 RGB-D rosbag 时复用同一节点，并把 `source_type` 设为 `4`
（`SOURCE_REPLAY`）、`source_name` 设为 bag/manifest 名称；几何与单位处理不分叉。

## 按需采集

另开终端观察 canonical topic；TransientLocal 可让迟加入订阅者收到最近样本：

```bash
source /opt/ros/jazzy/setup.bash
source camera_ws/install/setup.bash
source perception_ws/install/setup.bash
ros2 topic echo /strawberry/perception/observation \
  --qos-reliability reliable --qos-durability transient_local
```

请求一个新样本：

```bash
ros2 service call /strawberry/perception/capture_observation \
  strawberry_perception_interfaces/srv/CaptureObservation \
  "{scene_id: bench_001, not_before: {sec: 0, nanosec: 0}, \
  timeout: {sec: 2, nanosec: 0}, discard_frames: 3, \
  require_color: true, require_mask: true, require_pose: true}"
```

语义：

- 每次请求只考虑服务开始后到达的同步帧。
- 非零 `not_before` 进一步要求深度曝光时间不早于该值。
- `discard_frames` 丢弃指定数量的、已经通过尺寸/编码/5 ms 时差/深度质量
  检查的候选帧。当前固定采集流程显式传 3；请求值 0 仍严格表示不丢帧，
  不作为默认值哨兵。
- `scene_id` 必须非空，并原样写入 Observation。
- `require_color=false` 表示调用者不把彩色图列为输出完整性的额外要求；当前
  Gemini 适配器仍以 color + depth + 两份 CameraInfo 四路同步，用去畸变后的
  color 生成 mask，
  因而相机侧必须提供 color，成功的 Observation 也始终包含 `rgb8` color。
- ID 由深度曝光时间确定：`obs_<sec>_<nanosec>`，在同一 scene 内唯一且对
  相同输入可复现。
- `not_before.nanosec` 和 `timeout.nanosec` 必须小于 `1e9`；畸形 ROS 时间/
  Duration 会以 `INVALID_REQUEST` 拒绝。
- 需要 mask 时，未检测到至少 200 px 的红色最大连通域会继续等下一帧，
  直到超时。

canonical Observation 的 `header` 采用深度曝光时间；输出的是已校验的零畸变
注册深度 `CameraInfo`，其 stamp 归一化为同一深度曝光时间，同时保留注册后的
`camera_color_optical_frame`。color 与 depth 的真实时差保存在
`color_depth_skew_sec`。

同步订阅和服务回调位于不同 callback group，主程序使用三线程 executor；
服务等待新帧时不会阻塞相机消息回调。服务组自身互斥，避免并发 capture
争抢同一帧。

## 测试

```bash
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
python -m colcon --log-base perception_ws/test-log test \
  --base-paths perception_ws/src \
  --build-base perception_ws/build \
  --install-base perception_ws/install \
  --packages-select strawberry_observation
python -m colcon test-result \
  --test-result-base perception_ws/build/strawberry_observation --verbose
```

合成图像单测覆盖 16UC1/32FC1 转换、NaN 规则、BGR/RGB 转换、adapter 内
去畸变、基于去畸变图的双区间红色 mask、最大连通域、5 ms 时间校验、四路
尺寸/frame/stamp/K 校验、非有限彩色 D、非零深度 D 和可选 mask。launch
测试锁定最小相机参数，并实际执行 scoped `SetParameter`，验证得到
`time_domain=global`。

## 实机 transport 验收说明

实时 Python collector 即使把深度像素解码移到后台 worker，仍可能因大图反序列化和
callback 调度漏收 color；不能据此直接宣布相机掉帧。正式 30 分钟结果以并行 C++
`ros2 bag record` 的 topic count 和离线顺序解析的 sensor header stamp 为权威：在当前
USB 2.0 480M、640x400@10 Hz profile 下，color/depth 均约 9.97 Hz，估算丢帧约
0.0434%，sensor gap P99 约 100.73 ms，K/frame/encoding 稳定。该结果只通过 transport
子项；已知距离平面和 color-depth 边缘对齐尚未测，因此完整 G3 仍未通过。可提交摘要见
`validation/week2/artifacts/g3_camera_runtime_summary.json`，大型 bag 与五个完整 NPZ
Observation 样本保存在 Git 忽略的 `artifacts/week2/`。
