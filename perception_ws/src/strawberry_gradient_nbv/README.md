# strawberry_gradient_nbv

这是不依赖 ROS 1、MoveIt、ABB 或相机驱动的 Gradient-NBV 实现。纯核心只接收：

- 米制 `float32` optical-Z 深度，坏值为 NaN；
- 同像素网格的 0/255 `uint8` 目标 mask；
- 对应网格的 3x3 pinhole `K`；
- 明确的 `T_world_camera_optical` 4x4 位姿。

核心维护 occupancy/semantic log-odds 与独立的 `ever_observed` 位图。coverage 只表示
“目标 ROI 内曾被有效射线更新的体素比例”，不等同于真实物体表面覆盖率。默认把
640x400 输入以 `[::2, ::2]` 精确抽样成 320x200，K 同比例缩放；沿射线采样 128 点、
优化 10 次，最终平移被观察边界和 0.10 m 步长共同限制。姿态由稳定 look-at 构造，
相机光学 +Z 始终指向固定目标中心。

上游论文、commit、许可证边界和所有算法差异见 [NOTICE](NOTICE)。本包没有复制上游
ABB/MoveIt/ROS1/Gazebo 代码。

## 环境与离线运行

```bash
cd /home/yyt/strawberry_active_perception
python3 -m venv --system-site-packages --prompt sap-nbv .venv-nbv
source .venv-nbv/bin/activate
python -m pip install -r \
  perception_ws/src/strawberry_gradient_nbv/requirements-nbv.txt

export PYTHONPATH="$PWD/perception_ws/src/strawberry_gradient_nbv:$PYTHONPATH"
python -m strawberry_gradient_nbv.replay generate multiview \
  /tmp/strawberry_multiview.npz
python -m strawberry_gradient_nbv.replay inspect /tmp/strawberry_multiview.npz
python -m pytest -q perception_ws/src/strawberry_gradient_nbv/test
```

`plane` 和 `multiview` fixture、NPZ loader 都不使用 pickle，保存 color/depth/mask/K/
pose/stamp/scene config。五视角 fixture 的位姿是已知真值；它是 G2 的主定量验收来源。

把同一个 NPZ 逐帧发布成公共 `Observation`（保留 scene/observation ID、K、位姿、
米制深度和曝光时间）可运行：

```bash
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
ros2 run strawberry_gradient_nbv gradient_nbv_ros_replay \
  /tmp/strawberry_multiview.npz
```

发布器使用 Reliable + TransientLocal + KeepLast(1)，`source_type=SOURCE_REPLAY`；它不
隐式配置或更新 NBV 地图，计算仍需显式调用 `ConfigureNBV` 和 `ComputeNextView`。

### 四来源线级等价边界

`canonical_observation_from_fixture()` 是 offline、synthetic、NPZ replay 测试入口以及
Gemini adapter 输出模拟的共享 wire-level 工厂。同一规范化样本分别声明为
`SOURCE_REAL/OFFLINE/SYNTHETIC/REPLAY` 时，除 `source_type/source_name` 外，ROS CDR
序列化往返后的所有消息字段，以及解码得到的 rgb8、32FC1 米制深度（含 NaN）、mono8
mask、K、`T_world_camera_optical`、时间戳和 frame 必须完全相同；该性质由
`test_four_source_kinds_are_wire_equivalent_after_metadata` 交叉测试锁定。CDR 对齐填充字节
不属于消息语义，因此不参与比较。

这里的 `SOURCE_REAL` 路径从 **C++ adapter 已完成规范化的输出边界** 开始，用于证明
下游契约不会因来源标签改变；它不是实机采集测试。Gemini 的 16UC1 mm→32FC1 m、坏值
转 NaN 和 rgb/mask 规范化仍由 `strawberry_observation/test/test_observation_processor.cpp`
覆盖，硬件启动、同步和持续运行证据属于 G3。

## ROS 2 wrapper

先在 `.venv-nbv` 中构建 `perception_ws`，再启动：

```bash
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
ros2 launch strawberry_gradient_nbv gradient_nbv.launch.py
```

wrapper 订阅 `/strawberry/perception/observation`，提供 `/strawberry/nbv/configure`、
`/strawberry/nbv/reset_map` 和 `/strawberry/nbv/compute_next_view`，并在
`/strawberry/nbv/next_view` 发布 Reliable + TransientLocal 的结构化结果。同一
`(scene_id, observation_id)` 只融合一次；重复 Action goal 返回缓存结果。
