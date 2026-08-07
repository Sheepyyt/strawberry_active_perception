# NERO 单臂 Placo 控制

这个包完成第一周的目标：让 ROS 2 接收一个“末端 `link7` 要到达的位姿”，用 Placo 算出 NERO 的 7 个关节角，再生成平滑轨迹并交给机械臂执行。

## 先弄清楚三个角色

- **Placo 是计算员**：它负责 IK，也就是把末端位置和方向换成 7 个关节角。每次都从真实关节角开始算，并尽量靠近当前姿态，避免七自由度机械臂突然换到另一组解。
- **轨迹模块是路线绘制员**：它不会让关节从起点一下跳到终点，而是生成起止速度、加速度都为零的五次曲线。
- **`move_j` 是送信员**：它只接收已经算好的 7 个关节角并传给电机，**不会做 IK**。所以使用 `/control/move_j` 不等于使用原厂 IK。

本项目不启动 MoveIt，也不调用原厂笛卡尔接口 `/control/move_p`、`/control/move_l`、`/control/move_c`。同样禁用无平滑的 `/control/move_js` 和 MIT 控制。第一周没有环境避障，测试前必须由人清空工作空间。

## 构建

项目使用根目录的 `.venv`（提示符名称为 `sap-core`）。一定要让 `colcon` 也由这个 Python 运行，否则安装后的 ROS 节点可能找不到只装在虚拟环境里的 Placo。

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -c "import placo; print('Placo 可以导入')"

cd nero_ws
python -m colcon build --symlink-install \
  --packages-up-to strawberry_nero_control \
  --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
```

每次打开新终端都要依次执行：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash
```

## 最简单的纯 Python Placo 运动演示

如果想先把 ROS 2 完全放到一边，只确认“Placo 能否算出关节角并让 NERO 虚拟模型平滑运动”，运行下面的独立演示。它直接复用本项目的 `PlacoIKSolver` 和五次轨迹模块，但不启动 ROS 节点、不加载机械臂驱动、不连接 CAN，也不会发送任何真机命令。

```bash
cd /home/yyt/strawberry_active_perception
source .venv/bin/activate
cd nero_ws/src/strawberry_nero_control

python -m strawberry_nero_control.standalone_demo
```

终端会先用 ready 姿态计算 `link7` 沿基座 Z 方向移动 `-20 mm` 的目标，打印 Placo 关节解、末端残差、奇异性、最大关节变化和轨迹峰值，然后给出 MeshCat 地址。打开网页后回到终端按 Enter，模型会按 50 Hz 五次轨迹在 ready 与目标之间往返两次。

只计算、不打开网页：

```bash
python -m strawberry_nero_control.standalone_demo --dry-run
```

尝试另一个很小的相对位移（单位为米）：

```bash
python -m strawberry_nero_control.standalone_demo \
  --dx 0.01 --dy 0.00 --dz -0.02 --cycles 1
```

目标不可达、接近奇异点、关节越限、关节跨度超过 `0.35 rad` 或轨迹超限时，程序会拒绝播放并说明原因。这个演示验证的是运动学、连续性、平滑轨迹和模型显示，不验证电机、CAN、动力学、真实碰撞或环境避障。

## 第一步：只在网页仿真中检查

```bash
ros2 launch strawberry_nero_control sim.launch.py
```

该启动文件只运行 Placo 控制节点和 MeshCat 可视化器，**不会加载机械臂驱动，也不会连接 CAN**。终端会打印应在浏览器中打开的 MeshCat 地址，以该地址为准；`meshcat_port` 表示 MeshCat 的 ZMQ 端口，默认为 `0`，即自动选择空闲端口。如果手动指定的端口已被占用，查看器也会报告警告并自动换用空闲端口。先用只求解、不运动的服务检查接口：

```bash
ros2 interface show strawberry_nero_interfaces/srv/SolveIK
ros2 service type /strawberry_nero/solve_ik
ros2 action info /strawberry_nero/move_to_pose
```

MeshCat 在这里不是“带重力和碰撞的物理仿真器”，而是一块三维验算白板。它能显示当前关节姿态、目标坐标轴和规划出的末端轨迹；`sim` 控制节点还会真的运行 Placo IK、安全拒绝逻辑和五次关节轨迹，只是用“理想机械臂完全跟得上命令”来代替电机与 CAN。它不能验证重力、力矩、电机延迟、真实碰撞或环境避障。

仅仅打开网页时，机械臂会静止在 ready 姿态。要让整条仿真链实际运行，在第二个已加载环境的终端发送下面这个离线验证过的目标：

```bash
ros2 action send_goal /strawberry_nero/move_to_pose \
  strawberry_nero_interfaces/action/MoveToPose \
  "{target_pose: {header: {frame_id: base_link}, pose: {position: \
  {x: 0.14636685, y: -0.02563819, z: 0.41500512}, orientation: \
  {x: -0.33552712, y: 0.64814599, z: -0.62260026, w: -0.28230699}}}, \
  controlled_frame: link7, timeout: {sec: 10}}" --feedback
```

这时网页中的机械臂应平滑运动，并出现目标坐标轴与末端轨迹线；终端会依次报告检查目标、Placo 求解、生成轨迹、执行和稳定等待。默认采用完整的 STL 实体网格，以避开官方 `link4.dae` 在浏览器 Collada 渲染中的缺段问题。如果要对比官方彩色 DAE，可启动时覆盖参数：

```bash
ros2 launch strawberry_nero_control sim.launch.py \
  viewer_use_collision_meshes:=false
```

输入位姿的 `header.frame_id` 是参考坐标系，第一周应为 `base_link`；`controlled_frame` 应为 `link7`。相机虽然装在末端中心，但精确方向和毫米级偏移尚未标定，所以配置中的 `camera_transform_valid` 默认为 `false`，相机光学帧目标会被明确拒绝。

每次规划都会发布 `/strawberry_nero/planned_trajectory`（`trajectory_msgs/JointTrajectory`），便于记录、复查以及以后接入 VAMP。详细阈值见 `config/nero_control.yaml`。YAML 中的上下限是 pyAgxArm 提供的原始 SDK 限位；程序还会与 URDF 限位取交集，并只在最后统一向内缩进 2°，所得结果才是真正允许 IK 和轨迹使用的安全范围。

## 第二步：真机只读检查

只有网页验证和离线测试通过后，才连接电源与 CAN。先配置单个 CAN 适配器：

```bash
cd /home/yyt/strawberry_active_perception/nero_ws/src/agx_arm_ros
bash scripts/can_activate.sh can0 1000000
```

保证工作区无人、无杂物且急停随手可按，然后启动：

```bash
ros2 launch strawberry_nero_control real.launch.py can_port:=can0
```

也可添加 `launch_rviz:=true` 打开只读 RViz；RViz 的控制滑条始终关闭。

真机启动默认值被安全锁定为：

- 驱动 `auto_enable=false`、`control_enabled=false`、`fast_mode=false`；
- `speed_percent=10`、`effector_type=none`；
- Placo 执行开关 `execution_enabled_on_start=false`。

因此刚启动时只能读取 `/feedback/joint_states` 和机械臂状态，任何运动请求都应被拒绝。先检查反馈中的 `joint1` 到 `joint7` 顺序、弧度值、安装方向、固件状态与机械臂网页限位；启动程序**不会自动移动到 ready 位姿**。

确认无误后，需要分别打开原厂驱动的“允许收命令”开关、使能电机，以及本包的“允许执行”开关。先用下面命令确认服务的真实名称和类型，再按现场检查表操作，不要盲目复制使能命令：

```bash
ros2 service list -t | grep -E 'control_enable|enable_agx_arm|enable_execution|emergency'
ros2 topic echo /feedback/joint_states --once
ros2 topic echo /feedback/arm_status --once
```

停止时按相反顺序：先关闭本包执行开关，再关闭驱动控制门，最后失能电机。发生跟踪超差、反馈超时、奇异、越限、解跳变或取消时，控制器会拒绝或保持当前真实关节位置，并在 action 结果和诊断话题中说明原因。

## 已配置的安全起点

用户实测的 ready 关节角已写入 YAML，单位都是弧度：

```text
[0.0, -1.2698666571660342, 0.0, 1.8844843532558375,
 0.0, -0.00003490658503988659, 0.0]
```

它只是“已知安全的参考姿态”，不是上电后自动执行的 home 命令。尤其不要调用 `/move_home`：原厂 home 是全零姿态，而全零姿态接近本项目要避开的奇异构型。

## 后续 Gradient-NBV 怎么接

Gradient-NBV 只需把下一观察位姿送到 `MoveToPose` action。将来完成 `link7 -> camera optical` 外参标定后，把 `controlled_frame` 改成相机光学帧即可；IK、连续性检查、轨迹和真机执行接口都不用重写。相机驱动、Next Best View 和 VAMP 不在第一周范围内。
