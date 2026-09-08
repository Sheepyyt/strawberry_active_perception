# NERO 大幅度参观演示

这是一个独立目录，不需要加入 ROS 包、不需要重新 `colcon build`，也没有修改现有项目文件。
程序复用当前项目已经运行的 `MoveToPose → Placo IK → 关节限位/奇异性检查 → 50 Hz
五次平滑轨迹 → 真实反馈跟踪` 安全链，不直接发布 `/control/move_j`。

展示所需的宽松 IK 范围只保存在 `config/nero_exhibition.yaml`，并由
`start_control.sh real` 显式叠加。科研/NBV 默认配置仍保持 2 mm、2°、0.35 rad
等严格门槛；不要在科研实验中传入这个展示配置。

提供两种模式：

- `poses`：沿安全路点依次到达展开、左侧、中心、右侧等固定位姿，并在主要端点默认额外停留 1 秒，适合讲解或拍照；
- `trajectory`：经过同一组安全路点，沿左右跨度约 **0.66 m** 的大弧线分段平滑运动，但默认不增加展示停留。

当前安全接口一次只接收一个位姿，因此 `trajectory` 是逐段五次平滑、每个路点检查真实到位的
轨迹，不是绕过安全链的无停顿流式控制。

## 最短操作流程

### 1. 先用仿真排练（不连接 CAN）

终端 1：

```bash
cd /home/yyt/strawberry_active_perception/nero_exhibition_demo
./start_control.sh sim
```

打开终端输出的 MeshCat 地址。终端 2：

```bash
cd /home/yyt/strawberry_active_perception/nero_exhibition_demo
./run_demo.sh poses
```

查看分段大弧线：

```bash
./run_demo.sh trajectory
```

只做完整离线 IK、限位和轨迹检查，完全不连接 ROS：

```bash
./run_demo.sh poses --preview-only
./run_demo.sh trajectory --preview-only
```

### 2. 真机展示

先固定底座、理顺相机线缆、清空扫掠区，再给控制箱上电。

终端 1 只需一条命令；它会激活 CAN、检查 1 Mbps / `UP` / `LOWER_UP` /
`ERROR-ACTIVE`，然后以 10% 速度启动驱动和 Placo 控制器：

```bash
cd /home/yyt/strawberry_active_perception/nero_exhibition_demo
./start_control.sh real can0
```

根据提示输入 `START`，并保持终端 1 运行。如果有多个 CAN 模块，可在第三个参数传 USB
地址，例如：

```bash
./start_control.sh real can0 1-2:1.0
```

终端 2 启动固定姿态展示：

```bash
cd /home/yyt/strawberry_active_perception/nero_exhibition_demo
./run_demo.sh poses
```

程序会要求一次 `START` 确认，然后自动完成：

```text
安全区 recovery
→ 10 段 center-ready
→ 每一路点读取最新反馈并现场 SolveIK
→ 附加展示安全门
→ MoveToPose 执行和真实到位验收
→ 最后回到 ready（保持使能）
```

轨迹展示只需把最后一个参数改为：

```bash
./run_demo.sh trajectory
```

可重复两轮，固定姿态到位后停留 1.5 秒：

```bash
./run_demo.sh poses --cycles 2 --pause 1.5
```

`--yes` 可以跳过第二次 `START`，但只应在同一现场已经完成扫掠区核查、且始终有人监督时使用。

## 默认运动范围

大幅展示先从项目 ready 展开到：

```text
SHOW_CENTER = [0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0] rad
SHOW_LEFT   = [1.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0] rad
SHOW_RIGHT  = [-1.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0] rad
```

固定模式的主要展示姿态顺序为：

```text
READY → CENTER → LEFT → CENTER → RIGHT → CENTER → READY
```

程序会先用 3 个过渡点从项目 ready 向 base_link 的 **-X** 方向展开，再在主要姿态之间插入
过渡路点。主要展示弧线中 `link7.x` 约为 `-0.391～-0.191 m`，不会进入 +X；左右段只改变
`joint1`，机械臂形状保持不变。末端左右跨度约 0.657 m，但每一个实际 Action 的最大关节变化
不超过约 0.502 rad，低于展示程序附加的 0.55 rad 门和控制器普通模式的 1.50 rad 门。
默认仍使用当前项目的 0.30 rad/s、0.50 rad/s²、50 Hz 五次轨迹限制。

## 现场安全边界

- 当前控制器和 Placo **没有环境碰撞检查**。MeshCat 的碰撞网格只是显示；桌面、相机、支架、
  线缆和观众均不在模型内。
- 预设路线已做 URDF 非相邻连杆自碰撞抽样，但这不替代每套实物的低速 MeshCat/真机排练。
- 原项目 ready 本身仍占用 +X 空间；完整模型在 ready 的 +X 边界约为 0.33 m。新的展开过程只会
  将该边界逐步收回，不会比 ready 更向 +X 伸展。+X 障碍物不能侵入机械臂当前 ready 包络。
- 真机启动时自动执行的 `recover` 和 `center-ready` 仍是现有项目的恢复路径，不属于新的 -X
  展示弧线。如果 +X 障碍物可能与启动恢复路径相交，必须先移开障碍物或单独完成恢复路径验证，
  不能仅依据主要展示弧线位于 -X 就直接运行。
- 新路线模型扫掠包络的水平半径约 0.42 m、高度约 0-0.57 m。现场至少清空底座周围半径 0.6 m、
  高度 0-0.7 m；观众保持 1 m 以上距离。相机或线缆外伸更大时，以实测外廓再增加余量。
- 不要同时启动 MoveIt、RViz 关节滑块、原厂控制示例或任何其它 `/control/*` 发布者；现有真机
  安全门会检查 `/nero_control` 是唯一的 `/control/move_j` 发布者。
- NERO 没有独立机械急停。本项目的急停是电子阻尼，不是机械锁止；失能或断电可能使机械臂
  因重力下落。必须有观察员在工作区外守住控制箱，机械臂下方始终无人。
- 运动中按 `Ctrl+C` 会请求取消当前 Action，控制器会停止后续路点并保持当前姿态；程序不会
  自动失能，也不会在异常后强行尝试回位。

## 无硬件检查

运行：

```bash
cd /home/yyt/strawberry_active_perception/nero_exhibition_demo
./check_demo.sh
```

检查内容包括两套路线的：

- 7 关节完整性、幅度和保守限位余量；
- 每段 Placo IK、姿态参考、奇异性与残差；
- 0.30 rad/s / 0.50 rad/s² 五次轨迹；
- 所有采样点的限位、端点静止和单调无超调。

如需改动作，只编辑本目录的 `demo_profiles.py`，然后依次运行 `check_demo.sh`、MeshCat 排练和
清场后的低速真机排练。不要通过放宽附加门限来让一个被拒绝的动作强行通过。
