# Strawberry NERO Control

本包让 NERO 单臂按照给定的 `link7` 目标位姿运动：

```text
目标位置和姿态 → Placo IK → 安全检查 → 五次平滑轨迹 → move_j → NERO
```

Placo 是唯一的末端位姿反解器。系统不启动 MoveIt，也不调用原厂
`move_p/move_l/move_c` 笛卡尔 IK；`move_j` 只发送 Placo 已经算好的 7 个关节角。

## 1. 首次安装和构建

### 下载包含子模块的仓库

```bash
git clone --recurse-submodules \
  https://github.com/Sheepyyt/strawberry_active_perception.git
cd strawberry_active_perception
git submodule update --init --recursive
```

### 创建 Python 环境

ROS 2 使用系统 Python 包，因此虚拟环境必须带 `--system-site-packages`：

```bash
cd /home/yyt/strawberry_active_perception
python3 -m venv --system-site-packages --prompt sap-core .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 构建 ROS 2 包

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate

cd nero_ws
python -m colcon build --symlink-install \
  --packages-select strawberry_nero_interfaces strawberry_nero_control
source install/setup.bash
```

必须使用 `python -m colcon`，让 ROS 2 可执行脚本使用装有 Placo 的 `.venv`。工作区的
`colcon_defaults.yaml` 会跳过 `agx_arm_moveit`。

## 2. 每个新终端的公共环境

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash

unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77
```

## 3. 无 CAN 的 MeshCat 演示

终端 1 执行公共环境命令，然后启动：

```bash
ros2 launch strawberry_nero_control sim.launch.py \
  viewer_use_collision_meshes:=true
```

打开终端输出的 MeshCat 网页地址。终端 2 执行公共环境命令，然后：

```bash
# 读取当前虚拟 link7 位姿
ros2 run strawberry_nero_control nero_pose_demo current

# 只预览，不运动
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 30 0 15 \
  --rpy-deg 0 0 8 \
  --frame base

# 执行同一个目标，只移动虚拟模型
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 30 0 15 \
  --rpy-deg 0 0 8 \
  --frame base \
  --execute
```

MeshCat 用于检查模型、IK、安全拒绝和轨迹连续性，不模拟重力、电机、CAN 或真实碰撞。

## 4. 真机现场 Demo：从未上电到目标位姿

本节是当前**现场展示版**的标准真机流程，按实际使用状态设计：

```text
机械臂未上电、自然耷拉
→ 控制箱上电
→ 激活 CAN
→ 启动 real.launch.py
→ 电机使能并抱住当前耷拉姿态
→ 此时部分关节允许暂时仍在 Placo 保守安全范围外
→ recover --execute：先把越界关节收回 Placo 安全范围
→ center-ready --execute：回到统一 ready 起始安全位姿
→ nero_pose_demo ... --execute：执行各种目标位姿
→ 任意时刻需要回起点：
   recover --execute && center-ready --execute
```

这里必须区分两个概念：

- **Placo 安全范围**：关节已经回到当前软件允许的保守关节范围内，可以开始普通 Placo IK；
- **ready 起始安全位姿**：项目定义的统一展示起点。`recover` 只负责前者，
  `center-ready` 才负责后者。

当前普通 Demo 模式使用：

```text
first_motion_test_mode=false
precision_test_mode=false
speed_percent=10
allow_limit_recovery_execution=true
startup_enable=true
```

当前展示版还采用以下参数：

```text
普通 IK 最大单关节变化                 1.50 rad
IK 允许的最终近似残差                 50 mm / 5°
运动完成后的最终验收                  50 mm / 5°
Placo 关节保守 margin                 0.5°
center-ready                          10 段 Placo 回位
```

早期测试阶段额外的 `80 mm / 30°` Demo 单步限制、`0.12 rad` Demo 门限，以及
`center-ready` 的 `0.50 / 0.06 / 0.08 / 0.12 rad` 测试小步门已经不再用于当前展示流程。

机械臂原始关节范围、真实反馈、CAN/驱动状态、专用 recovery 边界、Placo 求解、
奇异性检查和运行中的轨迹跟踪检查仍然保留。

### 4.1 未上电时的机械臂状态

允许从机械臂**未上电、自然耷拉**的状态开始。

此时机械臂可能存在某些关节超出 Placo 当前设置的保守安全范围。这不要求在上电前人工
把机械臂摆到 ready，也不要在未使能状态下把它强行掰到某个精确关节角。

先清空机械臂周围和下方空间，再给 NERO 控制箱上电。

本项目所用 NERO 没有独立机械急停按钮。软件急停是通过 CAN 发送的电子阻尼急停，
不是机械锁止；直接失能或切断电源后机械臂可能因重力下落。

### 4.2 终端 1：加载环境并激活 CAN

打开终端 1：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash

unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77
```

激活 CAN：

```bash
cd /home/yyt/strawberry_active_perception/nero_ws/src/agx_arm_ros
sudo bash scripts/can_activate.sh can0 1000000
ip -details -statistics link show can0
```

正常应看到：

```text
UP
LOWER_UP
ERROR-ACTIVE
bitrate 1000000
```

如果看到 `BUS-OFF`、找不到 `can0`，或者 CAN 错误持续增加，不继续启动真机运动。

### 4.3 终端 1：启动驱动、使能机械臂和 Placo 控制器

回到项目根目录：

```bash
cd /home/yyt/strawberry_active_perception

ros2 launch strawberry_nero_control real.launch.py \
  can_port:=can0 \
  speed_percent:=10 \
  allow_limit_recovery_execution:=true \
  first_motion_test_mode:=false \
  precision_test_mode:=false \
  startup_enable:=true
```

当前现场 Demo 始终使用：

```text
first_motion_test_mode=false
precision_test_mode=false
```

不要为了 recovery 临时切回 Week 1 的监督测试模式。

`startup_enable=true` 会使能电机并抱住**当前真实姿态**。如果机械臂上电前处于耷拉状态，
那么刚使能后它仍可能保持在一个超出 Placo 保守安全范围的关节状态；这正是后面
`recover --execute` 要处理的情况。

这一阶段：

```text
机械臂已经上电
机械臂已经使能
机械臂已经有真实关节反馈
但关节不一定已经进入 Placo 保守安全范围
```

因此**此时不要直接运行 `center-ready` 或目标位姿命令**。

终端 1 必须保持运行。正常日志应包含类似：

```text
All joints enable status is True
Agx_arm feedback is ready, control is now enabled
NERO Placo controller ready
```

### 4.4 终端 2：加载环境，并确认控制节点和 Action 已在线

新开终端 2：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash

unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77
```

先检查节点：

```bash
ros2 node list
```

至少应看到：

```text
/agx_arm_ctrl_single_node
/nero_control
```

再检查 Action：

```bash
ros2 action list -t | grep strawberry_nero
```

至少应看到：

```text
/strawberry_nero/move_to_pose
/strawberry_nero/recover_to_safe
```

也可以单独检查：

```bash
ros2 action list | grep recover_to_safe
```

必须出现：

```text
/strawberry_nero/recover_to_safe
```

如果没有看到该 Action，**不要运行 recovery**。先回终端 1 确认 `real.launch.py`
正在运行。刚刚重新 `colcon build` 之后，也应重新启动 `real.launch.py`，
否则可能出现：

```text
恢复预览 Action 不可用
```

### 4.5 终端 2：检查真实机械臂反馈

```bash
ros2 topic echo /feedback/arm_status --once
ros2 topic echo /feedback/joint_states --once
```

需要有完整七关节反馈。

此处的关节角**可以暂时超出 Placo 保守安全范围**，因为下一步就是专用 recovery。
但如果存在驱动、通信或机械臂自身报警，应先停止排查，而不是直接运动。

### 4.6 第一步运动：把耷拉姿态恢复到 Placo 安全关节范围

执行：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute
```

当前展示版不再要求输入 `RECOVER`。

程序会先读取真实关节，生成专用 recovery 预览，然后根据当前状态分两种情况。

#### 情况 A：当前已经在 Placo 安全范围内

可能看到：

```text
code=1, success=True
executed=False, robot_is_safe=True
说明：所有关节已经位于保守安全范围内，没有发送命令
```

这是正常结果。因为无需恢复，所以不会运动，可以直接进入下一节的 `center-ready`。

#### 情况 B：当前耷拉姿态有支持恢复的关节超出保守范围

可能先看到：

```text
code=2, success=True
executed=False, robot_is_safe=False
说明：安全区恢复预览完成；没有发送任何真机命令
```

随后程序继续执行专用的 inward-only recovery，把越界关节向保守安全区内收回。

恢复成功后应确认最终结果表示：

```text
success=True
executed=True
robot_is_safe=True
```

然后才能进入 `center-ready`。

当前专用 recovery 是按照已验证的真机恢复模式设计的，主要处理当前耷拉起始状态中
**J2 下限侧和 J4 上限侧**的越界。如果未来出现其它关节的异常越界，程序可能主动拒绝，
此时不要绕过 recovery 拒绝直接进入普通 Placo 运动。

### 4.7 第二步运动：恢复到统一 ready 起始安全位姿

当上一节已经确认：

```text
robot_is_safe=True
```

执行：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

当前展示版不再要求：

```text
先手工运行一次 center-ready 预览
输入 CENTER_READY
```

`center-ready --execute` 会自动：

```text
再次检查当前关节已经位于 Placo 安全范围
→ 读取最新真实反馈
→ 生成 10 段进入 ready 的 Placo 预览
→ 每一段重新读取真实反馈并重新 SolveIK
→ 每段生成五次平滑关节轨迹
→ 依次执行
→ 检查最终 ready 误差
```

成功时最后会看到类似：

```text
=== ready 中心化结果 ===
最大关节差=...
末端位姿差=...

Placo 分段进入 ready 姿态测试通过。
```

2026-08-19 已完成一次实机验证：从最大关节差约 `1.5359 rad` 的姿态开始，
10/10 段全部 `code=0, success=True`，最终相对 ready 最大关节差约
`0.00356 rad`，末端位姿差约 `0.785 mm / 0.0296°`。

如果这里出现：

```text
恢复预览没有确认 ALREADY_SAFE；禁止进入 Placo 阶段
```

说明当前还没有确认位于 Placo 安全范围。不要继续 `center-ready`，重新运行：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute
```

### 4.8 到这里，机械臂已经位于统一展示起点

此时完成了：

```text
耷拉状态上电
→ 电机使能并保持真实姿态
→ 专用 recovery 收回越界关节
→ Placo 10 段回到 ready
```

从这里开始可以运行各种目标位姿。

可选地先读取当前 `link7` 位姿：

```bash
ros2 run strawberry_nero_control nero_pose_demo current
```

输出使用 `base_link` 坐标系；位置单位为米，四元数顺序为 `x y z w`。

### 4.9 执行相对目标位姿

例如：

```bash
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm -80 0 -40 \
  --rpy-deg 0 0 15 \
  --frame base \
  --execute
```

当前程序会自动：

```text
读取执行前最新真实 link7 位姿
→ 根据输入生成相对目标
→ Placo IK 预览
→ 预览被接受
→ 直接执行平滑关节轨迹
→ 等待真实关节稳定
→ 输出最终实测误差
```

不再要求输入 `MOVE_POSE`。

参数含义：

```text
--xyz-mm DX DY DZ
```

表示相对于当前末端的位置变化，单位为 mm。

```text
--rpy-deg DROLL DPITCH DYAW
```

表示相对于当前末端的姿态变化，单位为度。

```text
--frame base
```

表示这些变化量按 `base_link` 坐标系解释。

如果只希望检查 Placo 解而不运动，去掉 `--execute`：

```bash
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm -80 0 -40 \
  --rpy-deg 0 0 15 \
  --frame base
```

### 4.10 执行绝对目标位姿

使用位置 + 四元数：

```bash
ros2 run strawberry_nero_control nero_pose_demo absolute \
  --position-m X Y Z \
  --quat-xyzw QX QY QZ QW \
  --execute
```

或者位置 + RPY：

```bash
ros2 run strawberry_nero_control nero_pose_demo absolute \
  --position-m X Y Z \
  --rpy-deg R P Y \
  --execute
```

绝对位置单位为米，四元数顺序必须是：

```text
x y z w
```

### 4.11 连续运行多个展示目标

每一条 `relative` 命令都会重新读取**执行前的真实当前位姿**，因此可以连续运行不同目标：

```bash
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm DX DY DZ \
  --rpy-deg DROLL DPITCH DYAW \
  --frame base \
  --execute
```

例如完成一个目标后，可以直接再运行下一条新的相对目标命令。

如果某个目标的 Placo 预览返回：

```text
success=False
```

该目标不会发送运动命令。修改目标后重新运行即可。

### 4.12 随时恢复到统一 ready 起始安全位姿

展示过程中，无论当前机械臂位于哪个正常展示姿态，需要回到统一起点时，使用下面这组
**标准回位命令**：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute && \
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

这可以视为当前现场 Demo 的“回起点指令”。

执行逻辑是：

```text
recover --execute
    ↓
如果已经在 Placo 安全范围：不运动，直接成功返回
如果有支持恢复的越界：先执行专用 recovery
    ↓
只有 recovery 成功
    ↓
center-ready --execute
    ↓
回到统一 ready 起始安全位姿
```

使用 `&&` 的原因是：如果 recovery 本身失败，第二条 `center-ready` 不会继续执行。

因此不需要人工判断当前到底要不要先恢复安全区。需要回 ready 时直接运行这组命令即可。

### 4.13 从未上电到运行目标位姿：最短现场流程

#### 终端 1

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77

cd nero_ws/src/agx_arm_ros
sudo bash scripts/can_activate.sh can0 1000000

cd /home/yyt/strawberry_active_perception
ros2 launch strawberry_nero_control real.launch.py \
  can_port:=can0 \
  speed_percent:=10 \
  allow_limit_recovery_execution:=true \
  first_motion_test_mode:=false \
  precision_test_mode:=false \
  startup_enable:=true
```

终端 1 保持运行。

#### 终端 2

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77
```

确认控制器在线：

```bash
ros2 node list
ros2 action list -t | grep strawberry_nero
ros2 topic echo /feedback/joint_states --once
```

从耷拉姿态恢复到统一 ready：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute && \
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

然后执行目标，例如：

```bash
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm -80 0 -40 \
  --rpy-deg 0 0 15 \
  --frame base \
  --execute
```

继续执行其它目标时，只需要继续运行新的 `nero_pose_demo relative/absolute ... --execute`。

需要随时回起点：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute && \
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

### 4.14 如果刚刚修改并重新构建过代码

修改 Python/配置并运行：

```bash
python -m colcon build --symlink-install \
  --packages-select strawberry_nero_control
source install/setup.bash
```

之后，**不要只在终端 2 直接运行客户端**。

如果终端 1 仍然是旧的 `real.launch.py` 进程，应先 `Ctrl+C` 停止，然后按照 4.3
重新启动 `real.launch.py`。

再用：

```bash
ros2 action list | grep recover_to_safe
```

确认：

```text
/strawberry_nero/recover_to_safe
```

已经存在，然后再执行 recovery。

### 4.15 安全结束

展示结束后，建议先回统一 ready：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute && \
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

确认机械臂已经停止运动。

如果确实需要失能，必须先确保机械臂受到可靠支撑并且人员避开夹点，然后：

```bash
ros2 service call /enable_agx_arm \
  std_srvs/srv/SetBool \
  "{data: false}"
```

最后在终端 1 按 `Ctrl+C` 停止 launch。

运动中发现异常时可以请求电子阻尼急停；如果 CAN 通信已经失效，只能使用现场控制箱
断电手段，并注意失能或断电后机械臂可能因重力下落。


## 5. 输入接口

- `/strawberry_nero/solve_ik`（`SolveIK.srv`）：只计算，不运动；
- `/strawberry_nero/move_to_pose`（`MoveToPose.action`）：计算并执行，支持反馈和取消；
- `/strawberry_nero/recover_to_safe`（`RecoverToSafe.action`）：启动安全恢复；
- `/strawberry_nero/planned_trajectory`：发布完整 7 关节规划轨迹；
- `/strawberry_nero/diagnostics`：发布状态和拒绝原因。

`nero_pose_demo` 是人工展示客户端。后续 Gradient-NBV 可直接调用 `MoveToPose`，不需要
修改 IK、轨迹和安全执行模块。相机外参尚未标定，因此目前控制的是 `link7`，不是已经标定
的相机光学中心。

## 6. 代码结构

| 文件 | 功能 |
|---|---|
| `control_node.py` | ROS 2 接口、反馈、安全门和真机执行 |
| `ik_core.py` | Placo FK/IK、限位、连续性和奇异性检查 |
| `trajectory.py` | 五次平滑关节轨迹 |
| `pose_demo.py` | 人工相对/绝对目标 Demo |
| `real_smoke_test.py` | 安全恢复、ready 和真机检查工具 |
| `ros_utils.py` / `models.py` | 消息转换、关节顺序和结果结构 |
| `meshcat_viewer.py` / `standalone_demo.py` | 可视化和纯 Python 演示 |
| `offline_benchmark.py` / `axis_suite.py` | 离线与六方向测试 |
| `acceptance_dataset.py` / `week1_acceptance.py` | 30×3 真机数据集与记录工具 |
| `config/nero_control.yaml` | ready、IK、轨迹和安全参数 |
| `test/` | 不连接 CAN 的自动回归测试 |

`strawberry_nero_interfaces` 是独立的 ROS 2 消息/服务/Action 合同；本包是具体实现。外层
`strawberry_nero_control/` 是 ROS 2 Python 工程，内层同名目录是可以被 Python 导入的
源码模块，这是 `ament_python` 的标准结构，不是重复代码。

## 7. 测试记录

简明过程、数据和原始 CSV/JSON 说明见项目根目录：

```text
validation/week1/README.md
```

运行不连接 CAN 的自动测试：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash
python -m pytest -q nero_ws/src/strawberry_nero_control/test
```
