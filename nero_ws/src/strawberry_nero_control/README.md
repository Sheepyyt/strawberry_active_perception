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

## 4. 真机现场 Demo：从上电到结束

至少两人配合：一人操作终端，一人观察机械臂并能接触实体急停。清空机械臂周围和下方；
失能前如果没有支架，必须人工托稳并避开夹点。

### 4.1 终端 1：激活 CAN

先执行“每个新终端的公共环境”，再运行：

```bash
cd /home/yyt/strawberry_active_perception/nero_ws/src/agx_arm_ros
bash scripts/can_activate.sh can0 1000000
ip -details -statistics link show can0
```

必须看到 `UP`、`LOWER_UP`、`ERROR-ACTIVE` 和 `bitrate 1000000`。出现 `BUS-OFF`、接口
不存在或错误持续增长时停止操作。

### 4.2 终端 1：启动驱动和 Placo 控制器

本机失能时可能不持续发送完整反馈，因此使用已经实测成功的启动方式：

```bash
cd /home/yyt/strawberry_active_perception
ros2 launch strawberry_nero_control real.launch.py \
  can_port:=can0 \
  speed_percent:=10 \
  allow_limit_recovery_execution:=true \
  first_motion_test_mode:=false \
  precision_test_mode:=true \
  startup_enable:=true
```

`startup_enable=true` 会使能电机并抱住当前姿态，现场应准备应对抱闸释放/结合时的轻微动作；
它不会自动发送 ready 或 Placo 目标。日志应出现：

```text
All joints enable status is True
Agx_arm feedback is ready, control is now enabled
NERO Placo controller ready
```

终端 1 保持运行。若仍没有完整反馈，不要反复使能；停止并检查电源、急停、CAN 线和适配器。

### 4.3 终端 2：检查状态

新开终端，先执行“每个新终端的公共环境”，再运行：

```bash
ros2 topic echo /feedback/arm_status --once
ros2 topic echo /feedback/joint_states --once
```

确认 `arm_status: 0`，七关节反馈完整，限位/通信报警均为 `false`，关节速度接近零。

### 4.4 检查并恢复到安全范围

先预览：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover
```

- `code=1, robot_is_safe=True`：已经安全，直接继续；
- `code=2`：运行下面的恢复执行；
- `code=12` 或其他错误：没有可靠反馈，停止排查，禁止执行。

```bash
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute
```

输入 `RECOVER`，必须得到 `code=0, success=True, robot_is_safe=True`。

### 4.5 回到统一 ready 起点

先预览：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test center-ready
```

预览通过后执行：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

输入 `CENTER_READY`。该工具把路径分为 10 段，每段均使用最新真实反馈和 Placo 求解，
不调用原厂 IK。

### 4.6 读取当前位姿

```bash
ros2 run strawberry_nero_control nero_pose_demo current
```

输出使用 `base_link` 坐标系；位置单位为米，四元数顺序为 `x y z w`。

### 4.7 展示一个明显的位置和姿态目标

先预览，不运动：

```bash
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 30 0 15 \
  --rpy-deg 0 0 8 \
  --frame base
```

预览应显示 `code=0, success=True`。随后执行完全相同的目标：

```bash
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 30 0 15 \
  --rpy-deg 0 0 8 \
  --frame base \
  --execute
```

输入 `MOVE_POSE`。成功标志是：

```text
Action status=4, code=0, success=True
说明：目标已稳定到达
```

### 4.8 现场输入其他目标

相对目标：

```bash
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm DX DY DZ \
  --rpy-deg DROLL DPITCH DYAW \
  --frame base
```

绝对目标：

```bash
ros2 run strawberry_nero_control nero_pose_demo absolute \
  --position-m X Y Z \
  --quat-xyzw QX QY QZ QW
```

也可用 `--rpy-deg R P Y` 代替四元数。位置必须是 3 个数，四元数必须是 4 个数，不能把
位置 Z 复制成 QX。目标命令默认仅预览；只有预览通过后，才给同一条命令添加 `--execute`。

Demo 外层允许不超过 `80 mm / 30°` 的单步请求，但这不是“保证可达范围”。Placo 结果
还必须同时满足：最大关节变化 `0.12 rad`、限位余量 `0.010 rad`、残差、奇异性和真实反馈
检查。任意一项失败都应换目标，不能放宽阈值强行执行。

### 4.9 回程

最不容易抄错位姿的方法是重新使用 ready 工具：

```bash
ros2 run strawberry_nero_control nero_real_smoke_test center-ready
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

第二条命令按提示输入 `CENTER_READY`。如果需要展示绝对目标回程，应在去程前保存
`nero_pose_demo current` 的位置和四元数，回程先用 `absolute` 预览，再添加 `--execute`。

### 4.10 安全结束

确认机械臂静止。由另一人托稳机械臂并避开夹点后，才运行：

```bash
ros2 service call /enable_agx_arm \
  std_srvs/srv/SetBool \
  "{data: false}"
```

最后在终端 1 按 `Ctrl+C`。运动中发现异常时直接使用实体急停。

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
