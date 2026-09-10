# Strawberry NERO Control

本包把 `link7` 目标位姿变成 NERO 的安全关节轨迹：

```text
PoseStamped → Placo IK → 限位/奇异性/连续性检查
            → 五次平滑轨迹 → /control/move_j → 真实反馈验收
```

它不启动 MoveIt，也不调用原厂笛卡尔 IK。科研默认参数保持严格；大幅参观展示使用根目录
[`nero_exhibition_demo`](../../../nero_exhibition_demo/README.md) 的独立配置，不能把展示 YAML
传给 NBV 或普通科研实验。

## 构建与离线测试

先在仓库根目录应用固定 AGX 上游版本的安全补丁，再构建：

```bash
cd /home/yyt/strawberry_active_perception
./vendor_patches/agx_arm_ros/apply_checked.sh
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate

cd nero_ws
python -m colcon build --symlink-install \
  --packages-select agx_arm_msgs agx_arm_description agx_arm_ctrl \
    strawberry_nero_interfaces strawberry_nero_control
source install/setup.bash
```

必须使用装有 Placo 的 `.venv` 运行 `python -m colcon`。工作区配置会忽略
`agx_arm_moveit`。离线回归不会打开 CAN 或使机械臂运动：

```bash
cd /home/yyt/strawberry_active_perception
./verify_software.sh
```

## 科研默认值与展示模式的边界

[`config/nero_control.yaml`](config/nero_control.yaml) 是 NBV 和普通科研控制唯一默认配置：

- IK 位置/姿态误差：`2 mm / 2°`；
- 单次最大关节变化：`0.35 rad`；
- 最终真实到位误差：`10 mm / 5°`；
- Placo 关节范围在原厂/URDF 交集两侧各内缩 `2°`；
- 普通 `nero_pose_demo` 单步最多 `80 mm / 30°`，且仍检查关节变化、奇异性和残差；
- `recover`、`center-ready` 和普通真机运动均要求明确的现场确认词。

参观展示的 `50 mm / 5° / 1.50 rad / 0.5° margin` 等放宽值只存在于
[`nero_exhibition_demo/config/nero_exhibition.yaml`](../../../nero_exhibition_demo/config/nero_exhibition.yaml)，
并只由展示启动脚本显式叠加。展示路线、恢复方式和操作说明也只在展示目录维护。

5–10 cm NBV 实验仍属于科研控制，不使用展示参数。它只额外加载
[`config/large_nbv_experiment.yaml`](config/large_nbv_experiment.yaml)，把 precision 模式的
关节变化门从 `0.12 rad` 提到科研默认本来就允许的 `0.35 rad`；IK 误差、最终误差、速度、
奇异性、反馈监控和两道执行门保持科研值。监督器还会在运行时读取并核对这个参数，漏加载
或误加载展示配置都会在开门前拒绝。

## 每个终端的公共环境

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77
```

## 无 CAN 仿真

终端 1：

```bash
ros2 launch strawberry_nero_control sim.launch.py \
  viewer_use_collision_meshes:=true
```

终端 2 可读取、预览或只移动 MeshCat 虚拟模型：

```bash
ros2 run strawberry_nero_control nero_pose_demo current
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 30 0 15 --rpy-deg 0 0 8 --frame base
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 30 0 15 --rpy-deg 0 0 8 --frame base --execute
```

仿真不连接 CAN，也不模拟重力、真实电机或环境碰撞。

## 真机只读检查

先固定底座、清空机械臂完整扫掠区和下方空间、理顺线缆，并安排观察员。然后激活 CAN：

```bash
cd /home/yyt/strawberry_active_perception/nero_ws/src/agx_arm_ros
sudo bash scripts/can_activate.sh can0 1000000
```

根目录脚本只提供状态、只读启动和停止入口，不提供普通运动命令：

```bash
cd /home/yyt/strawberry_active_perception
./robot_operator.sh can
./robot_operator.sh start-readonly   # 保持这个终端运行
./robot_operator.sh status           # 另一个终端
./robot_operator.sh pose
```

`start-readonly` 不主动使能电机，且把原厂控制门和 Placo 执行门都保持关闭。NERO v1.11
在失能时可能不推送关节反馈；此时应按受监督启动流程处理，不能把“CAN 接口已 UP”误当成
“机械臂反馈正常”。

## 受监督恢复与普通小步测试

只有完成清场、线缆、绿灯、CAN、反馈和观察员检查后，才允许用 10% 速度启动受监督模式：

```bash
ros2 launch strawberry_nero_control real.launch.py \
  can_port:=can0 speed_percent:=10 startup_enable:=true \
  allow_limit_recovery_execution:=true \
  precision_test_mode:=true launch_rviz:=false
```

以下命令默认先预览；带 `--execute` 时还会要求输入准确确认词：

```bash
# 专用 inward-only 安全区恢复；执行确认词 RECOVER
ros2 run strawberry_nero_control nero_real_smoke_test recover
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute

# 10 段回到 ready；执行确认词 CENTER_READY
ros2 run strawberry_nero_control nero_real_smoke_test center-ready
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute

# 普通相对位姿；执行确认词 MOVE_POSE
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 5 0 0 --rpy-deg 0 0 0 --frame tool
ros2 run strawberry_nero_control nero_pose_demo relative \
  --xyz-mm 5 0 0 --rpy-deg 0 0 0 --frame tool --execute
```

NBV 真机闭环不由这些手工 Demo 命令驱动，而由
[`strawberry_active_perception_bridge`](../../../perception_ws/src/strawberry_active_perception_bridge/README.md)
对冻结计划、手眼外参、实时 IK、双软件门和一次/多步授权统一监督。

## 服务与 Action

- `/strawberry_nero/solve_ik`：只计算，不运动；
- `/strawberry_nero/move_to_pose`：通过安全检查后执行一条轨迹；
- `/strawberry_nero/recover_to_safe`：仅把受支持的轻微越界关节向内恢复；
- `/strawberry_nero/enable_execution`：Placo 执行门；
- `/control_enable`：原厂驱动控制门；
- `/electronic_emergency_stop`：NERO SDK 电子阻尼急停。

NERO 没有独立机械急停按钮。电子阻尼急停不是机械锁止，机械臂可能缓慢下降；直接失能或
断电可能立即下落。根目录的 `./robot_operator.sh e-stop` 会先请求电子阻尼急停再补关两道
软件门，但通信失效时仍只能依靠现场断电。任何异常后都不自动回位、不自动失能。

## 已知边界

- 当前控制器没有环境碰撞模型；桌面、支架、相机、线缆和人员只能靠清场与现场监督保证；
- MeshCat 的碰撞网格是可视化，不等于在线环境碰撞检测；
- 不要同时启动 MoveIt、关节滑块、原厂示例或第二个 `/control/move_j` 发布者；
- 历史真机结果是版本化证据，不代表本次设备仍在线或本次现场仍安全；每次只相信新鲜状态检查。
