# Strawberry active-perception bridge

这个包是 Gradient-NBV 相机位姿与 NERO Placo 控制之间的窄接口。普通 preview 节点
只调用 `/strawberry_nero/solve_ik`，不创建运动 Action client，也不发布关节/CAN 命令。
`real_nbv_supervisor` 是唯一允许真实闭环运动的入口。

相机目标通过正式手眼外参转换为：

```text
T_base_link7 = T_base_camera * inverse(T_link7_camera)
```

真实外参只能从 SHA 绑定的 30 姿态正式报告读取。仿真配置中的
`T_link7_camera` 只是 fixture，绝不能用于真机。

## 无运动仿真预检

```bash
source /opt/ros/jazzy/setup.bash
source /home/yyt/strawberry_active_perception/.venv/bin/activate
source /home/yyt/strawberry_active_perception/nero_ws/install/setup.bash
source /home/yyt/strawberry_active_perception/perception_ws/install/setup.bash
ros2 launch strawberry_active_perception_bridge simulation_preview.launch.py
```

第二个终端可运行固定变换/IK fixture：

```bash
ros2 run strawberry_active_perception_bridge nbv_ik_preview_fixture
ros2 run strawberry_active_perception_bridge gradient_placo_pipeline_fixture
```

两者都不会驱动机械臂。后者覆盖 synthetic Observation → Gradient-NBV → NextView →
SolveIK，并验证重复 ID 的幂等结果。

## 正式手眼外参的只读预检

```bash
ros2 launch strawberry_active_perception_bridge real_preview.launch.py
ros2 run strawberry_active_perception_bridge real_handeye_preview \
  --ros-args \
  --params-file /home/yyt/strawberry_active_perception/perception_ws/install/strawberry_active_perception_bridge/share/strawberry_active_perception_bridge/config/real_preview.yaml \
  -p preview_mode:=current
```

将 `preview_mode` 改为 `small_nbv` 可只读求解一个确定性的 5 mm 候选。它仍不会把
关节解发送给控制器。real 模式严格拒绝仿真相机 frame 和 synthetic Observation。

## 真实 NBV 监督器

监督器默认 `execute=false`、`max_motion_steps=1`。只读 preview 固定采两批、每批五帧：

- 每个深度像素至少 3/5 有效才取中位数，否则为 NaN；
- mask 至少 3/5 帧为前景才保留；
- 五个 ID/scene/stamp 必须唯一且时间递增；
- K、D、图像网格必须完全一致；
- 每帧都必须取得曝光时刻精确的 `base_link <- link7` TF；
- 批内相机位姿跨度不超过 0.25 mm / 0.10°。

preview 会把完整 ConfigureNBV 请求、体素维度和地图原点写入 v3 候选，并由整个 JSON
文件 SHA 绑定。execution 的新照片只能复核现场，不能替换第一步冻结目标。

### 由停止条件控制的持久地图

`max_motion_steps` 可设为 `1` 到 `10`。它是硬上限，不是要求必须运动这么多次。一个授权
允许程序在这个上限内持续观察，直到达到停止条件：

- 只在开始时调用一次 ConfigureNBV；全部动作共用同一 scene、配置 SHA、体素原点和地图；
- 每个动作关门并静止后，再采固定五帧并只融合一次；
- 后续目标来自累积地图，并在开门前原子写入 execution JSON；
- 每步相机平移 `(1, 5] mm`、旋转 `≤10°`、IK 最大关节变化 `≤0.08 rad`、
  残差 `≤3 mm / 2°`、`sigma_min≥0.10`、条件数 `≤20`；
- 相对会话起点累计相机变化 `≤15 mm / 30°`；
- 每一步结束都要让控制器门和驱动门分别获得两次关闭回执，并重新证明健康、静止；
- 目标少于 200 个有效 mask 深度像素、IK 失败、重复 Observation、coverage 下降、
  额外命令发布者或关门失败都会立即终止后续 Goal；不自动回位或失能；
- 达到可选的 `coverage_target` 时停止；设为 `0` 时禁用此条件；
- 默认连续两步 coverage 增量都小于 0.5 个百分点时停止；
- 下一位移不超过 1 mm 时停止；任何错误则立即停止。

多步实验的科学验收要求：coverage 单调不降；至少两步各增加 1 个百分点；最终比第一帧
增加至少 20 个百分点。安全完成但未达到这些数字时，artifact 会写明未通过科学验收。

默认的一步模式继续使用授权词 `EXECUTE_REAL_NBV_ONCE`。多步授权词为
`EXECUTE_REAL_NBV_SESSION_N`，其中 `N` 必须与 preview 的 `max_motion_steps` 完全相同，
例如 10 步上限使用 `EXECUTE_REAL_NBV_SESSION_10`。最大步数、coverage 目标、平台期阈值和
连续次数都进入 SHA，执行时不能临时改动。旧单帧/v1/v2 工具和 artifact 已删除，v2 schema 只保留
读取历史证据的兼容代码，永远不能执行。

完整现场命令和已完成的一步实测证据见
[`validation/week4/README.md`](../../../validation/week4/README.md)。

### 5–10 cm 独立实验

`large_workspace_experimental` 与上面的 1–5 mm 已验证配置完全分开。它最多执行三步，
单步只接受 `[5, 10] cm`，累计最多 30 cm / 45°，单步旋转≤15°，IK 最大关节变化≤0.35 rad。
Gradient-NBV 从 10 cm 开始做增益保持的回溯；Placo 会逐个拒绝不可达候选。
这项流程优先实验单独允许 5 mm / 2° 的 IK 与最终位置误差；默认小步科研配置仍保持
2 mm 求解和 3 mm 监督门，二者不会相互覆盖。
相机适配器仍要求 HSV 目标区域至少 200 像素；大步配置只把其中“通过深度离群点剔除”的
门槛设为 100（约一个 10×10 样本区域），以容忍小草莓轮廓上的深度缺失。原 1–5 mm
配置继续使用 200。

机械臂控制器必须显式加载配套 overlay：

```bash
ros2 launch strawberry_nero_control real.launch.py \
  profile_config_file:=/home/yyt/strawberry_active_perception/nero_ws/install/strawberry_nero_control/share/strawberry_nero_control/config/large_nbv_experiment.yaml \
  can_port:=can0 speed_percent:=10 startup_enable:=true \
  precision_test_mode:=true first_motion_test_mode:=false \
  allow_limit_recovery_execution:=false launch_rviz:=false
```

这条命令只应在现场清场、观察员就位后运行。它会使能电机以取得反馈，但两道运动门仍从
关闭状态启动，不会自行发送目标。先启动 Gradient-NBV，再生成无运动 preview：

```bash
ros2 run strawberry_active_perception_bridge real_nbv_supervisor \
  --ros-args \
  --params-file /home/yyt/strawberry_active_perception/perception_ws/install/strawberry_active_perception_bridge/share/strawberry_active_perception_bridge/config/real_nbv_supervisor_large_step.yaml \
  -p execute:=false
sha256sum artifacts/week6/large_step_nbv_preview.json
```

preview 必须明确为 `passed_preview_only`，而且要查看实际相机距离、旋转、关节变化和画面
边界。之后才允许使用同一配置执行；下面的 SHA 必须换成刚生成的完整 64 位值：

```bash
ros2 run strawberry_active_perception_bridge real_nbv_supervisor \
  --ros-args \
  --params-file /home/yyt/strawberry_active_perception/perception_ws/install/strawberry_active_perception_bridge/share/strawberry_active_perception_bridge/config/real_nbv_supervisor_large_step.yaml \
  -p execute:=true \
  -p execution_plan_path:=/home/yyt/strawberry_active_perception/artifacts/week6/large_step_nbv_preview.json \
  -p execution_plan_sha256:=<刚生成的64位SHA> \
  -p operator_workspace_clearance_confirmed:=true \
  -p execution_authorization_token:=EXECUTE_REAL_NBV_LARGE_SESSION_3
```

这不是对历史小步授权的放宽：profile 名、距离范围、累计范围、控制器 joint gate、preview
文件和新授权词都进入审计。默认小步配置及授权词不变。离线证据和 mask 候选见
[`validation/week6/README.md`](../../../validation/week6/README.md)。

2026-09-10 已按上述入口完成一次真实大步：计划 50.00 mm、实际 48.95 mm，coverage 从
2.0856% 增至 4.1400%，动作后两道门关闭。第二步约 10 cm 的原始建议在当时关节姿态下
不可达，因此会话在 1 个高层运动后停止。完整审计、地图快照和可视化见 Week 6 文档。

## 重要安全边界

控制器当前没有环境碰撞检查。软件只限制目标、IK、反馈和命令来源，不能识别桌面、支架、
线缆或人员。真实执行仍要求人工清场、工作区外人员守住控制箱断电位置。关门无法确认时
artifact 会标记 `onsite_stop_required=true` 并停止感知/后续动作。
