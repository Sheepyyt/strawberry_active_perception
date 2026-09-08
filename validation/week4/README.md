# Week 4：红色目标真实 NBV 闭环

## 已完成的事实

2026-08-14 已完成一次真实闭环：Gemini 五帧聚合、Gradient-NBV、正式手眼变换、
Placo SolveIK、一个真实小步、关门、再次拍摄并更新同一张体素地图。

- 高层 `MoveToPose` Goal：1 个；44 条 `/control/move_j` 是这一 Goal 的 50 Hz
  平滑轨迹采样，不是 44 次运动。
- 动作后相机目标误差：`0.421 mm / 0.0115°`。
- coverage：`30.568% → 43.581%`，增加 `13.013` 个百分点。
- 两道软件门各取得两次关闭回执，随后独立观察 2 分钟，无第二动作或报警。

保留的正式证据只有三份：

- [`real_nbv_frozen_config_v3_preview.json`](artifacts/real_nbv_frozen_config_v3_preview.json)，
  SHA256 `8e3d032a794a9a5a37dfed4e24964daffefde32fd17df11d97fba529f14e256f`；
- [`real_nbv_frozen_config_v3_execution.json`](artifacts/real_nbv_frozen_config_v3_execution.json)，
  SHA256 `a22c6db98dcaab0b04e2e28e76429e731ece286ceb842f0dc3338ac7c71a442c`；
- [`real_nbv_post_motion_summary.json`](artifacts/real_nbv_post_motion_summary.json)。

旧的单帧、v1 和 v2 规划工具及失败 artifact 已删除；它们已被五帧聚合和冻结完整
ConfigureNBV 语义的 v3 流程取代。

## 现在新增的最多三步会话

仍使用同一个 `real_nbv_supervisor`，没有复制第二套控制程序。默认
`max_motion_steps=1`，只有明确设置为 `3` 并使用三步授权词时才允许最多三个 Goal。

三步模式的行为是：

1. 先生成新的只读 v3 preview。文件 SHA 同时绑定第一步目标、唯一地图配置，以及
   三步会话策略。
2. 真机 execution 用新拍的两批五帧复核现场，但第一步只能执行 SHA 中冻结的目标。
3. 第一步关门并静止后，在新姿态再固定拍五帧；这组 Observation 只融合一次。
4. 第二、第三步目标由同一张、未 reset 的地图动态产生。每个目标及其 IK 会在开门前
   原子写入会话 JSON。
5. 每一步只发一个 Goal，随后两道门各关闭两次并复核静止。任一错误会终止整个会话；
   不自动回位，也不自动失能。

固定硬门包括：每步相机平移 `(1, 5] mm`、旋转不超过 `10°`、IK 最大关节变化
`0.08 rad`、残差 `3 mm / 2°`、`sigma_min≥0.10`、条件数不超过 20；整个会话相对
起点累计不超过 `15 mm / 30°`。红色目标必须始终有至少 200 个有效 mask 深度像素。

每步 coverage 不得下降。如果单步增量低于 `0.5` 个百分点，程序会认为接近收敛并
提前停止，不为凑满三步而继续运动。正式科学验收还要求至少两步各增加 1 个百分点，
且最终比第一帧增加至少 20 个百分点；未达到时 artifact 会明确写
`scientific_acceptance_not_met`，不会伪装成通过。

## 无硬件软件测试

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
source perception_ws/install/setup.bash
export PYTHONPATH="$PWD/perception_ws/src/strawberry_active_perception_bridge:$PYTHONPATH"
/usr/bin/python3 -m pytest -q \
  perception_ws/src/strawberry_active_perception_bridge/test \
  validation/week4/test
```

状态机测试覆盖第二步 IK 失败、目标丢失、重复 Observation、coverage 下降、关门失败和
额外命令发布者。失败会锁存终止原因，后续 Goal 数不能增加；关门无法证明时会标记
`onsite_stop_required`，而不是继续运行。

2026-09-08 软件收口结果（全程未启动相机流、CAN 或机械臂）：

- 相机、NERO、perception 三个工作区均从独立空目录构建成功；
- 科研 NERO 控制 `105/105`，AGX 安全服务 `7/7`，展示隔离检查 `6/6`；
- Gradient-NBV `51/51`；接口、相机 adapter、手眼和 bridge 的 clean-colcon 结果为
  `150` 项、`0` 失败（`4` 项工具性跳过）；
- bridge 加 Week 4 会话测试 `79/79`，Week 2–3 验证工具 `45/45`；
- 联合 `rosdep` 检查通过。未使用的 `agx_arm_moveit` 仍按工作区配置排除。

厂商 AGX 子模块自己的全仓格式检查仍会报告其上游原有的排版/版权头问题；这不属于上述
功能门，也没有通过大规模格式化去改写厂商源码。项目对 AGX 的 7 项安全服务测试和固定版本
补丁复核均已通过。

## 下次上电后的操作顺序

在软件通知“已就绪”前不要上电。之后先按根目录 `CAMERA_GUIDE_CN.md` 和机械臂说明
启动相机、adapter、Gradient-NBV、只读 TF、NERO 驱动和控制器。先运行根目录状态命令，
再生成三步 preview：

```bash
cd /home/yyt/strawberry_active_perception
export ROS_DOMAIN_ID=77
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash
source nero_ws/install/setup.bash
source perception_ws/install/setup.bash

ros2 run strawberry_active_perception_bridge real_nbv_supervisor \
  --ros-args \
  --params-file perception_ws/install/strawberry_active_perception_bridge/share/strawberry_active_perception_bridge/config/real_nbv_supervisor.yaml \
  -p execute:=false \
  -p max_motion_steps:=3 \
  -p output_path:=/home/yyt/strawberry_active_perception/artifacts/week4/real_nbv_three_step_preview.json

sha256sum artifacts/week4/real_nbv_three_step_preview.json
```

只读文件必须为 `status=passed_preview_only`。在人工复核该文件并再次确认清场后，才把
上一步 SHA 填入下面命令：

```bash
ros2 run strawberry_active_perception_bridge real_nbv_supervisor \
  --ros-args \
  --params-file perception_ws/install/strawberry_active_perception_bridge/share/strawberry_active_perception_bridge/config/real_nbv_supervisor.yaml \
  -p execute:=true \
  -p max_motion_steps:=3 \
  -p execution_plan_path:=/home/yyt/strawberry_active_perception/artifacts/week4/real_nbv_three_step_preview.json \
  -p execution_plan_sha256:=<填入刚才的64位SHA> \
  -p execution_output_path:=/home/yyt/strawberry_active_perception/artifacts/week4/real_nbv_three_step_execution.json \
  -p operator_workspace_clearance_confirmed:=true \
  -p execution_authorization_token:=EXECUTE_REAL_NBV_SESSION_3
```

不要提前运行最后一条命令。三步授权只对这一个 preview SHA 和最多三个动作有效；一旦
至少一个 Goal 被发送，artifact 会锁存 `authorization_consumed=true`。完成后的同一
授权文件不能再次执行。监督器会在第一个 Goal 发送前，在 preview 旁边原子写入
`<preview文件名>.consumed.json`；即使换一个 execution 输出路径，也不能绕过这张消费凭据。
