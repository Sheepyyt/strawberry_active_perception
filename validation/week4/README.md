# Week 4：红色目标真实 NBV 闭环

## 已完成的事实

2026-09-08 已从一个新的现场安全起点完成红色目标三步真实 NBV 闭环。旧实验起点与
现场手动恢复后的姿态相差约 24 cm，因此没有冒险做大幅“精确回位”；先将当前姿态定义为
新起点，并沿 `base_link +Z` 平移 10 mm 重新居中关节余量，然后从零建立新地图。

- 同一个 `scene_id`、ConfigureNBV 配置、体素原点和地图连续完成 3 个高层 Goal；
- coverage：`24.682% → 40.719% → 44.754% → 46.371%`，总增量
  `21.689` 个百分点；
- 三步红色有效深度像素为 `419 / 416 / 416`，均高于 200；
- 三步实际相机运动依次约为 `4.622 mm / 2.686°`、`1.417 mm / 0.084°`、
  `1.047 mm / 0.102°`；
- 第二、第三步来自前面累积的地图，并分别在开门前冻结动态目标 SHA；
- 每步结束两道门各取得两次关闭回执；最终无 CAN、限位或机械臂错误，额外观察
  4 秒没有新的 `/control/move_j`；
- 完整执行 artifact SHA256 为
  `41061c9e97a798958b1f64d875072bd82e54d850949eecd78d205217d5de8bd9`，小型可提交摘要见
  [`real_nbv_three_step_recentered_summary.json`](artifacts/real_nbv_three_step_recentered_summary.json)，
  完整审计记录见
  [`real_nbv_three_step_recentered_execution.json`](artifacts/real_nbv_three_step_recentered_execution.json)。

现场 USB2 彩色流偶有短暂停顿，原 2 秒 capture 超时曾使第一步后安全终止。默认超时已
改为 5 秒；它只增加等待时间，坏帧仍然不会被发布，也没有放宽机械臂运动限制。

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
2. 真机 execution 用新拍的两批五帧复核目标身份、起点和现场，但第一步只能执行 SHA
   中冻结的目标。单视角可能有多个近似等价的 NBV 方向，因此 fresh optimizer 是否再次
   选中同一方向只记录为诊断，不会替换或否决已冻结目标；即时 IK 和所有运动安全门仍是
   强制条件。
3. 第一步关门并静止后，在新姿态再固定拍五帧；这组 Observation 只融合一次。
4. 第二、第三步目标由同一张、未 reset 的地图动态产生。每个目标及其 IK 会在开门前
   原子写入会话 JSON。
5. 每一步只发一个 Goal，随后两道门各关闭两次并复核静止。任一错误会终止整个会话；
   不自动回位，也不自动失能。

固定硬门包括：每步相机平移 `(1, 5] mm`、旋转不超过 `10°`、IK 最大关节变化
`0.08 rad`、残差 `3 mm / 2°`、`sigma_min≥0.10`、条件数不超过 20，且 IK 目标的
任一关节必须离科研安全限位至少 `0.001 rad`；整个会话相对起点累计不超过
`15 mm / 30°`。红色目标必须始终有至少 200 个有效 mask 深度像素。平移和旋转分别
沿原始 NBV 的直线与最短旋转弧缩短，二者独立使用同一组确定性比例；这样不会因共用
一个比例而错过本来满足全部硬门的小步候选，也不会把目标留在编码器微小波动会反复
跨越的限位边界上。

Gradient-NBV 使用 `float32` 计算；若它声称已限幅到 5 mm、但监督器用 `float64`
复算时仅多出不超过 1 微米，监督器会沿原方向把平移精确缩回 5 mm，并把修正量写入
artifact。超过这项纯数值容差仍会拒绝，最终运动上限没有放宽。

五帧聚合中的单张图只记录目标深度支持度，不单独用“200 个有效像素”否决整批；这是
为了让预先规定的“至少 3/5 帧有值才取深度中位数”真正能够容忍最多两张局部坏帧。
最终聚合 Observation 仍必须有至少 200 个红色有效深度像素，门槛没有降低。聚合后
的目标深度先按大于 50 mm 的间隔分层，再选取离相机最近且具有连续像素支持的一层；
透明或镂空目标后方的远处背景不能用于凑满 200 个目标像素。

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

2026-09-09 软件收口复核结果（测试命令不启动相机流、CAN 或机械臂）：

- 相机、NERO、perception 三个工作区均从独立空目录构建成功；
- 科研 NERO 控制 `105/105`，AGX 安全服务 `7/7`，展示隔离检查 `6/6`；
- Gradient-NBV（含地图快照/可视化）`56/56`，手眼 `34/34`；
- bridge 加 Week 4 会话测试 `91/91`，Week 2–3 验证工具 `45/45`；
- 当前 perception 工作区 `colcon test-result` 汇总 `209` 项、`0` 错误、`0` 失败，
  `4` 项工具性跳过；
- 联合 `rosdep` 检查通过。未使用的 `agx_arm_moveit` 仍按工作区配置排除。

厂商 AGX 子模块自己的全仓格式检查仍会报告其上游原有的排版/版权头问题；这不属于上述
功能门，也没有通过大规模格式化去改写厂商源码。项目对 AGX 的 7 项安全服务测试和固定版本
补丁复核均已通过。

## 每次新实验的操作顺序

先按根目录 `CAMERA_GUIDE_CN.md` 和机械臂说明启动相机、adapter、只读 TF、NERO 驱动和
控制器，并重新核验现场状态。Gradient-NBV 必须监听监督器补入
真实 `base_link` 位姿后的专用话题，不能使用默认原始 Observation 配置：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
.venv-nbv/bin/python -m strawberry_gradient_nbv.ros_node --ros-args \
  --params-file perception_ws/install/strawberry_gradient_nbv/share/\
strawberry_gradient_nbv/config/real_nbv.yaml
```

确认 `/gradient_nbv` 已启动后，再运行根目录状态命令并生成三步 preview：

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

新版本的真实 NBV 配置还会把每次地图更新保存到 `artifacts/nbv_map_snapshots/`。会话结束后
可用 `strawberry_gradient_nbv.map_visualization render` 生成三视图 PNG 和动态 GIF；历史
三步 artifact 只保存体素统计，不能反推当时每个体素的完整状态。
