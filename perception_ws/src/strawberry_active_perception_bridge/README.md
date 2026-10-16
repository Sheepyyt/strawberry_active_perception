# Strawberry active-perception bridge

本包连接 Gradient-NBV 的相机目标位姿与 NERO 的 Placo IK / 运动接口。
`real_nbv_supervisor` 负责本轮预览、SHA 绑定、逐步执行、持续地图更新和停止记录。
日常从项目根目录使用统一入口，无需单独启动本包节点：

```bash
./demo_operator.sh prepare
./demo_operator.sh run --workspace-clear
./demo_operator.sh open
./demo_operator.sh analyze
```

上电、初始摆放、中止及安全断电步骤见[演示说明](../../../DEMO_GUIDE_CN.md)。
停止运动使用 `./demo_operator.sh stop`；扶稳机械臂后使用
`./demo_operator.sh shutdown --arm-supported` 完成失能与进程收尾。

## 当前闭环

演示入口加载 `config/real_nbv_supervisor_large_step.yaml`，叠加 YOLO11 mask 配置，
并匹配 NERO 的 `large_nbv_experiment.yaml` 控制器配置。
每轮根据当前测得的相机、关节和草莓位姿重新规划，不要求复用上轮起点。

1. 采集新观测，通过正式手眼报告和曝光时刻的机械臂 TF 得到相机世界位姿。
2. 生成本轮预览并冻结配置、第一步目标和文件 SHA；执行前用新观测复核现场。
3. 在 100、75、50、25、10、5 mm 半径的候选中筛选 Placo 可达解，
   根据地图收益选择目标；运动后静止采集，并更新同一张地图。
4. 达到覆盖率或增量停止条件、无可达且有收益的候选、达到 8 步上限，
   或观测、IK、反馈检查失败时停止，并记录原因。步数上限不是必须完成的次数。

候选需满足当前配置中的相机工作距离、关节、IK 残差和反馈条件。
这些数值由配置及对应测试维护；本包不做环境碰撞检测。
`config/real_nbv_supervisor.yaml` 保留保守默认参数及其回归验证，演示入口显式使用
上面的厘米级配置。

## 数据约定

- 坐标变换为 `T_base_link7 = T_base_camera × inverse(T_link7_camera)`。
  手眼外参仅从 SHA 校验通过的正式报告读取。
- 每次地图更新使用五帧聚合：深度至少 3/5 有效才取中位数，mask 至少 3/5 为前景。
  每帧都检查身份、时间、相机模型和曝光时刻 TF，批内位姿变化需满足配置。
- 本轮只初始化一次地图；随后保持相同场景、配置 SHA、体素原点和地图。
  执行前的新照片不能替换第一步冻结目标；后续目标在运动前写入执行记录。
- 每个动作完成后关闭控制器与驱动的软件运动门，确认静止后再采集。
  停止后不会自动回位；断电收尾由统一演示入口完成。
- ROI 射线覆盖率表示目标区域中被有效观测射线经过的体素比例，
  不等同于草莓表面完整度。

每轮证据写入 `results/demo/run_日期_时间_编号/`，包括预览、授权、执行记录、
实际观测、地图和图表。周报由 `analyze` 离线读取这些文件生成。
旧预览只能作为历史证据，不能作为新一轮运动授权。

## 保留模块与验证

`calibration_report.py` 校验手眼报告，`transforms.py` 处理坐标变换，
`real_nbv_contract.py` 检查观测、冻结计划和会话状态，
`real_nbv_supervisor.py` 实现 ROS 闭环。包内 `test/` 及项目 `tests/nbv`、
`tests/demo` 验证当前流程；项目根目录的 `./verify_software.sh` 运行离线检查。
