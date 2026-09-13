# 真实 NBV 闭环展示报告

- 程序状态：`executed_session_scientific_acceptance_passed`
- 实际高层运动次数：2
- 成功拍照并更新地图的运动次数：2
- coverage：24.06% → 44.87%
- 停止原因：no positive-gain reachable candidate remained in the predeclared multi-radius search lattice
- 运动结束时两道执行门：请查看审计 JSON

| 步骤 | 实际移动 | 实际转动 | coverage | 本步新增 | 最终误差 | 有效目标像素 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.8 mm | 1.74° | 37.87% | +13.81 pp | 0.30 mm / 0.010° | 523 |
| 2 | 4.8 mm | 0.64° | 44.87% | +7.00 pp | 0.27 mm / 0.025° | 553 |

## 图片怎么读

- `00_experiment_dashboard.png`：一页式总览，适合直接放汇报 PPT。
- `01_coverage_curve.png`：蓝线越往上，代表更多目标附近空间被相机射线看过。
- `02_camera_trajectory_3d.png`：绿线是相机真实走过的路，蓝箭头是每次朝向。
- `03_candidate_funnel.png`：灰色是机械臂去不了的点，橙色是能去但没新信息，蓝色是能去且有用。
- `04_motion_profile.png`：对比计划动作与真实动作。
- `05_accuracy_safety.png`：显示到位误差、奇异性和关节变化是否在门槛内。
- `06_target_map_quality.png`：显示草莓像素、体素地图大小和轨迹采样点。
- `candidate_step_*.png`：每一步完整备选观察点的三维分布。
- `map/nbv_map_progress.gif`：体素地图随观察次数增长的动画。
