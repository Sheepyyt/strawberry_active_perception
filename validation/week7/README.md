# Week 7：停止条件驱动的长闭环与展示材料

本阶段不再把“三步”当作算法停止条件。大空间 profile 现在最多允许 8 个高层动作，但 8 只是
防止程序无限运行的最后保险；正常情况下会因为以下任一条件提前停止：

1. 目标 ROI 的有效射线 coverage 达到 20%；
2. 连续 2 步的 coverage 增量都小于 0.5 个百分点；
3. 下一视角位移不超过 1 mm；
4. 固定的多方向、多距离候选中没有“机械臂可达且能增加信息”的点；
5. 累计路径、目标可见性、相机数据、IK、机械臂反馈或关门检查触发安全边界。

全部动作仍使用同一个 `scene_id`、ConfigureNBV 请求 SHA、体素原点和体素地图。每到一个
新姿态采固定 5 帧，深度至少 3/5 有效才取中位数，mask 至少 3/5 同意才保留。

## 2026-09-13 真实结果

最新实验真实发送了 3 个高层目标；前两次运动后的观测成功更新地图，coverage
`4.94% → 11.05% → 17.77%`，相机分别实际平移 `98.51 / 98.48 mm`。第三次动作也正常
完成，但动作后 mask 内有效深度为 0，随后只读检查确认 Gemini 已从 USB 总线消失。程序
关闭两道执行门并停止，没有执行第四步。本次不能标为达到 20% coverage 的科学验收，但
已经真实验证连续厘米级重规划、同一地图累积以及相机链路失效时的安全停止。

完整的通俗说明、数字、图片入口和离线复现命令见
[RESULT_20260913_CN.md](RESULT_20260913_CN.md)。

## 只读记录相机过程图

先完成 preview 并从 JSON 读取 `scene_id`。执行前，在单独终端启动只有一个订阅器、没有
控制接口的记录器：

```bash
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 \
  validation/week7/observation_recorder.py \
  --scene-id <preview里的scene_id> \
  --output-dir artifacts/week7/observations \
  --timeout-sec 1200
```

它保存的是 NBV 真正消费的规范数据：矫正彩色图、米制对齐深度、HSV mask、相机内参和
曝光时刻位姿。它只读 `/strawberry/perception/real_nbv_observation`，不能让机械臂运动。

## 生成展示包

真实会话结束后，先找该 scene 对应的体素快照，再运行：

```bash
PYTHONPATH=perception_ws/src/strawberry_gradient_nbv \
  /usr/bin/python3 validation/week7/session_visualization.py \
  artifacts/week7/long_session_execution.json \
  --output-dir artifacts/week7/presentation \
  --snapshots artifacts/nbv_map_snapshots/<scene>/generation_*/map_*.npz

/usr/bin/python3 validation/week7/observation_visualization.py \
  artifacts/week7/observations/observation_*.npz \
  --output-dir artifacts/week7/presentation/observations

PYTHONPATH=perception_ws/src/strawberry_gradient_nbv \
  /usr/bin/python3 validation/week7/voxel_cloud_visualization.py \
  artifacts/nbv_map_snapshots/<scene>/generation_*/map_*.npz \
  --output-dir artifacts/week7/presentation/voxel_3d
```

输出包括一页式总览、coverage 曲线、三维相机轨迹、每步候选筛选漏斗、计划/实际运动对比、
到位误差和奇异性、草莓像素与地图体素数量、每一步的候选点三维图、RGB/mask/depth/点云
四联图、接触图、两个动态 GIF、可在浏览器打开的 `index.html` 和中文 `REPORT_CN.md`。
新增的三维体素云工具还会生成高清 3-D 总览、地图增长 GIF 和 360° 旋转 GIF。
`manifest.json` 给每个输入和输出保存 SHA-256，保证图表能追溯到原始实验记录。

图表与记录器完全离线/只读。真实运动仍只能通过经过 preview SHA 绑定、健康检查、IK 门和
双执行门保护的 `real_nbv_supervisor` 完成。
