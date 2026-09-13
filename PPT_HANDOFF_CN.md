# 主动感知进展汇报 PPT：网页端 GPT 交接说明

## 建议提供的材料

如果网页端 GPT 能直接读取 GitHub 仓库，给它仓库地址并要求读取 `main` 分支的最新提交，
优先阅读以下文件：

1. `README.md`：项目总览、架构、当前进度与复现入口；
2. `validation/week10/` 整个目录：最新 YOLO11 真机实验、审计 JSON、图片与动画；
3. `validation/week3/HAND_EYE_RESULT_CN.md` 和
   `validation/week3/artifacts/stability_pose001_030_factory_raw_D.json`：正式手眼标定；
4. `validation/week2/README.md` 与其中的 G2 地图图片：MoveIt-free Gradient-NBV 离线验收；
5. `validation/REPRODUCIBILITY_MANIFEST.json`：上游版本、证据 SHA 和测试结果；
6. 原始研究计划
   `/home/yyt/Downloads/双臂协同主动感知草莓研究计划_杨雨婷_v2.docx`：用于对照最初目标与阶段安排。

如果只能上传少量文件，按下面的“最小材料包”提供：

- 原始研究计划 DOCX；
- 根目录 `README.md`；
- `validation/week10/README.md`；
- `validation/week10/artifacts/session/00_experiment_dashboard.png`；
- `validation/week10/artifacts/session/01_coverage_curve.png`；
- `validation/week10/artifacts/session/02_camera_trajectory_3d.png`；
- `validation/week10/artifacts/depth_failure_recovery_story.png`；
- `validation/week10/artifacts/voxel/voxel_cloud_final_3d.png`；
- `validation/week10/artifacts/evidence/yolo11_nbv_execution_target80.json`；
- `validation/week3/HAND_EYE_RESULT_CN.md`；
- `validation/REPRODUCIBILITY_MANIFEST.json`。

不要上传本机 `artifacts/week2/` 中约 9 GB 的相机长时间 rosbag、虚拟环境、build/install/log
目录或全部源码压缩包。它们对制作汇报没有帮助。GIF 如果网页端不能读取，可以只上传对应
PNG；完整视频可在本机把 GIF 转成 MP4 后再插入最终 PPT。

## 必须保持准确的边界

- 最新真实实验由固定 SHA 的 YOLO11m-seg 草莓 mask 驱动，并实际控制了两次 NBV 运动；
  HSV 五步闭环是较早的对照实验。
- coverage 是目标附近 ROI 的“有效深度射线覆盖率”，不是草莓表面重建完成度。
- 最新 YOLO11 实验完成 2 次运动，coverage 从 24.06% 到 37.87%、再到 44.87%；
  因没有剩余正收益可达候选而结束，不是固定写死走两步。
- 人为设置的 80% 是上限停止条件之一，并未达到；不能说本次重建完成了 80%。
- 软件会先用 Placo IK 删除不可达、贴近关节限位或过于奇异的候选，再在可达候选中比较
  Gradient-NBV 收益。
- 项目没有环境碰撞模型，真实实验仍依赖清场、观察员和受监督执行，不能描述为无人值守。
- 当前阶段是单臂、单相机、观察单颗真实草莓；学习式 mask 已接入，双臂和采摘尚未完成。

## 可直接复制给网页端 GPT 的 Prompt

```text
你是一名机器人主动感知方向的科研汇报顾问。请根据我提供的研究计划、GitHub 仓库文档、
正式 JSON 证据和实验图片，制作一份面向导师组会的中文进展汇报 PPT 方案。

项目主题：NERO 七自由度机械臂末端安装 Gemini 2 XL RGB-D 相机，复现并改造
Gradient-NBV，实现面向草莓的单臂主动多视角观察。当前核心链路为：RGB-D 与目标 mask
→ 统一 Observation → 持续更新同一张体素地图 → 在 Placo 可达候选中选择 Next Best View
→ 手眼变换 → IK → 机械臂运动 → 再观察，且不依赖 MoveIt。

请先阅读并交叉核对所有材料，数字以正式 JSON 和最新 Week 10 文档为准。不要根据文件名或
截图自行猜测数据。请把历史失败实验与最新成功实验分开，不要混在一起。

必须准确表达以下事实：
1. 最新真机实验使用 YOLO11m-seg 自动产生草莓 mask，完成 2 次高层运动，并持续更新同一张
   体素地图；coverage 依次为 24.06%、37.87%、44.87%，总增量 20.81 个百分点；
2. 两次实际相机平移约 9.83 mm 和 4.80 mm。随后没有剩余正收益可达候选，算法自动停止；
   不是写死只走两步，也不是报错；
3. coverage 指目标 ROI 中被有效深度射线观察过的体素比例，不是草莓表面真实完整度；
4. 当前 mask 来自用户提供、SHA-256 固定的 YOLO11m-seg 草莓权重；模型文件因来源许可未知
   不提交 Git，只提交哈希、软件配置和实验结果；
5. 已完成正式 eye-in-hand 手眼标定、MoveIt-free Gradient-NBV、Placo IK、可达候选筛选、
   双执行门和输入失效安全停止；
6. 当前没有环境碰撞模型，不能称为无人值守自主系统；
7. 第一次斜视尝试中 YOLO 轮廓仍正确，但草莓表面深度丢失；系统拒绝把背景融合进地图，
   返回有效视角后重新实验成功。这证明了输入失效安全停止，也暴露了深度可靠性问题；
8. 下一阶段是提高小型光滑草莓的深度可靠性、改进候选方向记忆，并把停止指标升级为更接近
   表面完整度的指标；双臂和采摘不属于当前已完成内容。

请输出 12～15 页 PPT 的逐页方案。每页包含：
- 页面标题；
- 3～5 条简洁正文；
- 推荐使用的现有图片文件名及摆放方式；
- 讲解备注（导师可能追问什么、我该怎样回答）；
- 如果需要图示，请给出可直接绘制的流程图内容，不要虚构实验照片。

建议结构：研究背景与问题、原计划与当前阶段、系统硬件、统一接口与坐标系、软件架构、
Gradient-NBV/体素地图的通俗原理、可达候选选择、手眼标定与安全门、实验设计与停止条件、
YOLO 两步定量结果、三维地图与轨迹可视化、深度失效及恢复、局限性、下一步计划、总结。

请额外提供：
A. 一页适合放在开头的“30 秒成果摘要”；
B. 一张从相机输入到机械臂再观察的 Mermaid 流程图；
C. 一张区分“已完成 / 正在做 / 尚未做”的表格；
D. 10 个导师可能提问的问题及严谨回答；
E. 最后一页可直接口头朗读的总结；
F. 对每张现有实验图说明它证明了什么、不能证明什么。

风格要求：中文、科研但易懂、少段落多图表、数字醒目、不要营销化。所有结论分成
“代码/离线测试证明”“真实硬件实验验证”“下一步计划”三类。对不确定或材料不足的内容
明确写“尚未验证”，不得补造实验结果、模型精度、引用或硬件指标。
```
