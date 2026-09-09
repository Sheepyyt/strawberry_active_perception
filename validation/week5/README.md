# Week 5：真实草莓（HSV mask）多步闭环

这一阶段把镜头前的红色测试物换成了真实草莓，但目标区域仍由 HSV 红色阈值产生，尚未接入
SAM 3 或其他学习式分割模型。因此它证明的是“真实草莓 + 真实深度 + 多步 NBV”能连起来，
不是“系统已经理解什么是草莓”。

## 已取得的结果

- 同一张体素地图只配置 1 次；
- 连续完成 2 个真实运动 Goal；
- ROI 有效射线覆盖率从 `23.94%` 增至 `35.21%`，再增至 `41.65%`；
- 总增量为 `17.71` 个百分点，两个动作后执行门均关闭；
- 第三个候选位移不超过 1 mm。当时的代码把它写成失败；现在的代码会在已经完成动作的
  前提下把它解释为“没有值得执行的新位移，提前收敛”，但这一行为修复尚未重新做真机实验。

精简摘要见
[`real_strawberry_hsv_multistep_summary.json`](artifacts/real_strawberry_hsv_multistep_summary.json)，
完整原始审计记录见
[`real_strawberry_hsv_multistep_execution.json`](artifacts/real_strawberry_hsv_multistep_execution.json)。

## 下一步

把“产生 mono8 mask”做成可替换的 provider：保留 HSV 作为快速基线，新增 SAM 3.1 文本提示
`strawberry` 的 provider。先离线比较两种 mask，再让通过质量门的 SAM mask 进入现有
Observation；NBV、手眼、IK 和运动监督器不需要重写。
