# Week 6：学习式 mask 候选与 5–10 cm 大步 NBV 实验

## 2026-09-13 多方向三步真机结果

可达候选选择已经完成真机验证。机械臂起始姿态因人工摆放使关节 2/4 轻微越过 2°保守区，
专用恢复程序先只把这两轴低速移回安全区。第一次 15°旋转上限的只读 preview 检查了
114 个点：49 个 IK 不可达，另 65 个 IK 可达但都需旋转 15–27°，所以没有运动。大空间
profile 随后单独改为 30°（普通科研 profile 仍为 10°），第二次 preview 通过。

冻结 preview SHA-256 为
`f4920998b6879c619ed40df735f6a3bc756d6d3d7ab8ed2717f421eb54316e1f`：114 个候选中
30 个通过 IK/安全门，2 个有正收益，选择了 10 cm / 24.66° 的候选。一次性三步授权随后
完成：

| 步 | 实际平移 | 实际旋转 | coverage | 增量 | 动作后误差 |
|---:|---:|---:|---:|---:|---:|
| 1 | 98.96 mm | 24.63° | 3.24% | +1.69 pct | 1.34 mm / 0.025° |
| 2 | 97.79 mm | 2.63° | 5.31% | +2.06 pct | 2.23 mm / 0.040° |
| 3 | 97.99 mm | 6.71° | 8.80% | +3.49 pct | 2.16 mm / 0.024° |

只调用一次 ConfigureNBV，三步使用同一 scene、地图原点和体素图；3 个高层 MoveToPose
对应控制器生成的 98/84/91 个平滑轨迹采样点，不是 273 个独立目标。每步后两道门各取得
两次关闭回执；最终 arm/motion/error 为 0，CAN、七轴通信与限位均正常。总 coverage 增量
7.24 个百分点，低于事先写下的 20 个百分点科学门槛，因此 artifact 状态诚实保留为
`executed_session_scientific_acceptance_not_met`。这不否定三步闭环已无报错跑通。

完整可复核文件：

- [冻结 preview](../../artifacts/week6/reachable_large_step_preview_r30_20260913.json)
- [三步执行审计](../../artifacts/week6/reachable_large_step_execution_r30_20260913.json)
- [一次性授权回执](../../artifacts/week6/reachable_large_step_authorization_r30_20260913.json)
- [114 个候选的可达/收益图](../../artifacts/week6/reachable_large_step_candidates_r30_20260913.png)
- [最终真实体素地图](../../artifacts/week6/reachable_large_step_map_r30_20260913/nbv_map_final.png)
- [四次地图更新动画](../../artifacts/week6/reachable_large_step_map_r30_20260913/nbv_map_progress.gif)

四个原始、无 pickle 的 NPZ 快照位于
`artifacts/nbv_map_snapshots/real_nbv_once_20260913T083800Z/generation_003/`，可按本文后面的
渲染命令重新生成 PNG/GIF。

## 2026-09-10 真机结果

本轮继续使用 HSV 红色 mask，在真实草莓模型、真实 Gemini 2 XL 和真实 NERO 上完成了
一个约 5 cm 的 Gradient-NBV 动作：

- 冻结 preview SHA-256：
  `b7d7155379f2fb0611c2de86fce64d71583806e02400715600eb19f273e1a1df`；
- 计划/实际相机平移 `50.00 / 48.95 mm`，实际旋转 `2.78°`；
- 末端最终位置误差 `1.12 mm`、姿态误差 `0.023°`；
- 同一张体素地图的 ROI 有效射线 coverage 从 `2.0856%` 增至 `4.1400%`，
  增加 `2.0544` 个百分点；
- 本步只发出 1 个高层 MoveToPose，控制器按约 50 Hz 生成 61 个轨迹采样点；
  动作完成后两道执行门均已关闭，机械臂、CAN 和 7 个关节反馈正常；
- 第二步 Gradient-NBV 给出了约 10 cm 的建议，但从当时关节姿态出发没有通过可达性门，
  因而会话安全停止，没有强行执行第二、第三步。

这里的 `status=failed` 指“最多三步的整场科学验收未完成”，不代表第一步运动失败。
第一步已经成功到达并更新地图；停止暴露出两个工程问题：当前监督器只会把同一 NBV 方向
逐级缩短，还不会比较其他高收益、可达的方向；实验配置又把 5 cm 当成硬下限，导致可能
可用的较短动作也被拒绝。这与“优先大步、必要时自动缩短”的真实实验意图不一致，下一版
已经改为 10/7.5/5/2.5/1/0.5 cm 分层候选；上述 2026-09-13 实验已完成真机验证。

可复核文件：

- [冻结的只读 preview](artifacts/real_large_step_nbv_preview.json)
- [完整真实执行审计](artifacts/real_large_step_nbv_execution.json)
- [运动前地图快照](artifacts/real_large_step_nbv_map_step_001.npz)
- [运动后地图快照](artifacts/real_large_step_nbv_map_step_002.npz)
- [最终体素地图](artifacts/real_large_step_nbv_map_final.png)
- [运动前后动态对比](artifacts/real_large_step_nbv_map_progress.gif)

重新渲染（只读，不连接硬件）：

```bash
cd /home/yyt/strawberry_active_perception
PYTHONPATH=perception_ws/src/strawberry_gradient_nbv \
  .venv-nbv/bin/python -m strawberry_gradient_nbv.map_visualization render \
  --output-dir /tmp/real-large-nbv-map \
  validation/week6/artifacts/real_large_step_nbv_map_step_001.npz \
  validation/week6/artifacts/real_large_step_nbv_map_step_002.npz
xdg-open /tmp/real-large-nbv-map/nbv_map_final.png
```

本阶段同时处理两个问题，但不把两项风险混在同一次首次实验里：

1. 真实大步实验先继续使用已经真机验证过的 HSV 红色 mask，只改变 NBV 运动尺度；
2. 网络分割权重先离线处理同一批图片，证明它能稳定给出正确 mask 后，下一轮才替换 HSV。

## 网络上能否直接找到草莓分割模型

能找到，但“网页上有一个 `.pt` 文件”和“可以安全、可复现地接入机器人”不是同一件事。
本轮核验结果固定在 [`mask_model_candidates.json`](mask_model_candidates.json)：

- 首选学习式候选是 `LCAS/aoc_fruit_detector`。它明确是草莓果实的 Mask R-CNN 实例分割，
  有 ROS 2 代码和 Apache-2.0 许可证；缺点是 Detectron2 依赖较旧，必须放在独立环境中。
- Hugging Face 的 `Ibrahim-Hegazi/strawberry-segmentation` 确实会输出 mask，但它主要区分
  健康叶、健康草莓和多种病害；仓库没有模型卡、精度、训练数据说明或许可证。因此只作为
  离线候选，不能直接成为真机默认输入。
- 普通 COCO Mask R-CNN 没有 `strawberry` 类；只有检测框的 YOLO 模型也不能代替像素 mask。

离线评估工具是 [`mask_model_evaluator.py`](mask_model_evaluator.py)。它没有 ROS publisher、
Service/Action client，也不能运动机械臂。它只输出 `mono8` 黑白 mask、绿色叠加图和 JSON。
第三方 `.pt` 可能包含 Python pickle，工具会先核对 SHA，并要求显式确认后才加载：

```bash
mkdir -p artifacts/models artifacts/week6/mask-eval
curl -L \
  https://huggingface.co/Ibrahim-Hegazi/strawberry-segmentation/resolve/main/best.pt \
  -o artifacts/models/hf_strawberry_segmentation_best.pt
sha256sum artifacts/models/hf_strawberry_segmentation_best.pt
# 必须等于 e89a33b2b53c89fad0deac0d7922ce53cfe780c230c593f569d0dd4852062adc

# 建立独立环境；不要装进 Placo 或 NBV 环境：
python3 -m venv .venv-mask
.venv-mask/bin/python -m pip install -r validation/week6/requirements-mask.txt

# 对一张保存的相机图离线运行：
.venv-mask/bin/python validation/week6/mask_model_evaluator.py \
  --checkpoint artifacts/models/hf_strawberry_segmentation_best.pt \
  --image /path/to/a_saved_camera_image.png \
  --output-directory artifacts/week6/mask-eval \
  --accept-untrusted-pickle
```

这个命令的结果仍会写 `safe_for_robot_use=false`。至少应在不同角度、远近和光照下保存
20 张图片，逐张查看叠加图，确认没有把叶片、桌面或红色背景大面积当成果实，才进入 ROS。

## 为什么大步配置是独立的

原来的 1–5 mm 配置已经过真机验证，不能为了一个新实验而偷偷改掉。新增的
`large_workspace_experimental` 当前配置为：

- 优先选择 5–10 cm 的相机动作，最多 3 步；若较大候选不可达，会换方向并依次回退到
  2.5/1/0.5 cm，而不是直接停止；真正停止死区仍是 1 mm；
- 累计平移不超过 30 cm、累计旋转不超过 45°；
- 单步相机旋转不超过 30°；该值只用于大空间展示配置，普通科研配置仍为 10°；
- Placo 最大关节变化不超过 0.35 rad；大步探索单独采用 5 mm / 2°，
  `sigma≥0.10`、条件数≤20（原小步配置仍是 2 mm 求解、3 mm 监督门）；
- HSV 连通域仍至少 200 像素；其中通过深度离群点剔除的像素在本配置中至少 100 个，
  用于容忍小草莓边缘的深度缺失（原 1–5 mm 配置仍要求 200 个）；
- 控制速度仍为 10%，两道执行门、五帧聚合、SHA 冻结和一次性授权全部保留。

Gradient-NBV 的线搜索也已修正：以前第一步固定只试一个体素（3 mm），所以即使配置写成
10 cm，实际候选仍容易停在毫米级；现在从配置的最大步长开始做回溯，并且只接受增益提高的
候选。确定性合成五视角实测每次约 9.98–10.00 cm，五步增益都为正。

在此基础上，真实监督器现在会围绕当前相机生成固定的 18 方向、6 距离候选。Placo 先删掉
不可达、贴近限位或接近奇异的点，NBV 再通过只读服务给剩余点打分；在收益增量达到当轮
最优值 90% 的候选中选择位移最大的一个。完整候选的 IK、关节余量、gain 与拒绝原因都会
进入 preview/execution JSON。连续空间不可能穷举，这个固定网格是可复现的工程近似。

生成新 preview 后，可以直接把“哪些点不可达、哪些点有收益、最终选了哪个点”画出来：

```bash
python3 validation/week6/reachable_candidate_visualization.py \
  artifacts/week6/large_step_nbv_preview.json \
  --output artifacts/week6/large_step_nbv_candidates.png
xdg-open artifacts/week6/large_step_nbv_candidates.png
```

灰色叉号是 IK/安全门拒绝的点，橙色是机械臂能到但没有正收益的点，蓝色是可达且有收益的
点，绿色星号是最终选择。该图和体素地图图分别回答“机械臂能去哪儿”和“地图看到了什么”。

硬件无关的 READY 姿态 Placo 预检见
[`large_step_ik_preflight.json`](large_step_ik_preflight.json)：5 cm 六个相机轴向全部可解，
10 cm 六个方向中三个可解。它只说明“大步不是数学上不可能”，真机仍必须从当时实测关节
重新 preview。

## 软件配置

- NBV 监督器：
  `perception_ws/src/strawberry_active_perception_bridge/config/real_nbv_supervisor_large_step.yaml`
- NERO 精度模式叠加：
  `nero_ws/src/strawberry_nero_control/config/large_nbv_experiment.yaml`

后者只把 precision 模式的关节变化上限从 0.12 rad 提到科研主控制已有的 0.35 rad；没有使用
展示程序的 1.50 rad，也没有放宽速度、误差、奇异性、反馈或门控。

## 上电后的顺序

当前先不要直接执行运动。上电、清场并启动相机和机械臂反馈后：

1. 运行根目录六项状态命令，确认 HSV 能稳定看到草莓；
2. 用大步配置生成 `execute=false` preview；
3. 查看 preview 的实际距离、方向、关节变化和画面边界；
4. 只有 preview 通过，才为这个新 JSON 的完整 SHA 做一次最多三步授权。

完整命令在 bridge README 的“优先厘米级、不可达时自适应回退”一节。USB 仍为 480M 不阻塞
`640×400@10 Hz`。本项目仍没有环境碰撞模型，所以 5–10 cm 实验必须比毫米级实验更认真地
清空整臂、相机、线缆的扫掠区域。
