# 手眼标定结果：现在做到哪里了

## 一句话结论

Gemini 2 XL 与 NERO `link7` 之间的手眼外参，已经用 30 个真实机械臂姿态完成离线标定，
并通过 20/20 个独立留出切分。相机模型修正后的独立实机数据也能复核这个结果。真实外参下
的当前位姿和相机光学 +X 方向 5 mm Placo `SolveIK` **只读预览**也均已通过；此后该外参
已经用于一次、三步和真实草莓 HSV-mask 多步 NBV 闭环。

这里的“标定报告允许用于机械臂”只表示矩阵通过了数值门槛。它不代表碰撞检查、运动控制、
线缆安全和现场人员确认已经自动完成，也不等于真实运动已经获准或发生。

## 我们具体做了什么

1. 在相机安装于机械臂末端的真实状态下，采集了 30 个不同机械臂姿态。每份样本同时保存
   棋盘角点、RGB-D、关节反馈和 `base_link -> link7`，采集程序发送的运动命令数为 0。
2. 找到并修正了一个相机图像模型问题：厂商 `image_undistorted` 不是实际去畸变图。
3. 保留原始 NPZ 不动，用相机序列号 `AYML241003A` 的真实原始 `K/D` 重新计算 30 份
   `T_camera_checkerboard`，并把每个输入文件和数组的哈希写入派生数据集。
4. 用鲁棒 `AX=XB` 求解和 20 个确定性训练/留出切分检查外参；20 次全部通过。
5. 另外采集一组新的 20 帧固定标定板图像，检查 adapter 输出的实际 RGB-D 几何，并用
   固定板重复位姿再次核对标定结果。普通棋盘旋转 180° 后图案仍等价，因此比较前先按
   黑色原点格约定消除了这项 180° 对称，避免把同一个棋盘误认为翻转了 180°。
6. 把正式外参接入真实 Placo `SolveIK` 计算链，分别检查当前姿态和相机光学 +X 方向
   5 mm 候选。两项都通过，两个独立运动命令计数都保持为 0，机械臂没有移动。

## 相机图像问题，用通俗话说明

相机镜头会使原图产生轻微弯曲，算法必须知道图像是“原始弯曲图”还是“已经拉直的图”。
厂商节点虽然发布了名为 `image_undistorted` 的图，但我们把它和同一时刻的原图逐字节比较，
两者完全相同；源码检查也确认，程序在真正校正前把畸变参数清零了，所以这一路实际上没有
拉直图像。旧数据却把这张原图标成了 `D=0`，这会系统性影响棋盘位姿和手眼标定。

现在的处理方式是：

- adapter 订阅原始 RGB、HW-D2C 对齐深度以及两份 CameraInfo；
- 先检查彩色与深度的分辨率、像素网格和 `K` 确实一致；
- adapter 使用原始 RGB 的非零 `D`，自己调用 OpenCV 做真正的 rectification；
- 输出的统一 RGB 和 CameraInfo 才标记为 `D=0`；
- HW-D2C depth 已由相机硬件放到正确的彩色网格，保持原样，不做第二次 remap。

最后一点很重要：对 depth 再校正一次不是“更精确”，而是会把已经对齐的数据再次扭曲。

## 真实相机的 20 帧几何结果

修正后的 adapter 连续采集了 20 个不同 Observation ID、分辨率均为 `640x400` 的静止实机
样本。主要结果如下：

- 彩色与深度最大时间差：`0.113 ms`；
- 有效深度比例中位数：`74.70%`；
- 实测单格边长中位数：`30.078 mm`（标称 `30 mm`）；
- 水平/竖直尺度最大误差：`0.416% / 0.312%`；
- RGB PnP 与注册深度的棋盘中心差中位数：`1.766 mm`；
- RGB PnP 与注册深度的棋盘法向夹角中位数：`0.215°`；
- 四条实体边的彩深偏差：中位数 `0.879 px`、P95 `2.126 px`，全部 accepted profile
  均在 `3 px` 内。

这组数字说明“adapter 真正校正 RGB、HW-D2C depth 不二次 remap”的组合在真实数据上是
一致的。USB 链路目前仍是 USB 2.0（协商速率 480 Mbps），但它已经能稳定运行本项目固定的
`640x400@10 Hz`。没有 USB 3 不会阻塞当前阶段；只有未来提高分辨率/帧率、增加数据流，
或机械臂运动时出现线缆断流，才需要把 USB 3 升级列为硬件前置条件。插口是蓝色只能作为
外观线索；这里说“USB 2.0”依据的是系统实际协商出的 `480 Mbps`，不是插口颜色。

## 30 姿态正式手眼结果

矩阵约定为 `T_link7_camera_optical`：把相机光学坐标中的点变换到 `link7`；平移单位为米。

```text
T_link7_camera_optical =
[  0.002514086749113  -0.003111263784683   0.999991999670737   0.064551880692857
  -0.010401145783000   0.999940985158437   0.003137254655803   0.002329651399496
  -0.999942746128107  -0.010408949900768   0.002481577672488   0.000198876908971
   0.000000000000000   0.000000000000000   0.000000000000000   1.000000000000000 ]
```

可以把平移部分直观理解成：相机光学原点相对 `link7` 约为
`[64.55, 2.33, 0.20] mm`。矩阵方向不能凭这三个数字手工交换；程序必须严格按上面的
`T_link7_camera_optical` 约定使用。

正式门禁结果：

- 30 个姿态，20 个独立训练/留出切分，`20/20` 全部通过；
- 所有切分中最差的 holdout translation P95：`6.10 mm`；
- 所有切分中最差的 holdout rotation P95：`0.52°`；
- 20 个切分所得外参两两比较，最大平移差：`3.386 mm`；
- 20 个切分所得外参两两比较，最大旋转差：`0.513°`；
- 正式报告字段：`status=passed`、`safe_for_robot_use=true`，且验收未使用人为物理先验。

独立的新 20 帧固定板复核采用了另一条计算链：固定使用上面的正式外参、`pose_030` 的
`T_base_link7`，再对 20 帧新图逐帧运行 IPPE，得到每帧的 `T_base_checkerboard`。对
`11x8` 内角格网，先用 `S = Rz(pi), t = [0.30, 0.21, 0] m` 把棋盘 180° 等价方向归一，
再与原 30 姿态通过正式外参得到的固定板 medoid 比较。结果为：

- 平移差 median / P95 / max：`3.3550 / 3.4159 / 3.4603 mm`；
- 旋转差 median / P95 / max：`0.23214 / 0.30131 / 0.31105°`。

因此简写为“约 `3.36 mm / 0.23°`”。这是额外的独立一致性复核，不是上一节的
RGB-PnP 与 depth 几何对齐指标，也不替代 20 个留出切分，更不表示已经做了真实 NBV 运动。

## 可审计证据

这些大文件位于 Git 忽略的本机 `artifacts/` 目录。SHA-256 用来确认复现时读到的是同一份
内容；任何文件被改动，SHA 都会变化。

| 内容 | 本机路径 | SHA-256 |
|---|---|---|
| 原始彩色相机模型 | `artifacts/week3/handeye_session_001/gemini2xl_raw_color_camera_model.json` | `b66b71b5e291abd865db7fd4da408272ab56a2d5b6a262a8b5e9d8448032f696` |
| 修正相机模型后的 30 姿态派生数据集 | `artifacts/week3/handeye_session_001/handeye_samples_pose001_030_factory_raw_D.json` | `2a18d4bc69b13d01e28dcfad8cf62186bde4fe31e925f87c9c4f8db6b6718aef` |
| 30 姿态正式稳定性报告 | `validation/week3/artifacts/stability_pose001_030_factory_raw_D.json` | `31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f` |
| adapter 修正后的 20 帧几何报告 | `artifacts/week3/handeye_session_001/corrected_adapter_checkerboard_20_geometry.json` | `2a847e733fb9bd9f52e3fd7d8814f5586afbf616d5fdaa7149cf927ac77713cf` |
| 20 帧实机图像/深度数据 | `artifacts/week3/handeye_session_001/corrected_adapter_checkerboard_20.npz` | `5dff526e555a339fbdde53ea78da425c048abb9b2dfa9d0649cd7745f155cd2a` |
| 独立 20 帧固定板复核摘要 | `validation/week3/artifacts/handeye_live_repeatability_summary.json` | `fd84fcefe142c675ded6918ff797661afe69e073d58adcba5d731650f797e4a5` |
| 当前位姿真实只读 SolveIK | [`validation/week3/artifacts/real_handeye_current_solveik_preview.json`](artifacts/real_handeye_current_solveik_preview.json) | `5b2c3d6096acc3caa12e49acfc0ddb0a9be4ca6bc481f81a1615a8a51f946413` |
| optical +X 5 mm 真实只读 SolveIK | [`validation/week3/artifacts/real_handeye_small_nbv_solveik_preview.json`](artifacts/real_handeye_small_nbv_solveik_preview.json) | `5ca282f883f2b0affe9b7719c6554843ab45c401216046e3eb3dadc0447f421e` |

原始 `pose_001.npz` 到 `pose_030.npz` 没有被覆盖。派生数据集内还逐样本记录了原始 NPZ、
角点、物点、旧棋盘位姿和 `T_base_link7` 的 SHA，可追溯每一项重计算输入。

## 这份手眼结果后来怎样被使用

手眼完成后先做了两次真实、只读的 SolveIK 预检：

两次预检都读取正式报告及其 SHA，并确认 bridge 使用的矩阵与报告完全一致：

- 当前位姿检查为 `passed`；
- 相机光学 `+X` 方向 5 mm 候选为 `passed`；
- 5 mm 候选在完整步长 `alpha=1` 第一次求解就成功，最大关节解变化为 `0.012 rad`；
- 5 mm 候选的位置误差为 `0.993657 mm`，姿态误差为 `0.000576818 rad`；
- bridge 自身和外部独立 topic 观察器记录的运动命令数均为 0；
- 控制器 `execution_enabled=false`，驱动 `control_enabled=false`，检查时关节速度为 0。

这两次只读结果当时只证明计算链正确，没有下发运动。此后，正式外参已经进入真实监督器：
先完成 1 次冻结目标闭环，再完成红色目标 3 步持续地图闭环，以及真实草莓 HSV-mask 的
2 步闭环。后续运动证据分别在 `validation/week4` 和 `validation/week5`，不要再把本页的
早期“只读”结论误读成项目仍停留在只读阶段。

手眼报告通过并不替代环境碰撞、线缆、工作空间和现场安全检查。每次真实实验仍需读取曝光
时刻的 `T_base_link7`，与这里的 `T_link7_camera_optical` 相乘得到相机世界位姿；不能使用
固定单位位姿，也不能复用过期现场状态或历史授权。
