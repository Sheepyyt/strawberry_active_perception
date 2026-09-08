# 末端相机手眼标定（离线核心）

这个包只做一件事：根据多组“机械臂末端位姿 + 相机看到的标定板位姿”，计算相机相对 `link7` 的精确安装位置和方向。它不连接 CAN、不使能机械臂、不发送运动命令，也不依赖 ROS、MoveIt 或 Placo。

## 输入和输出方向

每个样本必须同时包含：

- `T_base_link7`：把 `link7` 中的点变换到 `base_link`，即
  `p_base = T_base_link7 · p_link7`。
- `T_camera_checkerboard`：把标定板中的点变换到相机光学坐标系；这是 OpenCV `solvePnP` 直接给出的方向，即
  `p_camera = T_camera_checkerboard · p_checkerboard`。

输出固定为：

- `T_link7_camera_optical`：把相机光学坐标系中的点变换到 `link7`，即
  `p_link7 = T_link7_camera_optical · p_camera`。

对静止不动的标定板，每个合格样本都应满足：

```text
T_base_checkerboard
  = T_base_link7 · T_link7_camera_optical · T_camera_checkerboard
```

JSON 中保存了完整的方向说明，方向不一致的数据会在进入求解器之前被拒绝。

## 为什么需要多个不同姿态

只在一个位置拍摄，无法分清“相机安装偏差”和“标定板位置”。需要大约 15～20 个不同姿态，并且末端要绕至少两个不同方向发生转动。仅沿直线平移，或者只绕同一根轴转动，程序会明确返回 `failed_degenerate_motion`，不会给出看似正常但实际不可靠的结果。

求解时会执行以下检查：

1. 留出约 20% 姿态，完全不参与求解，只用于最终验证。
2. 比较 OpenCV 的 TSAI、PARK、HORAUD、ANDREFF 和 DANIILIDIS 五种方法，先用
   `AX=XB` 残差诊断各初值，再要求至少三个方法形成 10 mm / 2° 内的一致簇。
3. 对一致簇的中值初值做成对 `AX=XB` 的 SE(3) 鲁棒优化。每个姿态对同时使用正、反
   两个方向，避免文件排序改变答案。
4. 鲁棒权重作用于完整的六维残差组，使用显式的平移/旋转归一化和 group soft-L1
   IRLS；不会对六个坐标分量分别加权，因此不会因旋转坐标基而改变答案。
5. 分别从所有能给出有限结果的 OpenCV 方法开始优化，要求最终落入同一个解；即使
   ANDREFF 给出较小的静止标定板残差，只要其 `AX=XB` 平移残差大且偏离方法簇，也会被
   明确诊断并排除。
6. 根据静止标定板的一致性识别训练集异常值，并用正常样本重新计算。
7. 在留出姿态上检查平移和旋转残差；默认 P95 必须分别不超过 10 mm 和 2°。
8. 正式验收必须完成至少 20 个确定性独立留出划分，20 次全部通过，并要求各次外参
   两两相差不超过 10 mm / 2°。
9. 还会使用六组预先固定的残差尺度（平移 1～3 mm、旋转 0.125～1°）重算完整数据，
   尺度之间的外参差异也必须不超过 10 mm / 2°。

## 离线自检

构建工作区后，可生成一个已知答案的确定性数据集：

```bash
source /opt/ros/jazzy/setup.bash
source /home/yyt/strawberry_active_perception/perception_ws/install/setup.bash
ros2 run strawberry_handeye_calibration handeye_make_fixture \
  --output /tmp/handeye_fixture.json
```

然后离线求解：

```bash
ros2 run strawberry_handeye_calibration handeye_calibrate \
  --input /tmp/handeye_fixture.json \
  --output /tmp/handeye_report.json
```

这两个命令只读写 JSON 文件，不会启动相机或机械臂。

## 真实数据文件结构

数据集采用 `strawberry_handeye_samples/v1`。核心字段如下（矩阵均为 4×4、行优先、长度单位为米）：

```json
{
  "schema_version": "strawberry_handeye_samples/v1",
  "transform_convention": {"...": "文件中必须保留完整的固定说明"},
  "session_id": "handeye_2026_08_13",
  "frames": {
    "base": "base_link",
    "link": "link7",
    "camera_optical": "camera_color_optical_frame",
    "checkerboard": "checkerboard"
  },
  "checkerboard": {
    "columns": 11,
    "rows": 8,
    "square_size_m": 0.03
  },
  "samples": [
    {
      "sample_id": "pose_000",
      "timestamp_sec": 0.0,
      "T_base_link7": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "T_camera_checkerboard": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "detection": {"corner_count": 88, "reprojection_rms_px": 0.25}
    }
  ],
  "metadata": {}
}
```

真实采集还可以另外保留原始彩色图、深度图、角点和关节角作为审计材料；核心求解器只读取上述两种已验证的位姿，不会悄悄根据关节角或 TF 猜测变换方向。

## 把现场 NPZ 转为核心 JSON

`validation/week3/capture_handeye_sample.py` 每次生成一个只读 NPZ。先把一组已经冻结的
样本转换成上面的核心 JSON，再运行求解器：

```bash
ros2 run strawberry_handeye_calibration handeye_import_npz \
  --input-directory artifacts/week3/handeye_session_001 \
  --output artifacts/week3/handeye_session_001/session.json \
  --sample-ids pose_001 pose_002 pose_003 pose_004 pose_005

ros2 run strawberry_handeye_calibration handeye_calibrate \
  --input artifacts/week3/handeye_session_001/session.json \
  --output artifacts/week3/handeye_session_001/report.json
```

省略 `--sample-ids` 时会按文件名导入目录中全部 NPZ；现场仍在继续采集时应显式列出样本
ID，避免同一轮分析的数据集合在运行间发生变化。转换器使用 `allow_pickle=False`，并复核
每个样本的 schema、刚体矩阵、时间戳、相机内参、棋盘物点、PnP 重投影、真实相机来源、
`motion_commands_sent=0` 和整场一致性。JSON 的 metadata 会保留每个 NPZ 的 SHA-256、
Observation ID、关节位置和只读采集审计字段。这个命令只读文件，不导入 ROS，也不接触机械臂。

单次固定留出划分通过后，还必须执行跨划分稳定性验收：

```bash
ros2 run strawberry_handeye_calibration handeye_validate_stability \
  --input artifacts/week3/handeye_session_001/session.json \
  --output artifacts/week3/handeye_session_001/stability_report.json
```

默认重复 20 个确定性的训练/留出划分，要求 20 次全部通过，并要求各次求出的
`T_link7_camera_optical` 两两相差不超过 10 mm 和 2°。任一划分失败、无有限外参，或外参
随数据划分或预先声明的残差尺度明显漂移时，报告会返回
`failed_cross_split_stability`，把
`safe_for_robot_use` 设为 false，且不会给出可用于机械臂的正式外参。报告仍保留带有
`diagnostic_` 前缀的中值样本解，纯粹用于诊断下一轮缺少哪类姿态；物理安装尺寸等先验不参与
通过判定。单次 `handeye_calibrate` 即使通过，也始终标记
`safe_for_robot_use=false`；只有上述完整稳定性验收有权输出正式外参。

## 用修正后的畸变参数重算已有 NPZ

如果现场 NPZ 保存了正确的棋盘角点和机械臂位姿，但当时错误地把相机畸变 `D` 记成了
全零，不需要重新移动机械臂，也不能直接修改这些原始 NPZ。先准备一个独立的相机模型
JSON，再运行纯离线重处理：

```json
{
  "schema_version": "strawberry_camera_model/v1",
  "source_name": "gemini2xl_AYML241003A",
  "image_size": {"width": 640, "height": 400},
  "K": [
    [304.12066650390625, 0.0, 313.2231140136719],
    [0.0, 304.1894226074219, 197.83775329589844],
    [0.0, 0.0, 1.0]
  ],
  "D": [0.01, -0.02, 0.0001, -0.0002, 0.003, 0.0, 0.0, 0.0],
  "distortion_model": "rational_polynomial",
  "provenance": {
    "method": "live color CameraInfo read and frozen offline audit",
    "recorded_at": "2026-08-13T12:00:00+08:00"
  }
}
```

`plumb_bob` 必须恰好给 5 个系数，`rational_polynomial` 必须恰好给 8 个系数；`D` 必须
有限且至少有一个非零值。`provenance` 必须是非空对象，用来说明这些参数来自哪里。命令为：

```bash
ros2 run strawberry_handeye_calibration handeye_reprocess_camera_model \
  --input-directory artifacts/week3/handeye_session_001 \
  --camera-model artifacts/week3/handeye_session_001/color_camera_model.json \
  --output artifacts/week3/handeye_session_001/handeye_samples_corrected_D.json \
  --sample-ids pose_001 pose_002 pose_003
```

这个命令会逐个执行以下检查和处理：

1. 按原导入器完整复核 Capture v1、刚体位姿、棋盘尺寸、角点、物点、原 PnP 和采集审计。
2. 要求所有 NPZ 的 `source_name`（包含相机序列号）、640×400 图像尺寸、原 `K` 和原
   `D` 完全一致，并且原 `D` 必须确实全为零。
3. 默认要求新模型的 `K` 与 NPZ 原 `K` 在 `1e-9` 像素绝对容差内一致。本工作流默认
   只修复错误的 `D=0`，不能顺手静默更换内参。
4. 用新 `K,D` 和 `cv2.solvePnP(..., SOLVEPNP_IPPE)` 重算每个
   `T_camera_checkerboard` 及重投影 RMS，并拒绝非有限或落在相机背后的解。
5. 输出仍是 `strawberry_handeye_samples/v1`。metadata 保存相机模型文件 SHA-256、模型
   全文、每个原 NPZ SHA-256、角点/物点/机器人位姿字段哈希，以及新旧相机位姿差异的
   最小值、中位数、P95 和最大值。

程序只读取 NPZ，并在读前、读后复核文件 SHA-256；不会改写原始 NPZ，不导入 ROS，也
不会连接相机或机械臂。输出只是重新计算后的手眼求解输入，始终标记
`safe_for_robot_use=false`；仍需继续运行 `handeye_calibrate` 和完整的
`handeye_validate_stability`。

只有在做独立的“同时更换 K”诊断实验时，才可以显式加 `--allow-k-change`。此时 metadata
会把 `K_changed_beyond_tolerance` 和 `diagnostic_only_due_to_K_change` 设为 true，并明确
禁止把结果直接当作真机外参。正常修正 D 的流程不要使用这个选项。

## 冻结前 25 个姿态，检查后 5 个新姿态

已经提前约定训练集和后来采集的测试集时，可以做一次不“偷看答案”的前瞻检查：程序只用
前 25 个姿态运行一次 `ROBUST_PAIRWISE_AX_XB`，固定求出的
`T_link7_camera_optical` 和训练集棋盘参考位姿，再评价后 5 个姿态。后 5 个姿态不会参与
求解，也不会使相机外参或棋盘参考重新拟合。

```bash
ros2 run strawberry_handeye_calibration handeye_validate_prospective \
  --input artifacts/week3/handeye_session_001/handeye_samples_pose001_030_factory_raw_D.json \
  --output artifacts/week3/handeye_session_001/prospective_pose026_030_factory_raw_D.json \
  --train-ids pose_{001..025} \
  --test-ids pose_{026..030}
```

报告记录输入文件的 SHA-256 和字节数、明确的训练/测试 ID、固定外参、固定棋盘参考、每个
测试姿态的平移/旋转误差，以及 P95、中位数和最大值。默认只用平移 P95≤10 mm、旋转
P95≤2°作为该项检查的门槛，最大误差另外完整报告。即使这项检查通过，报告也始终写入
`safe_for_robot_use=false`；它只是独立诊断证据，正式真机外参仍只能来自全部 20 个划分
和残差尺度试验都通过的稳定性报告。
