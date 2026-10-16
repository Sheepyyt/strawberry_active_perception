# Gemini 2 XL → NERO link7 标定

`handeye_report.json` 是真实机械臂运行读取的正式手眼外参报告。报告来自 30 个真实姿态，
通过 20 个固定训练/留出切分；迁移只改变路径，没有重新拟合或改动报告内容。

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `handeye_report.json` | 运行时外参及验收记录 | `31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f` |
| `handeye_samples.json` | 相机模型修正后的 30 姿态标定数据集 | `2a18d4bc69b13d01e28dcfad8cf62186bde4fe31e925f87c9c4f8db6b6718aef` |
| `gemini2xl_raw_color_camera_model.json` | 序列号 AYML241003A 的原始彩色内参与畸变参数 | `b66b71b5e291abd865db7fd4da408272ab56a2d5b6a262a8b5e9d8448032f696` |

矩阵方向为 `T_link7_camera_optical`：`p_link7 = T_link7_camera_optical @ p_camera`。
平移单位是米；相机光学坐标 x 向右、y 向下、z 向前。报告中的历史文件路径仅用于追溯原始
实验，不需要为迁移改写它们；改写会破坏运行时校验的 SHA。

相机相对末端的安装位置发生变化时，必须重新标定。可用
`tools/calibration/capture_handeye_sample.py` 采集新数据，再调用
`strawberry_handeye_calibration` 包重新求解与验证。清理后保留本目录的报告、数值样本与
相机模型；旧照片和实验输出不属于当前运行输入，已不在项目中维护。
