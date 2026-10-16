# 最终运行工具

所有命令从项目根目录执行。日常使用 `./demo_operator.sh`，操作顺序见
[演示说明](../DEMO_GUIDE_CN.md)，结果直接阅读 [中文结果说明](../results/结果说明.txt)，或执行 `./demo_operator.sh open` 用浏览器看图。

| 目录 | 用途 |
|---|---|
| `demo/` | 统一启动、检查、关节恢复、闭环执行和安全收尾 |
| `analysis/` | 离线核对执行观测、地图增量、深度质量与候选敏感性，生成 Word / PDF 周报 |
| `camera/` | USB、帧率与时间戳检查、标定板 RGB-D 几何检查 |
| `calibration/` | 手眼原始样本采集和只读机械臂 TF |
| `nbv/` | 无硬件的确定性 NBV 验证、Placo 相机位移可达性预检 |
| `mask/` | 保存当前 YOLO mask、RGB、深度与质量报告 |
| `recording/` | 保存闭环实际使用的标准 Observation 数据 |
| `visualization/` | 从执行 JSON、Observation 和地图 NPZ 生成图表与动画 |

记录与绘图使用现有脚本，运行 `--help` 可查看输入输出参数：

```bash
/usr/bin/python3 tools/recording/observation_recorder.py --help
/usr/bin/python3 tools/visualization/session_visualization.py --help
/usr/bin/python3 tools/visualization/observation_visualization.py --help
PYTHONPATH=perception_ws/src/strawberry_gradient_nbv \
  /usr/bin/python3 tools/visualization/voxel_cloud_visualization.py --help
```

测试位于 `tests/` 和各 ROS 包的 `test/`；固定小样本 `tests/fixtures/` 随 Git 提交，其余测试数据临时生成。每轮观测、执行证据及图表保存在本机的 `results/demo/run_日期_时间_编号/`，不随 Git 提交；需要复盘时另行保留完整结果目录。旧实验计划不能用作新一轮运动授权。

周报入口为 `./demo_operator.sh analyze [已有结果目录]`。省略目录时读取最近一轮；输出位于 `results/weekly/运行编号/`，顶层仅含 Word、PDF，`figures/` 为三张图，`data/` 为统计表、分析数据及输入/代码/输出 SHA 清单。需要系统 LibreOffice、`pdfinfo`（`poppler-utils`）、NumPy、Matplotlib、Pillow，以及 Noto CJK 字体；无需额外 Python 环境或 ROS 服务。
