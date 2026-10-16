# 草莓主动感知：NERO + Gemini 2 XL + YOLO11 + Placo

本项目复现 Gradient-NBV：从当前 RGB-D 观测定位单颗草莓，在同一张体素地图中选择下一视角，由 Placo 求解关节目标并执行，然后重新观察。

## 日常演示只看这里

机械臂未使能时，把相机大致朝向草莓，建议距离 45–55 cm。每次起点和草莓位置可以不同，程序会重新计算；没有自动搜果或环境避障能力。

```bash
./demo_operator.sh prepare
# 确认整臂扫掠区清空、线缆有余量、有人现场看护后：
./demo_operator.sh run --workspace-clear
```

结束时终端会直接打印中文结论。**看图只需运行 `./demo_operator.sh open`，脚本会用浏览器打开报告。** 也可直接点开 [中文结果说明](results/结果说明.txt)，在编辑器里同样能读懂。报告说明实际做了几步、为什么停止、地图增加了多少信息；启动关节恢复与主动感知运动分开计数。

需要周报时运行 `./demo_operator.sh analyze`：仅读取最近一轮已保存的观测、地图与候选记录，生成 Word、PDF、三张分析图和统计表，位于 `results/weekly/运行编号/`。也可在命令后指定已有结果目录。此操作无需连接硬件，需要系统已安装 LibreOffice 和 `poppler-utils`；数据不完整时会明确报错。

收尾：

```bash
./demo_operator.sh stop
# 可靠承托整臂，确认失去电机力矩也不会下落后：
./demo_operator.sh shutdown --arm-supported
```

等到显示“失能收尾完成，本脚本所属进程已全部退出”后再人工断电。下次仍从 `prepare` 开始。详细操作见 [DEMO_GUIDE_CN.md](DEMO_GUIDE_CN.md)。

## 为什么不保证运动很多步

最多 8 步是上限。达到覆盖率目标、连续两次新增信息很少，或当前候选中没有可达且有正收益的新视角，都会提前结束；输入或执行检查失败也会停止。**先看结果首页的停止原因，不能只凭运动次数判断算法是否完成。**

候选相机位移使用算法现有的 5 / 10 / 25 / 50 / 75 / 100 mm 范围，结合收益、IK 与相机工作距离选择。覆盖率指目标附近 ROI 体素被有效深度射线观测的比例，不是草莓表面完整率。

## 保留的目录

| 内容 | 用途 |
|---|---|
| `demo_operator.sh`、`tools/demo/` | 一键启动、恢复、运行、收尾及中文结果入口 |
| `camera_ws/`、`nero_ws/`、`perception_ws/` | 相机、机械臂与感知源码；本机另保留构建产物 |
| `config/calibration/` | 正式手眼外参、相机模型与标定样本 |
| `config/models/` | Git 保留模型清单；草莓 YOLO 权重另行提供 |
| `.venv/`、`.venv-nbv/`、`.venv-mask/` | Placo、NBV、YOLO 的隔离运行环境，不能作为缓存删除 |
| `tools/`、`tests/`、包内 `test/` | 采集、离线复现、绘图和必要回归测试 |
| `vendor_patches/` | NERO v1.11 驱动补丁 |
| `results/` | 本轮观测、执行证据与图表；阅读 `结果说明.txt`，或用 `./demo_operator.sh open` 看图 |

`build/` 与 `install/` 之间有软链接，删除前者会破坏现有安装。历史周报、旧实验大包和独立展示程序已移除。

远程仓库保留源码、固定子模块与补丁、标定 JSON、模型清单和测试样本。`results/`、模型 `.pt`、三个 Python 环境及构建产物均不提交 Git：迁移时另带模型、已验证的机械臂 SDK 和需要复盘的结果目录，并按安装说明重建环境。上面的结果链接在本机完成运行后生成，刚克隆时尚不存在。

重新安装与复现见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。日常无需逐个运行底层工具；排错时可用 `camera_operator.sh`、`mask_operator.sh`、`robot_operator.sh`，其中电子阻尼停止入口为 `./robot_operator.sh e-stop`。

```bash
./verify_software.sh  # 离线软件检查，不开启相机或机械臂运动
```
