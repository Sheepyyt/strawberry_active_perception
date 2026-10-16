# Gemini 2 XL 相机检查

完整主动感知使用 [演示流程](DEMO_GUIDE_CN.md)。本页只用于单独检查相机；正式目标分割使用 YOLO11。所有命令均从项目根目录执行。

## 连接与供电

```text
原装直流电源 → OBox DC 12V
电脑 USB → OBox “to Host”
相机 → OBox “to Device”
```

使用原装电源，或已经确认电压、插头和极性匹配的 `12V、至少 2A` 电源；不要使用 `12V 1A` 适配器。检查线缆有无损伤及关节运动所需余量。出现 USB 过流提示时断开相机供电和 USB，先排查连接，避免换端口反复尝试。

## 独立检查

若已经通过 `demo_operator.sh prepare` 启动相机，直接使用 `view`、`status` 或 `usb`，无需再次 `start`。不要同时用两个入口管理同一相机。

```bash
./camera_operator.sh start   # 独立启动；保持此终端打开
```

在另一终端检查：

```bash
./camera_operator.sh view    # 实时彩色画面
./camera_operator.sh status  # 检查节点及新深度帧
./camera_operator.sh usb     # 查看实际 USB 速度
```

当前使用 640×400、10 帧/秒。`480M` 表示 USB 2，`5000M` 或更高表示 USB 3；插口颜色不能代替实际速度检查。出现掉帧或断连时，优先检查供电、插头和线缆。

由相机脚本独立启动时，用 `./camera_operator.sh stop` 收尾，或在启动终端按一次 Ctrl+C。由演示脚本启动时，沿用 `demo_operator.sh stop` 的收尾流程。

## 目标分割与标定

准备演示时，将相机大致朝向一颗草莓；初始距离可参考 45–55 cm，是否可用由本轮深度检查判断。`demo_operator.sh prepare` / `check` 会保存新采集的分割画面。

`./mask_operator.sh test` 用于单独检查已启动的 YOLO11 链路。`./camera_operator.sh test-strawberry`（别名 `test-red`）只检查适配器的 HSV 备用 mask，不代表正式 YOLO11 分割结果。

正式手眼外参、标定数值样本和相机模型在 [config/calibration/](config/calibration/README.md)。相机相对末端的安装改变后需要重新标定；采集及几何检查工具在 `tools/calibration/`、`tools/camera/`。

## 首次构建

换电脑、删除构建产物或更新源码后，按 [重新安装说明](docs/REPRODUCE.md) 安装驱动依赖和 udev 规则、构建工作区。日常运行不需要重新构建。
