# Gemini 2 XL 相机简明操作指南

这份说明面向第一次使用本项目的人，不要求理解 ROS、话题、坐标系或像素等术语。

## 我们现在已经做到了什么

目前的软件链路可以完成五件事：

1. 打开 Gemini 2 XL，同时读取普通彩色画面和每个像素的距离。
2. 在需要时拍一组“标准观测”：彩色图、米制深度、目标 mask、相机参数和相机位姿。
3. Gradient-NBV 根据观测计算“相机下一步最好从哪里看”。这部分不依赖 MoveIt，也不需要
   机械臂真的运动。
4. 用正式手眼外参把相机位置换算成机械臂末端位置，并交给 Placo 检查是否有 IK 解。
5. 在人工清场和一次性授权下，执行最多三步小运动；每步再拍照并更新同一张地图。

相机已经安装在机械臂末端，手眼标定和真实多步闭环已经完成。尚未完成的是：学习式草莓
分割、环境碰撞模型、按观察目标自动决定循环次数，以及双臂协同。

## 已验证的软件/硬件能力（不是当前开机状态）

- `libgoogle-glog-dev` 已安装，全部系统依赖检查通过。
- 相机与感知软件包均可从源码重新构建。
- 相机能稳定输出 640×400、每秒 10 帧的彩色和深度数据。
- 验收所用固定线缆当时协商为 480M，也就是 USB 2 速度。它已经在 640×400、每秒 10 帧的配置下通过
  约 30 分钟稳定性测试，可以继续用于当前低速移动实验。5000M 或更高的 USB 3 是
  推荐升级，不是当前阶段必须先解决的条件。

本节不代表相机此刻正在运行。每次都用 `./camera_operator.sh status` 查看当次状态。

## 每次使用相机

打开第一个终端：

```bash
cd /home/yyt/strawberry_active_perception
./camera_operator.sh start
```

这个终端要保持打开。看到下面这句就表示相机成功启动：

```text
Device Orbbec Gemini 2 XL connected
```

再打开第二个终端，看实时画面：

```bash
cd /home/yyt/strawberry_active_perception
./camera_operator.sh view
```

如果只想让电脑自动检查是否工作，不必看日志：

```bash
cd /home/yyt/strawberry_active_perception
./camera_operator.sh status
```

正常结果会包含“相机程序：正在运行”和“深度数据：已实际收到一帧”。

使用结束后，先关闭实时画面窗口，然后在任一终端运行：

```bash
cd /home/yyt/strawberry_active_perception
./camera_operator.sh stop
```

也可以回到第一个终端按一次 `Ctrl+C`。不要直接拔相机。

## 当前目标 mask：真实草莓的 HSV 红色基线

当前程序已经可以对真实草莓运行，但它仍然按 HSV 红色范围生成 mask。也就是说，放一颗
真实草莓足以跑通当前几何闭环，却不能证明程序真的理解“草莓”类别；红杯子也可能被选中。
下一阶段会把这个 mask provider 换成 SAM 3.1 等学习式模型，NBV 后半段无需重写。

请按以下动作准备：

1. 不要启动机械臂，也不要让机械臂上电运动。
2. 把相机镜头朝向桌面或桌上的物体，不要再朝向天花板。
3. 在镜头前约 0.5～1 米处放一颗颜色清楚的真实草莓，避免强反光和大面积其他红色物体。
4. 在实时画面中，让草莓靠近中央，并且至少有“一枚硬币那么显眼”；不要让它小得像
   一个红点。所谓“200 像素”只是程序内部阈值，你不需要计算。
5. 保持相机和物体静止，然后执行：

```bash
cd /home/yyt/strawberry_active_perception
./camera_operator.sh test-strawberry
```

看到“草莓 mask 测试：成功”就表示真实单视角观测已经生成。若未识别到，程序会提示把
草莓移近、放到中央或减少反光。`test-red` 仍作为兼容别名保留。

早期红色测试已经验证单帧接口；随后红色目标完成三步闭环，真实草莓在 HSV mask 下完成
两步并提前进入 1 mm 位移死区。最新数据分别见 `validation/week4` 和 `validation/week5`。

## 已完成的标定板检查

你提供的标定板黑白格区域是 360×270 毫米，共 12×9 格，所以每格正好是 30 毫米。
程序识别出了全部 88 个内部角点，并得到：

- 深度测得的每格中值为 30.24 毫米，误差约 0.80%；
- 横向误差约 0.45%，纵向误差约 1.59%，都小于 2% 门限；
- 彩色图和深度图分别算出的板中心只相差约 2.23 毫米；
- 两者算出的板面朝向只相差约 0.45°。

因此，“相机测出的物体尺寸是否正确”这一项已经通过。

第一次平放测试时，标定板和桌面的高度差只有约 1～4 毫米，无法形成可靠深度边缘；报告
诚实地将该项标成 `not_testable`。随后将板竖起并让四边与后方物体形成明显距离差，自动
采集了 20 组同步图像。最终结果为：

- 20/20 帧都识别出全部 88 个角点；
- 四条边的可靠测量覆盖率约为 100%、81%、76%、80%；
- 四条边的彩深中值错位约为 1.65、0.81、2.00、0.13 像素；
- 最差一条边的 P95 为 2.88 像素；
- 所有可靠剖面都不超过 3 像素。

因此彩色—深度实体边缘对齐正式通过。绿色支撑箱遮住了底边中央一部分，但底边两侧仍有
超过门限的有效长度和至少约 54 毫米的前后距离差，未把被遮挡部分当成成功样本。相机程序
在采集结束后已正常关闭，整个检查没有启动或移动机械臂。

## 蓝色 USB 插口是否换对了

插口颜色不是可靠判据。历史稳定性实验中的链路为 480M；本次实际速度只需运行：

```bash
cd /home/yyt/strawberry_active_perception
./camera_operator.sh usb
```

- 显示 `480M`：仍是 USB 2，但本阶段 10 帧/秒测试可以继续。
- 显示 `5000M` 或更高：才是真正进入 USB 3 高速模式。

如果之后继续排查，请先正常停止相机，再按以下顺序做：

1. 优先插到主机背面、紧挨网线口或显示器接口的蓝色/青色 USB 口，而不是机箱前面或顶部。
2. 两端插头都重新插到底；如果相机端是 USB-C，可把 USB-C 插头翻转一次。
3. 再试另一个主机背面的 USB 3 口。
4. 如果多个已知支持 USB 3 的插口都只有 480M，固定线缆或相机端的高速触点可能是原因。

不需要反复拔插。现有 480M 连接已经通过约 30 分钟、每秒约 10 帧的稳定性测试；同样的数据
完整到达时，USB 3 不会让深度自动变得更准确。我们会继续保持这个低带宽配置，并在机械臂
低速运动时重新检查线缆弯折是否造成断流。只有需要更高分辨率/帧率、额外开启 IR/点云，
或者运动中出现断连和严重掉帧时，USB 3 才会成为必须解决的问题。

## 首次构建（通常不需要重复）

只有删除了 `build/install`、源码有更新或换电脑时才需要重新运行：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash

cd camera_ws
colcon build --symlink-install \
  --packages-select orbbec_camera_msgs orbbec_description orbbec_camera

cd ..
source camera_ws/install/setup.bash
source nero_ws/install/setup.bash
source .venv-nbv/bin/activate
python -m colcon --log-base perception_ws/log build \
  --symlink-install --base-paths perception_ws/src \
  --build-base perception_ws/build --install-base perception_ws/install
```
