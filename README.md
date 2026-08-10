# 双臂主动感知草莓项目空间

当前已实现 NERO 单臂的 Placo 末端位姿 IK、ROS 2 接口、平滑关节轨迹、安全拒绝和真机
执行。系统不使用 MoveIt，也不调用 NERO 原厂笛卡尔 IK。相机、双臂协同、Gradient-NBV
和 VAMP 尚未接入运行链路。

## 下载

机械臂驱动和相机驱动是 Git 子模块：

```bash
git clone --recurse-submodules \
  https://github.com/Sheepyyt/strawberry_active_perception.git
cd strawberry_active_perception
git submodule update --init --recursive
```

GitHub 网页会把子模块显示成可点击的提交链接，而不是复制第三方仓库的全部文件，这是正常
现象。

## 创建环境并构建

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash

python3 -m venv --system-site-packages --prompt sap-core .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cd nero_ws
python -m colcon build --symlink-install \
  --packages-select strawberry_nero_interfaces strawberry_nero_control
source install/setup.bash
```

工作区的 `colcon_defaults.yaml` 会忽略 `agx_arm_moveit`。必须使用 `python -m colcon`，让
生成的 ROS 2 Python 程序使用安装了 Placo 的 `.venv`。

## 目录

```text
strawberry_active_perception/
├── camera_ws/src/OrbbecSDK_ROS2/        相机驱动子模块
├── nero_ws/
│   ├── src/agx_arm_ros/                 NERO 官方驱动子模块
│   ├── src/strawberry_nero_interfaces/  ROS 2 消息、服务和 Action 定义
│   ├── src/strawberry_nero_control/     Placo IK、轨迹、安全控制和 Demo
│   └── colcon_defaults.yaml             构建时忽略 MoveIt
├── validation/week1/                    保留的测试 CSV、JSON 和数据摘要
├── requirements.txt                     Python 运行依赖
└── README.md
```

`.venv/`、`nero_ws/build/`、`nero_ws/install/`、日志和 Python 缓存均为本机生成内容，不提交
Git。删除 `build/install` 后，重新执行上面的构建命令即可恢复。

## 两个 Strawberry 包为什么分开

- `strawberry_nero_interfaces` 只定义其他节点怎样请求 IK、运动和安全恢复，以及返回哪些字段；
- `strawberry_nero_control` 实现 Placo 求解、连续性检查、五次轨迹和 NERO 执行。

接口独立后，未来 Gradient-NBV 只需依赖稳定的小接口包，不必依赖控制程序内部实现。

`nero_ws/src/strawberry_nero_control/strawberry_nero_control/` 这个内层同名目录是 Python 源码
模块；外层目录是 ROS 2 Python 工程。它们不是两份重复代码，而是 `ament_python` 的标准
结构。

## 复现 Demo 和查看结果

- 从上电、CAN、恢复、ready、目标预览、真机执行、回程到安全失能的完整教程：
  [`nero_ws/src/strawberry_nero_control/README.md`](nero_ws/src/strawberry_nero_control/README.md)
- 测试过程、数据表和原始结果说明：
  [`validation/week1/README.md`](validation/week1/README.md)
