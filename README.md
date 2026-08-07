# 双臂主动感知草莓项目空间

当前第一周只做一件事：用 Placo 为一只 NERO 机械臂求 IK，生成平滑关节轨迹，并通过 ROS 2 的普通关节位置接口执行。相机、双臂协同、Gradient-NBV 和 VAMP 暂时保留接口，不在本周实现。

## 重新下载仓库

机械臂驱动和相机驱动是 Git 子模块。GitHub 网页会把它们显示成一个可点击的提交链接，而不是把第三方源码复制进主仓库，这是正常现象。新电脑上应这样下载：

```bash
git clone --recurse-submodules \
  https://github.com/Sheepyyt/strawberry_active_perception.git
cd strawberry_active_perception
git submodule update --init --recursive
```

如果已经普通 `git clone`，只需补执行最后一条 `git submodule update` 命令。`agx_arm_ros` 自带 MoveIt 包，但本项目不编译也不启动它；主仓库跟踪的 `nero_ws/colcon_defaults.yaml` 会让 colcon 跳过该包。

## 目录地图

```text
strawberry_active_perception/
├── .venv/                         sap-core Python 虚拟环境（本机生成，不进 Git）
├── camera_ws/                     相机 ROS 2 工作空间，本周暂不开发
│   └── src/OrbbecSDK_ROS2/        奥比中光官方驱动 Git 子模块
├── nero_ws/                       NERO ROS 2 工作空间
│   ├── src/
│   │   ├── agx_arm_ros/           松灵官方驱动 Git 子模块
│   │   ├── strawberry_nero_interfaces/  对外 ROS 2 接口合同
│   │   └── strawberry_nero_control/     Placo 与安全执行的具体实现
│   ├── colcon_defaults.yaml       固定忽略 MoveIt 包的工作区配置
│   ├── build/                     colcon 编译中间文件（可再生成）
│   ├── install/                   编译后的可运行文件（可再生成）
│   └── log/                       构建日志（可删除）
├── .gitmodules                    两个第三方仓库的子模块地址
├── .gitignore                     不上传虚拟环境、日志和构建产物
└── README.md                      本说明
```

`build/` 和 `install/` 目前保留，因为接下来马上要做真机检查；以后怀疑构建缓存损坏时可以删除并重新 `colcon build`。`log/`、`__pycache__/`、`.pytest_cache/` 和 VS Code 的 `browse.vc.db` 只是自动生成的记录或缓存，已从 Git 排除，可以安全删除。

`colcon_defaults.yaml` 只有从 `nero_ws` 目录运行 `colcon` 时才会自动生效。本项目文档中的构建命令都遵守这个约定；它比放在第三方子模块内部、无法由主仓库上传的 `COLCON_IGNORE` 更容易复现。

## `interfaces` 与 `control` 有什么区别

- `strawberry_nero_interfaces` 像一张统一的“表格和约定”：它只定义别人怎样请求 IK、怎样请求运动，以及程序要返回哪些结果。以后 Gradient-NBV 只需按这张表提交目标位姿，不必知道 Placo 内部怎么计算。
- `strawberry_nero_control` 像真正干活的“工作人员”：它读取这张表，调用 Placo、检查安全条件、生成五次轨迹，并在允许时把 7 个关节角发给驱动。

接口单独成包是 ROS 2 的标准做法。ROS 2 必须先把 `.msg/.srv/.action` 生成为 Python/C++ 类型，控制包才能导入并使用它们。这样将来相机节点、Gradient-NBV 节点或 VAMP 节点都只依赖小而稳定的接口包，不会反过来依赖整个控制实现。

## 为什么里面还有一个同名文件夹

```text
strawberry_nero_control/                 外层：ROS 2 / Python 软件包工程
├── package.xml                          ROS 2 依赖清单
├── setup.py / setup.cfg                 Python 安装规则
├── resource/                            ament 软件包索引标记
├── launch/                              启动文件
├── config/                              参数
├── test/                                自动安全测试
└── strawberry_nero_control/             内层：可被 Python import 的源码模块
```

这不是两份重复代码。可以把外层理解为“包装箱”，内层理解为“箱子里的机器”。因为 Python 要支持：

```python
from strawberry_nero_control.ik_core import PlacoIKSolver
```

所以内层模块通常与外层软件包同名，这是 `ament_python` 的标准结构。

## 控制包中保留的代码

| 文件或目录 | 功能 |
|---|---|
| `config/nero_control.yaml` | ready 姿态、IK 权重、限位、死区、轨迹和跟踪阈值 |
| `launch/sim.launch.py` | 不连接 CAN，启动 Placo 控制器和 MeshCat |
| `launch/real.launch.py` | 以三道安全门均关闭的方式启动真机读取 |
| `ik_core.py` | Placo FK/IK、关节限位、奇异和连续性检查 |
| `trajectory.py` | 生成零起止速度/加速度的五次关节轨迹 |
| `control_node.py` | ROS 2 Service/Action、真实反馈、TF、安全门和 NERO 执行 |
| `ros_utils.py` | ROS 位姿、关节消息和矩阵之间的可靠转换 |
| `models.py` | 关节名称、ready 角度及通用结果数据结构 |
| `meshcat_viewer.py` | 显示当前模型、目标坐标轴和末端轨迹 |
| `standalone_demo.py` | 完全不启动 ROS 2 的纯 Python Placo 演示 |
| `offline_benchmark.py` | 批量离线 IK 验证并输出 CSV/JSON 指标 |
| `test/` | 防止限位、连续性、轨迹和消息转换在修改后悄悄失效 |

## 接口包中保留的文件

| 文件 | 功能 |
|---|---|
| `srv/SolveIK.srv` | 只计算、不运动，返回 IK 结果 |
| `action/MoveToPose.action` | 计算并执行，支持进度反馈和取消 |
| `msg/IKResult.msg` | 成功、误差、奇异性以及拒绝原因的统一格式 |
| `CMakeLists.txt`、`package.xml` | 让 ROS 2 生成上述接口代码 |

更详细的仿真、构建和真机命令见 [`nero_ws/src/strawberry_nero_control/README.md`](nero_ws/src/strawberry_nero_control/README.md)。
