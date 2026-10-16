# NERO Placo 控制

本包将 `link7` 目标位姿转为关节轨迹，并使用实时反馈检查执行结果：

```text
PoseStamped → Placo IK → 关节/奇异性/连续性检查 → 平滑轨迹 → NERO 反馈
```

日常运行使用仓库根目录的 `demo_operator.sh`，见
[演示说明](../../../DEMO_GUIDE_CN.md)。启动时按当前反馈检查关节；需要恢复时只让允许的
越界关节向内回到控制范围，不依赖固定初始姿态。

## 构建与测试

完整环境及三个工作区的构建顺序见 [重新安装说明](../../../docs/REPRODUCE.md)。
本包必须使用安装了 Placo 的 `.venv` 构建，原厂驱动先应用 `vendor_patches/agx_arm_ros/`
中的固定版本补丁。当前流程不使用 MoveIt。

```bash
./verify_software.sh   # 离线测试，不连接 CAN 或相机
```

## 当前运行配置

`config/nero_control.yaml` 为控制器基础配置；统一演示入口额外加载
`config/large_nbv_experiment.yaml`。参数以这两个文件以及监督器的运行核对为准，
每轮使用的配置会随结果保存。关节恢复与普通 NBV 运动分别检查、分别记录。

## 接口与必要诊断

| 接口 | 用途 |
|---|---|
| `/strawberry_nero/solve_ik` | 只计算关节解 |
| `/strawberry_nero/move_to_pose` | 检查后执行目标位姿 |
| `/strawberry_nero/recover_to_safe` | 预览或执行关节向内恢复 |
| `/strawberry_nero/enable_execution` | Placo 执行门 |
| `/control_enable` | 原厂驱动控制门 |
| `/electronic_emergency_stop` | SDK 电子阻尼急停 |

必要时从项目根目录运行：

```bash
./robot_operator.sh can
./robot_operator.sh status
./robot_operator.sh pose
./robot_operator.sh e-stop
```

`real_smoke_test.py` 的恢复客户端仍由演示入口使用；`pose_demo.py` 支持当前位姿诊断；
`axis_suite.py` 是这两个客户端的公共依赖。它们属于现有运行链。
`sim.launch.py` 和 `meshcat_viewer.py` 保留为无 CAN 的模型/IK 调试工具。
早期 Week-1 批量验收、离线 benchmark 和独立展示脚本已移除。

## 运行边界

控制器没有环境避障能力，MeshCat 网格也不代表在线碰撞检测。真机运行仍需现场清场和看护，
并避免同时启动第二个控制命令发布者。电子阻尼急停不是机械锁止；失能、断电前需要可靠承托。
正常收尾使用 `./demo_operator.sh shutdown --arm-supported`，具体顺序见演示说明。
