# Placo 与 NERO 原生 IK 对比实验

这个目录只放一次性实验，不修改控制节点、驱动或后续研究接口。后续项目仍使用 Placo。

## 比较什么

直接复用 `../real/real_30x3.json` 中已经真机验证过的绝对目标位姿，默认选取 5 个目标：

| 目标 | 位置变化 | 姿态变化 |
|---|---:|---:|
| P28 | 10.3 mm | 4.8° |
| P20 | 21.3 mm | 3.4° |
| P01 | 17.4 mm | 8.0° |
| P21 | 27.0 mm | 12.5° |
| P03 | 42.0 mm | 12.4° |

每个目标执行 3 次“锚点 → 目标 → 锚点”。Placo 侧直接读取已完成实验中的 15 次去程和
15 次回程；NERO 侧执行相同的 30 段。比较：

- 成功率；
- 基于真实编码器和同一 URDF 正运动学得到的位置、姿态误差；
- 指令发出到编码器开始变化的延迟；
- 整段运动用时；
- 单段最大关节变化和回到锚点后的关节误差。
- NERO 实际到达后的 7 个关节角，与“从同一真实起点求出的 Placo 解”的最大关节差。

两侧并非只更换一个数学函数：Placo 数据来自“Placo IK + 五次关节轨迹 + `move_j`”，
NERO 数据来自原生 `move_p`，其中包含“原生 IK + 原生轨迹”。因此运动延迟和总时长是完整
系统对比。

当前机械臂固件为 1.11。该版本没有受支持的 `get_ik_joint_angles()` 接口，无法读取原生 IK
选中的关节解和纯 IK 计算时间；该接口从 1.12 才提供。因此结果中 NERO 的
`ik_solve_time_ms` 留空，不能把整段 `move_p` 时间错误地叫作 IK 求解时间。实验仍会在运动
完成后记录真实 7 关节角，并额外从相同起点算一次 Placo 对照解，用来比较两种 IK 选择的
关节姿态是否不同；这次 Placo 计算只作测量，不参与 NERO 的运动控制。

NERO 状态码 `arm_status=2` 的官方含义是“原生 IK 无解”，不是通信、碰撞或关节硬件故障。
实验会把它保留为失败结果。如果去程已经到达、但原生 IK 无法从目标返回锚点，程序会用
已知安全锚点的 7 个关节角通过 `move_j` 复位，然后继续下一次实验。这个复位只恢复统一起点，
不计入 IK 性能，原生 IK 的无解记录也不会被改成成功。

## 最终结果

精度、延迟和运动时长只统计成功到达的样本；被连续性保护中止的 P21 不作为最终精度样本。

| 指标 | Placo | NERO 原生 v1.11 |
|---|---:|---:|
| 去程成功 | 15/15（100%） | 12/15（80%） |
| 回程成功 | 15/15（100%） | 0/15（0%） |
| 实际发送的原生回程 | — | 12次：11次无解、1次无有效运动超时 |
| 未发送的回程 | 0 | 3次（P21去程被连续性保护拒绝） |
| 成功去程位置误差 p50 / p95 | 0.424 / 0.666 mm | 0.078 / 0.128 mm |
| 成功去程姿态误差 p50 / p95 | 0.023 / 0.047° | 0.027 / 0.064° |
| 成功去程启动延迟 p50 / p95 | 123.6 / 248.8 ms | 53.3 / 59.0 ms |
| 成功去程总时长 p50 / p95 | 1.241 / 1.293 s | 0.566 / 0.801 s |
| 成功去程最大关节变化 p50 / p95 | 0.0656 / 0.0840 rad | 0.0910 / 0.1741 rad |
| NERO解与同一起点Placo解最大差 p50 / p95 | — | 0.0601 / 0.1327 rad |
| 纯 IK 时间 p50 / p95 | 0.414 / 0.657 ms | 固件1.11无法读取 |

逐目标去程结果：

| 目标 | Placo | NERO | NERO现象 |
|---|---:|---:|---|
| P28 | 3/3 | 3/3 | NERO关节变化约为Placo的1.9倍 |
| P20 | 3/3 | 3/3 | NERO关节变化约为Placo的2.2倍 |
| P01 | 3/3 | 3/3 | NERO关节变化约为Placo的2.6倍 |
| P21 | 3/3 | 0/3 | NERO约0.202 rad，超过0.20 rad连续性上限 |
| P03 | 3/3 | 3/3 | 两者关节变化接近 |

结论：NERO在成功去程中模型内位置精度更高、整条 `move_p` 链路响应更快，但它选择的冗余
关节解通常变化更大，P21三次均触发连续性保护，而且没有一次原生回程成功。Placo速度更
保守，但5个目标的去程、回程全部成功，关节变化更小、连续性更稳定。因此后续研究继续使用
Placo；NERO原生IK只保留为这次性能对比。

这里的速度差包含不同轨迹策略：Placo使用五次关节轨迹，NERO使用原生 `move_p`。不能据此
声称NERO的数学IK计算更快。误差由编码器和同一URDF正运动学得到，不代表外部测量设备下的
绝对末端精度。

## 第一步：离线预览（不需要真机）

```bash
cd /home/yyt/strawberry_active_perception
source .venv/bin/activate
python validation/week1/ik_comparison/compare_ik.py preview
```

它只读取原有数据，显示目标范围和 Placo 子集统计，不连接 CAN、不运动。

## 第二步：真机启动

先清空工作区，保证急停可触及。接通电源和 CAN 后，在终端 1 执行：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77

cd nero_ws/src/agx_arm_ros
bash scripts/can_activate.sh can0 1000000

cd /home/yyt/strawberry_active_perception
ros2 launch strawberry_nero_control real.launch.py \
  can_port:=can0 \
  speed_percent:=10 \
  allow_limit_recovery_execution:=true \
  first_motion_test_mode:=false \
  precision_test_mode:=true \
  startup_enable:=true
```

保持终端 1 运行。实验必须使用 `speed_percent:=10`、无末端执行器和零 TCP 偏置。

## 第三步：回到同一实验锚点

在终端 2 执行相同的四个 `source/export` 设置，然后先预览、再进入 ready：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source nero_ws/install/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=77

ros2 run strawberry_nero_control nero_real_smoke_test recover
ros2 run strawberry_nero_control nero_real_smoke_test recover --execute
ros2 run strawberry_nero_control nero_real_smoke_test center-ready
ros2 run strawberry_nero_control nero_real_smoke_test center-ready --execute
```

只有预览通过后才运行对应的 `--execute`。按终端提示输入确认文字。

## 第四步：NERO 原生实验预检（不运动）

```bash
python validation/week1/ik_comparison/compare_ik.py native
```

程序会核对：7 关节反馈、驱动故障、关节限位、当前姿态是否接近锚点，以及 URDF `link7`
与 NERO TCP 是否对齐。它不会打开软件控制门，也不会发送 `move_p`。

如果这里提示坐标未对齐，不要加大阈值强行运行，应停止并检查 TCP 偏置和坐标定义。

## 第五步：采集 NERO 原生结果

预检通过且空间仍安全时执行：

```bash
python validation/week1/ik_comparison/compare_ik.py native --execute
```

输入 `NERO_COMPARE` 后，机械臂执行 5 个目标、每个 3 次往返。程序逐段写入
`native_results.csv`，任一故障、反馈过期、关节跨度大于 0.20 rad 或运动超时都会停止。无论
完成还是退出，程序都会尝试关闭 `/control_enable` 软件门，但不会自动失能电机。

原生 IK 报告“无解”属于实验结果，不等于硬件故障：程序记录失败后会用 `move_j` 返回已知
锚点，再继续后续目标。NERO 关节解跨度超过 `0.20 rad` 同样作为连续性拒绝记录，并在安全
复位后继续；其他非零状态仍会立即停止。正式 IK 连续性标准始终保持 `0.20 rad`。

如中途失败，先保留 `native_results.csv` 并查明原因。程序只允许从“去程和回程记录都已写入”
的完整往返边界续跑；机械臂必须先恢复到锚点。复位和预检通过后执行：

```bash
python validation/week1/ik_comparison/compare_ik.py native --execute --resume
```

程序会显示已经读取多少次完整往返，并跳过它们。`--resume` 与 `--overwrite` 不能同时使用。
只有决定丢弃当前断点、从头重做时才使用：

```bash
python validation/week1/ik_comparison/compare_ik.py native --execute --overwrite
```

### 如果程序停在目标位姿并显示 `arm_status=2`

先预览已知锚点关节复位：

```bash
python validation/week1/ik_comparison/compare_ik.py reset-anchor
```

确认显示的锚点复位变化不超过 `0.25 rad`，再执行：

```bash
python validation/week1/ik_comparison/compare_ik.py reset-anchor --execute
```

按提示输入 `RESET_ANCHOR`。它使用已知锚点关节角和 `move_j`，不调用任何 IK，也不自动失能。
复位上限比 IK 连续性上限多出的 `0.05 rad` 只用于覆盖触发保护后的制动距离，不会放宽正式
IK 对比的 `0.20 rad` 标准。
复位成功后重新运行 `native` 预检。若断点包含完整的去程/回程记录，使用
`native --execute --resume`；否则在排查后选择完整重做。

## 第六步：查看结果

完整采集后程序会自动生成：

| 文件 | 内容 |
|---|---|
| `native_results.csv` | NERO 原生 30 段逐项记录 |
| `comparison_results.csv` | Placo 与 NERO 的同格式逐段数据 |
| `comparison_summary.json` | 两种方法按去程/回程汇总的 p50、p95、最大值和成功率 |

如果只需要重新汇总，不运动：

```bash
python validation/week1/ik_comparison/compare_ik.py summarize
```

## 停止和失能

按 `Ctrl+C` 会停止实验、发送当前关节保持并关闭软件控制门。软件门不是物理急停，危险情况应
立即使用硬件急停。NERO 失能会因重力下落；实验结束后不要直接失能，必须先由另一人在避开
夹点的位置托稳机械臂，再由操作员执行失能。
