# YOLO11 草莓分割 mask

这个包把已经同步、矫正的 canonical `Observation.color` 送入 YOLO11 实例分割，只替换
`target_mask`，然后发布新的 Observation。RGB、米制深度、内参、时间戳、scene/ID 和
曝光时刻相机位姿全部保持不变。

## 当前权重

- 用户提供文件：`best.pt`，45,192,111 bytes；
- SHA-256：`7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357`；
- 权重内嵌 Ultralytics 版本：`8.3.74`；
- 模型：`yolo11m-seg` / segmentation；
- 内嵌类别：只有 `0: strawberry`。

`.pt` 是可包含 Python pickle 的归档。节点先核对完整 SHA，再允许 Ultralytics 加载。权重
本身没有提交到 Git；其训练数据来源和再分发许可证也尚未获得，因此这里只提交哈希、接口
和本地使用配置。权重内嵌的 Ultralytics 元数据声明 AGPL-3.0，这不等于训练数据也自动拥有
同一许可证。

## 为什么默认只选一颗

当前 NBV 会话的 `target_center`、ROI 和地图围绕一个目标定义。如果画面里检测出多颗草莓，
默认 `highest_confidence` 只保留置信度最高的一颗，避免把相距很远的多颗草莓合成一个奇怪
目标。以后做多果实观察时，应先为每颗实例分配稳定 ID，再逐颗建立或调度 NBV 会话。

## 环境和构建

从项目根目录执行：

```bash
mkdir -p artifacts/models
cp /home/yyt/Downloads/best.pt artifacts/models/yolo11m_strawberry_best.pt

python3 -m venv --system-site-packages --prompt sap-mask .venv-mask
.venv-mask/bin/python -m pip install -r validation/week9/requirements-yolo11.txt

source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
source .venv-mask/bin/activate
python -m colcon --log-base perception_ws/log build \
  --symlink-install --base-paths perception_ws/src \
  --build-base perception_ws/build --install-base perception_ws/install \
  --packages-select strawberry_learned_mask
deactivate
```

当前环境固定 CPU 版 PyTorch，避免与 `.venv-nbv` 的 CUDA 13 栈冲突，也避免重复下载数 GB
的 CUDA 12 运行库。按需拍照时单帧延迟可接受；若以后要处理连续视频，再建立经过单独回归
的 GPU 推理环境，不直接改动 NBV 环境。

## 先离线看结果

下面命令不能连接或控制机械臂：

```bash
source .venv-mask/bin/activate
PYTHONPATH=perception_ws/src/strawberry_learned_mask \
python -m strawberry_learned_mask.offline \
  --checkpoint artifacts/models/yolo11m_strawberry_best.pt \
  --checkpoint-sha256 7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357 \
  --input artifacts/week8/run_20260913_2025/observations/observation_010_agg5_29655c33de25b1d05b49bf7c.npz \
  --output-directory artifacts/week9/offline \
  --confidence 0.70 --labels strawberry --accept-pickle-checkpoint
```

绿色是 YOLO11 mask，橙色是“HSV 有、YOLO 没有”的部分。JSON 固定写
`safe_for_robot_use=false`，离线脚本永远不会发布 ROS 控制消息。

## 相机只读运行

相机适配器运行后，另开一个终端：

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
source perception_ws/install/setup.bash
source .venv-mask/bin/activate
export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python -m strawberry_learned_mask.ros_node --ros-args \
  --params-file perception_ws/install/strawberry_learned_mask/share/strawberry_learned_mask/config/yolo11m_strawberry.yaml
```

输入 `/strawberry/perception/observation`，输出
`/strawberry/perception/learned_observation`。QoS 为 Reliable、TransientLocal。节点没有
Service/Action client，也不发布任何机器人控制 topic。

## 接入 NBV

先生成只读 preview。除了原来的大空间配置，再叠加：

```text
--params-file perception_ws/install/strawberry_learned_mask/share/strawberry_learned_mask/config/real_nbv_use_learned_mask.yaml
```

这让监督器等待学习式 Observation。Capture 服务和全部安全门保持原样。如果模型没有找到
置信度至少 0.70 的 `strawberry`、mask 少于 200 像素或其中有效深度少于 100 像素，输出
mask 会被清空，NBV 会话确定性拒绝该帧，不会偷偷回退到 HSV 后继续运动。

第一次学习式 mask 真机实验仍必须先生成新的 preview；历史 HSV 计划和授权不可复用。
