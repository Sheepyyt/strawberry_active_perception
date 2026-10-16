# 重新安装与离线复现

日常操作使用根目录 `DEMO_GUIDE_CN.md`。本文用于重建当前软件环境，不要求每轮演示都执行。

## 依赖与固定文件

当前验证环境为 Ubuntu 24.04、Python 3.12、ROS 2 Jazzy；先安装 ROS 2 Jazzy。以下命令从克隆后的项目根目录执行。设备驱动由 Git 子模块固定，使用记录的提交，不直接切到最新分支。

```bash
git submodule update --init --recursive
sudo apt install python3-venv python3-pip python3-opencv python3-pytest \
  python3-flake8 python3-colcon-common-extensions python3-rosdep \
  can-utils ethtool
./vendor_patches/agx_arm_ros/apply_checked.sh

source /opt/ros/jazzy/setup.bash
# 首次配置 rosdep 时执行 sudo rosdep init；已初始化的机器跳过。
rosdep update
rosdep install --rosdistro jazzy --ignore-src -r -y --from-paths \
  camera_ws/src perception_ws/src \
  nero_ws/src/strawberry_nero_interfaces nero_ws/src/strawberry_nero_control \
  nero_ws/src/agx_arm_ros/src/agx_arm_msgs \
  nero_ws/src/agx_arm_ros/src/agx_arm_description \
  nero_ws/src/agx_arm_ros/src/agx_arm_ctrl

sudo bash camera_ws/src/OrbbecSDK_ROS2/orbbec_camera/scripts/install_udev_rules.sh
```

上述依赖检查只包含当前链路，不包含 MoveIt。相机安装细节见子模块的 `README_CN.MD`。

Git 中必须保留：

- `config/calibration/handeye_report.json`：正式手眼外参，SHA-256 为 `31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f`。相机相对末端的安装改变后需要重新标定。
- `config/calibration/handeye_samples.json` 和相机模型：支持检查、重新求解标定。
- `config/models/yolo11_manifest.json`：权重路径、大小与 SHA 校验清单。
- `tests/fixtures/`：离线回归需要的固定样本，不是可执行的真机授权计划。

正式标定报告内的历史采集路径是来源信息；运行只依赖这里保存的报告及样本。不要修改报告内容，否则 SHA 校验会失败。

迁移时另带 `config/models/yolo11m_strawberry_best.pt`，SHA-256 为 `7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357`。模型不提交 Git，也没有公开下载来源。需要离线复盘时，另带完整的 `results/demo/run_…/`；Git 不保存运行结果，测试使用固定小样本或临时生成数据。

离线周报导出另需系统组件：`sudo apt install libreoffice-writer poppler-utils fonts-noto-cjk python3-numpy python3-matplotlib python3-pil`。使用 `./demo_operator.sh analyze [已有结果目录]` 重新生成 Word / PDF 和分析表；不依赖 ROS 节点或设备连接。

## 三个 Python 环境

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
# 将路径换成另行带来的、已验证支持 NERO v1.11 的 SDK 源码目录。
.venv/bin/python -m pip install /path/to/pyAgxArm 'python-can==4.6.1'
.venv/bin/python -c "from importlib.metadata import version; assert version('pyAgxArm') == '1.0.0'; print('pyAgxArm 1.0.0')"

python3 -m venv --system-site-packages .venv-nbv
.venv-nbv/bin/python -m pip install -r perception_ws/src/strawberry_gradient_nbv/requirements-nbv.txt

python3 -m venv --system-site-packages .venv-mask
.venv-mask/bin/python -m pip install -r requirements-yolo11.txt
```

本机已验证的 SDK 元数据为 `pyAgxArm 1.0.0`、`python-can 4.6.1`。SDK 来自用户提供的本地源码，未记录可核对的上游提交，且不能用 `pip install pyAgxArm==1.0.0` 从 PyPI 重建。因此迁移必须另带已验证的 SDK；版本号检查本身不能证明两份源码相同。

Placo、PyTorch/NBV、YOLO 使用不同环境，避免 NumPy、CUDA 和系统 ROS/OpenCV 的版本互相污染。相机采集与结果入口使用系统 Python。`.venv*`、`build/`、`install/` 不随 Git 提交，克隆后按此重建。

## 构建

```bash
source /opt/ros/jazzy/setup.bash

cd camera_ws
colcon build --symlink-install --packages-select orbbec_camera_msgs orbbec_description orbbec_camera
cd ..

source .venv/bin/activate
cd nero_ws
python -m colcon build --symlink-install --packages-select agx_arm_msgs agx_arm_description agx_arm_ctrl strawberry_nero_interfaces strawberry_nero_control
cd ..
deactivate

source camera_ws/install/setup.bash
source nero_ws/install/setup.bash
source .venv-nbv/bin/activate
python -m colcon --log-base perception_ws/log build --symlink-install --base-paths perception_ws/src --build-base perception_ws/build --install-base perception_ws/install
deactivate

source perception_ws/install/setup.bash
source .venv-mask/bin/activate
python -m colcon --log-base perception_ws/log build --symlink-install --base-paths perception_ws/src --build-base perception_ws/build --install-base perception_ws/install --packages-select strawberry_learned_mask
deactivate

./verify_software.sh
```

`nero_ws/colcon_defaults.yaml` 排除 MoveIt，当前流程不依赖 MoveIt。

## 复现一轮结果

每轮保存 `run.json` 与配置副本、预览计划、执行 JSON、实际使用的观测 NPZ、地图 NPZ 和模型审计。新一轮必须使用新的观测和计划，不能执行旧结果目录中的授权。

只重新生成图表，不连接硬件：

```bash
./demo_operator.sh report results/demo/run_实际目录名
```

运行 `./demo_operator.sh open` 用浏览器查看最近一轮图文报告，或直接阅读 `results/结果说明.txt`。网页的展开区域提供原始证据。未取得动作后有效观测时，不将该动作计作地图更新，也不把启动关节恢复计作 NBV 步数。
