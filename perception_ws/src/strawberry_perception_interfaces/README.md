# strawberry_perception_interfaces

This ROS 2 Jazzy package is the dependency-light contract between camera
adapters, Gradient-NBV, replay/simulation sources, and later robot bridges. It
contains no Orbbec SDK, MoveIt, Placo, or algorithm implementation.

## Contract

- `Observation.msg` carries one registered RGB-D sample. `header.stamp` is the
  depth exposure time and `header.frame_id` is the shared optical frame.
- Color is `rgb8`; depth is `32FC1` in metres with NaN invalid values; the
  optional target mask is `mono8` with values 0/255. All populated image fields
  and `CameraInfo` describe the same pixel grid.
- `camera_pose` is `T_world_camera_optical` at the observation stamp.
  `pose_valid` is authoritative; default pose values never imply validity.
- `scene_id` prevents accidental map mixing. `observation_id` is unique within
  a scene and map consumers process it idempotently.
- `CaptureObservation.srv` requires the destination `scene_id` and returns an ID
  and stamp; the large Observation is published on a canonical topic instead of
  copied through the service.
- A successful `ConfigureNBV.srv` call atomically applies the configuration and
  resets that scene's map. All geometry and depth values are SI metres.
- `ComputeNextView.action` references one Observation by ID and returns a
  structured `NextView`; its target pose is the camera optical pose in the
  configured world frame. `NextView` repeats `scene_id` and `observation_id` so
  the same result remains self-describing when published on an independent or
  transient-local topic.

Source kinds are stable numeric values: unknown `0`, real `1`, offline `2`,
synthetic `3`, and replay `4`. Status responses follow the repository convention
of `success`, numeric `code`, and human-readable `reason`; code `255` is reserved
for unexpected internal errors.

The optical frame follows the ROS camera convention: +X right, +Y down, +Z
forward. Depth is optical-axis Z, not Euclidean ray length. ROS quaternions are
always `xyzw`; `camera_pose.header.frame_id` names the world/map frame and the
implicit child is `Observation.header.frame_id`. Color and mask may retain the
color exposure stamp so their signed skew is auditable, while the Observation,
depth, CameraInfo, and camera pose use the depth exposure stamp. Adapters must
reject an absolute color/depth skew over 5 ms.

`CameraInfo.K` describes the registered color/depth pixel grid carried in that
same message. Version 1 requires a rectified pinhole grid: every supplied
distortion coefficient is zero, because depth back-projection consumers use K
directly. K must be rescaled whenever an adapter resizes or strides the images.
Invalid metric depth is NaN on the wire—zero, negative, infinity, and
values outside the configured adapter interval must not escape as valid data.
Version 1 intentionally carries no `PointCloud2`; consumers reconstruct points
from depth and K, avoiding vendor-specific layout and frame assumptions.

The canonical observation topic is
`/strawberry/perception/observation` with Reliable + TransientLocal +
KeepLast(1). `CaptureObservation` returns only the accepted ID and depth stamp;
large images travel once on that topic.

## Build and test

```bash
cd /home/yyt/strawberry_active_perception
source /opt/ros/jazzy/setup.bash
python -m colcon --log-base perception_ws/log build \
  --base-paths perception_ws/src \
  --build-base perception_ws/build \
  --install-base perception_ws/install \
  --packages-select strawberry_perception_interfaces
python -m colcon --log-base perception_ws/test-log test \
  --base-paths perception_ws/src \
  --build-base perception_ws/build \
  --install-base perception_ws/install \
  --packages-select strawberry_perception_interfaces
python -m colcon test-result --test-result-base perception_ws/build
```
