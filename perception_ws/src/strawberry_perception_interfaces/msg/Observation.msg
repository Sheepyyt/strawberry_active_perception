# One synchronized, canonical RGB-D observation consumed by Gradient-NBV.
#
# header.stamp is the depth exposure time. header.frame_id names the camera
# optical frame shared by the registered depth, color, mask, and camera_info.
std_msgs/Header header

# A map/configuration generation. Observations from different scenes must never
# update the same NBV map.
string scene_id

# Unique within scene_id. Consumers must process a given ID at most once.
string observation_id

# Adapter/source kinds. source_name identifies the concrete device, bag, or
# dataset without coupling consumers to its ROS topic names.
uint8 SOURCE_UNKNOWN=0
uint8 SOURCE_REAL=1
uint8 SOURCE_OFFLINE=2
uint8 SOURCE_SYNTHETIC=3
uint8 SOURCE_REPLAY=4
uint8 source_type
string source_name

# Registered image grid. color uses rgb8. depth uses 32FC1 metres, with NaN for
# invalid samples. target_mask uses mono8 (0 background, 255 target); it may be
# empty only when the caller did not require a mask.
sensor_msgs/Image color
sensor_msgs/Image depth
sensor_msgs/Image target_mask

# Intrinsics for the registered image grid and optical frame above. Canonical
# v1 observations are already rectified: distortion_model is retained for
# provenance, but every value in camera_info.d must be finite and exactly zero.
sensor_msgs/CameraInfo camera_info

# T_world_camera_optical at header.stamp. camera_pose.header.frame_id is the
# fixed world frame and the described child is header.frame_id. pose_valid=false
# explicitly marks an unavailable transform; consumers must not infer validity
# from zero/default pose values.
geometry_msgs/PoseStamped camera_pose
bool pose_valid

# Fraction of registered depth pixels containing finite values in the accepted
# depth interval, in [0, 1]. Skew is color stamp minus depth stamp, in seconds;
# its sign is retained for diagnostics.
float32 valid_depth_fraction
float64 color_depth_skew_sec
