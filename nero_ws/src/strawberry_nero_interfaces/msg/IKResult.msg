# Result codes shared by the IK service and motion action.
# Codes 0-9 are successful outcomes.
uint16 SUCCESS=0
uint16 ALREADY_AT_TARGET=1

# Codes 10-19 describe invalid or unavailable inputs.
uint16 INVALID_TARGET=10
uint16 TF_UNAVAILABLE=11
uint16 FEEDBACK_STALE=12

# Codes 20-29 describe rejected IK solutions.
uint16 UNREACHABLE=20
uint16 JOINT_LIMIT_VIOLATION=21
uint16 NEAR_SINGULARITY=22
uint16 DISCONTINUOUS_SOLUTION=23
uint16 JOINT_DELTA_TOO_LARGE=24

# Codes 30-39 describe trajectory or execution failures.
uint16 TRAJECTORY_LIMIT_VIOLATION=30
uint16 TRACKING_ERROR=31
uint16 TIMEOUT=32
uint16 CANCELED=33
uint16 DRIVER_FAULT=34

# An unexpected implementation failure. The reason field must contain details.
uint16 INTERNAL_ERROR=255

# True only for SUCCESS and ALREADY_AT_TARGET outcomes.
bool success
uint16 code
string reason

# A complete, ordered joint1 through joint7 solution. Implementations must fill
# name and position with exactly seven entries; velocity and effort are optional.
sensor_msgs/JointState solution_joint_state

# FK pose of controlled_frame for solution_joint_state, expressed in the target
# pose reference frame.
geometry_msgs/PoseStamped solved_pose

# Cartesian residuals between solved_pose and the requested target pose.
float64 position_error_m
float64 orientation_error_rad

# IK diagnostics.
float64 solve_time_ms
float64 sigma_min
float64 condition_number
float64 max_joint_delta_rad
