# Structured Gradient-NBV outcome and diagnostics.
# These keys identify the exact canonical Observation that produced this
# result, including when NextView is replayed on its transient-local topic.
string scene_id
string observation_id

# Codes 0-9 are successful outcomes.
uint16 SUCCESS=0

# Codes 10-19 describe invalid or unavailable inputs/configuration.
uint16 INVALID_REQUEST=10
uint16 OBSERVATION_NOT_FOUND=11
uint16 NOT_CONFIGURED=12
uint16 SCENE_MISMATCH=13

# Codes 20-29 describe map or next-view computation failures.
uint16 MAP_UPDATE_FAILED=20
uint16 NO_VALID_CANDIDATE=21
uint16 OPTIMIZATION_FAILED=22

# Codes 30-39 describe interrupted computation.
uint16 TIMEOUT=32
uint16 CANCELED=33

# Unexpected implementation failure. reason must contain actionable details.
uint16 INTERNAL_ERROR=255

# True only for SUCCESS.
bool success
uint16 code
string reason

# Target pose T_world_camera_optical. pose.header.frame_id is the configured
# world frame. Position is in metres and orientation is a unit quaternion.
geometry_msgs/PoseStamped pose

# Predicted information gain and map coverage after accepting this observation.
# coverage is a dimensionless fraction in [0, 1].
float64 gain
float32 coverage

# Voxel-map diagnostics after the map update and before any future observation.
uint64 total_voxel_count
uint64 observed_voxel_count
uint64 occupied_voxel_count
uint64 unknown_voxel_count

# Optimizer work and total ComputeNextView wall-clock time.
uint32 optimization_iterations
float64 compute_time_ms
