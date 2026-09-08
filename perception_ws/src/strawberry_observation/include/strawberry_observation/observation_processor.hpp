#ifndef STRAWBERRY_OBSERVATION__OBSERVATION_PROCESSOR_HPP_
#define STRAWBERRY_OBSERVATION__OBSERVATION_PROCESSOR_HPP_

#include <array>
#include <cstdint>
#include <string>

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace strawberry_observation
{

enum class FrameStatus : uint16_t
{
  kSuccess = 0,
  kMissingColor = 12,
  kMissingDepth = 13,
  kMissingCameraInfo = 14,
  kMissingMask = 15,
  kInvalidEncoding = 17,
  kDimensionMismatch = 18,
  kTimeSkewExceeded = 19,
  kDepthQualityLow = 20,
  kCameraModelMismatch = 21,
  kInternalError = 255,
};

struct ProcessingConfig
{
  double depth_16uc1_scale{0.001};
  double depth_32fc1_scale{1.0};
  double depth_min_m{0.2};
  double depth_max_m{2.5};
  double min_valid_depth_fraction{0.05};
  double max_time_skew_sec{0.005};

  std::array<int, 3> red_hsv_low_1{0, 80, 50};
  std::array<int, 3> red_hsv_high_1{10, 255, 255};
  std::array<int, 3> red_hsv_low_2{170, 80, 50};
  std::array<int, 3> red_hsv_high_2{179, 255, 255};
  int morphology_kernel_size{5};
  int min_mask_pixels{200};
};

struct ProcessedFrame
{
  sensor_msgs::msg::Image color;
  sensor_msgs::msg::Image depth;
  sensor_msgs::msg::Image target_mask;
  sensor_msgs::msg::CameraInfo camera_info;
  float valid_depth_fraction{0.0F};
  double color_depth_skew_sec{0.0};
  int mask_pixels{0};
};

struct ProcessResult
{
  FrameStatus status{FrameStatus::kInternalError};
  std::string reason;
  ProcessedFrame frame;

  bool ok() const {return status == FrameStatus::kSuccess;}
};

class ObservationProcessor
{
public:
  explicit ObservationProcessor(ProcessingConfig config);

  ProcessResult process(
    const sensor_msgs::msg::Image & color,
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & color_camera_info,
    const sensor_msgs::msg::CameraInfo & depth_camera_info,
    bool require_mask) const;

  const ProcessingConfig & config() const {return config_;}

private:
  ProcessingConfig config_;
};

}  // namespace strawberry_observation

#endif  // STRAWBERRY_OBSERVATION__OBSERVATION_PROCESSOR_HPP_
