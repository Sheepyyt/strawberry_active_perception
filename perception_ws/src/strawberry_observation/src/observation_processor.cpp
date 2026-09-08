#include "strawberry_observation/observation_processor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace strawberry_observation
{
namespace
{

ProcessResult failure(FrameStatus status, std::string reason)
{
  ProcessResult result;
  result.status = status;
  result.reason = std::move(reason);
  return result;
}

int64_t stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return rclcpp::Time(stamp, RCL_ROS_TIME).nanoseconds();
}

bool valid_stamp(const builtin_interfaces::msg::Time & stamp)
{
  return stamp.sec >= 0 && stamp.nanosec < 1000000000U;
}

bool finite_intrinsics(const sensor_msgs::msg::CameraInfo & info)
{
  return std::all_of(info.k.begin(), info.k.end(), [](double value) {
             return std::isfinite(value);
    }) &&
         info.k[0] > 0.0 && info.k[4] > 0.0 &&
         std::abs(info.k[1]) <= 1.0e-12 &&
         std::abs(info.k[3]) <= 1.0e-12 &&
         std::abs(info.k[6]) <= 1.0e-12 &&
         std::abs(info.k[7]) <= 1.0e-12 &&
         std::abs(info.k[8] - 1.0) <= 1.0e-12;
}

bool finite_supported_pinhole_distortion(const sensor_msgs::msg::CameraInfo & info)
{
  const bool supported_layout =
    (info.distortion_model == "rational_polynomial" && info.d.size() == 8U) ||
    (info.distortion_model == "plumb_bob" && info.d.size() == 5U);
  return supported_layout &&
         std::all_of(info.d.begin(), info.d.end(), [](double value) {
             return std::isfinite(value);
         });
}

bool is_rectified_camera_model(const sensor_msgs::msg::CameraInfo & info)
{
  return finite_supported_pinhole_distortion(info) &&
         std::all_of(info.d.begin(), info.d.end(), [](double value) {
             return std::abs(value) <= 1.0e-12;
         });
}

bool matching_intrinsics(
  const sensor_msgs::msg::CameraInfo & first,
  const sensor_msgs::msg::CameraInfo & second)
{
  for (size_t index = 0U; index < first.k.size(); ++index) {
    if (std::abs(first.k[index] - second.k[index]) > 1.0e-12) {
      return false;
    }
  }
  return true;
}

cv::Mat camera_matrix(const sensor_msgs::msg::CameraInfo & info)
{
  cv::Mat matrix(3, 3, CV_64FC1);
  std::copy(info.k.begin(), info.k.end(), matrix.ptr<double>());
  return matrix;
}

cv::Mat distortion_vector(const sensor_msgs::msg::CameraInfo & info)
{
  cv::Mat distortion(1, static_cast<int>(info.d.size()), CV_64FC1);
  std::copy(info.d.begin(), info.d.end(), distortion.ptr<double>());
  return distortion;
}

std::string dimensions(uint32_t width, uint32_t height)
{
  return std::to_string(width) + "x" + std::to_string(height);
}

}  // namespace

ObservationProcessor::ObservationProcessor(ProcessingConfig config)
: config_(std::move(config))
{
  if (!std::isfinite(config_.depth_16uc1_scale) || config_.depth_16uc1_scale <= 0.0) {
    throw std::invalid_argument("depth_16uc1_scale must be finite and positive");
  }
  if (!std::isfinite(config_.depth_32fc1_scale) || config_.depth_32fc1_scale <= 0.0) {
    throw std::invalid_argument("depth_32fc1_scale must be finite and positive");
  }
  if (!std::isfinite(config_.depth_min_m) || !std::isfinite(config_.depth_max_m) ||
    config_.depth_min_m < 0.0 || config_.depth_max_m <= config_.depth_min_m)
  {
    throw std::invalid_argument("depth range must satisfy 0 <= min < max");
  }
  if (!std::isfinite(config_.min_valid_depth_fraction) ||
    config_.min_valid_depth_fraction < 0.0 || config_.min_valid_depth_fraction > 1.0)
  {
    throw std::invalid_argument("min_valid_depth_fraction must be in [0, 1]");
  }
  if (!std::isfinite(config_.max_time_skew_sec) || config_.max_time_skew_sec < 0.0) {
    throw std::invalid_argument("max_time_skew_sec must be finite and non-negative");
  }
  if (config_.morphology_kernel_size <= 0 || config_.morphology_kernel_size % 2 == 0) {
    throw std::invalid_argument("morphology_kernel_size must be a positive odd integer");
  }
  if (config_.min_mask_pixels < 1) {
    throw std::invalid_argument("min_mask_pixels must be positive");
  }

  const auto valid_hsv = [](const std::array<int, 3> & value) {
      return value[0] >= 0 && value[0] <= 179 &&
             value[1] >= 0 && value[1] <= 255 &&
             value[2] >= 0 && value[2] <= 255;
    };
  if (!valid_hsv(config_.red_hsv_low_1) || !valid_hsv(config_.red_hsv_high_1) ||
    !valid_hsv(config_.red_hsv_low_2) || !valid_hsv(config_.red_hsv_high_2))
  {
    throw std::invalid_argument("HSV thresholds are outside OpenCV HSV ranges");
  }
}

ProcessResult ObservationProcessor::process(
  const sensor_msgs::msg::Image & color,
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::CameraInfo & color_camera_info,
  const sensor_msgs::msg::CameraInfo & depth_camera_info,
  bool require_mask) const
{
  if (color.width == 0U || color.height == 0U || color.data.empty()) {
    return failure(FrameStatus::kMissingColor, "color image is empty");
  }
  if (depth.width == 0U || depth.height == 0U || depth.data.empty()) {
    return failure(FrameStatus::kMissingDepth, "depth image is empty");
  }
  if (color_camera_info.width == 0U || color_camera_info.height == 0U ||
    depth_camera_info.width == 0U || depth_camera_info.height == 0U ||
    !finite_intrinsics(color_camera_info) || !finite_intrinsics(depth_camera_info))
  {
    return failure(FrameStatus::kMissingCameraInfo,
        "color or depth CameraInfo is empty or has invalid intrinsics");
  }
  if (!finite_supported_pinhole_distortion(color_camera_info)) {
    return failure(
      FrameStatus::kCameraModelMismatch,
      "raw color CameraInfo must use rational_polynomial with 8 finite coefficients "
      "or plumb_bob with 5 finite coefficients");
  }
  if (!is_rectified_camera_model(depth_camera_info)) {
    return failure(
      FrameStatus::kCameraModelMismatch,
      "registered depth CameraInfo must use rational_polynomial with 8 or plumb_bob "
      "with 5 finite, zero distortion coefficients");
  }
  if (!matching_intrinsics(color_camera_info, depth_camera_info)) {
    return failure(
      FrameStatus::kCameraModelMismatch,
      "raw color and registered depth CameraInfo K matrices must match exactly");
  }
  if (color.width != depth.width || color.height != depth.height ||
    color.width != color_camera_info.width || color.height != color_camera_info.height ||
    depth.width != depth_camera_info.width || depth.height != depth_camera_info.height)
  {
    std::ostringstream reason;
    reason << "registered grid mismatch: color=" << dimensions(color.width, color.height)
           << ", depth=" << dimensions(depth.width, depth.height)
           << ", color_info=" << dimensions(
      color_camera_info.width, color_camera_info.height)
           << ", depth_info=" << dimensions(
      depth_camera_info.width, depth_camera_info.height);
    return failure(FrameStatus::kDimensionMismatch, reason.str());
  }
  if (color.header.frame_id.empty() || depth.header.frame_id.empty() ||
    color_camera_info.header.frame_id.empty() || depth_camera_info.header.frame_id.empty() ||
    color.header.frame_id != depth.header.frame_id ||
    color.header.frame_id != color_camera_info.header.frame_id ||
    color.header.frame_id != depth_camera_info.header.frame_id)
  {
    return failure(
      FrameStatus::kDimensionMismatch,
      "color, depth, and both CameraInfo messages must share one registered optical frame_id");
  }

  if (!valid_stamp(color.header.stamp) || !valid_stamp(depth.header.stamp) ||
    !valid_stamp(color_camera_info.header.stamp) ||
    !valid_stamp(depth_camera_info.header.stamp))
  {
    return failure(
      FrameStatus::kTimeSkewExceeded,
      "color, depth, and both CameraInfo stamps must have sec >= 0 and nanosec < 1000000000");
  }

  // Subtract integer nanosecond stamps before converting the small delta to
  // seconds.  Converting epoch-sized stamps to double first loses tens or
  // hundreds of nanoseconds and makes the canonical skew disagree with the
  // original headers.
  const int64_t color_stamp = stamp_nanoseconds(color.header.stamp);
  const int64_t depth_stamp = stamp_nanoseconds(depth.header.stamp);
  const int64_t color_info_stamp = stamp_nanoseconds(color_camera_info.header.stamp);
  const int64_t depth_info_stamp = stamp_nanoseconds(depth_camera_info.header.stamp);
  if (color_info_stamp != color_stamp || depth_info_stamp != depth_stamp) {
    return failure(
      FrameStatus::kTimeSkewExceeded,
      "each CameraInfo stamp must exactly equal its corresponding image stamp");
  }
  const double color_depth_skew =
    static_cast<double>(color_stamp - depth_stamp) * 1.0e-9;
  const double maximum_skew = std::abs(color_depth_skew);
  if (!std::isfinite(maximum_skew) || maximum_skew > config_.max_time_skew_sec) {
    std::ostringstream reason;
    reason << "timestamp skew " << maximum_skew << " s exceeds "
           << config_.max_time_skew_sec << " s";
    return failure(FrameStatus::kTimeSkewExceeded, reason.str());
  }

  cv_bridge::CvImagePtr color_cv;
  cv_bridge::CvImagePtr depth_cv;
  cv::Mat rectified_color;
  try {
    color_cv = cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::RGB8);
    if (depth.encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      depth_cv = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_16UC1);
    } else if (depth.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      depth_cv = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
    } else {
      return failure(
        FrameStatus::kInvalidEncoding,
        "depth encoding must be 16UC1 or 32FC1, got '" + depth.encoding + "'");
    }
    const cv::Mat intrinsics = camera_matrix(color_camera_info);
    const cv::Mat distortion = distortion_vector(color_camera_info);
    cv::undistort(color_cv->image, rectified_color, intrinsics, distortion, intrinsics);
  } catch (const cv_bridge::Exception & error) {
    return failure(FrameStatus::kInvalidEncoding, error.what());
  } catch (const cv::Exception & error) {
    return failure(FrameStatus::kCameraModelMismatch, error.what());
  }

  cv::Mat depth_m(depth_cv->image.rows, depth_cv->image.cols, CV_32FC1);
  const float invalid = std::numeric_limits<float>::quiet_NaN();
  uint64_t valid_count = 0U;
  const auto accept_depth = [this](double value) {
      return std::isfinite(value) && value >= config_.depth_min_m && value <= config_.depth_max_m;
    };

  if (depth.encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    for (int row = 0; row < depth_cv->image.rows; ++row) {
      const auto * input = depth_cv->image.ptr<uint16_t>(row);
      auto * output = depth_m.ptr<float>(row);
      for (int col = 0; col < depth_cv->image.cols; ++col) {
        const double value_m = static_cast<double>(input[col]) * config_.depth_16uc1_scale;
        if (input[col] != 0U && accept_depth(value_m)) {
          output[col] = static_cast<float>(value_m);
          ++valid_count;
        } else {
          output[col] = invalid;
        }
      }
    }
  } else {
    for (int row = 0; row < depth_cv->image.rows; ++row) {
      const auto * input = depth_cv->image.ptr<float>(row);
      auto * output = depth_m.ptr<float>(row);
      for (int col = 0; col < depth_cv->image.cols; ++col) {
        const double value_m = static_cast<double>(input[col]) * config_.depth_32fc1_scale;
        if (accept_depth(value_m)) {
          output[col] = static_cast<float>(value_m);
          ++valid_count;
        } else {
          output[col] = invalid;
        }
      }
    }
  }

  const auto pixel_count = static_cast<uint64_t>(depth.width) * depth.height;
  const float valid_fraction = pixel_count == 0U ? 0.0F :
    static_cast<float>(static_cast<double>(valid_count) / static_cast<double>(pixel_count));
  if (valid_fraction < config_.min_valid_depth_fraction) {
    std::ostringstream reason;
    reason << "valid depth fraction " << valid_fraction << " is below "
           << config_.min_valid_depth_fraction;
    return failure(FrameStatus::kDepthQualityLow, reason.str());
  }

  cv::Mat hsv;
  cv::cvtColor(rectified_color, hsv, cv::COLOR_RGB2HSV);
  cv::Mat mask_1;
  cv::Mat mask_2;
  cv::inRange(
    hsv,
    cv::Scalar(
      config_.red_hsv_low_1[0], config_.red_hsv_low_1[1], config_.red_hsv_low_1[2]),
    cv::Scalar(
      config_.red_hsv_high_1[0], config_.red_hsv_high_1[1], config_.red_hsv_high_1[2]),
    mask_1);
  cv::inRange(
    hsv,
    cv::Scalar(
      config_.red_hsv_low_2[0], config_.red_hsv_low_2[1], config_.red_hsv_low_2[2]),
    cv::Scalar(
      config_.red_hsv_high_2[0], config_.red_hsv_high_2[1], config_.red_hsv_high_2[2]),
    mask_2);

  cv::Mat mask = mask_1 | mask_2;
  const cv::Mat kernel = cv::getStructuringElement(
    cv::MORPH_RECT,
    cv::Size(config_.morphology_kernel_size, config_.morphology_kernel_size));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int component_count = cv::connectedComponentsWithStats(
    mask, labels, stats, centroids, 8, CV_32S);
  int largest_label = 0;
  int largest_area = 0;
  for (int label = 1; label < component_count; ++label) {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (area > largest_area) {
      largest_label = label;
      largest_area = area;
    }
  }

  cv::Mat largest_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
  if (largest_area >= config_.min_mask_pixels) {
    cv::compare(labels, largest_label, largest_mask, cv::CMP_EQ);
  } else {
    largest_area = 0;
  }
  if (require_mask && largest_area == 0) {
    return failure(
      FrameStatus::kMissingMask,
      "no red connected component reached " + std::to_string(config_.min_mask_pixels) +
      " pixels");
  }

  ProcessResult result;
  result.status = FrameStatus::kSuccess;
  result.reason = "ok";
  result.frame.color = *cv_bridge::CvImage(
    color.header, sensor_msgs::image_encodings::RGB8, rectified_color).toImageMsg();
  result.frame.depth = *cv_bridge::CvImage(
    depth.header, sensor_msgs::image_encodings::TYPE_32FC1, depth_m).toImageMsg();
  result.frame.target_mask = *cv_bridge::CvImage(
    color.header, sensor_msgs::image_encodings::MONO8, largest_mask).toImageMsg();
  result.frame.camera_info = depth_camera_info;
  // CameraInfo describes the accepted registered depth grid. Normalize its
  // authoritative stamp after validating the source skew above so every
  // canonical field uses the depth exposure time.
  result.frame.camera_info.header.stamp = depth.header.stamp;
  result.frame.valid_depth_fraction = valid_fraction;
  result.frame.color_depth_skew_sec = color_depth_skew;
  result.frame.mask_pixels = largest_area;
  return result;
}

}  // namespace strawberry_observation
