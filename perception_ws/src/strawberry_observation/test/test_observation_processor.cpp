#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "cv_bridge/cv_bridge.hpp"
#include "gtest/gtest.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "strawberry_observation/observation_processor.hpp"

namespace strawberry_observation
{
namespace
{

constexpr uint32_t kWidth = 40U;
constexpr uint32_t kHeight = 30U;
constexpr char kFrame[] = "camera_color_optical_frame";

builtin_interfaces::msg::Time stamp(int32_t sec, uint32_t nanosec = 0U)
{
  builtin_interfaces::msg::Time value;
  value.sec = sec;
  value.nanosec = nanosec;
  return value;
}

sensor_msgs::msg::Image image_message(
  const cv::Mat & image, const std::string & encoding,
  const builtin_interfaces::msg::Time & image_stamp)
{
  std_msgs::msg::Header header;
  header.stamp = image_stamp;
  header.frame_id = kFrame;
  return *cv_bridge::CvImage(header, encoding, image).toImageMsg();
}

sensor_msgs::msg::Image red_rgb_image(
  const builtin_interfaces::msg::Time & image_stamp, bool bgr_encoding = false)
{
  cv::Mat image(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_8UC3, cv::Scalar(0, 0, 0));
  const cv::Scalar red = bgr_encoding ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
  cv::rectangle(image, cv::Rect(5, 5, 24, 20), red, cv::FILLED);
  cv::rectangle(image, cv::Rect(34, 2, 3, 3), red, cv::FILLED);
  return image_message(
    image, bgr_encoding ? sensor_msgs::image_encodings::BGR8 :
    sensor_msgs::image_encodings::RGB8, image_stamp);
}

sensor_msgs::msg::CameraInfo camera_info(const builtin_interfaces::msg::Time & info_stamp)
{
  sensor_msgs::msg::CameraInfo info;
  info.header.stamp = info_stamp;
  info.header.frame_id = kFrame;
  info.width = kWidth;
  info.height = kHeight;
  info.k[0] = 300.0;
  info.k[2] = 20.0;
  info.k[4] = 300.0;
  info.k[5] = 15.0;
  info.k[8] = 1.0;
  info.distortion_model = "rational_polynomial";
  info.d.assign(8U, 0.0);
  return info;
}

ProcessingConfig test_config()
{
  ProcessingConfig config;
  config.depth_min_m = 0.1;
  config.depth_max_m = 3.0;
  config.min_valid_depth_fraction = 0.5;
  config.max_time_skew_sec = 0.005;
  config.min_mask_pixels = 200;
  return config;
}

TEST(ObservationProcessor, Converts16BitMillimetresAndBuildsLargestRedMask)
{
  const auto depth_stamp = stamp(42, 100000000U);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1, cv::Scalar(1250));
  depth.at<uint16_t>(0, 0) = 0U;
  depth.at<uint16_t>(0, 1) = 3500U;

  ObservationProcessor processor(test_config());
  const ProcessResult result = processor.process(
    red_rgb_image(stamp(42, 102000000U), true),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, depth_stamp),
    camera_info(stamp(42, 102000000U)), camera_info(depth_stamp), true);

  ASSERT_TRUE(result.ok()) << result.reason;
  EXPECT_EQ(result.frame.color.encoding, sensor_msgs::image_encodings::RGB8);
  EXPECT_EQ(result.frame.depth.encoding, sensor_msgs::image_encodings::TYPE_32FC1);
  EXPECT_EQ(result.frame.target_mask.encoding, sensor_msgs::image_encodings::MONO8);
  EXPECT_EQ(result.frame.depth.header.stamp, depth_stamp);
  EXPECT_NEAR(result.frame.color_depth_skew_sec, 0.002, 1.0e-9);
  EXPECT_NEAR(result.frame.valid_depth_fraction, 1198.0 / 1200.0, 1.0e-6);
  EXPECT_GE(result.frame.mask_pixels, 400);
  EXPECT_LE(result.frame.mask_pixels, 480);

  const cv::Mat canonical_depth = cv_bridge::toCvCopy(
    result.frame.depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
  EXPECT_TRUE(std::isnan(canonical_depth.at<float>(0, 0)));
  EXPECT_TRUE(std::isnan(canonical_depth.at<float>(0, 1)));
  EXPECT_NEAR(canonical_depth.at<float>(1, 1), 1.25F, 1.0e-6F);

  const cv::Mat canonical_color = cv_bridge::toCvCopy(
    result.frame.color, sensor_msgs::image_encodings::RGB8)->image;
  EXPECT_EQ(canonical_color.at<cv::Vec3b>(10, 10), cv::Vec3b(255, 0, 0));
  const cv::Mat mask = cv_bridge::toCvCopy(
    result.frame.target_mask, sensor_msgs::image_encodings::MONO8)->image;
  EXPECT_EQ(mask.at<uint8_t>(10, 10), 255U);
  EXPECT_EQ(mask.at<uint8_t>(3, 35), 0U);
}

TEST(ObservationProcessor, DilatesOnlyTheAcceptedLargestRedComponent)
{
  auto config = test_config();
  config.mask_dilation_kernel_size = 5;
  const auto common_stamp = stamp(43);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));

  ObservationProcessor processor(config);
  const ProcessResult result = processor.process(
    red_rgb_image(common_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp),
    camera_info(common_stamp), camera_info(common_stamp), true);

  ASSERT_TRUE(result.ok()) << result.reason;
  EXPECT_EQ(result.frame.mask_pixels, 28 * 24);
  const cv::Mat mask = cv_bridge::toCvCopy(
    result.frame.target_mask, sensor_msgs::image_encodings::MONO8)->image;
  EXPECT_EQ(mask.at<uint8_t>(3, 3), 255U);
  EXPECT_EQ(mask.at<uint8_t>(2, 2), 0U);
  EXPECT_EQ(mask.at<uint8_t>(3, 35), 0U);
}

TEST(ObservationProcessor, Normalizes32BitDepthAndInvalidValuesToNan)
{
  auto config = test_config();
  config.depth_32fc1_scale = 1.0;
  config.min_valid_depth_fraction = 0.0;
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_32FC1, cv::Scalar(0.8F));
  depth.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
  depth.at<float>(0, 1) = std::numeric_limits<float>::infinity();
  depth.at<float>(0, 2) = 0.05F;
  depth.at<float>(0, 3) = 4.0F;

  ObservationProcessor processor(config);
  const auto common_stamp = stamp(10);
  const ProcessResult result = processor.process(
    red_rgb_image(common_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_32FC1, common_stamp),
    camera_info(common_stamp), camera_info(common_stamp), false);

  ASSERT_TRUE(result.ok()) << result.reason;
  const cv::Mat canonical_depth = cv_bridge::toCvCopy(
    result.frame.depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
  for (int col = 0; col < 4; ++col) {
    EXPECT_TRUE(std::isnan(canonical_depth.at<float>(0, col)));
  }
  EXPECT_NEAR(canonical_depth.at<float>(1, 0), 0.8F, 1.0e-6F);
}

TEST(ObservationProcessor, RejectsTimestampSkewOverFiveMilliseconds)
{
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1, cv::Scalar(1000));
  ObservationProcessor processor(test_config());
  const ProcessResult result = processor.process(
    red_rgb_image(stamp(2, 6000000U)),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, stamp(2)),
    camera_info(stamp(2, 6000000U)), camera_info(stamp(2)), true);

  EXPECT_EQ(result.status, FrameStatus::kTimeSkewExceeded);
  EXPECT_FALSE(result.ok());
}

TEST(ObservationProcessor, NormalizesCameraInfoStampToDepthExposure)
{
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  ObservationProcessor processor(test_config());
  const auto depth_stamp = stamp(3, 100000000U);
  const auto color_stamp = stamp(3, 102000000U);
  const ProcessResult result = processor.process(
    red_rgb_image(color_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, depth_stamp),
    camera_info(color_stamp), camera_info(depth_stamp), true);

  ASSERT_TRUE(result.ok()) << result.reason;
  EXPECT_EQ(result.frame.camera_info.header.stamp, depth_stamp);
  EXPECT_EQ(result.frame.camera_info.header.frame_id, kFrame);
}

TEST(ObservationProcessor, PreservesNanosecondSkewAtEpochSizedStamps)
{
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  ObservationProcessor processor(test_config());
  const auto depth_stamp = stamp(1786589021, 969401088U);
  const auto color_stamp = stamp(1786589021, 969515008U);
  const ProcessResult result = processor.process(
    red_rgb_image(color_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, depth_stamp),
    camera_info(color_stamp), camera_info(depth_stamp), true);

  ASSERT_TRUE(result.ok()) << result.reason;
  EXPECT_DOUBLE_EQ(result.frame.color_depth_skew_sec, 113920.0e-9);
}

TEST(ObservationProcessor, PreservesNegativeSkewAndEnforcesExactFiveMillisecondLimit)
{
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  ObservationProcessor processor(test_config());
  const auto depth_stamp = stamp(1786589021, 969515008U);
  const ProcessResult negative = processor.process(
    red_rgb_image(stamp(1786589021, 969401088U)),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, depth_stamp),
    camera_info(stamp(1786589021, 969401088U)), camera_info(depth_stamp), true);
  ASSERT_TRUE(negative.ok()) << negative.reason;
  EXPECT_DOUBLE_EQ(negative.frame.color_depth_skew_sec, -113920.0e-9);

  const auto boundary_depth_stamp = stamp(1786589021, 900000000U);
  const ProcessResult boundary = processor.process(
    red_rgb_image(stamp(1786589021, 905000000U)),
    image_message(
      depth, sensor_msgs::image_encodings::TYPE_16UC1, boundary_depth_stamp),
    camera_info(stamp(1786589021, 905000000U)),
    camera_info(boundary_depth_stamp), true);
  ASSERT_TRUE(boundary.ok()) << boundary.reason;
  EXPECT_DOUBLE_EQ(boundary.frame.color_depth_skew_sec, 0.005);

  const ProcessResult over_boundary = processor.process(
    red_rgb_image(stamp(1786589021, 905000001U)),
    image_message(
      depth, sensor_msgs::image_encodings::TYPE_16UC1, boundary_depth_stamp),
    camera_info(stamp(1786589021, 905000001U)),
    camera_info(boundary_depth_stamp), true);
  EXPECT_EQ(over_boundary.status, FrameStatus::kTimeSkewExceeded);
}

TEST(ObservationProcessor, RejectsRegisteredGridAndFrameMismatch)
{
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth - 1U), CV_16UC1,
    cv::Scalar(1000));
  ObservationProcessor processor(test_config());
  const auto common_stamp = stamp(5);
  const ProcessResult size_result = processor.process(
    red_rgb_image(common_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp),
    camera_info(common_stamp), camera_info(common_stamp), true);
  EXPECT_EQ(size_result.status, FrameStatus::kDimensionMismatch);

  depth = cv::Mat(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1, cv::Scalar(1000));
  auto info = camera_info(common_stamp);
  info.header.frame_id = "camera_depth_optical_frame";
  const ProcessResult frame_result = processor.process(
    red_rgb_image(common_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp),
    camera_info(common_stamp), info, true);
  EXPECT_EQ(frame_result.status, FrameStatus::kDimensionMismatch);
}

TEST(ObservationProcessor, RectifiesRawColorAndPublishesRegisteredDepthModel)
{
  const auto common_stamp = stamp(6);
  cv::Mat color(
    static_cast<int>(kHeight), static_cast<int>(kWidth), CV_8UC3,
    cv::Scalar(0, 0, 0));
  cv::rectangle(color, cv::Rect(27, 5, 11, 20), cv::Scalar(255, 0, 0), cv::FILLED);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  auto color_info = camera_info(common_stamp);
  color_info.k[0] = 25.0;
  color_info.k[4] = 25.0;
  color_info.distortion_model = "plumb_bob";
  color_info.d = {-0.35, 0.08, 0.001, -0.002, 0.0};
  auto depth_info = color_info;
  depth_info.d.assign(5U, 0.0);

  cv::Mat K(3, 3, CV_64FC1, color_info.k.data());
  cv::Mat D(1, static_cast<int>(color_info.d.size()), CV_64FC1, color_info.d.data());
  cv::Mat expected;
  cv::undistort(color, expected, K, D, K);

  auto config = test_config();
  config.morphology_kernel_size = 1;
  config.min_mask_pixels = 1;
  ObservationProcessor processor(config);
  const ProcessResult result = processor.process(
    image_message(color, sensor_msgs::image_encodings::RGB8, common_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp),
    color_info, depth_info, true);

  ASSERT_TRUE(result.ok()) << result.reason;
  const cv::Mat actual = cv_bridge::toCvCopy(
    result.frame.color, sensor_msgs::image_encodings::RGB8)->image;
  EXPECT_EQ(cv::norm(actual, expected, cv::NORM_INF), 0.0);
  EXPECT_EQ(result.frame.camera_info.k, depth_info.k);
  EXPECT_EQ(result.frame.camera_info.d, depth_info.d);

  cv::Mat expected_hsv;
  cv::cvtColor(expected, expected_hsv, cv::COLOR_RGB2HSV);
  cv::Mat expected_mask;
  cv::inRange(expected_hsv, cv::Scalar(0, 80, 50), cv::Scalar(10, 255, 255), expected_mask);
  const cv::Mat actual_mask = cv_bridge::toCvCopy(
    result.frame.target_mask, sensor_msgs::image_encodings::MONO8)->image;
  EXPECT_EQ(cv::norm(actual_mask, expected_mask, cv::NORM_INF), 0.0);

  cv::Mat raw_hsv;
  cv::cvtColor(color, raw_hsv, cv::COLOR_RGB2HSV);
  cv::Mat raw_mask;
  cv::inRange(raw_hsv, cv::Scalar(0, 80, 50), cv::Scalar(10, 255, 255), raw_mask);
  EXPECT_GT(cv::countNonZero(actual_mask != raw_mask), 0);
}

TEST(ObservationProcessor, AcceptsWhitelistedRationalAndPlumbBobModels)
{
  const auto common_stamp = stamp(6);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  const auto color = red_rgb_image(common_stamp);
  const auto depth_image = image_message(
    depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp);
  ObservationProcessor processor(test_config());

  ProcessResult result = processor.process(
    color, depth_image, camera_info(common_stamp), camera_info(common_stamp), true);
  ASSERT_TRUE(result.ok()) << result.reason;

  auto color_info = camera_info(common_stamp);
  auto depth_info = camera_info(common_stamp);
  color_info.distortion_model = "plumb_bob";
  color_info.d.assign(5U, 0.0);
  depth_info.distortion_model = "plumb_bob";
  depth_info.d.assign(5U, 0.0);
  result = processor.process(color, depth_image, color_info, depth_info, true);
  ASSERT_TRUE(result.ok()) << result.reason;
}

TEST(ObservationProcessor, RejectsEquidistantAndMismatchedDistortionLayouts)
{
  const auto common_stamp = stamp(6);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  const auto color = red_rgb_image(common_stamp);
  const auto depth_image = image_message(
    depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp);
  ObservationProcessor processor(test_config());

  auto color_info = camera_info(common_stamp);
  const auto depth_info = camera_info(common_stamp);
  color_info.distortion_model = "equidistant";
  color_info.d.assign(4U, 0.0);
  ProcessResult result = processor.process(
    color, depth_image, color_info, depth_info, true);
  EXPECT_EQ(result.status, FrameStatus::kCameraModelMismatch);
  EXPECT_NE(result.reason.find("rational_polynomial"), std::string::npos);

  color_info = camera_info(common_stamp);
  color_info.d.assign(5U, 0.0);
  result = processor.process(color, depth_image, color_info, depth_info, true);
  EXPECT_EQ(result.status, FrameStatus::kCameraModelMismatch);

  color_info = camera_info(common_stamp);
  color_info.distortion_model = "plumb_bob";
  color_info.d.assign(8U, 0.0);
  result = processor.process(color, depth_image, color_info, depth_info, true);
  EXPECT_EQ(result.status, FrameStatus::kCameraModelMismatch);

  color_info = camera_info(common_stamp);
  color_info.distortion_model = "unknown_model";
  result = processor.process(color, depth_image, color_info, depth_info, true);
  EXPECT_EQ(result.status, FrameStatus::kCameraModelMismatch);
}

TEST(ObservationProcessor, RejectsInvalidOrInconsistentCameraModels)
{
  const auto common_stamp = stamp(6);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  const auto color = red_rgb_image(common_stamp);
  const auto depth_image = image_message(
    depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp);
  ObservationProcessor processor(test_config());

  auto color_info = camera_info(common_stamp);
  auto depth_info = camera_info(common_stamp);
  depth_info.d[0] = -0.1;
  ProcessResult result = processor.process(
    color, depth_image, color_info, depth_info, true);

  EXPECT_EQ(result.status, FrameStatus::kCameraModelMismatch);
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.reason.find("zero distortion"), std::string::npos);

  depth_info = camera_info(common_stamp);
  color_info.d[0] = std::numeric_limits<double>::quiet_NaN();
  result = processor.process(color, depth_image, color_info, depth_info, true);
  EXPECT_EQ(result.status, FrameStatus::kCameraModelMismatch);

  color_info = camera_info(common_stamp);
  depth_info.k[0] += 1.0e-9;
  result = processor.process(color, depth_image, color_info, depth_info, true);
  EXPECT_EQ(result.status, FrameStatus::kCameraModelMismatch);
  EXPECT_NE(result.reason.find("K matrices"), std::string::npos);
}

TEST(ObservationProcessor, RejectsCameraInfoThatDoesNotMatchItsImageStamp)
{
  const auto common_stamp = stamp(6);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  ObservationProcessor processor(test_config());
  const ProcessResult result = processor.process(
    red_rgb_image(common_stamp),
    image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp),
    camera_info(stamp(6, 1U)), camera_info(common_stamp), true);

  EXPECT_EQ(result.status, FrameStatus::kTimeSkewExceeded);
  EXPECT_NE(result.reason.find("corresponding image stamp"), std::string::npos);
}

TEST(ObservationProcessor, RejectsInvalidStampFieldsForAllFourInputsWithoutThrowing)
{
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1,
    cv::Scalar(1000));
  ObservationProcessor processor(test_config());
  const auto valid = stamp(6, 123U);
  const auto negative_sec = stamp(-1, 123U);
  const auto invalid_nanosec = stamp(6, 1000000000U);

  const auto expect_rejected = [&](
    const builtin_interfaces::msg::Time & color_stamp,
    const builtin_interfaces::msg::Time & depth_stamp,
    const builtin_interfaces::msg::Time & color_info_stamp,
    const builtin_interfaces::msg::Time & depth_info_stamp)
    {
      const ProcessResult result = processor.process(
        red_rgb_image(color_stamp),
        image_message(depth, sensor_msgs::image_encodings::TYPE_16UC1, depth_stamp),
        camera_info(color_info_stamp), camera_info(depth_info_stamp), true);
      EXPECT_EQ(result.status, FrameStatus::kTimeSkewExceeded) << result.reason;
      EXPECT_FALSE(result.ok());
      EXPECT_NE(result.reason.find("sec >= 0"), std::string::npos);
    };

  expect_rejected(negative_sec, valid, valid, valid);
  expect_rejected(invalid_nanosec, valid, valid, valid);
  expect_rejected(valid, negative_sec, valid, valid);
  expect_rejected(valid, invalid_nanosec, valid, valid);
  expect_rejected(valid, valid, negative_sec, valid);
  expect_rejected(valid, valid, invalid_nanosec, valid);
  expect_rejected(valid, valid, valid, negative_sec);
  expect_rejected(valid, valid, valid, invalid_nanosec);
}

TEST(ObservationProcessor, EnforcesMinimumMaskAreaOnlyWhenRequested)
{
  const auto common_stamp = stamp(7);
  cv::Mat color(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_8UC3, cv::Scalar(0, 0, 0));
  cv::rectangle(color, cv::Rect(4, 4, 10, 10), cv::Scalar(255, 0, 0), cv::FILLED);
  cv::Mat depth(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_16UC1, cv::Scalar(1000));
  ObservationProcessor processor(test_config());

  const auto color_message = image_message(
    color, sensor_msgs::image_encodings::RGB8, common_stamp);
  const auto depth_message = image_message(
    depth, sensor_msgs::image_encodings::TYPE_16UC1, common_stamp);
  const ProcessResult required = processor.process(
    color_message, depth_message, camera_info(common_stamp),
    camera_info(common_stamp), true);
  EXPECT_EQ(required.status, FrameStatus::kMissingMask);

  const ProcessResult optional = processor.process(
    color_message, depth_message, camera_info(common_stamp),
    camera_info(common_stamp), false);
  ASSERT_TRUE(optional.ok()) << optional.reason;
  EXPECT_EQ(optional.frame.mask_pixels, 0);
}

}  // namespace
}  // namespace strawberry_observation
