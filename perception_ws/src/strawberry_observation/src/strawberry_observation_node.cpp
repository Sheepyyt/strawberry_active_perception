#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "strawberry_observation/observation_processor.hpp"
#include "strawberry_perception_interfaces/msg/observation.hpp"
#include "strawberry_perception_interfaces/srv/capture_observation.hpp"

namespace strawberry_observation
{

class StrawberryObservationNode : public rclcpp::Node
{
public:
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using Observation = strawberry_perception_interfaces::msg::Observation;
  using CaptureObservation = strawberry_perception_interfaces::srv::CaptureObservation;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    Image, Image, CameraInfo, CameraInfo>;

  StrawberryObservationNode()
  : Node("strawberry_observation")
  {
    const std::string color_topic = declare_parameter<std::string>(
      "input_color_topic", "/camera/color/image_raw");
    const std::string depth_topic = declare_parameter<std::string>(
      "input_depth_topic", "/camera/depth/image_raw");
    const std::string color_camera_info_topic = declare_parameter<std::string>(
      "input_color_camera_info_topic", "/camera/color/camera_info");
    const std::string depth_camera_info_topic = declare_parameter<std::string>(
      "input_depth_camera_info_topic", "/camera/depth/camera_info");
    const std::string observation_topic = declare_parameter<std::string>(
      "observation_topic", "/strawberry/perception/observation");
    const std::string capture_service = declare_parameter<std::string>(
      "capture_service", "/strawberry/perception/capture_observation");

    source_name_ = declare_parameter<std::string>(
      "source_name", "gemini2xl_AYML241003A");
    const int64_t source_type = declare_parameter<int64_t>("source_type", Observation::SOURCE_REAL);
    pose_world_frame_ = declare_parameter<std::string>("pose_world_frame", "camera_session");
    fixed_pose_enabled_ = declare_parameter<bool>("fixed_pose_enabled", true);
    default_capture_timeout_sec_ = declare_parameter<double>("default_capture_timeout_sec", 2.0);
    sync_max_interval_sec_ = declare_parameter<double>("sync_max_interval_sec", 0.02);
    const int64_t sync_queue_size = declare_parameter<int64_t>("sync_queue_size", 30);
    const int64_t sample_queue_capacity = declare_parameter<int64_t>("sample_queue_capacity", 100);
    max_discard_frames_ = declare_parameter<int64_t>("max_discard_frames", 100);

    if (color_topic.empty() || depth_topic.empty() || color_camera_info_topic.empty() ||
      depth_camera_info_topic.empty() ||
      observation_topic.empty() || capture_service.empty())
    {
      throw std::invalid_argument("input, output, and service topic names must be non-empty");
    }
    if (source_name_.empty()) {
      throw std::invalid_argument("source_name must be non-empty");
    }
    if (source_type < Observation::SOURCE_REAL || source_type > Observation::SOURCE_REPLAY) {
      throw std::invalid_argument(
              "source_type must be REAL(1), OFFLINE(2), SYNTHETIC(3), or REPLAY(4)");
    }
    source_type_ = static_cast<uint8_t>(source_type);
    if (fixed_pose_enabled_ && pose_world_frame_.empty()) {
      throw std::invalid_argument("pose_world_frame must be non-empty when fixed pose is enabled");
    }
    if (!std::isfinite(default_capture_timeout_sec_) || default_capture_timeout_sec_ <= 0.0) {
      throw std::invalid_argument("default_capture_timeout_sec must be finite and positive");
    }
    if (!std::isfinite(sync_max_interval_sec_) || sync_max_interval_sec_ <= 0.0) {
      throw std::invalid_argument("sync_max_interval_sec must be finite and positive");
    }
    if (sync_queue_size < 2 || sample_queue_capacity < 1 || max_discard_frames_ < 0) {
      throw std::invalid_argument(
              "sync_queue_size >= 2, sample_queue_capacity >= 1, and max_discard_frames >= 0 required");
    }
    sample_queue_capacity_ = static_cast<size_t>(sample_queue_capacity);

    ProcessingConfig processing_config;
    processing_config.depth_16uc1_scale = declare_parameter<double>(
      "depth_16uc1_scale", 0.001);
    processing_config.depth_32fc1_scale = declare_parameter<double>(
      "depth_32fc1_scale", 1.0);
    processing_config.depth_min_m = declare_parameter<double>("depth_min_m", 0.2);
    processing_config.depth_max_m = declare_parameter<double>("depth_max_m", 2.5);
    processing_config.min_valid_depth_fraction = declare_parameter<double>(
      "min_valid_depth_fraction", 0.05);
    processing_config.max_time_skew_sec = declare_parameter<double>(
      "max_time_skew_sec", 0.005);
    processing_config.red_hsv_low_1 = hsv_parameter("red_hsv_low_1", {0, 80, 50});
    processing_config.red_hsv_high_1 = hsv_parameter("red_hsv_high_1", {10, 255, 255});
    processing_config.red_hsv_low_2 = hsv_parameter("red_hsv_low_2", {170, 80, 50});
    processing_config.red_hsv_high_2 = hsv_parameter("red_hsv_high_2", {179, 255, 255});
    processing_config.morphology_kernel_size = static_cast<int>(
      declare_parameter<int64_t>("morphology_kernel_size", 5));
    processing_config.mask_dilation_kernel_size = static_cast<int>(
      declare_parameter<int64_t>("mask_dilation_kernel_size", 1));
    processing_config.min_mask_pixels = static_cast<int>(
      declare_parameter<int64_t>("min_mask_pixels", 200));
    processor_ = std::make_unique<ObservationProcessor>(processing_config);

    observation_publisher_ = create_publisher<Observation>(
      observation_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

    subscription_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    service_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group = subscription_group_;
    color_subscriber_.subscribe(
      this, color_topic, rmw_qos_profile_sensor_data, subscription_options);
    depth_subscriber_.subscribe(
      this, depth_topic, rmw_qos_profile_sensor_data, subscription_options);
    color_camera_info_subscriber_.subscribe(
      this, color_camera_info_topic, rmw_qos_profile_sensor_data, subscription_options);
    depth_camera_info_subscriber_.subscribe(
      this, depth_camera_info_topic, rmw_qos_profile_sensor_data, subscription_options);

    SyncPolicy policy(static_cast<uint32_t>(sync_queue_size));
    policy.setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_max_interval_sec_));
    synchronizer_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      static_cast<const SyncPolicy &>(policy),
      color_subscriber_, depth_subscriber_, color_camera_info_subscriber_,
      depth_camera_info_subscriber_);
    synchronizer_->registerCallback(
      std::bind(
        &StrawberryObservationNode::synchronized_callback, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4));

    capture_service_ = create_service<CaptureObservation>(
      capture_service,
      std::bind(
        &StrawberryObservationNode::capture_callback, this,
        std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(),
      service_group_);

    RCLCPP_INFO(
      get_logger(),
      "Ready: sync [%s, %s, %s, %s], capture %s, publish %s (Reliable/TransientLocal)",
      color_topic.c_str(), depth_topic.c_str(), color_camera_info_topic.c_str(),
      depth_camera_info_topic.c_str(),
      capture_service.c_str(), observation_topic.c_str());
  }

  ~StrawberryObservationNode() override
  {
    {
      std::lock_guard<std::mutex> lock(sample_mutex_);
      shutting_down_ = true;
    }
    sample_condition_.notify_all();
  }

private:
  struct SynchronizedSample
  {
    uint64_t sequence{0U};
    Image::ConstSharedPtr color;
    Image::ConstSharedPtr depth;
    CameraInfo::ConstSharedPtr color_camera_info;
    CameraInfo::ConstSharedPtr depth_camera_info;
  };

  std::array<int, 3> hsv_parameter(
    const std::string & name, const std::vector<int64_t> & default_value)
  {
    const auto values = declare_parameter<std::vector<int64_t>>(name, default_value);
    if (values.size() != 3U) {
      throw std::invalid_argument(name + " must contain exactly [H, S, V]");
    }
    return {
      static_cast<int>(values[0]), static_cast<int>(values[1]), static_cast<int>(values[2])};
  }

  void synchronized_callback(
    const Image::ConstSharedPtr & color,
    const Image::ConstSharedPtr & depth,
    const CameraInfo::ConstSharedPtr & color_camera_info,
    const CameraInfo::ConstSharedPtr & depth_camera_info)
  {
    {
      std::lock_guard<std::mutex> lock(sample_mutex_);
      SynchronizedSample sample;
      sample.sequence = ++latest_sequence_;
      sample.color = color;
      sample.depth = depth;
      sample.color_camera_info = color_camera_info;
      sample.depth_camera_info = depth_camera_info;
      samples_.push_back(std::move(sample));
      while (samples_.size() > sample_queue_capacity_) {
        samples_.pop_front();
      }
    }
    sample_condition_.notify_all();
  }

  static bool is_zero_stamp(const builtin_interfaces::msg::Time & stamp)
  {
    return stamp.sec == 0 && stamp.nanosec == 0U;
  }

  static bool scene_id_valid(const std::string & scene_id)
  {
    return !scene_id.empty() && std::any_of(
      scene_id.begin(), scene_id.end(), [](unsigned char value) {return !std::isspace(value);});
  }

  static std::string deterministic_observation_id(const builtin_interfaces::msg::Time & stamp)
  {
    std::ostringstream id;
    id << "obs_" << stamp.sec << "_" << std::setw(9) << std::setfill('0') << stamp.nanosec;
    return id.str();
  }

  static double duration_seconds(const builtin_interfaces::msg::Duration & duration)
  {
    return static_cast<double>(duration.sec) + static_cast<double>(duration.nanosec) * 1.0e-9;
  }

  void set_error(
    CaptureObservation::Response & response, uint16_t code, const std::string & reason) const
  {
    response.success = false;
    response.code = code;
    response.reason = reason;
    response.observation_id.clear();
    response.stamp = builtin_interfaces::msg::Time();
  }

  void capture_callback(
    const std::shared_ptr<CaptureObservation::Request> request,
    std::shared_ptr<CaptureObservation::Response> response)
  {
    if (!scene_id_valid(request->scene_id)) {
      set_error(*response, CaptureObservation::Response::INVALID_REQUEST,
          "scene_id must be non-empty");
      return;
    }
    if (request->not_before.sec < 0 || request->not_before.nanosec >= 1000000000U) {
      set_error(
        *response, CaptureObservation::Response::INVALID_REQUEST,
        "not_before must be zero or a valid non-negative ROS time with nanosec < 1e9");
      return;
    }
    if (request->timeout.sec < 0 || request->timeout.nanosec >= 1000000000U) {
      set_error(
        *response, CaptureObservation::Response::INVALID_REQUEST,
        "timeout must be zero or a valid positive duration with nanosec < 1e9");
      return;
    }
    if (request->discard_frames > static_cast<uint32_t>(max_discard_frames_)) {
      set_error(
        *response, CaptureObservation::Response::INVALID_REQUEST,
        "discard_frames exceeds configured max_discard_frames=" +
        std::to_string(max_discard_frames_));
      return;
    }
    if (request->require_pose && !fixed_pose_enabled_) {
      set_error(
        *response, CaptureObservation::Response::POSE_UNAVAILABLE,
        "fixed identity pose is disabled");
      return;
    }

    double timeout_sec = duration_seconds(request->timeout);
    if (timeout_sec == 0.0) {
      timeout_sec = default_capture_timeout_sec_;
    }
    if (!std::isfinite(timeout_sec) || timeout_sec <= 0.0) {
      set_error(
        *response, CaptureObservation::Response::INVALID_REQUEST,
        "timeout must resolve to a finite positive duration");
      return;
    }

    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(timeout_sec));
    uint64_t next_sequence = 0U;
    {
      std::lock_guard<std::mutex> lock(sample_mutex_);
      next_sequence = latest_sequence_ + 1U;
    }

    uint32_t remaining_discard = request->discard_frames;
    uint64_t seen_samples = 0U;
    FrameStatus last_rejection = FrameStatus::kSuccess;
    std::string last_rejection_reason;

    while (rclcpp::ok()) {
      SynchronizedSample sample;
      {
        std::unique_lock<std::mutex> lock(sample_mutex_);
        const bool woke = sample_condition_.wait_until(
          lock, deadline,
          [this, next_sequence]() {
            return shutting_down_ ||
                   (!samples_.empty() && samples_.back().sequence >= next_sequence);
          });
        if (shutting_down_) {
          set_error(*response, CaptureObservation::Response::CANCELED, "node is shutting down");
          return;
        }
        if (!woke) {
          break;
        }
        const auto candidate = std::find_if(
          samples_.begin(), samples_.end(),
          [next_sequence](const SynchronizedSample & value) {
            return value.sequence >= next_sequence;
          });
        if (candidate == samples_.end()) {
          continue;
        }
        sample = *candidate;
        next_sequence = sample.sequence + 1U;
      }
      ++seen_samples;

      if (!is_zero_stamp(request->not_before)) {
        const rclcpp::Time depth_stamp(sample.depth->header.stamp, RCL_ROS_TIME);
        const rclcpp::Time lower_bound(request->not_before, RCL_ROS_TIME);
        if (depth_stamp < lower_bound) {
          continue;
        }
      }

      const ProcessResult processed = processor_->process(
        *sample.color, *sample.depth, *sample.color_camera_info,
        *sample.depth_camera_info, request->require_mask);
      if (!processed.ok()) {
        last_rejection = processed.status;
        last_rejection_reason = processed.reason;
        RCLCPP_DEBUG(
          get_logger(), "Rejected synchronized frame: %s", processed.reason.c_str());
        continue;
      }
      if (remaining_discard > 0U) {
        --remaining_discard;
        continue;
      }

      Observation observation;
      observation.header = processed.frame.depth.header;
      observation.scene_id = request->scene_id;
      observation.observation_id = deterministic_observation_id(observation.header.stamp);
      observation.source_type = source_type_;
      observation.source_name = source_name_;
      observation.color = processed.frame.color;
      observation.depth = processed.frame.depth;
      observation.target_mask = processed.frame.target_mask;
      observation.camera_info = processed.frame.camera_info;
      observation.camera_pose.header.stamp = observation.header.stamp;
      observation.camera_pose.header.frame_id = pose_world_frame_;
      observation.camera_pose.pose.orientation.w = 1.0;
      observation.pose_valid = fixed_pose_enabled_;
      observation.valid_depth_fraction = processed.frame.valid_depth_fraction;
      observation.color_depth_skew_sec = processed.frame.color_depth_skew_sec;

      observation_publisher_->publish(observation);
      response->success = true;
      response->code = CaptureObservation::Response::SUCCESS;
      response->reason = "captured and published canonical observation";
      response->observation_id = observation.observation_id;
      response->stamp = observation.header.stamp;
      RCLCPP_INFO(
        get_logger(), "Published %s/%s: valid_depth=%.3f mask_pixels=%d skew=%.6f s",
        observation.scene_id.c_str(), observation.observation_id.c_str(),
        observation.valid_depth_fraction, processed.frame.mask_pixels,
        observation.color_depth_skew_sec);
      return;
    }

    if (!rclcpp::ok()) {
      set_error(*response, CaptureObservation::Response::CANCELED, "ROS context stopped");
    } else if (seen_samples == 0U) {
      set_error(
        *response, CaptureObservation::Response::NO_NEW_FRAME,
        "no synchronized frame set arrived before timeout");
    } else if (last_rejection != FrameStatus::kSuccess) {
      std::ostringstream reason;
      reason << "no acceptable frame before timeout; last rejected frame: "
             << last_rejection_reason;
      set_error(*response, static_cast<uint16_t>(last_rejection), reason.str());
    } else {
      std::ostringstream reason;
      reason << "capture timed out after " << timeout_sec << " s";
      if (remaining_discard > 0U) {
        reason << "; still needed " << remaining_discard << " discard frame(s)";
      } else if (!is_zero_stamp(request->not_before)) {
        reason << "; no frame reached not_before";
      }
      set_error(*response, CaptureObservation::Response::TIMEOUT, reason.str());
    }
  }

  std::string source_name_;
  uint8_t source_type_{Observation::SOURCE_REAL};
  std::string pose_world_frame_;
  bool fixed_pose_enabled_{true};
  double default_capture_timeout_sec_{2.0};
  double sync_max_interval_sec_{0.02};
  int64_t max_discard_frames_{100};
  size_t sample_queue_capacity_{100U};

  std::unique_ptr<ObservationProcessor> processor_;
  rclcpp::Publisher<Observation>::SharedPtr observation_publisher_;
  rclcpp::Service<CaptureObservation>::SharedPtr capture_service_;
  rclcpp::CallbackGroup::SharedPtr subscription_group_;
  rclcpp::CallbackGroup::SharedPtr service_group_;

  message_filters::Subscriber<Image> color_subscriber_;
  message_filters::Subscriber<Image> depth_subscriber_;
  message_filters::Subscriber<CameraInfo> color_camera_info_subscriber_;
  message_filters::Subscriber<CameraInfo> depth_camera_info_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;

  std::mutex sample_mutex_;
  std::condition_variable sample_condition_;
  std::deque<SynchronizedSample> samples_;
  uint64_t latest_sequence_{0U};
  bool shutting_down_{false};
};

}  // namespace strawberry_observation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<strawberry_observation::StrawberryObservationNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3U);
  executor.add_node(node);
  executor.spin();
  executor.remove_node(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}
