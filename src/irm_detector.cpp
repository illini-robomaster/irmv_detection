#include <chrono>
#include <cmath>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.h>
#include <fmt/format.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <rcl/time.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "irmv_detection/armor.hpp"
#include "irmv_detection/irm_detector.hpp"
#include "irmv_detection/magic_enum.hpp"
#include "irmv_detection/yolo_engine.hpp"

constexpr bool ALLOW_DEBUG_AND_PROFILING = true;

namespace irmv_detection
{
IrmDetector::IrmDetector(const rclcpp::NodeOptions & options)
{
  node_ = std::make_shared<rclcpp::Node>("irmv_detector", options);

  // Declare parameters
  declare_parameters();

  // Initialize YOLO engine
  auto model_path =
    ament_index_cpp::get_package_share_directory("irmv_detection") + "/models/yolov7.onnx";
  yolo_engine_ =
    std::make_unique<YoloEngine>(node_, model_path, image_input_size_, enable_profiling_);

  RCLCPP_INFO(node_->get_logger(), "YOLOEngine initialized");

  // Warmup
  cv::Mat dummy_image = cv::Mat::zeros(image_input_size_, CV_8UC3);
  for (int i = 0; i < 100; i++) {
    yolo_engine_->detect(dummy_image);
  }

  RCLCPP_INFO(node_->get_logger(), "YOLOEngine warmed up");

  auto camera_info_manager =
    std::make_unique<camera_info_manager::CameraInfoManager>(node_.get(), "mv_camera");
  auto camera_info_url = node_->declare_parameter(
    "camera_info_url", std::string("package://irmv_detection/config/camera_info.yaml"));
  if (!camera_info_manager->validateURL(camera_info_url)) {
    RCLCPP_ERROR(node_->get_logger(), "Invalid camera info URL");
    exit(0);
  }
  camera_info_manager->loadCameraInfo(camera_info_url);
  auto camera_info_msg = camera_info_manager->getCameraInfo();
  pnp_solver_ = std::make_unique<PnPSolver>(camera_info_msg.k, camera_info_msg.d);

  RCLCPP_INFO(node_->get_logger(), "PnPSolver initialized");

  // Handle parameter changes
  param_event_handle_ = node_->add_on_set_parameters_callback(
    std::bind_front(&IrmDetector::param_event_callback, this));

  // Initialize publishers and subscribers
  armors_pub_ = node_->create_publisher<auto_aim_interfaces::msg::Armors>(
    "/detector/armors", rclcpp::SensorDataQoS());
  if constexpr (ALLOW_DEBUG_AND_PROFILING) {
    create_debug_publishers();
  }

  // Initialize camera
  camera_ = std::make_unique<VirtualCamera>(
    "/mnt/d/RMUL23_Vision_data/3v3/Italy_Torino_group/video_28.mp4",
    yolo_engine_->get_src_image_buffer(), std::bind_front(&IrmDetector::message_callback, this),
    100);
}

void IrmDetector::create_debug_publishers()
{
  if (enable_profiling_) {
    total_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
      "/detector/total_latency", rclcpp::SystemDefaultsQoS());
    comm_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
      "/detector/comm_latency", rclcpp::SystemDefaultsQoS());
    processing_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
      "/detector/processing_latency", rclcpp::SystemDefaultsQoS());
    inference_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
      "/yolo_engine/inference_latency", rclcpp::SystemDefaultsQoS());
    pnp_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>(
      "/pnp_solver/pnp_latency", rclcpp::SystemDefaultsQoS());
  }
  if (enable_debug_) {
    binary_img_pub_ = image_transport::create_publisher(
      node_.get(), "/image/binary_image", rmw_qos_profile_sensor_data);
    visualized_img_pub_ = image_transport::create_publisher(
      node_.get(), "/image/visualized_image", rmw_qos_profile_sensor_data);
  }
  if (enable_rviz_) {
    armor_marker_.ns = "armor";
    armor_marker_.action = visualization_msgs::msg::Marker::ADD;
    armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
    armor_marker_.scale.x = 0.05;
    armor_marker_.scale.z = 0.125;
    armor_marker_.color.a = 1.0;
    armor_marker_.color.g = 0.5;
    armor_marker_.color.b = 1.0;
    armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    text_marker_.ns = "classification";
    text_marker_.action = visualization_msgs::msg::Marker::ADD;
    text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker_.scale.z = 0.1;
    text_marker_.color.a = 1.0;
    text_marker_.color.r = 1.0;
    text_marker_.color.g = 1.0;
    text_marker_.color.b = 1.0;
    text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    marker_array_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>("/detector/marker", 10);
  }
}

void IrmDetector::declare_parameters()
{
  auto param_desc = rcl_interfaces::msg::ParameterDescriptor();

  param_desc.description = "Enable debug mode";
  param_desc.additional_constraints = "Must be true or false";
  enable_debug_ = node_->declare_parameter<bool>("debug", false, param_desc);

  param_desc.description = "Enable profiling";
  param_desc.additional_constraints = "Must be true or false";
  enable_profiling_ = node_->declare_parameter<bool>("profiling", false, param_desc);
  if (enable_debug_) enable_profiling_ = true;

  param_desc.description = "Enable Rviz visualization";
  param_desc.additional_constraints = "Must be true or false";
  enable_rviz_ = node_->declare_parameter<bool>("rviz", false, param_desc);
  if (enable_debug_) enable_rviz_ = true;

  param_desc.description = "Input size of the YOLO model";
  param_desc.additional_constraints = "Must be a list of two integers";
  auto image_input_size = node_->declare_parameter<std::vector<long>>(
    "image_input_size", std::vector<long>{1280, 1024}, param_desc);
  image_input_size_ = cv::Size(image_input_size[0], image_input_size[1]);

  param_desc.description = "Binary threshold for light extraction";
  param_desc.additional_constraints = "Must be an integer ranging from 0 to 255";
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 255;
  param_desc.integer_range[0].step = 1;
  binary_threshold_ = node_->declare_parameter<int>("binary_threshold", 150, param_desc);

  param_desc.description = "Enemy color, 0 for blue, 1 for red";
  param_desc.additional_constraints = "Must be 0 or 1";
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  param_desc.integer_range[0].step = 1;
  enemy_color_ = node_->declare_parameter<int>("enemy_color", 0, param_desc);

  light_min_ratio_ = node_->declare_parameter<double>("light.min_ratio", 0.1);
  light_max_ratio_ = node_->declare_parameter<double>("light.max_ratio", 0.4);
  light_max_angle_ = node_->declare_parameter<double>("light.max_angle", 40.0);

  armor_min_small_center_distance_ =
    node_->declare_parameter<double>("armor.min_small_center_distance", 0.8);
  armor_max_small_center_distance_ =
    node_->declare_parameter<double>("armor.max_small_center_distance", 3.2);
  armor_min_large_center_distance_ =
    node_->declare_parameter<double>("armor.min_large_center_distance", 3.2);
  armor_max_large_center_distance_ =
    node_->declare_parameter<double>("armor.max_large_center_distance", 5.5);
}

void IrmDetector::message_callback(
  cv::Mat & image, std::chrono::time_point<std::chrono::system_clock> time_stamp)
{
  rclcpp::Time extraction_end_time;
  rclcpp::Time pnp_end_time;

  std::vector<YoloEngine::bbox> bboxes = yolo_engine_->detect(image);

  std::vector<Armor> armors = extract_armors(yolo_engine_->get_rotated_image(), bboxes);

  if (pnp_solver_ == nullptr) return;  // This could happen if camera_info topic is not received yet

  if constexpr (ALLOW_DEBUG_AND_PROFILING) {
    if (enable_profiling_) extraction_end_time = node_->now();
  }

  rclcpp::Time time_stamp_ros(time_stamp.time_since_epoch().count(), RCL_ROS_TIME);
  std_msgs::msg::Header header;
  header.stamp = time_stamp_ros;
  header.frame_id = "camera_optical_frame";
  auto_aim_interfaces::msg::Armors armors_msg;
  armors_msg.header = header;
  if (enable_rviz_) {
    armor_marker_.header = header;
    text_marker_.header = header;
    armor_marker_.id = 0;
    text_marker_.id = 0;
  }

  for (const auto & armor : armors) {
    cv::Mat rvec;
    cv::Mat tvec;
    if (!pnp_solver_->solvePnP(armor, rvec, tvec)) {
      continue;
    }

    auto_aim_interfaces::msg::Armor armor_msg;

    // Fill in pose
    armor_msg.pose.position.x = tvec.at<double>(0);
    armor_msg.pose.position.y = tvec.at<double>(1);
    armor_msg.pose.position.z = tvec.at<double>(2);

    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);
    tf2::Matrix3x3 tf2_rot_mat(
      rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1), rot_mat.at<double>(0, 2),
      rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),
      rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2));
    tf2::Quaternion tf2_quat;
    tf2_rot_mat.getRotation(tf2_quat);
    armor_msg.pose.orientation = tf2::toMsg(tf2_quat);

    // Fill in message
    armor_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);
    armors_msg.armors.emplace_back(armor_msg);

    if (enable_rviz_) {
      armor_marker_.id++;
      armor_marker_.pose = armor_msg.pose;
      armor_marker_.scale.y = armor.size == ArmorSize::SMALL ? 0.135 : 0.23;
      text_marker_.id++;
      text_marker_.pose.position = armor_msg.pose.position;
      text_marker_.pose.position.y -= 0.1;
      text_marker_.text = magic_enum::enum_name(armor.armor_class);
      marker_array_.markers.push_back(armor_marker_);
      marker_array_.markers.push_back(text_marker_);
    }
  }

  armors_pub_->publish(armors_msg);

  if constexpr (ALLOW_DEBUG_AND_PROFILING) {
    if (enable_profiling_) {
      pnp_end_time = node_->now();
      // Publish profiling data
      const auto inference_time = yolo_engine_->get_profiling_time();
      std_msgs::msg::Float64 total_latency_msg, inference_latency_msg, pnp_latency_msg;
      total_latency_msg.data = (pnp_end_time - time_stamp_ros).seconds() * 1000;
      total_latency_pub_->publish(total_latency_msg);
      inference_latency_msg.data = inference_time;
      inference_latency_pub_->publish(inference_latency_msg);
      pnp_latency_msg.data = (pnp_end_time - extraction_end_time).seconds() * 1000;
      pnp_latency_pub_->publish(pnp_latency_msg);
      // Publish debug images
      if (enable_debug_) {
        cv::Mat visualized_image = yolo_engine_->get_rotated_image().clone();
        visualize_armors(visualized_image, armors);
        yolo_engine_->visualize_bboxes(visualized_image, bboxes);
        cv::putText(
          visualized_image, fmt::format("Total latency: {} ms", total_latency_msg.data),
          cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(
          visualized_image, fmt::format("Inference latency: {} ms", inference_latency_msg.data),
          cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        visualized_img_pub_.publish(
          cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", visualized_image).toImageMsg());

        cv::Mat binary_image = yolo_engine_->get_rotated_image().clone();
        cv::cvtColor(binary_image, binary_image, cv::COLOR_BGR2GRAY);
        cv::threshold(binary_image, binary_image, binary_threshold_, 255, cv::THRESH_BINARY);
        cv::cvtColor(binary_image, binary_image, cv::COLOR_GRAY2BGR);
        yolo_engine_->visualize_bboxes(binary_image, bboxes);
        binary_img_pub_.publish(
          cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", binary_image).toImageMsg());
      }
    }
    if (enable_rviz_) {
      armor_marker_.action = armors_msg.armors.empty() ? visualization_msgs::msg::Marker::DELETE
                                                       : visualization_msgs::msg::Marker::ADD;
      marker_array_.markers.push_back(armor_marker_);
      marker_array_pub_->publish(marker_array_);
      marker_array_.markers.clear();
    }
  }
}

std::vector<Armor> IrmDetector::extract_armors(
  const cv::Mat & image, const std::vector<YoloEngine::bbox> & bboxes) const
{
  std::vector<Armor> armors;

  for (const auto & bbox : bboxes) {
    // Get the ROI, note ROI generated by YOLO is not always inside the image
    float min_x = std::max(bbox.xyxy[0], 0.0f);
    float min_y = std::max(bbox.xyxy[1], 0.0f);
    float max_x = std::min(bbox.xyxy[2], static_cast<float>(image.cols));
    float max_y = std::min(bbox.xyxy[3], static_cast<float>(image.rows));
    if (min_x >= max_x || min_y >= max_y) continue;
    cv::Mat roi = image(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y));

    // Get the binary version of the ROI
    cv::Mat roi_gray;
    cv::Mat roi_binary;
    cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(roi_gray, roi_binary, binary_threshold_, 255, cv::THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(roi_binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find lights
    std::vector<Light> light_list;
    for (const auto & contour : contours) {
      if (contour.size() < 5) continue;

      cv::RotatedRect r_rect = cv::minAreaRect(contour);
      Light light(r_rect);

      if (!light.is_light(light_min_ratio_, light_max_ratio_, light_max_angle_)) continue;

      light.offset_bbox(min_x, min_y);
      light_list.emplace_back(std::move(light));
    }

    if (light_list.size() < 2) continue;

    Armor armor;
    armor = Armor(light_list[0], light_list[1]);
    armor.armor_class = bbox.class_id;
    armor.confidence = bbox.score;
    double avg_light_length = (light_list[0].length + light_list[1].length) / 2;
    double center_distance =
      cv::norm(armor.left_light.center - armor.right_light.center) / avg_light_length;
    armor.size =
      center_distance > armor_min_large_center_distance_ ? ArmorSize::LARGE : ArmorSize::SMALL;
    // Check for center distance
    if (
      armor.size == ArmorSize::SMALL && (armor_min_small_center_distance_ > center_distance ||
                                         armor_max_small_center_distance_ < center_distance))
      continue;
    if (
      armor.size == ArmorSize::LARGE && (armor_min_large_center_distance_ > center_distance ||
                                         armor_max_large_center_distance_ < center_distance))
      continue;
    armors.emplace_back(std::move(armor));
  }

  return armors;
}

void IrmDetector::visualize_armors(cv::Mat & image, const std::vector<Armor> & armors) const
{
  for (const auto & armor : armors) {
    cv::circle(image, armor.left_light.top, 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(image, armor.left_light.bottom, 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(image, armor.right_light.top, 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(image, armor.right_light.bottom, 5, cv::Scalar(0, 255, 0), 2);

    cv::line(image, armor.left_light.top, armor.left_light.bottom, cv::Scalar(0, 255, 0), 2);
    cv::line(image, armor.right_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
    cv::line(image, armor.left_light.top, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    cv::line(image, armor.left_light.bottom, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
  }
}

rcl_interfaces::msg::SetParametersResult IrmDetector::param_event_callback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  for (const auto & parameter : parameters) {
    std::string name = parameter.get_name();
    if (name == "debug") {
      enable_debug_ = parameter.as_bool();
    } else if (name == "binary_threshold") {
      binary_threshold_ = static_cast<int>(parameter.as_int());
    } else if (name == "enemy_color") {
      enemy_color_ = static_cast<int>(parameter.as_int());
    } else if (name == "light.min_ratio") {
      light_min_ratio_ = parameter.as_double();
    } else if (name == "light.max_ratio") {
      light_max_ratio_ = parameter.as_double();
    } else if (name == "light.max_angle") {
      light_max_angle_ = parameter.as_double();
    } else if (name == "armor.min_small_center_distance") {
      armor_min_small_center_distance_ = parameter.as_double();
    } else if (name == "armor.max_small_center_distance") {
      armor_max_small_center_distance_ = parameter.as_double();
    } else if (name == "armor.min_large_center_distance") {
      armor_min_large_center_distance_ = parameter.as_double();
    } else if (name == "armor.max_large_center_distance") {
      armor_max_large_center_distance_ = parameter.as_double();
    }
  }
  return result;
}
}  // namespace irmv_detection

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(irmv_detection::IrmDetector)
