#include "irm_detection/irm_detector.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/convert.h"
#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include "irm_detection/armor.hpp"
#include "irm_detection/magic_enum.hpp"
#include "irm_detection/yolo_engine.hpp"

#define ALLOW_DEBUG_AND_PROFILING 1

namespace irm_detection
{
  IrmDetector::IrmDetector(const rclcpp::NodeOptions & options)
  {
    node_ = std::make_shared<rclcpp::Node>("irm_detector", options);

    // Declare parameters
    declare_parameters();

    // Initialize YOLO engine
    yolo_engine_ = std::make_unique<YoloEngine>("/home/niceme/workspaces/irm_ros-dev/src/iRM_Vision_2023/irm_detection/models/yolov7.onnx", enable_profiling_);

    // Initialize PnP solver
    camera_info_sub_ = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/image/camera_info", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        pnp_solver_ = std::make_unique<PnPSolver>(msg->k, msg->d);
        camera_info_sub_.reset(); // Unsubscribe
      });

    // Handle parameter changes
    param_event_handle_ = node_->add_on_set_parameters_callback(std::bind(&IrmDetector::param_event_callback, this, std::placeholders::_1));

    // Initialize publishers and subscribers
    armors_pub_ = node_->create_publisher<auto_aim_interfaces::msg::Armors>("/detector/armors", rclcpp::SensorDataQoS());
#if ALLOW_DEBUG_AND_PROFILING
    if (enable_profiling_) {
      total_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/detector/total_latency", rclcpp::SystemDefaultsQoS());
      comm_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/detector/comm_latency", rclcpp::SystemDefaultsQoS());
      preprocess_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/yolo_engine/preprocess_latency", rclcpp::SystemDefaultsQoS());
      inference_latency_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/yolo_engine/inference_latency", rclcpp::SystemDefaultsQoS());
    }
    if (enable_debug_) {
      binary_img_pub_ = image_transport::create_publisher(node_.get(), "/image/binary_image", rmw_qos_profile_sensor_data);
      visualized_img_pub_ = image_transport::create_publisher(node_.get(), "/image/visualized_image", rmw_qos_profile_sensor_data);
    }
#endif
    img_sub_ = image_transport::create_subscription(node_.get(), "/image/image_raw", std::bind(&IrmDetector::message_callback, this, std::placeholders::_1), "raw", rmw_qos_profile_sensor_data);
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

    light_min_ratio_ = node_->declare_parameter<float>("light.min_ratio", 0.1);
    light_max_ratio_ = node_->declare_parameter<float>("light.max_ratio", 0.4);
    light_max_angle_ = node_->declare_parameter<float>("light.max_angle", 40.0);

    armor_min_small_center_distance_ = node_->declare_parameter<float>("armor.min_small_center_distance", 0.8);
    armor_max_small_center_distance_ = node_->declare_parameter<float>("armor.max_small_center_distance", 3.2);
    armor_min_large_center_distance_ = node_->declare_parameter<float>("armor.min_large_center_distance", 3.2);
    armor_max_large_center_distance_ = node_->declare_parameter<float>("armor.max_large_center_distance", 5.5);
  }

  void IrmDetector::message_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    rclcpp::Time callback_start_time, rotation_end_time, inference_end_time, extraction_end_time;

#if ALLOW_DEBUG_AND_PROFILING
    if (enable_profiling_)
      callback_start_time = node_->now();
#endif

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    cv::rotate(cv_ptr->image, cv_ptr->image, cv::ROTATE_180);

#if ALLOW_DEBUG_AND_PROFILING
    if (enable_profiling_)
      rotation_end_time = node_->now();
#endif

    std::vector<YoloEngine::bbox> bboxes = yolo_engine_->detect(cv_ptr->image);

    std::vector<Armor> armors = extract_armors(cv_ptr->image, bboxes);

#if ALLOW_DEBUG_AND_PROFILING
    if (enable_debug_) {
      cv::Mat visualized_image = cv_ptr->image.clone();
      visualize_armors(visualized_image, armors);
      yolo_engine_->visualize_bboxes(visualized_image, bboxes);
      visualized_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", visualized_image).toImageMsg());

      cv::Mat binary_image = cv_ptr->image.clone();
      cv::cvtColor(binary_image, binary_image, cv::COLOR_BGR2GRAY);
      cv::threshold(binary_image, binary_image, binary_threshold_, 255, cv::THRESH_BINARY);
      cv::cvtColor(binary_image, binary_image, cv::COLOR_GRAY2BGR);
      yolo_engine_->visualize_bboxes(binary_image, bboxes);
      binary_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", binary_image).toImageMsg());
    }
#endif

    if (pnp_solver_ == nullptr) {
      return;
    }

    auto_aim_interfaces::msg::Armors armors_msg;
    for (const auto &armor : armors) {
      cv::Mat rvec, tvec;
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
    }

    armors_pub_->publish(armors_msg);

#if ALLOW_DEBUG_AND_PROFILING
    if (enable_profiling_) {
      const auto [preprocess_time, inference_time] = yolo_engine_->get_profiling_time();
      const double rotation_time = (rotation_end_time - callback_start_time).seconds() * 1000;
      std_msgs::msg::Float64 total_latency_msg, comm_latency_msg, preprocess_latency_msg, inference_latency_msg;
      total_latency_msg.data = (node_->now() - msg->header.stamp).seconds() * 1000;
      total_latency_pub_->publish(total_latency_msg);
      comm_latency_msg.data = (callback_start_time - msg->header.stamp).seconds() * 1000;
      comm_latency_pub_->publish(comm_latency_msg);
      preprocess_latency_msg.data = preprocess_time + rotation_time;
      preprocess_latency_pub_->publish(preprocess_latency_msg);
      inference_latency_msg.data = inference_time;
      inference_latency_pub_->publish(inference_latency_msg);
    }
#endif
  }

  std::vector<Armor> IrmDetector::extract_armors(const cv::Mat &image, const std::vector<YoloEngine::bbox> &bboxes)
  {
    std::vector<Armor> armors;

    for (const auto &bbox : bboxes) {
      // Get the ROI, note ROI generated by YOLO is not always inside the image
      float min_x = std::max(bbox.xyxy[0], 0.0f);
      float min_y = std::max(bbox.xyxy[1], 0.0f);
      float max_x = std::min(bbox.xyxy[2], static_cast<float>(image.cols));
      float max_y = std::min(bbox.xyxy[3], static_cast<float>(image.rows));
      if (min_x >= max_x || min_y >= max_y) continue;
      cv::Mat roi = image(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y));
      
      // Get the binary version of the ROI
      cv::Mat roi_gray, roi_binary;
      cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
      cv::threshold(roi_gray, roi_binary, binary_threshold_, 255, cv::THRESH_BINARY);

      // Find contours
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(roi_binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      // Find lights
      std::vector<Light> light_list;
      for (const auto &contour : contours) {
        if (contour.size() < 5) continue;

        cv::RotatedRect r_rect = cv::minAreaRect(contour);
        Light light(r_rect);

        if (!light.is_light(light_min_ratio_, light_max_ratio_, light_max_angle_)) continue;

        light.offset_bbox(min_x, min_y);
        light_list.emplace_back(light);
      }

      if (light_list.size() < 2) continue;

      Armor armor;
      armor = Armor(light_list[0], light_list[1]);
      armor.armor_class = bbox.class_id;
      armor.confidence = bbox.score;
      float avg_light_length = (light_list[0].length + light_list[1].length) / 2;
      float center_distance = cv::norm(armor.left_light.center - armor.right_light.center) / avg_light_length;
      armor.size = center_distance > armor_min_large_center_distance_ ? ArmorSize::LARGE : ArmorSize::SMALL;
      bool center_distance_valid = (armor.size == ArmorSize::SMALL && armor_min_small_center_distance_ < center_distance && center_distance < armor_max_small_center_distance_) || 
                                   (armor.size == ArmorSize::LARGE && armor_min_large_center_distance_ < center_distance && center_distance < armor_max_large_center_distance_);
      if (!center_distance_valid) continue;
      armors.emplace_back(armor);
    }

    return armors;
  }

  void IrmDetector::visualize_armors(cv::Mat &image, const std::vector<Armor> &armors)
  {
    for (const auto &armor : armors) {
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

  rcl_interfaces::msg::SetParametersResult IrmDetector::param_event_callback(const std::vector<rclcpp::Parameter> &parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    for (const auto & parameter : parameters) {
      std::string name = parameter.get_name();
      if (name == "debug") {
        enable_debug_ = parameter.as_bool();
      } else if (name == "binary_threshold") {
        binary_threshold_ = parameter.as_int();
      } else if (name == "enemy_color") {
        enemy_color_ = parameter.as_int();
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
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(irm_detection::IrmDetector)
