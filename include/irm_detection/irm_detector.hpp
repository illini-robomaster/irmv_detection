#pragma once

#include "irm_detection/yolo_engine.hpp"
#include "irm_detection/armor.hpp"
#include "irm_detection/pnp_solver.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float64.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include <memory>


namespace irm_detection
{
  class IrmDetector
  {
    public:
      explicit IrmDetector(const rclcpp::NodeOptions & options);
      rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_node_base_interface() const
      {
        return node_->get_node_base_interface();
      }

    private:
      void declare_parameters();
      void message_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
      std::vector<Armor> extract_armors(const cv::Mat &image, const std::vector<YoloEngine::bbox> &bboxes);
      void visualize_armors(cv::Mat &image, const std::vector<Armor> &armors);
      rcl_interfaces::msg::SetParametersResult param_event_callback(const std::vector<rclcpp::Parameter> &parameters);

      rclcpp::Node::SharedPtr node_;
      std::unique_ptr<YoloEngine> yolo_engine_;
      std::unique_ptr<PnPSolver> pnp_solver_;
      rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;
      rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
      image_transport::Subscriber img_sub_;
      rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_event_handle_;

      // Parameters
      bool enable_debug_;
      bool enable_profiling_;
      int binary_threshold_;
      int enemy_color_;
      float light_min_ratio_;
      float light_max_ratio_;
      float light_max_angle_;
      float armor_min_small_center_distance_;
      float armor_max_small_center_distance_;
      float armor_min_large_center_distance_;
      float armor_max_large_center_distance_;

      // Debug & profiling
      image_transport::Publisher visualized_img_pub_;
      image_transport::Publisher binary_img_pub_;
      rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr total_latency_pub_;
      rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr comm_latency_pub_;
      rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr processing_latency_pub_;
      rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr inference_latency_pub_;
      rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pnp_latency_pub_;
  };
}
