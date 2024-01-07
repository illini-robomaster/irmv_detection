
#include <chrono>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "irmv_detection/camera.hpp"
#include "irmv_detection/yolo_engine.hpp"

TEST(irmv_detection, virtual_camera)
{
  std::string model_path =
    ament_index_cpp::get_package_share_directory("irmv_detection") + "/models/yolov7.onnx";
  irmv_detection::YoloEngine yolo_engine(nullptr, model_path, cv::Size(1280, 1024), false);
  auto callback = [&yolo_engine](
                    const cv::Mat & image,
                    std::chrono::time_point<std::chrono::system_clock> time_stamp) {
    (void) image;
    auto bboxes = yolo_engine.detect();
    auto cur_time = std::chrono::system_clock::now();
    auto processing_time = std::chrono::duration<double, std::milli>(cur_time - time_stamp).count();
    if (processing_time > 10) {
      std::cout
        << "[WARN] Latency exceeded desired value, camera fetch may be blocked. Processing time: "
        << processing_time << "ms" << std::endl;
    }
  };
  irmv_detection::VirtualCamera virtual_camera(
    "/mnt/d/RMUL23_Vision_data/3v3/Italy_Torino_group/video_28.mp4",
    yolo_engine.get_src_image_buffer(), callback, 100);

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(100));
  }
}

TEST(irmv_detection, mv_camera)
{
  std::string model_path =
    ament_index_cpp::get_package_share_directory("irmv_detection") + "/models/yolov7.onnx";
  irmv_detection::YoloEngine yolo_engine(nullptr, model_path, cv::Size(1280, 1024), false);
  auto callback = [&yolo_engine](
                    const cv::Mat & image,
                    std::chrono::time_point<std::chrono::system_clock> time_stamp){
    (void) image;
    auto bboxes = yolo_engine.detect();
    auto cur_time = std::chrono::system_clock::now();
    auto processing_time = std::chrono::duration<double, std::milli>(cur_time - time_stamp).count();
    if (processing_time > 10) {
      std::cout
        << "[WARN] Latency exceeded desired value, camera fetch may be blocked. Processing time: "
        << processing_time << "ms" << std::endl;
    }
  };
  irmv_detection::MVCamera mv_camera(yolo_engine.get_src_image_buffer(), cv::Size(1280, 1024), callback);

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(100));
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}