
#include "irmv_detection/camera.hpp"
#include <chrono>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include "irmv_detection/yolo_engine.hpp"

TEST(irmv_detection, virtual_camera)
{
  irmv_detection::YoloEngine yolo_engine(
    nullptr, "/home/niceme/workspaces/irm_ros-dev/src/irmv_detection/models/yolov7.engine",
    cv::Size(1280, 1024), false);
  irmv_detection::VirtualCamera virtual_camera(
    "/mnt/d/RMUL23_Vision_data/3v3/Italy_Torino_group/video_28.mp4", yolo_engine.get_src_image_buffer(), 100);

  auto callback = [&yolo_engine](
                    cv::Mat & image,
                    std::chrono::time_point<std::chrono::system_clock> time_stamp) {
    auto bboxes = yolo_engine.detect(image);
    auto cur_time = std::chrono::system_clock::now();
    auto processing_time = std::chrono::duration<double, std::milli>(cur_time - time_stamp).count();
    if (processing_time > 10) {
      std::cout << "[WARN] Latency exceeded desired value, camera fetch may be blocked. Processing time: " << processing_time << "ms" << std::endl;
    }
  };
  virtual_camera.set_camera_callback(callback);
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(100));
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}