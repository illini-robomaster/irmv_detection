#include <chrono>
#include <cstdint>
#include <numeric>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "gtest/gtest.h"
#include "irmv_detection/magic_enum.hpp"
#include "irmv_detection/yolo_engine.hpp"


TEST(irmv_detection, yolo_engine_demo)
{
  cudaSetDevice(0);
  std::string model_path =
    ament_index_cpp::get_package_share_directory("irmv_detection") + "/models/yolov7.onnx";
  irmv_detection::YoloEngine yolo_engine(model_path, cv::Size(1280, 1024));

  std::string image_path =
    ament_index_cpp::get_package_share_directory("irmv_detection") + "/test/rm_test.jpg";

  cv::Mat image = cv::imread(image_path);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::namedWindow("result", cv::WINDOW_NORMAL);

  uint8_t * src_image_buffer = yolo_engine.get_src_image_buffer();
  memcpy(src_image_buffer, image.data, image.total() * image.elemSize());
  std::vector<irmv_detection::YoloEngine::bbox> bboxes = yolo_engine.detect();

  std::cout << "bboxes.size(): " << bboxes.size() << std::endl;

  // Usually there shouldn't be more than 20 bboxes.
  // If there are more than 20 bboxes, it's very likely that the model is not working properly.
  ASSERT_LT(bboxes.size(), 20U);

  for (auto & bbox : bboxes) {
    std::cout << bbox.xyxy[0] << ", " << bbox.xyxy[1] << ", " << bbox.xyxy[2] << ", "
              << bbox.xyxy[3] << std::endl;
    std::cout << bbox.score << std::endl;
    std::cout << magic_enum::enum_name(bbox.class_id) << std::endl;
  }

  cv::Mat visualized_image = yolo_engine.get_rotated_image().clone();
  yolo_engine.visualize_bboxes(visualized_image, bboxes);

  cv::imshow("result", visualized_image);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

TEST(irmv_detection, yolo_engine_benchmark)
{
  cudaSetDevice(0);
  std::string model_path =
    ament_index_cpp::get_package_share_directory("irmv_detection") + "/models/yolov7.onnx";
  irmv_detection::YoloEngine yolo_engine(model_path, cv::Size(1280, 1024), false);

  std::string image_path =
    ament_index_cpp::get_package_share_directory("irmv_detection") + "/test/rm_test.jpg";

  cv::Mat image = cv::imread(image_path);

  std::cout << "Benchmarking..." << std::endl;

  // Warmpup
  uint8_t * src_image_buffer = yolo_engine.get_src_image_buffer();
  for (int i = 0; i < 100; i++) {
    memcpy(src_image_buffer, image.data, image.total() * image.elemSize());
    std::vector<irmv_detection::YoloEngine::bbox> bboxes = yolo_engine.detect();
  }

  std::vector<double> avg_times;

  for (int run = 0; run < 30; run++) {
    std::chrono::high_resolution_clock::time_point begin =
      std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++) {
      memcpy(src_image_buffer, image.data, image.total() * image.elemSize());
      std::vector<irmv_detection::YoloEngine::bbox> bboxes = yolo_engine.detect();
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    double avg_time =
      static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /
      10000.0;
    avg_times.emplace_back(avg_time);
  }

  for (int i = 0; i < static_cast<int>(avg_times.size()); i++) {
    std::cout << "Run " << i << ": " << avg_times[i] << " ms" << std::endl;
  }

  double avg_time = std::accumulate(avg_times.begin(), avg_times.end(), 0.0) / avg_times.size();
  std::cout << "Average detection time: " << avg_time << " ms" << std::endl;
  double max_time = *std::ranges::max_element(avg_times);
  std::cout << "Max detection time: " << max_time << " ms" << std::endl;
  double min_time = *std::ranges::min_element(avg_times);
  std::cout << "Min detection time: " << min_time << " ms" << std::endl;

  // A detection time larger than 30 ms generally means very bad performance.
  ASSERT_LT(max_time, 30);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
