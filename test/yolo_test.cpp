#include "gtest/gtest.h"
#include "irmv_detection/yolo_engine.hpp"
#include <chrono>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include "irmv_detection/magic_enum.hpp"

namespace fs = std::filesystem;

constexpr char TEST_MODEL_PATH[] = "/home/niceme/workspaces/irm_ros-dev/src/irmv_detection/models/yolov7.onnx";
constexpr char TEST_IMAGE_PATH[] = "/home/niceme/workspaces/irm_ros-dev/src/irmv_detection/test/rm_test.jpg";

static void visualize_bboxes(cv::Mat &image, std::vector<irmv_detection::YoloEngine::bbox> &bboxes)
{
  for (auto &bbox : bboxes) {
    cv::Point p1(static_cast<int>(bbox.xyxy[0]), bbox.xyxy[1]);
    cv::Point p2(bbox.xyxy[2], bbox.xyxy[3]);
    cv::Scalar color;
    if (magic_enum::enum_name(bbox.class_id)[0] == 'B') {
      color = cv::Scalar(255, 0, 0);
    } else {
      color = cv::Scalar(0, 0, 255);
    }
    cv::rectangle(image, p1, p2, color, 2);
    cv::putText(image, std::string(magic_enum::enum_name(bbox.class_id)), p1, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
  }
}

TEST(irmv_detection, yolo_engine_demo)
{
  cudaSetDevice(0);
  irmv_detection::YoloEngine yolo_engine(nullptr, TEST_MODEL_PATH, cv::Size(1280, 1024));

  fs::path image_path(TEST_IMAGE_PATH);

  cv::Mat image = cv::imread(image_path.string());
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::namedWindow("result", cv::WINDOW_NORMAL);

  std::vector<irmv_detection::YoloEngine::bbox> bboxes = yolo_engine.detect(image);
  
  std::cout << "bboxes.size(): " << bboxes.size() << std::endl;

  // Usually there shouldn't be more than 20 bboxes.
  // If there are more than 20 bboxes, it's very likely that the model is not working properly.
  ASSERT_LT(bboxes.size(), 20U);

  for (auto &bbox : bboxes) {
    std::cout << bbox.xyxy[0] << ", " << bbox.xyxy[1] << ", " << bbox.xyxy[2] << ", " << bbox.xyxy[3] << std::endl;
    std::cout << bbox.score << std::endl;
    std::cout << magic_enum::enum_name(bbox.class_id) << std::endl;
  }

  cv::Mat visualized_image = yolo_engine.get_rotated_image().clone();
  visualize_bboxes(visualized_image, bboxes);

  cv::imshow("result", visualized_image);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

TEST(irmv_detection, yolo_engine_benchmark)
{
  cudaSetDevice(0);
  irmv_detection::YoloEngine yolo_engine(nullptr, TEST_MODEL_PATH, cv::Size(1280, 1024), false);

  fs::path image_path(TEST_IMAGE_PATH);

  cv::Mat image = cv::imread(image_path.string());

  std::cout << "Benchmarking..." << std::endl;

  // Warmpup
  for (int i = 0; i < 100; i++) {
    std::vector<irmv_detection::YoloEngine::bbox> bboxes = yolo_engine.detect(image);
  }

  std::vector<double> avg_times;

  for (int run = 0; run < 30; run++) {
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++) {
      std::vector<irmv_detection::YoloEngine::bbox> bboxes = yolo_engine.detect(image);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    double avg_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 10000.0;
    avg_times.emplace_back(avg_time);
  }

  for (int i = 0; i < static_cast<int>(avg_times.size()); i++) {
    std::cout << "Run " << i << ": " << avg_times[i] << " ms" << std::endl;
  }

  double avg_time = std::accumulate(avg_times.begin(), avg_times.end(), 0.0) / avg_times.size();
  std::cout << "Average detection time: " << avg_time << " ms" << std::endl;
  double max_time = *std::ranges::max_element(avg_times.begin(), avg_times.end());
  std::cout << "Max detection time: " << max_time << " ms" << std::endl;
  double min_time = *std::min_element(avg_times.begin(), avg_times.end());
  std::cout << "Min detection time: " << min_time << " ms" << std::endl;

  // A detection time larger than 30 ms generally means very bad performance.
  // ASSERT_LT(avg_time, 30);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
