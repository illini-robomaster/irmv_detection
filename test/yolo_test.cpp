#include "gtest/gtest.h"
#include "irm_detection/yolo_engine.hpp"
#include <filesystem>
#include <opencv2/highgui.hpp>
#include "irm_detection/magic_enum.hpp"

namespace fs = std::filesystem;

constexpr char TEST_MODEL_PATH[] = "/home/niceme/workspaces/irm_ros-dev/src/iRM_Vision_2023/irm_detection/models/yolov7.onnx";
constexpr char TEST_IMAGE_PATH[] = "/home/niceme/workspaces/irm_ros-dev/src/iRM_Vision_2023/irm_detection/test/rm_test.png";

static void visualize_bboxes(cv::Mat &image, std::vector<irm_detection::YoloEngine::bbox> &bboxes)
{
  for (auto &bbox : bboxes) {
    cv::Point p1(bbox.xyxy[0], bbox.xyxy[1]);
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

TEST(irm_detection, yolo_engine_demo)
{
  cudaSetDevice(0);
  irm_detection::YoloEngine yolo_engine(TEST_MODEL_PATH);

  fs::path image_path(TEST_IMAGE_PATH);

  cv::Mat image = cv::imread(image_path.string());
  cv::Mat visualized_image = image.clone();
  cv::namedWindow("result", cv::WINDOW_NORMAL);

  std::vector<irm_detection::YoloEngine::bbox> bboxes = yolo_engine.detect(image);
  
  std::cout << "bboxes.size(): " << bboxes.size() << std::endl;

  // Usually there shouldn't be more than 20 bboxes.
  // If there are more than 20 bboxes, it's very likely that the model is not working properly.
  ASSERT_LT(bboxes.size(), 20U);

  for (auto &bbox : bboxes) {
    std::cout << bbox.xyxy[0] << ", " << bbox.xyxy[1] << ", " << bbox.xyxy[2] << ", " << bbox.xyxy[3] << std::endl;
    std::cout << bbox.score << std::endl;
    std::cout << magic_enum::enum_name(bbox.class_id) << std::endl;
  }

  visualize_bboxes(visualized_image, bboxes);

  cv::imshow("result", visualized_image);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

TEST(irm_detection, yolo_engine_benchmark)
{
  cudaSetDevice(0);
  irm_detection::YoloEngine yolo_engine(TEST_MODEL_PATH, true);

  fs::path image_path(TEST_IMAGE_PATH);

  cv::Mat image = cv::imread(image_path.string());

  std::cout << "Benchmarking..." << std::endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (int i = 0; i < 1000; i++) {
    std::vector<irm_detection::YoloEngine::bbox> bboxes = yolo_engine.detect(image);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  float avg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0;

  // A detection time larger than 30 ms generally means very bad performance.
  ASSERT_LT(avg_time, 30);

  std::cout << "Average detection time: " << avg_time << " ms" << std::endl;
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
