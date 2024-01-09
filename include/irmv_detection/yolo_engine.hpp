#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <NvInferRuntime.h>
#include <nppdefs.h>
#include <opencv2/opencv.hpp>

#include "irmv_detection/armor.hpp"

namespace irmv_detection
{
using namespace nvinfer1;
class YoloEngine
{
public:
  struct bbox
  {
    std::array<float, 4> xyxy;
    float score;
    ArmorClass class_id;

    bool operator==(const bbox & other) const = default;
  };

  YoloEngine(
    const std::string & onnx_file_path, cv::Size src_image_size, bool enable_profiling = false);
  ~YoloEngine();
  std::vector<bbox> detect();
  void visualize_bboxes(cv::Mat & image, const std::vector<bbox> & bboxes) const;
  double get_profiling_time() const { return inference_time_.count(); }
  const cv::Mat & get_rotated_image() const { return src_image_; }
  uint8_t * get_src_image_buffer() const { return src_image_buffer_; }

private:
  void load_engine_file(const std::string & engine_file_path);
  void preprocess();
  std::vector<bbox> parse_output(float scale_x, float scale_y) const;

  // OpenCV related
  cv::Mat src_image_;
  cv::Size src_image_size_;

  // TensorRT and NPP related
  IRuntime * runtime_;
  ICudaEngine * engine_;
  IExecutionContext * context_;
  cudaStream_t stream_;
  NppStreamContext npp_context_;
  uint8_t * src_image_buffer_;
  uint8_t * resized_image_buffer_;
  float * input_buffer_hwc_;
  float * input_buffer_;
  struct
  {
    int32_t * num_dets;
    float * bboxes;
    float * scores;
    int32_t * labels;
  } output_buffer_;
  cudaGraph_t graph_;
  cudaGraphExec_t graph_exec_;
  Npp32f * aDst_[3] = {nullptr, nullptr, nullptr};

  // Configurations
  bool enable_profiling_ = false;

  // Profiling related
  std::chrono::duration<double, std::milli> preprocess_time_;
  std::chrono::duration<double, std::milli> inference_time_;
};
}  // namespace irmv_detection
