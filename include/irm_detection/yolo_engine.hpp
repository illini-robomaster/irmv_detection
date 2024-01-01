#pragma once

#include <NvInferRuntime.h>
#include <array>
#include <cstdint>
#include <string>

#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "irm_detection/armor.hpp"
#include "npp.h"
#include "cuda_runtime.h"

using namespace nvinfer1;

namespace irm_detection
{
  class YoloEngine
  {
    public:
      struct bbox
      {
        std::array<float, 4> xyxy;
        float score;
        ArmorClass class_id;

        bool operator==(const bbox &other) const
        {
          return xyxy == other.xyxy && score == other.score && class_id == other.class_id;
        }

        bool operator!=(const bbox &other) const
        {
          return xyxy != other.xyxy || score != other.score || class_id != other.class_id;
        }
      };

      YoloEngine(rclcpp::Node::ConstSharedPtr node, const std::string &onnx_file_path, cv::Size image_input_size, bool enable_profiling = false);
      ~YoloEngine();
      std::vector<bbox> detect(const cv::Mat &image);
      void visualize_bboxes(cv::Mat &image, const std::vector<bbox> &bboxes);
      float get_profiling_time() const
      {
        return inference_time_;
      }
      const cv::Mat & get_rotated_image() const
      {
        return rotated_image_;
      }

    private:
      void load_engine_file(const std::string &engine_file_path);
      void preprocess();
      std::vector<bbox> parse_output(float scale_x, float scale_y);

      rclcpp::Node::ConstSharedPtr node_;

      // OpenCV related
      cv::Mat rotated_image_;
      cv::Size image_input_size_;

      // TensorRT and NPP related
      IRuntime *runtime_;
      ICudaEngine *engine_;
      IExecutionContext *context_;
      cudaStream_t stream_;
      NppStreamContext npp_context_;
      uint8_t *rotated_image_buffer_;
      uint8_t *resized_image_buffer_;
      float *input_buffer_hwc_;
      float *input_buffer_;
      struct {
        int32_t *num_dets;
        float *bboxes;
        float *scores;
        int32_t *labels;
      } output_buffer_;
      cudaGraph_t graph_;
      cudaGraphExec_t graph_exec_;
      Npp32f * aDst_[3] = {nullptr, nullptr, nullptr};

      // Configurations
      bool enable_profiling_ = false;

      // Profiling related
      double preprocess_time_ = 0;
      double inference_time_ = 0;
  };
}
