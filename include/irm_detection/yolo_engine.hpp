#pragma once

#include <NvInferRuntime.h>
#include <array>
#include <cstdint>
#include <string>

#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "irm_detection/armor.hpp"

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

      YoloEngine(const std::string &onnx_file_path, bool enable_profiling = false);
      ~YoloEngine();
      std::vector<bbox> detect(const cv::Mat &image);
      void visualize_bboxes(cv::Mat &image, const std::vector<bbox> &bboxes);
      std::tuple<float, float> get_profiling_time() const
      {
        return std::make_tuple(preprocess_time_, inference_time_);
      }

    private:
      void load_engine_file(const std::string &engine_file_path);
      void preprocess(const cv::Mat &image, cv::Mat &preprocessed_image);
      std::vector<bbox> parse_output(float scale_x, float scale_y);

      // TensorRT related
      IRuntime *runtime_;
      ICudaEngine *engine_;
      IExecutionContext *context_;
      cudaStream_t stream_;
      float *input_buffer_;
      struct {
        int32_t *num_dets;
        float *bboxes;
        float *scores;
        int32_t *labels;
      } output_buffer_;

      // Configurations
      bool enable_profiling_ = false;

      // Profiling related
      double preprocess_time_ = 0;
      double inference_time_ = 0;
  };
}
