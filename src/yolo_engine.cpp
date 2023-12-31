#include "irm_detection/yolo_engine.hpp"
#include "irm_detection/trt_logger.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "opencv2/opencv.hpp"
#include "irm_detection/magic_enum.hpp"

using namespace nvinfer1;
using namespace nvonnxparser;

namespace irm_detection
{
  static Logger gLogger;

  namespace fs = std::filesystem;

  YoloEngine::YoloEngine(const std::string &onnx_file_path, bool enable_profiling)
  {
    fs::path onnx_file(onnx_file_path);
    fs::path engine_file(onnx_file);

    engine_file.replace_extension(".engine");

    initLibNvInferPlugins(&gLogger, "");

    if (fs::exists(engine_file)) {
      load_engine_file(engine_file.string());
    } else {
      std::cout << "Please build the engine file with trtexec first." << std::endl;
      exit(0);
    }

    context_ = engine_->createExecutionContext();

    auto get_dim_size = [](const Dims &dims) {
      size_t size = 1;
      for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
      }
      return size;
    };

    Dims input_dims = engine_->getTensorShape("images");
    Dims num_dets_dims = engine_->getTensorShape("num_dets");
    Dims bboxes_dims = engine_->getTensorShape("det_boxes");
    Dims scores_dims = engine_->getTensorShape("det_scores");
    Dims labels_dims = engine_->getTensorShape("det_classes");

    cudaMallocManaged((void **)&input_buffer_,  get_dim_size(input_dims)* sizeof(float));
    cudaMallocManaged((void **)&output_buffer_.num_dets, get_dim_size(num_dets_dims) * sizeof(int));
    cudaMallocManaged((void **)&output_buffer_.bboxes, get_dim_size(bboxes_dims) * sizeof(float));
    cudaMallocManaged((void **)&output_buffer_.scores, get_dim_size(scores_dims) * sizeof(float));
    cudaMallocManaged((void **)&output_buffer_.labels, get_dim_size(labels_dims) * sizeof(int));

    context_->setInputTensorAddress("images", input_buffer_);
    context_->setTensorAddress("num_dets", output_buffer_.num_dets);
    context_->setTensorAddress("det_boxes", output_buffer_.bboxes);
    context_->setTensorAddress("det_scores", output_buffer_.scores);
    context_->setTensorAddress("det_classes", output_buffer_.labels);

    cudaStreamCreate(&stream_);

    enable_profiling_ = enable_profiling;
  }

  YoloEngine::~YoloEngine()
  {
    cudaStreamDestroy(stream_);
    cudaFree(input_buffer_);
    cudaFree(output_buffer_.num_dets);
    cudaFree(output_buffer_.bboxes);
    cudaFree(output_buffer_.scores);
    cudaFree(output_buffer_.labels);
    delete context_;
    delete engine_;
    delete runtime_;
  }

  void YoloEngine::load_engine_file(const std::string &engine_file_path)
  {
    // Read engine file
    std::ifstream engine_file(engine_file_path, std::ios::binary);
    engine_file.seekg(0, std::ios::end);
    const size_t engine_file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_file_size);
    engine_file.read(engine_data.data(), engine_file_size);
    engine_file.close();

    // Deserialize engine
    runtime_ = createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_file_size);
  }

  std::vector<YoloEngine::bbox> YoloEngine::detect(const cv::Mat &image)
  {
    std::chrono::system_clock::time_point preprocess_start, preprocess_end, inference_start, inference_end;
    if (enable_profiling_) {
      preprocess_start = std::chrono::high_resolution_clock::now();
    }
    cv::Mat preprocessed_image(640, 640, CV_32FC3);
    // Preprocess image [host side]
    preprocess(image, preprocessed_image);
    float scale_x = static_cast<float>(image.cols) / 640;
    float scale_y = static_cast<float>(image.rows) / 640;
    if (enable_profiling_) {
      preprocess_end = std::chrono::high_resolution_clock::now();
      preprocess_time_ =  std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000.0;
    }

    // Copy image to input buffer
    if (enable_profiling_) {
      inference_start = std::chrono::high_resolution_clock::now();
    }
    cudaMemcpyAsync(input_buffer_, preprocessed_image.ptr(), preprocessed_image.total() * preprocessed_image.elemSize(), cudaMemcpyDefault, stream_);

    // Inference [device side]
    context_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);
    if (enable_profiling_) {
      inference_end = std::chrono::high_resolution_clock::now();
      inference_time_ = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count() / 1000.0;
    }

    std::vector<bbox> bboxes;
    bboxes = parse_output(scale_x, scale_y);

    // Parse output [host side]
    return bboxes;
  }

  void YoloEngine::preprocess(const cv::Mat &image, cv::Mat &preprocessed_image)
  {
    cv::dnn::blobFromImage(image, preprocessed_image, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false, CV_32F);
  }

  std::vector<YoloEngine::bbox> YoloEngine::parse_output(float scale_x, float scale_y)
  {
    std::vector<bbox> bboxes;
    int32_t num_dets = *output_buffer_.num_dets;
    for (int i = 0; i < num_dets; i++) {
      float *bbox_ptr = output_buffer_.bboxes + i * 4;
      float score = output_buffer_.scores[i];
      int32_t class_id = output_buffer_.labels[i];
      bbox b;
      b.xyxy[0] = bbox_ptr[0] * scale_x;
      b.xyxy[1] = bbox_ptr[1] * scale_y;
      b.xyxy[2] = bbox_ptr[2] * scale_x;
      b.xyxy[3] = bbox_ptr[3] * scale_y;
      b.score = score;
      b.class_id = magic_enum::enum_cast<ArmorClass>(class_id).value_or(ArmorClass::UNKNOWN);
      bboxes.emplace_back(b);
    }
    return bboxes;
  }

  void YoloEngine::visualize_bboxes(cv::Mat &image, const std::vector<irm_detection::YoloEngine::bbox> &bboxes)
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

}  // namespace irm_detection
