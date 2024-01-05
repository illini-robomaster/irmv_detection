#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <npp.h>
#include <opencv2/opencv.hpp>

#include "irmv_detection/magic_enum.hpp"
#include "irmv_detection/trt_logger.hpp"
#include "irmv_detection/yolo_engine.hpp"

using namespace nvinfer1;

namespace irmv_detection
{
static Logger gLogger;

namespace fs = std::filesystem;

YoloEngine::YoloEngine(
  rclcpp::Node::ConstSharedPtr node, const std::string & onnx_file_path, cv::Size image_input_size,
  bool enable_profiling)
: node_(node), image_input_size_(image_input_size), enable_profiling_(enable_profiling)
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

  // Not sure why nVidia doesn't provide a function to get the size of a Dims object
  auto get_dim_size = [](const Dims & dims) {
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

  // Use unified memory to gain better performance on Jetson devices
  cudaMallocManaged(
    std::bit_cast<void **>(&rotated_image_buffer_),
    image_input_size.height * image_input_size.width * 3);
  cudaMallocManaged(
    std::bit_cast<void **>(&output_buffer_.num_dets), get_dim_size(num_dets_dims) * sizeof(int));
  cudaMallocManaged(
    std::bit_cast<void **>(&output_buffer_.bboxes), get_dim_size(bboxes_dims) * sizeof(float));
  cudaMallocManaged(
    std::bit_cast<void **>(&output_buffer_.scores), get_dim_size(scores_dims) * sizeof(float));
  cudaMallocManaged(
    std::bit_cast<void **>(&output_buffer_.labels), get_dim_size(labels_dims) * sizeof(int));
  // Below buffers won't be used by CPU,
  // so device memory is used to avoid unnecessary caching on CPU side
  // (this has a huge impact on dGPU devices, maybe not too much on Jetson)
  cudaMalloc(std::bit_cast<void **>(&resized_image_buffer_), get_dim_size(input_dims));
  cudaMalloc(std::bit_cast<void **>(&input_buffer_hwc_), get_dim_size(input_dims) * sizeof(float));
  cudaMalloc(std::bit_cast<void **>(&input_buffer_), get_dim_size(input_dims) * sizeof(float));

  rotated_image_ = cv::Mat(
    cv::Size(image_input_size_.width, image_input_size_.height), CV_8UC3, rotated_image_buffer_);

  // Compliant with enqueueV3 API
  context_->setInputTensorAddress("images", input_buffer_);
  context_->setTensorAddress("num_dets", output_buffer_.num_dets);
  context_->setTensorAddress("det_boxes", output_buffer_.bboxes);
  context_->setTensorAddress("det_scores", output_buffer_.scores);
  context_->setTensorAddress("det_classes", output_buffer_.labels);

  cudaStreamCreate(&stream_);
  // Code below comes from https://github.com/JaapSuter/npp_context_repro/blob/56a7aec8b0b71129d6090cdaea3f27f8fd45a2b9/main.cpp#L37C1-L40C6
  // Currently NPP documentation does not mention anything about the use of NppStreamContext.
  nppGetStreamContext(&npp_context_);
  npp_context_.hStream = stream_;

  // Create CUDA graph
  aDst_[0] = input_buffer_;
  aDst_[1] = input_buffer_ + 640 * 640;
  aDst_[2] = input_buffer_ + 640 * 640 * 2;
  context_->enqueueV3(
    stream_);  // Update engine internal states (bindings etc.), see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#cuda-graphs
  cudaGraphCreate(&graph_, 0);
  cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
  preprocess();
  context_->enqueueV3(stream_);
  cudaStreamEndCapture(stream_, &graph_);
  cudaGraphInstantiate(&graph_exec_, graph_, 0);
}

YoloEngine::~YoloEngine()
{
  cudaStreamDestroy(stream_);
  cudaGraphExecDestroy(graph_exec_);
  cudaGraphDestroy(graph_);
  cudaFree(rotated_image_buffer_);
  cudaFree(resized_image_buffer_);
  cudaFree(input_buffer_hwc_);
  cudaFree(input_buffer_);
  cudaFree(output_buffer_.num_dets);
  cudaFree(output_buffer_.bboxes);
  cudaFree(output_buffer_.scores);
  cudaFree(output_buffer_.labels);
  delete context_;
  delete engine_;
  delete runtime_;
}

void YoloEngine::load_engine_file(const std::string & engine_file_path)
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

std::vector<YoloEngine::bbox> YoloEngine::detect(const cv::Mat & image)
{
  static const float scale_x = static_cast<float>(image_input_size_.width) / 640;
  static const float scale_y = static_cast<float>(image_input_size_.height) / 640;
  if (image.cols != image_input_size_.width || image.rows != image_input_size_.height) {
    if (node_ != nullptr)
      RCLCPP_ERROR(
        node_->get_logger(),
        "YOLOEngine: Input image size does not match the input size specified in the "
        "constructor.\nInput image size: %dx%d\nInput size specified in the constructor: %dx%d",
        image.cols, image.rows, image_input_size_.width, image_input_size_.height);
    else
      std::cout << "YOLOEngine: Input image size does not match the input size specified in the "
                   "constructor.\nInput image size: "
                << image.cols << "x" << image.rows
                << "\nInput size specified in the constructor: " << image_input_size_.width << "x"
                << image_input_size_.height << std::endl;
    exit(0);
  }

  std::chrono::high_resolution_clock::time_point preprocess_start;
  std::chrono::high_resolution_clock::time_point inference_end;
  if (enable_profiling_) {
    preprocess_start = std::chrono::high_resolution_clock::now();
  }
  // Preprocess & Inference [device side]
  cudaMemcpyAsync(
    rotated_image_buffer_, image.ptr(), image_input_size_.height * image_input_size_.width * 3,
    cudaMemcpyDefault, stream_);
  cudaGraphLaunch(graph_exec_, stream_);
  cudaStreamSynchronize(stream_);

  // Parse output [host side]
  // This computation is very fast on CPU
  std::vector<bbox> bboxes;
  bboxes = parse_output(scale_x, scale_y);
  if (enable_profiling_) {
    inference_end = std::chrono::high_resolution_clock::now();
    inference_time_ = inference_end - preprocess_start;
  }

  return bboxes;
}

void YoloEngine::preprocess()
{
  // Rotate 180 degrees
  nppiMirror_8u_C3IR_Ctx(
    rotated_image_buffer_, image_input_size_.width * 3,
    NppiSize{image_input_size_.width, image_input_size_.height}, NPP_BOTH_AXIS, npp_context_);
  // Resize to 640x640
  nppiResize_8u_C3R_Ctx(
    rotated_image_buffer_, image_input_size_.width * 3,
    NppiSize{image_input_size_.width, image_input_size_.height},
    NppiRect{0, 0, image_input_size_.width, image_input_size_.height}, resized_image_buffer_,
    640 * 3, NppiSize{640, 640}, NppiRect{0, 0, 640, 640}, NPPI_INTER_LINEAR, npp_context_);
  // Convert to float and remap values from [0, 255] to [0.0, 1.0] (normalize)
  nppiScale_8u32f_C3R_Ctx(
    resized_image_buffer_, 640 * 3, input_buffer_hwc_, 640 * 3 * sizeof(float), NppiSize{640, 640},
    0.0, 1.0, npp_context_);
  // Convert HWC to CHW
  // TODO: I can't find NPP function to do this conversion directly, below is a workaround using NPP Packed To Planar Channel Copy
  nppiCopy_32f_C3P3R_Ctx(
    input_buffer_hwc_, 640 * 3 * sizeof(float), aDst_, 640 * sizeof(float), NppiSize{640, 640},
    npp_context_);
}

std::vector<YoloEngine::bbox> YoloEngine::parse_output(float scale_x, float scale_y) const
{
  std::vector<bbox> bboxes;
  int32_t num_dets = *output_buffer_.num_dets;
  for (int i = 0; i < num_dets; i++) {
    const float * bbox_ptr = output_buffer_.bboxes + i * 4;
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

void YoloEngine::visualize_bboxes(
  cv::Mat & image, const std::vector<irmv_detection::YoloEngine::bbox> & bboxes) const
{
  for (auto & bbox : bboxes) {
    cv::Point p1(bbox.xyxy[0], bbox.xyxy[1]);
    cv::Point p2(bbox.xyxy[2], bbox.xyxy[3]);
    cv::Scalar color;
    if (magic_enum::enum_name(bbox.class_id)[0] == 'B') {
      color = cv::Scalar(0, 0, 255);
    } else {
      color = cv::Scalar(255, 0, 0);
    }
    cv::rectangle(image, p1, p2, color, 2);
    cv::putText(
      image, std::string(magic_enum::enum_name(bbox.class_id)), p1, cv::FONT_HERSHEY_SIMPLEX, 1,
      color, 2);
  }
}

}  // namespace irmv_detection
