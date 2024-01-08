
#include "irmv_detection/camera.hpp"
#include <chrono>
#include <mutex>
#include <stdexcept>

#include <fmt/format.h>

namespace irmv_detection
{
VirtualCamera::VirtualCamera(
  const Config & config, const std::string & video_path, const CameraCallback & callback, int fps)
: config_(config), camera_callback_(callback), video_path_(video_path), fps_(fps)
{
  cap_ = cv::VideoCapture(video_path_.string());
  if (!cap_.isOpened()) {
    throw invalid_camera_error("Cannot open video file");
  }
  cap_ >> frame_;
  cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
  if (config_.image_size != frame_.size()) {
    throw std::invalid_argument("Image size does not match");
  }
  frame_ = cv::Mat(config_.image_size, CV_8UC3, config_.image_buffer);
  stream_thread_ = std::jthread(&VirtualCamera::stream_thread, this);
  receive_thread_ = std::jthread(&VirtualCamera::receive_thread, this);
}

void VirtualCamera::stream_thread()
{
  namespace chrono = std::chrono;
  auto interval = chrono::milliseconds(1000 / fps_);
  while (!shutdown_) {
    auto start_time = chrono::system_clock::now();
    std::unique_lock lock(frame_buffer_mutex_);
    producer_cv_.wait(lock, [this] { return !frame_ready_ || shutdown_; });
    if (!cap_.grab()) {
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      cap_.grab();
    }
    cap_.retrieve(frame_);
    cv::cvtColor(frame_, frame_, cv::COLOR_BGR2RGB);
    time_stamp_ = start_time;
    frame_ready_ = true;
    lock.unlock();
    consumer_cv_.notify_one();
    std::this_thread::sleep_until(start_time + interval);
  }
}

void VirtualCamera::receive_thread()
{
  namespace chrono = std::chrono;
  int frame_count = 0;
  auto starting_time = chrono::system_clock::now();
  while (!shutdown_) {
    std::unique_lock lock(frame_buffer_mutex_);
    consumer_cv_.wait(lock, [this] { return frame_ready_ || shutdown_; });
    camera_callback_(frame_, time_stamp_);
    frame_ready_ = false;
    lock.unlock();
    producer_cv_.notify_one();

    frame_count++;
    if (frame_count == 100) {
      auto cur_time = chrono::system_clock::now();
      fmt::print("FPS: {}\n", 100 / (chrono::duration<double>(cur_time - starting_time).count()));
      starting_time = cur_time;
      frame_count = 0;
    }
  }
}

VirtualCamera::~VirtualCamera()
{
  shutdown_ = true;
  // Force wake up the consumer thread
  consumer_cv_.notify_all();
  producer_cv_.notify_all();
}
}  // namespace irmv_detection
