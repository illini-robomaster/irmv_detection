
#include "irmv_detection/camera.hpp"
#include <chrono>
#include <mutex>

namespace irmv_detection
{
VirtualCamera::VirtualCamera(const std::string & video_path, int fps)
: video_path_(video_path), fps_(fps)
{
  cap_ = cv::VideoCapture(video_path_.string());
  if (!cap_.isOpened()) {
    throw std::runtime_error("Cannot open video file");
  }
  stream_thread_ = std::jthread(&VirtualCamera::stream_thread, this);
  receive_thread_ = std::jthread(&VirtualCamera::receive_thread, this);
  stream_thread_.detach();
  receive_thread_.detach();
}

[[noreturn]] void VirtualCamera::stream_thread()
{
  namespace chrono = std::chrono;
  auto interval = chrono::milliseconds(1000 / fps_);
  while (true) {
    auto start_time = chrono::high_resolution_clock::now();
    std::unique_lock lock(frame_buffer_mutex_);
    producer_cv_.wait(lock, [this] { return !frame_ready_; });
    cap_ >> frame_;
    time_stamp_ = start_time;
    if (frame_.empty()) {
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      cap_ >> frame_;
      time_stamp_ = start_time;
    }
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
  auto starting_time = chrono::high_resolution_clock::now();
  while (true) {
    std::unique_lock lock(frame_buffer_mutex_);
    consumer_cv_.wait(lock, [this] { return frame_ready_; });
    if (camera_callback_) {
      camera_callback_(frame_, time_stamp_);
    }
    frame_ready_ = false;

    lock.unlock();
    producer_cv_.notify_one();
    frame_count++;
    if (frame_count == 100) {
      auto cur_time = chrono::high_resolution_clock::now();
      std::cout << "FPS: "
                << 100 / (chrono::duration<double>(cur_time - starting_time).count())
                << std::endl;
      starting_time = cur_time;
      frame_count = 0;
    }
  }
}

void VirtualCamera::set_camera_callback(const CameraCallback & callback)
{
  camera_callback_ = callback;
}

}  // namespace irmv_detection
