
#include "irmv_detection/camera.hpp"
#include <chrono>
#include <memory>
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
  cap_ >> stamped_img_buf_[0].image;
  cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
  if (config_.image_size != stamped_img_buf_[0].image.size()) {
    throw std::invalid_argument("Image size does not match");
  }
  for (int i = 0; i < 3; i++) {
    stamped_img_buf_[i].image = cv::Mat(config_.image_size, CV_8UC3, config_.image_buffers[i]);
    // fmt::print("Buffer: {}\n", (void *)config_.image_buffers[i]);
    stamped_img_buf_[i].id = i;
  }
  triple_buffer_ = std::make_unique<TripleBuffer<StampedImage>>(stamped_img_buf_);
  stream_thread_ = std::jthread(&VirtualCamera::stream_thread, this);
  receive_thread_ = std::jthread(&VirtualCamera::receive_thread, this);
}

void VirtualCamera::stream_thread()
{
  namespace chrono = std::chrono;
  int frame_count = 0;
  auto starting_time = chrono::system_clock::now();
  auto interval = chrono::duration<double>(1.0 / fps_);
  while (!shutdown_) {
    auto start_time = chrono::system_clock::now();
    auto stamped_image = triple_buffer_->get_producer_buffer();
    // fmt::print("Producer: {}\n", stamped_image->id);
    if (!cap_.grab()) {
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      cap_.grab();
    }
    // fmt::print("Producer Buffer: {}\n", (void *)stamped_image->image.data);
    cap_.retrieve(stamped_image->image);
    stamped_image->time_stamp = start_time;
    triple_buffer_->producer_commit();
    std::this_thread::sleep_until(start_time + interval);
    frame_count++;
    if (frame_count == 200) {
      auto cur_time = chrono::system_clock::now();
      fmt::print(
        "Producer FPS: {}\n", 200 / (chrono::duration<double>(cur_time - starting_time).count()));
      starting_time = cur_time;
      frame_count = 0;
    }
  }
}

void VirtualCamera::receive_thread()
{
  namespace chrono = std::chrono;
  int frame_count = 0;
  auto starting_time = chrono::system_clock::now();
  while (!shutdown_) {
    auto stamped_image = triple_buffer_->get_consumer_buffer();
    // fmt::print("Consumer: {}\n", stamped_image->id);
    // fmt::print("Consumer Buffer: {}\n", (void *)stamped_image->image.data);
    camera_callback_(*stamped_image);
    // fmt::print("Consumer done\n");
    frame_count++;
    if (frame_count == 200) {
      auto cur_time = chrono::system_clock::now();
      fmt::print(
        "Consumer FPS: {}\n", 200 / (chrono::duration<double>(cur_time - starting_time).count()));
      starting_time = cur_time;
      frame_count = 0;
    }
  }
}

VirtualCamera::~VirtualCamera()
{
  shutdown_ = true;
  stream_thread_.join();
  triple_buffer_->producer_commit();  // Commit a dummy to wake up the consumer thread
}
}  // namespace irmv_detection
