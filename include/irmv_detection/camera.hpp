#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <chrono>

namespace irmv_detection
{
class Camera
{
public:
  using CameraCallback = std::function<void (cv::Mat &, std::chrono::time_point<std::chrono::high_resolution_clock>)>;
  Camera() = default;
  virtual void set_camera_callback(const CameraCallback & callback) = 0;
  virtual ~Camera() = default;

protected:
  CameraCallback camera_callback_;
};

class VirtualCamera : public Camera
{
public:
  explicit VirtualCamera(const std::string & video_path, int fps = 100);
  void set_camera_callback(const CameraCallback & callback) override;
  ~VirtualCamera() override = default;

private:
  [[noreturn]] void stream_thread();
  [[noreturn]] void receive_thread();
  std::filesystem::path video_path_;
  std::jthread stream_thread_;
  std::jthread receive_thread_;
  cv::VideoCapture cap_;
  int fps_;
  cv::Mat frame_;
  std::chrono::time_point<std::chrono::high_resolution_clock> time_stamp_;
  bool frame_ready_ = false;
  std::mutex frame_buffer_mutex_;
  std::condition_variable consumer_cv_;
  std::condition_variable producer_cv_;
};

class MVCamera : public Camera
{
public:
  MVCamera();
  void set_camera_callback(const CameraCallback & callback) override;
  ~MVCamera() override = default;
};
}  // namespace irmv_detection