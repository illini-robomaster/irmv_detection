#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <CameraDefine.h>
#include <chrono>

namespace irmv_detection
{
class Camera
{
public:
  using CameraCallback = std::function<void (cv::Mat &, std::chrono::time_point<std::chrono::system_clock>)>;
  Camera() = default;
  virtual ~Camera() = default;
};

class VirtualCamera : public Camera
{
public:
  explicit VirtualCamera(const std::string & video_path, uint8_t * buffer, const CameraCallback & callback, int fps = 100);
  ~VirtualCamera() override = default;

private:
  CameraCallback camera_callback_;
  [[noreturn]] void stream_thread();
  [[noreturn]] void receive_thread();
  std::filesystem::path video_path_;
  std::jthread stream_thread_;
  std::jthread receive_thread_;
  cv::VideoCapture cap_;
  int fps_;
  cv::Mat frame_;
  std::chrono::time_point<std::chrono::system_clock> time_stamp_;
  bool frame_ready_ = false;
  std::mutex frame_buffer_mutex_;
  std::condition_variable consumer_cv_;
  std::condition_variable producer_cv_;
};

class MVCamera : public Camera
{
public:
  explicit MVCamera(const CameraCallback & callback);
  void trigger_callback(CameraHandle hCamera, BYTE * pFrameBuffer, tSdkFrameHead * pFrameHead);
  ~MVCamera() override = default;

private:
  CameraCallback camera_callback_;
  [[noreturn]] void receive_thread();
  CameraHandle h_camera_;
  std::jthread receive_thread_;
  cv::Mat frame_;
  std::chrono::time_point<std::chrono::system_clock> time_stamp_;
  bool frame_ready_ = false;
  std::mutex frame_buffer_mutex_;
  std::condition_variable consumer_cv_;
  std::condition_variable producer_cv_;
};
}  // namespace irmv_detection