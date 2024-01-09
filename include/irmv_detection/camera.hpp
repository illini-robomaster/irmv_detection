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
  struct Config
  {
    double exposure_time;
    int analog_gain;
    int saturation;
    int gamma;
    cv::Size image_size = cv::Size(0, 0);
    uint8_t * image_buffer = nullptr;
  };
  class invalid_camera_error : public std::runtime_error
  {
  public:
    using runtime_error::runtime_error;
  };
  using CameraCallback = std::function<void (cv::Mat &, std::chrono::time_point<std::chrono::system_clock>)>;
  Camera() = default;
  virtual ~Camera() = default;
};

class VirtualCamera : public Camera
{
public:
  explicit VirtualCamera(const Config & config, const std::string & video_path, const CameraCallback & callback, int fps = 100);
  ~VirtualCamera() override;

private:
  void stream_thread();
  void receive_thread();
  Config config_;
  CameraCallback camera_callback_;
  std::filesystem::path video_path_;
  cv::VideoCapture cap_;
  int fps_;
  cv::Mat frame_;
  std::chrono::time_point<std::chrono::system_clock> time_stamp_;
  bool frame_ready_ = false;
  std::mutex frame_buffer_mutex_;
  std::condition_variable consumer_cv_;
  std::condition_variable producer_cv_;
  std::jthread stream_thread_;
  std::jthread receive_thread_;
  std::atomic<bool> shutdown_ = false;
};

class MVCamera : public Camera
{
public:
  explicit MVCamera(const Config & config, const CameraCallback & callback);
  void trigger_callback(CameraHandle hCamera, BYTE * pFrameBuffer, tSdkFrameHead * pFrameHead);
  ~MVCamera() override;

private:
  void receive_thread();
  Config config_;
  CameraCallback camera_callback_;
  CameraHandle h_camera_;
  cv::Mat frame_;
  std::chrono::time_point<std::chrono::system_clock> time_stamp_;
  bool frame_ready_ = false;
  std::mutex frame_buffer_mutex_;
  std::condition_variable consumer_cv_;
  std::condition_variable producer_cv_;
  std::jthread receive_thread_;
  std::atomic<bool> shutdown_ = false;
};
}  // namespace irmv_detection