#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <CameraDefine.h>
#include "irmv_detection/triple_buffer.hpp"
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
    std::array<uint8_t *, 3> image_buffers;
  };
  struct StampedImage
  {
    cv::Mat image;
    std::chrono::time_point<std::chrono::system_clock> time_stamp;
    int id;
  };
  class invalid_camera_error : public std::runtime_error
  {
  public:
    using runtime_error::runtime_error;
  };
  using CameraCallback = std::function<void (StampedImage &)>;
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
  std::array<StampedImage, 3> stamped_img_buf_;
  std::unique_ptr<TripleBuffer<StampedImage>> triple_buffer_;
  std::atomic<bool> shutdown_ = false;
  std::jthread stream_thread_;
  std::jthread receive_thread_;
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
  std::array<StampedImage, 3> stamped_img_buf_;
  std::unique_ptr<TripleBuffer<StampedImage>> triple_buffer_;
  std::atomic<bool> shutdown_ = false;
  std::jthread receive_thread_;
};
}  // namespace irmv_detection