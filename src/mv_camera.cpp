#include <CameraApi.h>
#include <CameraDefine.h>

#include "CameraStatus.h"
#include "irmv_detection/camera.hpp"

namespace irmv_detection
{

// The pContext is a workaround for the callback function.
// The callback function is a C-style function, so it cannot be binded to a class member function.
void CameraCallbackFunction(
  CameraHandle hCamera, BYTE * pFrameBuffer, tSdkFrameHead * pFrameHead, PVOID pContext)
{
  auto mv_camera_ptr = static_cast<MVCamera *>(pContext);
  if (mv_camera_ptr) {
    mv_camera_ptr->trigger_callback(hCamera, pFrameBuffer, pFrameHead);
  }
}

MVCamera::MVCamera(uint8_t * buffer, cv::Size image_size, const CameraCallback & callback) : camera_callback_(callback)
{
  CameraSdkInit(0);

  int iCameraCounts = 1;
  tSdkCameraDevInfo tCameraEnumList;
  int iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
  std::cout << "Camera count: " << iCameraCounts << std::endl;
  if (iStatus != CAMERA_STATUS_SUCCESS) {
    std::cout << "CameraEnumerateDevice error: " << iStatus << std::endl;
    exit(0);
  }

  if (iCameraCounts == 0) {
    std::cout << "No camera detected" << std::endl;
    exit(0);
  }

  iStatus = CameraInit(&tCameraEnumList, -1, -1, &h_camera_);
  if (iStatus != CAMERA_STATUS_SUCCESS) {
    std::cout << "CameraInit error: " << iStatus << std::endl;
    exit(0);
  }

  CameraSetTriggerMode(h_camera_, 2);

  CameraSetAeState(h_camera_, FALSE);

  tSdkCameraCapbility tCapability;
  CameraGetCapability(h_camera_, &tCapability);
  std::cout << "pImageSizeDesc->iWidth: " << tCapability.pImageSizeDesc << std::endl;
  std::cout << "pImageSizeDesc->iHeight: " << tCapability.pImageSizeDesc->iHeight << std::endl;
  double exposure_line_time;
  CameraGetExposureLineTime(h_camera_, &exposure_line_time);
  CameraSetExposureTime(h_camera_, 5000);

  int analog_gain;
  CameraGetAnalogGain(h_camera_, &analog_gain);
  CameraSetAnalogGain(h_camera_, analog_gain);

  CameraPlay(h_camera_);
  CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_RGB8);

  int trigger_mode;
  CameraGetTriggerMode(h_camera_, &trigger_mode);
  std::cout << "trigger_mode: " << trigger_mode << std::endl;

  CameraSetCallbackFunction(h_camera_, &CameraCallbackFunction, this, nullptr);

  frame_ = cv::Mat(1024, 1280, CV_8UC3, buffer);
  receive_thread_ = std::jthread(&MVCamera::receive_thread, this);
  receive_thread_.detach();
}

MVCamera::~MVCamera()
{
  CameraStop(h_camera_);
  CameraUnInit(h_camera_);
}

void MVCamera::trigger_callback(
  CameraHandle hCamera, BYTE * pFrameBuffer, tSdkFrameHead * pFrameHead)
{
  namespace chrono = std::chrono;
  auto start_time = chrono::system_clock::now();
  std::unique_lock lock(frame_buffer_mutex_);
  producer_cv_.wait(lock, [this] { return !frame_ready_; });
  CameraImageProcess(hCamera, pFrameBuffer, frame_.ptr(), pFrameHead);
  time_stamp_ = start_time;
  frame_ready_ = true;
  lock.unlock();
  consumer_cv_.notify_one();
}

void MVCamera::receive_thread()
{
  namespace chrono = std::chrono;
  int frame_count = 0;
  auto starting_time = chrono::system_clock::now();
  while (true) {
    std::unique_lock lock(frame_buffer_mutex_);
    consumer_cv_.wait(lock, [this] { return frame_ready_; });
    camera_callback_(frame_, time_stamp_);
    frame_ready_ = false;
    lock.unlock();
    producer_cv_.notify_one();

    frame_count++;
    if (frame_count == 100) {
      auto cur_time = chrono::system_clock::now();
      std::cout << "FPS: " << 100 / (chrono::duration<double>(cur_time - starting_time).count())
                << std::endl;
      starting_time = cur_time;
      frame_count = 0;
    }
  }
}

}  // namespace irmv_detection