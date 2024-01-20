#include <stdexcept>

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

MVCamera::MVCamera(const Config & config, const CameraCallback & callback)
: config_(config), camera_callback_(callback)
{
  CameraSdkInit(0);

  int iCameraCounts = 1;
  tSdkCameraDevInfo tCameraEnumList;
  int iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
  std::cout << "Camera count: " << iCameraCounts << std::endl;
  if (iStatus != CAMERA_STATUS_SUCCESS) {
    throw invalid_camera_error("CameraEnumerateDevice error");
  }

  iStatus = CameraInit(&tCameraEnumList, -1, -1, &h_camera_);
  if (iStatus != CAMERA_STATUS_SUCCESS) {
    std::cout << "CameraInit error: " << iStatus << std::endl;
    throw invalid_camera_error("CameraInit error");
  }

  CameraSetTriggerMode(h_camera_, 2);

  CameraSetAeState(h_camera_, FALSE);

  tSdkCameraCapbility tCapability;
  CameraGetCapability(h_camera_, &tCapability);
  if (
    config_.image_size !=
    cv::Size(tCapability.pImageSizeDesc->iWidth, tCapability.pImageSizeDesc->iHeight)) {
    CameraUnInit(h_camera_);
    throw std::invalid_argument("Image size does not match");
  }

  double exposure_line_time;
  CameraGetExposureLineTime(h_camera_, &exposure_line_time);
  CameraSetExposureTime(h_camera_, config_.exposure_time);

  int analog_gain;
  CameraGetAnalogGain(h_camera_, &analog_gain);
  CameraSetAnalogGain(h_camera_, config_.analog_gain);

  CameraPlay(h_camera_);
  CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_RGB8);

  int trigger_mode;
  CameraGetTriggerMode(h_camera_, &trigger_mode);
  std::cout << "Confirming trigger_mode: " << trigger_mode << std::endl;

  CameraSetCallbackFunction(h_camera_, &CameraCallbackFunction, this, nullptr);

  for (int i = 0; i < 3; i++) {
    stamped_img_buf_[i].image = cv::Mat(config_.image_size, CV_8UC3, config_.image_buffers[i]);
    stamped_img_buf_[i].id = i;
  }
  triple_buffer_ = std::make_unique<TripleBuffer<StampedImage>>(stamped_img_buf_);
  receive_thread_ = std::jthread(&MVCamera::receive_thread, this);
}

MVCamera::~MVCamera()
{
  CameraStop(h_camera_);
  CameraUnInit(h_camera_);
  shutdown_ = true;
  triple_buffer_->producer_commit();
}

void MVCamera::trigger_callback(
  CameraHandle hCamera, BYTE * pFrameBuffer, tSdkFrameHead * pFrameHead)
{
  static auto starting_time = std::chrono::system_clock::now();
  static int frame_count = 0;
  namespace chrono = std::chrono;
  auto start_time = chrono::system_clock::now();
  auto stamped_image = triple_buffer_->get_producer_buffer();
  CameraImageProcess(hCamera, pFrameBuffer, stamped_image->image.data, pFrameHead);
  stamped_image->time_stamp = start_time;
  triple_buffer_->producer_commit();
  frame_count++;
  if (frame_count == 100) {
    auto cur_time = chrono::system_clock::now();
    std::cout << "Producer FPS: "
              << 100 / (chrono::duration<double>(cur_time - starting_time).count()) << std::endl;
    starting_time = cur_time;
    frame_count = 0;
  }
}

void MVCamera::receive_thread()
{
  namespace chrono = std::chrono;
  int frame_count = 0;
  auto starting_time = chrono::system_clock::now();
  while (!shutdown_) {
    auto stamped_image = triple_buffer_->get_consumer_buffer();
    camera_callback_(*stamped_image);
    frame_count++;
    if (frame_count == 100) {
      auto cur_time = chrono::system_clock::now();
      std::cout << "Consumer FPS: "
                << 100 / (chrono::duration<double>(cur_time - starting_time).count()) << std::endl;
      starting_time = cur_time;
      frame_count = 0;
    }
  }
}

}  // namespace irmv_detection