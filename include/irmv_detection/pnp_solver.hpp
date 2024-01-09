#pragma once

#include <array>
#include <vector>

#include <opencv2/opencv.hpp>

#include "irmv_detection/armor.hpp"

namespace irmv_detection
{
class PnPSolver
{
public:
  PnPSolver(
    const std::array<double, 9> & camera_matrix,
    const std::vector<double> & distortion_coefficients);

  // Get 3d position
  bool solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec) const;

  // Calculate the distance between armor center and image center
  float calculateDistanceToCenter(const cv::Point2f & image_point);

private:
  cv::Mat camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
  cv::Mat dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);

  // Unit: mm
  static constexpr float SMALL_ARMOR_WIDTH = 135;
  static constexpr float SMALL_ARMOR_HEIGHT = 55;
  static constexpr float LARGE_ARMOR_WIDTH = 225;
  static constexpr float LARGE_ARMOR_HEIGHT = 55;

  // Four vertices of armor in 3d
  std::vector<cv::Point3d> small_armor_points_;
  std::vector<cv::Point3d> large_armor_points_;
};

}  // namespace irmv_detection
