#include <vector>

#include "irmv_detection/pnp_solver.hpp"

namespace irmv_detection
{
PnPSolver::PnPSolver(
  const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs)
{
  for (int i = 0; i < 9; i++) {
    camera_matrix_.at<double>(i / 3, i % 3) = camera_matrix[i];
  }
  for (int i = 0; i < 5; i++) {
    dist_coeffs_.at<double>(0, i) = dist_coeffs[i];
  }

  // Unit: m
  constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Start from bottom left in clockwise order
  // Model coordinate: x forward, y left, z up
  small_armor_points_.emplace_back(0, small_half_y, -small_half_z);
  small_armor_points_.emplace_back(0, small_half_y, small_half_z);
  small_armor_points_.emplace_back(0, -small_half_y, small_half_z);
  small_armor_points_.emplace_back(0, -small_half_y, -small_half_z);

  large_armor_points_.emplace_back(0, large_half_y, -large_half_z);
  large_armor_points_.emplace_back(0, large_half_y, large_half_z);
  large_armor_points_.emplace_back(0, -large_half_y, large_half_z);
  large_armor_points_.emplace_back(0, -large_half_y, -large_half_z);
}

bool PnPSolver::solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec) const
{
  std::vector<cv::Point2f> image_armor_points;

  // Fill in image points
  image_armor_points.emplace_back(armor.left_light.bottom);
  image_armor_points.emplace_back(armor.left_light.top);
  image_armor_points.emplace_back(armor.right_light.top);
  image_armor_points.emplace_back(armor.right_light.bottom);

  // Solve pnp
  bool small_armor = true;
  auto object_points = small_armor ? small_armor_points_ : large_armor_points_;
  return cv::solvePnP(
    object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
}

float PnPSolver::calculateDistanceToCenter(const cv::Point2f & image_point)
{
  float cx = camera_matrix_.at<float>(0, 2);
  float cy = camera_matrix_.at<float>(1, 2);
  return cv::norm<float>(image_point - cv::Point2f(cx, cy));
}

}  // namespace irmv_detection
