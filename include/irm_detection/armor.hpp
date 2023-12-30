#pragma once

#include "opencv2/opencv.hpp"

namespace irm_detection
{
  enum class ArmorClass
  {
    B1, B2, B3, B4, B5, BO, BS,
    R1, R2, R3, R4, R5, RO, RS,
    UNKNOWN
  };

  enum class ArmorSize
  {
    SMALL, LARGE, UNKNOWN
  };

  struct Light : public cv::RotatedRect
  {
    Light() = default;
    explicit Light(cv::RotatedRect box) : cv::RotatedRect(box)
    {
      cv::Point2f p[4];
      box.points(p);
      std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
      top = (p[0] + p[1]) / 2;
      bottom = (p[2] + p[3]) / 2;

      length = cv::norm(top - bottom);
      width = cv::norm(p[0] - p[1]);

      tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
      tilt_angle = tilt_angle / CV_PI * 180;
    }

    bool is_light(float min_ratio, float max_ratio, float max_angle) const
    {
      float ratio = width / length;
      bool ratio_ok = min_ratio < ratio && ratio < max_ratio;
      bool angle_ok = tilt_angle < max_angle;

      return ratio_ok && angle_ok;
    }

    void offset_bbox(float min_x, float min_y)
    {
      center.x += min_x;
      center.y += min_y;
      top.x += min_x;
      top.y += min_y;
      bottom.x += min_x;
      bottom.y += min_y;
    }

    cv::Point2f top, bottom;
    double length;
    double width;
    float tilt_angle;
  };

  struct Armor
  {
    Armor() = default;
    Armor(const Light & l1, const Light & l2)
    {
      if (l1.center.x < l2.center.x) {
        left_light = l1, right_light = l2;
      } else {
        left_light = l2, right_light = l1;
      }
      center = (left_light.center + right_light.center) / 2;
    }

    Light left_light, right_light;

    ArmorSize size;
    ArmorClass armor_class;
    float confidence;
    cv::Point2f center;
  };
}