#pragma once

#include <iostream>
#include "NvInfer.h"

using namespace nvinfer1;

namespace irmv_detection
{
  class Logger : public ILogger
  {
    public:
      void log(Severity severity, const char* msg) noexcept override
      {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
          std::cout << msg << std::endl;
      }
  };

}
