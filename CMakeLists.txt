cmake_minimum_required(VERSION 3.8)
project(irm_detection)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add -pg to CMAKE_CXX_FLAGS_DEBUG to enable profiling
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# importing CUDA
set(CUDA_MIN_VERSION "11.4")
find_package(CUDA REQUIRED)
if( ${CUDA_VERSION} GREATER_EQUAL 11.8)
  set(CMAKE_CUDA_ARCHITECTURES "89;87;86;80;75;72;70;61;60")
else()
  set(CMAKE_CUDA_ARCHITECTURES "87;86;80;75;72;70;61;60")
endif()

# irm_detection
ament_auto_add_library(irm_detection SHARED
  src/irm_detector.cpp
  src/yolo_engine.cpp
  src/pnp_solver.cpp
)
ament_target_dependencies(irm_detection
  rclcpp
  rclcpp_components
  sensor_msgs
  OpenCV
  cv_bridge
  image_transport
  auto_aim_interfaces
  tf2_geometry_msgs
  tf2
)
target_link_libraries(irm_detection
  nvinfer
  nvonnxparser
  nvinfer_plugin
  ${CUDA_LIBRARIES}
  ${CUDA_npp_LIBRARY}
  ${CUDA_nppc_LIBRARY}
  ${CUDA_nppi_LIBRARY}
)
target_include_directories(irm_detection PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
)
rclcpp_components_register_node(irm_detection 
  PLUGIN "irm_detection::IrmDetector"
  EXECUTABLE irm_detector
)

if(BUILD_TESTING)
  # find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  # set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  # set(ament_cmake_cpplint_FOUND TRUE)
  # ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(irm_detection_yolo_test
    test/yolo_test.cpp
  )
  target_include_directories(irm_detection_yolo_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CUDA_INCLUDE_DIRS}
  )
  ament_target_dependencies(irm_detection_yolo_test
    OpenCV
  )
  target_link_libraries(irm_detection_yolo_test
    irm_detection
  )
endif()

ament_auto_package()
