# irmv_detection
A hardware-accelerated (for NVIDIA GPUs) armor detection ROS2 node for RoboMaster competition.

## Requirements
- ROS2 Humble
- JetPack 6

## Performance Benchmark
| Model | Input Size | Inference Time* <br> (Jetson Orin Nano 8GB) | Inference Time* <br> (RTX 3060 Laptop 115W) |
| :---: | :---: | :---: | :---: |
| YOLOv8n | 640x640 | - | ~ 4-5ms |
| YOLOv8n <br> (Shufflenet backbone)** | 640x640 | ~5-6ms | ~ 3-4ms |

*: The inference time includes the time for image preprocessing and postprocessing (NMS).
**: https://github.com/zRzRzRzRzRzRzR/YOLO-of-RoboMaster-Keypoints-Detection-2023

Note that due to use of Unified Memory, the performance on dGPU devices is not ideal. However this enables true zero-copy on Tegra (e.g. Jetson) devices, potentially reducing the latency of image transfer between CPU and GPU.

## Acceleration Techniques
Below are techniques used or planned to use in this package to accelerate the inference process (some of them are still under development):
- [x] TensorRT (FP16)
- [x] GPU Postprocessing (with EfficientNMS plugin from TensorRT)
- [x] GPU Preprocessing (with NPP from CUDA Toolkit)
- [x] CUDA Streams
- [x] CUDA Graphs
- [ ] Custom CUDA kernel for resizing
- [ ] INT8 Quantization

## Troubleshooting

1. Model inference taking much longer than expected.
   By default, NVIDIA GPUs will run at a low clock rate when the GPU is idle. This is not ideal for competition environment. To solve this problem, you can use the following command (on Jetson only) to set the GPU clock rate to the maximum value:
    ```bash
    sudo jetson_clocks
    ```
    If you're using a dGPU (e.g. RTX 3060), you can use the following command to force the GPU clock rate to the maximum value:
    ```bash
    sudo nvidia-smi -lgc <MAX CLOCK RATE>
    ```
    Use `nvidia-smi -q -d SUPPORTED_CLOCKS` to check the supported clock rates.
    Fore more info on this: [https://stackoverflow.com/questions/64701751/can-i-fix-my-gpu-clock-rate-to-ensure-consistent-profiling-results](https://stackoverflow.com/questions/64701751/can-i-fix-my-gpu-clock-rate-to-ensure-consistent-profiling-results)