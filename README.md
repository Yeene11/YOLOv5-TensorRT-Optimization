YOLOv5 TensorRT 推理加速与 INT8 量化实践



📌 项目简介

本项目针对 YOLOv5 目标检测模型，实现了从 PyTorch 到 ONNX 再到 TensorRT 引擎的完整端到端部署流程。为了解决官方脚本在 INT8 量化时的静默失败问题，本项目手写了基于 `IInt8EntropyCalibrator2` 的熵校准器，成功生成 9MB 的 INT8 推理引擎，并显著降低了推理延迟。



📊 性能基准测试 (Benchmark)

测试环境：NVIDIA GeForce RTX 2080 Ti, TensorRT 8.5.1.7, CUDA 11.8



| 模型版本 | 精度 | 模型体积 | 推理延迟 (单帧) | 加速比 |

| PyTorch | FP32 | \~14 MB | 13.2 ms | 1x (基线) |

| TensorRT | FP16 | 35 MB | 3.6 ms | 3.6x |

| TensorRT | INT8 | 9 MB | 1.4 ms | 9.4x |



🛠️ 技术栈

部署框架：TensorRT 8.5, ONNX, PyTorch

编程语言：Python (开发与校准)

硬件加速：CUDA, cuDNN, PyCUDA

图像处理：OpenCV



🚀 核心实现与优化

1\.  模型中间格式转换：利用 `export.py` 将 YOLOv5 权重导出为 ONNX，并强制指定 `--opset 12` 以兼容老版本 TensorRT。

2\.  自定义 INT8 量化：官方 `export.py` 脚本在遇到 cuDNN 版本冲突时会静默退回 FP32。为此，本项目编写了 `int8\_quant.py`，直接调用 TensorRT Python API，继承 `IInt8EntropyCalibrator2` 实现了一套自定义的熵校准流程。

3\.  数据校准管线：利用 `PyCUDA` 直接在 GPU 上分配显存，使用 OpenCV 对 `coco128` 数据集进行预处理（Resize、归一化），通过 `cudaMemcpy` 完成 Host-to-Device 的数据搬运，配合 TensorRT 完成 KL 散度校准。



🕳️ 踩坑与工程排查记录 

本项目在部署过程中解决了一系列底层工程兼容性问题：

cuDNN 版本冲突：系统自带 cuDNN 8.2.0，而 TensorRT 链接的是 8.6.0，导致 INT8 校准失败。通过绕过官方脚本，改用底层 API 解决。

Numpy 版本不兼容：报错 `AttributeError: module 'numpy' has no attribute 'bool'`，通过降级 `numpy==1.23.5` 解决。

PyCUDA 语法报错：Python 3.8 无法识别新版 PyCUDA 的 `str | Sequence\[str]` 类型标注，通过安装历史兼容版本 `pycuda==2022.2.2` 解决。

环境与依赖网络问题：利用 AutoDL 的“无卡开机”模式挂机，解决了 1.3GB NVIDIA 依赖库下载极慢（250KB/s）的问题。



💡 未来优化方向

将 Python 部署链路重写为 C++ 版本，进一步消除解释器开销。

引入 CUDA Kernel 重写后处理（NMS），减少 Host-Device 数据拷贝。

尝试使用多线程/异步推理提升吞吐量（Throughput）。

