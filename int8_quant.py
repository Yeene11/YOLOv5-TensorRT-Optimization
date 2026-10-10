import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import numpy as np
import cv2

# --- 配置区 ---
ONNX_MODEL_PATH = "yolov5s.onnx"
ENGINE_PATH = "yolov5s_int8.engine"
CALIB_IMG_DIR = "datasets/coco128/images/train2017/"
INPUT_SHAPE = (1, 3, 640, 640) # YOLOv5的输入尺寸
BATCH_SIZE = 1
# --------------

# 1. 定义校准器类
class YOLOv5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, img_dir, batch_size, input_shape, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')][:100] # 取前100张即可
        self.current_index = 0
        # 分配设备内存
        self.device_input = cuda.mem_alloc(trt.volume(input_shape) * batch_size * np.dtype(np.float32).itemsize)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.img_paths):
            return None
        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.img_paths[self.current_index + i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_imgs.append(img)
        self.current_index += self.batch_size
        batch_data = np.ascontiguousarray(np.stack(batch_imgs))
        cuda.memcpy_htod(self.device_input, batch_data)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# 2. 构建引擎
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
config.set_flag(trt.BuilderFlag.INT8)

# 加载ONNX
print("Loading ONNX model...")
with open(ONNX_MODEL_PATH, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed to parse ONNX.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

# 设置INT8校准器
cache_file = "yolov5s.calibration"
calibrator = YOLOv5EntropyCalibrator(CALIB_IMG_DIR, BATCH_SIZE, INPUT_SHAPE, cache_file)
config.int8_calibrator = calibrator

# 构建引擎
print("Building INT8 engine...")
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    print("Failed to build engine.")
    exit()

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_engine)
print(f"Successfully saved INT8 engine to {ENGINE_PATH}")