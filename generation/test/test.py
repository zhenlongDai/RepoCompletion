import os
import torch

# 检查 CUDA 是否可用
def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return False
    else:
        print(f"CUDA is available. {torch.cuda.device_count()} device(s) found.")
        return True

# 获取 CUDA_VISIBLE_DEVICES 设置
def get_cuda_visible_devices():
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
        devices = cuda_visible_devices.split(',')
        print(f"Devices accessible: {devices}")
    else:
        print("CUDA_VISIBLE_DEVICES is not set, all available devices should be accessible.")

# 测试 GPU 设备信息
def test_gpu_devices():
    # 检查可用的 CUDA 设备
    if check_cuda():
        get_cuda_visible_devices()
        
        # 打印每个设备的物理 ID 和名称
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            print(f"Device {i} -> {torch.cuda.get_device_name(i)}")
            print(f"Device {i} Physical ID: {torch.cuda.device(i)}")

# 运行测试
test_gpu_devices()
