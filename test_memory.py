import threading
from pynvml import *
import numpy as np
import time
class GPUProfiler:
    def __init__(self, device_id: int = 0, sampling_interval: float = 0.01):
        """
        :param device_id: 要监控的 GPU 设备索引。
        :param sampling_interval: 采样时间间隔（秒）。
        """
        self.device_id = device_id
        self.sampling_interval = sampling_interval
        self._stop_event = threading.Event()
        self._thread = None
        
        self.util_samples = []  # 存储 GPU 利用率采样值
        self.vram_samples = []  # 存储已用显存采样值 (MB)

    def _monitor_loop(self):
        """后台监控循环，直到停止事件被设置。"""
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(self.device_id)
            
            while not self._stop_event.is_set():
                # 获取 GPU 利用率
                util = nvmlDeviceGetUtilizationRates(handle)
                self.util_samples.append(util.gpu)
                
                # 获取显存信息
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                self.vram_samples.append(mem_info.used / 1024 / 1024/1024)  # 转换为 GB
                
                time.sleep(self.sampling_interval)
        finally:
            nvmlShutdown()

    def __enter__(self):
        """启动后台监控线程。"""
        print("[Profiler] Starting GPU monitoring...")
        self.util_samples.clear()
        self.vram_samples.clear()
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止后台监控线程并等待其结束。"""
        self._stop_event.set()
        self._thread.join()
        print("[Profiler] Stopped GPU monitoring.")

    def get_stats(self) -> dict:
        """
        计算并返回监控期间的平均和峰值统计数据。
        """
        if not self.util_samples or not self.vram_samples:
            return {
                "avg_gpu_util_%": 0, "max_gpu_util_%": 0,
                "avg_vram_used_mb": 0, "max_vram_used_mb": 0
            }
        
        return {
            "avg_gpu_util_%": np.mean(self.util_samples),
            "max_gpu_util_%": np.max(self.util_samples),
            "avg_vram_used_mb": np.mean(self.vram_samples),
            "max_vram_used_mb": np.max(self.vram_samples)
        }
