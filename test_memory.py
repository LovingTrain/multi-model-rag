# 文件名: stress_test_refactored.py
import os
import time
import argparse
from typing import List, Dict, Tuple, Union
import numpy as np

import threading
from pynvml import (
    nvmlInit, 
    nvmlShutdown, 
    nvmlDeviceGetHandleByIndex, 
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)
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
# -------------------------------------------------------------------
# 1. 核心组件定义 (已解耦)
# -------------------------------------------------------------------
from stress_test import EmbeddingComponent, SearchComponent,measure_embedding_only,\
      measure_search_only, measure_end2end,expand_batch,make_search_only_queries,parse_args

def run_benchmark_memory(
    target_component: Union[EmbeddingComponent, SearchComponent, Tuple[EmbeddingComponent, SearchComponent]],
    mode: str,
    base_queries: List[str], # 这里仍然是字符串列表，可以是文本或路径
    query_type: str,
    batch_sizes: List[int],
    repeats: int,
    warmup: int,
    search_only_source: str = "encode",
    image_seed_path: str = None # 传递图片种子路径
) -> List[Dict]:
    results = []
    
    dim = 768
    if mode == 'search': dim = target_component.dim
    elif mode == 'end2end': dim = target_component[1].dim

    for bsz in batch_sizes:
        if bsz == 0: continue
        totals = []
        
        if mode in ("end2end", "embedding"):
            queries = expand_batch(base_queries, bsz)

        # 预热
        for _ in range(warmup):
            if mode == "end2end":
                measure_end2end(target_component[0], target_component[1], queries, query_type)
            elif mode == "embedding":
                measure_embedding_only(target_component, queries, query_type)
            else: # search
                qv = make_search_only_queries(dim, bsz, search_only_source, image_seed_path=image_seed_path)
                measure_search_only(target_component, qv)
            
        # 正式重复
        with GPUProfiler(device_id=0) as gpu_profiler:
            for _ in range(repeats):
                if mode == "end2end":
                    total = measure_end2end(target_component[0], target_component[1], queries, query_type)
                elif mode == "embedding":
                    total = measure_embedding_only(target_component, queries, query_type)
                else: # search
                    embedding_comp_for_query_gen = target_component[0] if mode == 'end2end' else None
                    qv = make_search_only_queries(dim, bsz, search_only_source, embedding_comp=embedding_comp_for_query_gen, image_seed_path=image_seed_path)
                    total = measure_search_only(target_component, qv)
        # 获取 GPU 统计数据
        gpu_stats = gpu_profiler.get_stats()
        
        rec = {
            "mode": mode, 
            "batch_size": bsz,
            "avg_gpu_util": gpu_stats['avg_gpu_util_%'],
            "max_gpu_util": gpu_stats['max_gpu_util_%'],
            "avg_vram_used_gb": gpu_stats['avg_vram_used_mb'], # 键名来自Profiler，但值是GB
            "max_vram_used_gb": gpu_stats['max_vram_used_mb'], # 键名来自Profiler，但值是GB
        }
        results.append(rec)
            
        # row = (
        #     f"{rec['mode']:<10} | {rec['batch_size']:>10d} | "
        #     f"{rec['avg_gpu_util']:>11.2f} | {rec['max_gpu_util']:>11.2f} | "
        #     f"{rec['avg_vram_used_gb']:>12.3f} | {rec['max_vram_used_gb']:>13.3f}" # 使用 .3f 以更好地显示GB单位
        # )
        # print(row)
    return results




def main():
    args = parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    # === 主要改动点 4: 根据 query_type 选择基础查询内容 ===
    if args.query_type == "text":
        base_queries = [args.text_query]
        print(f"测试模式: 文本查询, 内容: '{args.text_query}'")
    elif args.query_type == "image":
        base_queries = [args.image_query]
        print(f"测试模式: 图片查询, 路径: '{args.image_query}'")
        if not os.path.exists(args.image_query):
            print(f"错误: 找不到指定的图片文件: {args.image_query}")
            return
    else:
        # 这个分支理论上不会被执行，因为 argparse 的 choices 会拦截
        print(f"错误: 不支持的 query_type '{args.query_type}'")
        return

    # --- 按需初始化组件 ---
    embedding_comp = None
    search_comp = None
    target_component = None

    print("-" * 40)
    print(f"开始执行模式: {args.mode.upper()}")
    print("-" * 40)

    if args.mode in ["embedding", "end2end"]:
        embedding_comp = EmbeddingComponent(args.model_path)
    
    if args.mode in ["search", "end2end"]:
        if args.search_only_source == 'encode' and not embedding_comp:
            print("[注意] search-only 模式使用 'encode' 源，需要临时加载 embedding 模型...")
            embedding_comp = EmbeddingComponent(args.model_path)
        search_comp = SearchComponent(args.embedding_file, args.metadata_file)

    if args.mode == "embedding":
        target_component = embedding_comp
    elif args.mode == "search":
        target_component = search_comp
    elif args.mode == "end2end":
        target_component = (embedding_comp, search_comp)
    
    if target_component is None:
        raise ValueError(f"模式 '{args.mode}' 初始化失败，无有效组件。")
    
    # --- 运行基准测试 ---
    results = run_benchmark_memory(
        target_component=target_component,
        mode=args.mode,
        base_queries=base_queries,
        query_type=args.query_type,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        warmup=args.warmup,
          search_only_source=args.search_only_source,
        image_seed_path=args.image_query if args.query_type == 'image' else None
    )
    # <<< --- 重点修改: 格式化输出 (已移除延迟列，更新单位为GB) --- >>>
    print("\n" + "="*84)
    print(" " * 30 + "GPU 资源占用测试结果")
    print("="*84)
    
    # 打印表头
    header = (
        f"{'Mode':<10} | {'Batch Size':>10} | "
        f"{'Avg GPU(%)':>11} | {'Peak GPU(%)':>11} | {'Avg VRAM(GB)':>12} | {'Peak VRAM(GB)':>13}"
    )
    print(header)
    print("-" * 84)

    # 打印每一行数据
    for rec in results:
        row = (
            f"{rec['mode']:<10} | {rec['batch_size']:>10d} | "
            f"{rec['avg_gpu_util']:>11.2f} | {rec['max_gpu_util']:>11.2f} | "
            f"{rec['avg_vram_used_gb']:>12.3f} | {rec['max_vram_used_gb']:>13.3f}" # 使用 .3f 以更好地显示GB单位
        )
        print(row)
        
    print("-" * 84)

if __name__ == "__main__":
    main()
