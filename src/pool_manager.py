import logging
from concurrent.futures import ProcessPoolExecutor
from .worker_runtime import init_searcher, warm_up
import multiprocessing as mp

ctx = mp.get_context("spawn")

# 每类一池：键是 search_type
_pools: dict[str, ProcessPoolExecutor] = {}
_worker_counts = {}


def init_pools(
    cpu_workers: int = 2,
    gpu_workers: int = 1,
    hybrid_workers: int = 1,
    hybrid_ivf_workers: int = 1,
):
    """
    在应用启动时调用，一次性创建并预热各类型进程池。
    """
    logging.info(
        f"Start init pool workers: cpu={cpu_workers} gpu={gpu_workers} hybrid={hybrid_workers} hybrid_ivf={hybrid_ivf_workers}"
    )
    global _pools, _worker_counts
    if _pools:
        return  # 已初始化

    if cpu_workers > 0:
        _pools["cpu"] = ProcessPoolExecutor(
            max_workers=cpu_workers,
            initializer=init_searcher,
            initargs=("cpu",),
        )
        _worker_counts["cpu"] = cpu_workers

    if gpu_workers > 0:
        _pools["gpu"] = ProcessPoolExecutor(
            max_workers=gpu_workers,
            mp_context=ctx,
            initializer=init_searcher,
            initargs=("gpu",),
        )
        _worker_counts["gpu"] = gpu_workers

    if hybrid_workers > 0:
        _pools["hybrid"] = ProcessPoolExecutor(
            max_workers=hybrid_workers,
            mp_context=ctx,
            initializer=init_searcher,
            initargs=("hybrid",),
        )
        _worker_counts["hybrid"] = hybrid_workers

    if hybrid_ivf_workers > 0:
        _pools["hybrid_ivf"] = ProcessPoolExecutor(
            max_workers=hybrid_ivf_workers,
            mp_context=ctx,
            initializer=init_searcher,
            initargs=("hybrid_ivf",),
        )
        _worker_counts["hybrid_ivf"] = hybrid_ivf_workers

    # 预热所有进程池（提交 N 个空任务）
    for kind, pool in _pools.items():
        count = _worker_counts[kind]
        print(f"预热 {kind} 池...")
        futures = [pool.submit(warm_up) for _ in range(count)]
        for f in futures:
            f.result()  # 等待任务完成


def get_pool(type: str) -> ProcessPoolExecutor:
    pool = _pools.get(type)
    if pool is None:
        raise ValueError(f"pool for type '{type}' not initialized")
    return pool


def shutdown_pools(wait: bool = True, cancel_futures: bool = False):
    """
    在应用关闭时调用，优雅关停所有进程池。
    """
    global _pools
    for pool in _pools.values():
        pool.shutdown(wait=wait, cancel_futures=cancel_futures)
    _pools.clear()
