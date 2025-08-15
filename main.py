import logging
from mcp.server.fastmcp import FastMCP

from src.pool_manager import init_pools, shutdown_pools, get_pool
from src.worker_runtime import process_request

logging.basicConfig(level=logging.DEBUG)

m = FastMCP("multimodal_knowledge", port=9200)


@m.tool()
def search_multimodal_knowledge(
    search_type: str, query_type: str, query_content: str, top_k: int
):
    # 同步阻塞等待结果
    pool = get_pool(search_type)
    return pool.submit(process_request, query_type, query_content, top_k).result(
        timeout=None
    )


if __name__ == "__main__":  # ← 关键：只在主进程里初始化
    # 建议：pool_manager.py 里 GPU 池用 mp.get_context("spawn")
    init_pools(cpu_workers=1, gpu_workers=1, hybrid_workers=1, hybrid_ivf_workers=1)
    try:
        m.run(transport="streamable-http")
    finally:
        shutdown_pools()
