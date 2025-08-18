import logging
from mcp.server.fastmcp import FastMCP

from src.pool_manager import init_pools, shutdown_pools, get_pool
from src.worker_runtime import process_request

logging.basicConfig(level=logging.DEBUG)

m = FastMCP("multimodal_knowledge", host="0.0.0.0", port=9200)


@m.tool()
def search_multimodal_knowledge(
    search_type: str, query_type: str, query_content: str, top_k: int
):
    """
    在多模态知识库中进行相似度检索。

    参数:
        search_type (str): 指定使用的检索引擎类型。
            可选值:
              - "cpu": 仅使用 CPU 检索
              - "gpu": 仅使用 GPU 检索
              - "hybrid": 推荐，自动结合 CPU/GPU
              - "hybrid_ivf": 使用 IVF 聚类加速的混合检索
            推荐使用 "hybrid"。

        query_type (str): 查询内容的类型。
            可选值:
              - "text": 文本查询，将 query_content 视为自然语言文本
              - "image": 图像查询，将 query_content 视为图像路径

        query_content (str): 查询的具体内容。
            - 当 query_type="text" 时，此处应为一段文本
            - 当 query_type="image" 时，此处应为图片文件路径

        top_k (int): 返回相似度最高的前 k 条结果。

    返回:
        list: 相似度最高的检索结果列表（含文本或图像信息）。
    """
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
