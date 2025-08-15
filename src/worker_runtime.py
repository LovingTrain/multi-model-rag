import logging
from .searchers import (
    FaissSearcher,
    CPUSearcher,
    GPUSearcher,
    HybridSearcher,
    HybridIVFSearcher,
)
from .encode_mode import EmbeddingMode


FAISS_SEARCHER_CLASSES = {
    "cpu": CPUSearcher,
    "gpu": GPUSearcher,
    "hybrid": HybridSearcher,
    "hybrid_ivf": HybridIVFSearcher,
}

COCO_TRAIN_DIR = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L"
COCO_EMBEDDINGS_PATH = f"{COCO_TRAIN_DIR}/all_embeddings.npy"
COCO_METADATA_PATH = f"{COCO_TRAIN_DIR}/all_metadata.feather"

SEARCHER: FaissSearcher


def init_searcher(type: str):
    global SEARCHER
    searcher_class = FAISS_SEARCHER_CLASSES.get(type)
    if not searcher_class:
        raise ValueError(f"Invalid search type '{type}'")
    SEARCHER = searcher_class(
        embedding_path=COCO_EMBEDDINGS_PATH, metadata_path=COCO_METADATA_PATH
    )
    SEARCHER.initialize()
    logging.info(f"Finish init searcher: {type}")


def warm_up():
    return


def process_request(query_type: str, query_content: str, top_k: int):
    """
    子进程执行的任务函数：使用进程内全局 SEARCHER 处理请求。
    """
    if SEARCHER is None:
        raise RuntimeError("SEARCHER not initialized. Did you set initializer?")
    if query_type not in ["text", "image"]:
        return None
    encoder = EmbeddingMode()
    query_vector = (
        encoder.encoding_query_text(query_content)
        if query_type == "text"
        else encoder.encoding_query_image(query_content)
    )
    results = SEARCHER.search(query_vector, k=top_k)
    # 你也可以在这里统一封装返回格式/打日志
    return results
