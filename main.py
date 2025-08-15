import multiprocessing
import time

from src.encode_mode import EmbeddingMode
from src.searchers import (
    CPUSearcher,
    FaissSearcher,
    GPUSearcher,
    HybridIVFSearcher,
    HybridSearcher,
)

# 假设您的其他模块 (dataset, etc.) 都在正确的路径下

# --- 步骤1: 定义详细的、可配置的模拟任务 ---
# 每个字典代表一个独立的用户/场景
# 注意：这里的路径和配置需要根据您的实际情况进行修改
SIMULATION_CONFIGS = [
    {
        "user_id": "User_A (Text on CPU)",
        "embedding_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy",
        "metadata_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather",
        "query_type": "text",
        "query_content": "a dog playing on the beach",
        "search_type": "cpu",
    },
    {
        "user_id": "User_B (Image on GPU)",
        "embedding_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy",  # 示例：可以指向不同的数据集
        "metadata_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather",
        "query_type": "image",
        "query_content": "/home/wangjingjing/SimpleRAG/Locally/data/small/ai.png",
        "search_type": "gpu",
    },
    {
        "user_id": "User_C (Text on Hybrid)",
        "embedding_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy",
        "metadata_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather",
        "query_type": "text",
        "query_content": "a cat sitting on a couch",
        "search_type": "hybrid",
    },
    {
        "user_id": "User_D (Image on Hybrid_IVF)",
        "embedding_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy",
        "metadata_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather",
        "query_type": "image",
        "query_content": "/home/wangjingjing/SimpleRAG/Locally/data/small/ai.png",  # 假设有另一张图
        "search_type": "hybrid_ivf",
    },
]


COCO_TRAIN_DIR = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L"
COCO_EMBEDDINGS_PATH = f"{COCO_TRAIN_DIR}/all_embeddings.npy"
COCO_METADATA_PATH = f"{COCO_TRAIN_DIR}/all_metadata.feather"


FAISS_SEARCHER_CLASSES = {
    "cpu": CPUSearcher,
    "gpu": GPUSearcher,
    "hybrid": HybridSearcher,
    "hybrid_ivf": HybridIVFSearcher,
}


# --- 步骤2: 创建独立的“工作单元”函数 ---
def simulation_worker(config: dict):
    """
    这个函数在独立的进程中运行，模拟单个用户的完整流程。
    """
    user_id = config["user_id"]
    process_id = multiprocessing.current_process().pid
    print(f"[{user_id} - PID: {process_id}] 开始执行任务...")

    start_time = time.time()

    # 1. 在进程内部独立加载资源
    print(f"[{user_id}] 正在加载模型和索引...")

    # 每个进程创建自己的 encoder 实例
    encoder = EmbeddingMode()

    search_type = config["search_type"]

    # 每个进程创建自己的 searcher 和 index 实例
    searcher: FaissSearcher = FAISS_SEARCHER_CLASSES[search_type](
        embedding_path=config["embedding_file"], metadata_path=config["metadata_file"]
    )
    searcher.initialize()

    load_duration = time.time() - start_time
    print(f"[{user_id}] 资源加载完成，耗时: {load_duration:.2f} 秒。")

    # 2. 编码查询
    query_vector = (
        encoder.encoding_query_text(config["query_content"])
        if config["query_type"] == "text"
        else encoder.encoding_query_image(config["query_content"])
    )

    # 3. 执行搜索 (这里使用了修正和简化的逻辑)
    print(f"[{user_id}] 正在执行 '{search_type}' 搜索...")
    search_start_time = time.time()

    # 统一调用一个搜索方法，或者根据类型分别调用
    # 假设您的 searcher 类有统一的 search 方法
    # distances, indices = searcher.search(index, query_vector, k=5)
    distances, indices = searcher.search(query_vector, k=5)

    search_duration = time.time() - search_start_time
    print(f"[{user_id}] 搜索完成，耗时: {search_duration:.2f} 秒。")

    total_duration = time.time() - start_time
    return f"[{user_id}] 任务成功完成，总耗时: {total_duration:.2f} 秒。"


# --- 步骤3: 修改主函数以启动多进程 ---
def main():
    """
    主函数，负责启动和管理多进程模拟。
    """

    print(f"主进程启动，准备模拟 {len(SIMULATION_CONFIGS)} 个并发用户...")
    start_time = time.time()

    # 创建一个进程池，进程数量等于任务数量
    with multiprocessing.Pool(processes=len(SIMULATION_CONFIGS)) as pool:
        # map 函数会将 SIMULATION_CONFIGS 列表中的每一项作为参数，传递给 simulation_worker 函数
        # 它会阻塞直到所有进程完成
        results = pool.map(simulation_worker, SIMULATION_CONFIGS)

    print("\n" + "=" * 50)
    print("所有模拟任务已执行完毕。")
    print("=" * 50)
    for res in results:
        print(res)

    end_time = time.time()
    print(f"\n模拟总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == "__main__":
    # 在使用 CUDA 和 multiprocessing 时，建议设置启动方法为 'spawn'
    # 这能确保子进程有一个干净的、未初始化的CUDA环境
    multiprocessing.set_start_method("spawn", force=True)
    main()
