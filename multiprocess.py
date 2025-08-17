import time
import multiprocessing
from src.encode_mode import EmbeddingMode
from src.search_faiss import FaissSearcher
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
        "search_type": "cpu"
    },
    {
        "user_id": "User_B (Image on GPU)",
        "embedding_file": "/mnt/sdc/multi_model_data/data/index-data/another_dataset/all_embeddings.npy",  # 示例：可以指向不同的数据集
        "metadata_file": "/mnt/sdc/multi_model_data/data/index-data/another_dataset/all_metadata.feather",
        "query_type": "image",
        "query_content": "/home/wangjingjing/SimpleRAG/Locally/data/small/ai.png",
        "search_type": "gpu"
    },
    {
        "user_id": "User_C (Text on Hybrid)",
        "embedding_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy",
        "metadata_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather",
        "query_type": "text",
        "query_content": "a cat sitting on a couch",
        "search_type": "hybrid"
    },
    {
        "user_id": "User_D (Image on Hybrid_IVF)",
        "embedding_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy",
        "metadata_file": "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather",
        "query_type": "image",
        "query_content": "/home/wangjingjing/SimpleRAG/Locally/data/small/another_image.png",  # 假设有另一张图
        "search_type": "hybrid_ivf"
    }
]


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
    try:
        # 每个进程创建自己的 encoder 实例
        encoder = EmbeddingMode()

        # 每个进程创建自己的 searcher 和 index 实例
        searcher = FaissSearcher(embedding_path=config["embedding_file"],
                                 metadata_path=config["metadata_file"])

        # 根据任务配置，只初始化需要的索引
        index = None
        search_type = config["search_type"]
        if search_type == "cpu":
            index = searcher.init_cpu_index()
        elif search_type == "gpu":
            index = searcher.init_gpu_index()
        elif search_type == "hybrid":
            index = searcher.init_hybrid_index()
        elif search_type == "hybrid_ivf":
            index = searcher.init_hybrid_index_ivf()
        else:
            raise ValueError(f"不支持的 search_type: {search_type}")

        load_duration = time.time() - start_time
        print(f"[{user_id}] 资源加载完成，耗时: {load_duration:.2f} 秒。")

        # 2. 编码查询
        query_vector = None
        if config["query_type"] == "text":
            query_vector = encoder.encoding_query_text(config["query_content"])
        elif config["query_type"] == "image":
            query_vector = encoder.encoding_query_image(
                config["query_content"])

        # 3. 执行搜索 (这里使用了修正和简化的逻辑)
        print(f"[{user_id}] 正在执行 '{search_type}' 搜索...")
        search_start_time = time.time()

        # 统一调用一个搜索方法，或者根据类型分别调用
        # 假设您的 searcher 类有统一的 search 方法
        # distances, indices = searcher.search(index, query_vector, k=5)
        # 或者根据您的原始代码结构：
        if search_type == "cpu":
            distances, indices = searcher.cpu_search(index, query_vector, k=5)
        elif search_type == "gpu":
            distances, indices = searcher.gpu_search(index, query_vector, k=5)
        elif search_type in ["hybrid", "hybrid_ivf"]:
            distances, indices = searcher.hybrid_search(
                index, query_vector, k=5)

        search_duration = time.time() - search_start_time
        print(f"[{user_id}] 搜索完成，耗时: {search_duration:.2f} 秒。")

        total_duration = time.time() - start_time
        return f"[{user_id}] 任务成功完成，总耗时: {total_duration:.2f} 秒。"

    except Exception as e:
        return f"[{user_id} - PID: {process_id}] 任务失败: {e}"


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

    print("\n" + "="*50)
    print("所有模拟任务已执行完毕。")
    print("="*50)
    for res in results:
        print(res)

    end_time = time.time()
    print(f"\n模拟总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == "__main__":
    # 在使用 CUDA 和 multiprocessing 时，建议设置启动方法为 'spawn'
    # 这能确保子进程有一个干净的、未初始化的CUDA环境
    multiprocessing.set_start_method("spawn", force=True)
    main()
