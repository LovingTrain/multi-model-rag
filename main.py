from src.dataset import convert_dataset_to_folder_format, process_embeddings
from src.search_faiss import FaissSearcher
from src.encode_mode import EmbeddingMode
import concurrent.futures
import time
def get_index():
    EMBEDDING_FILE = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy"
    METADATA_FILE = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather"
    searcher = FaissSearcher(embedding_path = EMBEDDING_FILE,
                                metadata_path = METADATA_FILE)
    
    cpu_index = searcher.init_cpu_index()
    gpu_index = searcher.init_gpu_index()
    hybrid_index = searcher.init_hybrid_index()
    hybrid_ivf_index = searcher.init_hybrid_index_ivf()
    return cpu_index,gpu_index,hybrid_index,hybrid_ivf_index
def main(encoder,cpu_index,gpu_index,hybrid_index,hybrid_ivf_index):
    start= time.time()
    queries = [
        {"type": "text","search_type": "cpu", "content": "a dog playing on the beach"},
        {"type": "image","search_type": "gpu", "content": "/home/wangjingjing/SimpleRAG/Locally/data/small/ai.png"},
        {"type": "text","search_type": "hybrid", "content": "a dog playing on the beach"},
        {"type": "image","search_type": "hybrid_ivf", "content": "/home/wangjingjing/SimpleRAG/Locally/data/small/ai.png"}
    ]
    # 定义一个函数，用于处理单个查询
    
    
    def process_single_query(query):
        if query["type"] == "text":
            query_vector = encoder.encoding_query_text(query["content"])
            print(f"\n[并行查询-文本] {query['content']}")
            if query["search_type"] !="cpu" :
                distances, indices = searcher.cpu_search(hybrid_index, query_vector,k=5)
            else :
                distances, indices = searcher.hybrid_search(cpu_index, query_vector,k=5)
    
        elif query["type"] == "image":
            query_vector = encoder.encoding_query_image(query["content"])
            print(f"\n[并行查询-图片] {query['content']}")
            
            if query["search_type"] =="gpu" :
                distances, indices = searcher.gpu_search(gpu_index, query_vector,k=5)
            else :
                distances, indices = searcher.hybrid_search (hybrid_ivf_index, query_vector,k=5)
    
    # 线程池之前，先验证是否有问题
    for q in queries:
        process_single_query(q)
    # # 使用线程池并行执行.执行失败是看不出来的
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = [executor.submit(process_single_query, q) for q in queries]
    #     concurrent.futures.wait(futures)
    end= time.time()
    print(f"cost time {end-start} s")


if __name__ == "__main__":
    cpu_index,gpu_index,hybrid_index,hybrid_ivf_index = get_index()
    encoder = EmbeddingMode()
    main()
