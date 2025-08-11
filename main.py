from src.dataset import convert_dataset_to_folder_format, process_embeddings
from src.search_faiss import FaissSearcher
from src.encode_mode import EmbeddingMode
import concurrent.futures
import time

def main():
    # 可以在这里调用上面导入的函数或类
    # 例如:
    convert_dataset_to_folder_format(...)
    process_embeddings(...)

    searcher = FaissSearcher(embedding_path="/path/to/all_embeddings.npy",
                             metadata_path="/path/to/all_metadata.feather")
    cpu_index = searcher.init_cpu_index()
    gpu_index = searcher.init_gpu_index()
    encoder = EmbeddingMode()
    print("索引已创建，开始并行处理查询...")


    queries = [
        {"type": "text", "content": "a dog playing on the beach"},
        {"type": "image", "content": "/home/wangjingjing/SimpleRAG/Multi-Modal-RAG-Pipeline-on-Images-and-Text-Locally/data/small/space.png"},
        {"type": "text", "content": "sunset over mountains"},
        {"type": "image", "content": "/home/wangjingjing/SimpleRAG/Multi-Modal-RAG-Pipeline-on-Images-and-Text-Locally/data/small/city.png"}
    ]
    # 定义一个函数，用于处理单个查询


    def process_single_query(query):
        if query["type"] == "text":
            query_vector = encoder.encoding_query_text(query["content"])
            print(f"\n[并行查询-文本] {query['content']}")
            distances, indices = searcher.cpu_search(cpu_index, query_vector)

        elif query["type"] == "image":
            query_vector = encoder.encoding_query_image(query["content"])
            print(f"\n[并行查询-图片] {query['content']}")
            distances, indices = searcher.gpu_search(gpu_index, query_vector)


    # 使用线程池并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_query, q) for q in queries]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
