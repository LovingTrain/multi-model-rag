import time
import multiprocessing
from src.encode_mode import EmbeddingMode
from src.search_faiss import FaissSearcher


class PipelineRAG():
    def __init__(self, embedding_file_path, metadata_file_path, embedding_model_path, search_type):

        self.embedding_model = EmbeddingMode(model_path=embedding_model_path)
        self.searcher = FaissSearcher(embedding_path=embedding_file_path,
                                      metadata_path=metadata_file_path)
        self.index = self.searcher.init_gpu_index()  # 多态
        print("init finished ~ ")

        if search_type == "cpu":
            index = self.searcher.init_cpu_index()
        elif search_type == "gpu":
            index = self.searcher.init_gpu_index()
        elif search_type == "hybrid":
            index = self.searcher.init_hybrid_index()
        elif search_type == "hybrid_ivf":
            index = self.searcher.init_hybrid_index_ivf()
        else:
            raise ValueError(f"不支持的 search_type: {search_type}")
        self.index = index
        self.search_type = search_type

    def batch_rag(self, query, query_type):
        start = time.time()
        if query_type == "text":
            query_vector = self.encoder.encoding_query_text(query)
        elif query_type == "image":
            query_vector = self.encoder.encoding_query_batch_images(
                query)
        if self.search_type == "cpu":
            _, _ = self.searcher.cpu_search()
        elif self.search_type == "gpu":
            _, _ = self.searcher.gpu_search()
        elif self.search_type == "hybrid":
            _, _ = self.searcher.cpu_search()
        elif self.search_type == "hybrid_ivf":
            _, _ = self.searcher.cpu_search()
        else:
            raise ValueError(f"不支持的 search_type: {self.search_type}")
        end = time.time()

        return (end - start)/query.shape[0]

# --- 步骤3: 修改主函数以启动多进程 ---


def main():
    ef = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_embeddings.npy"
    mf = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L/all_metadata.feather"
    em = '/mnt/sdc/multi_model_data/weight/ViT-L-14.pt'
    query = ["a dog playing on the beach"]
    pipelineA = PipelineRAG(ef, mf, em, 'gpu')
    cost_time = []
    for i in range(1, 17):
        query_lists = query * i
        res = pipelineA.batch_rag(query_lists, 'text')
        cost_time.append(res)
    #  plt save png


if __name__ == "__main__":

    main()
