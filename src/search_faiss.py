import os
import torch
import clip
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import glob
import json
from tqdm import tqdm  # 用于显示漂亮的进度条


class FaissSearcher():
    def __init__(self, embedding_path, metadata_path) -> None:

        all_embeddings = np.load(embedding_path)
        self.metadata_df = pd.read_feather(metadata_path)

        self.all_embeddings = all_embeddings
        N = all_embeddings.shape[0]
        self.split_idx = int(0.7 * N)
    # IVF 索引,有聚类

    def init_cpu_index(self):
        d = self.all_embeddings.shape[1]
        nlist = int(4 * np.sqrt(len(self.all_embeddings)))
        quantizer = faiss.IndexFlatL2(d)
        cpu_index_ivfflat = faiss.IndexIVFFlat(
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        # 需要先 train,训练索引 无监督的聚类，将数据划分成若干个分区。
        cpu_index_ivfflat.train(self.all_embeddings)
        cpu_index_ivfflat.add(self.all_embeddings)
        cpu_index_ivfflat.nprobe = 16
        return cpu_index_ivfflat
    # IVF 索引,有聚类

    def init_gpu_index(self):

        cpu_index_ivfflat = self.init_cpu_index()
        # 准备 GPU 资源 (一次性)
        res = faiss.StandardGpuResources()

        gpu_search_index = faiss.index_cpu_to_gpu(
            res, 0, cpu_index_ivfflat)
        gpu_search_index.nprobe = 16

        return gpu_search_index
    #  Flat 索引——没有聚类，也没有 nprobe 的概念

    def init_hybrid_index(self):

        data_cpu = self.all_embeddings[:self.split_idx, :]    # 前 70%
        data_gpu = self.all_embeddings[self.split_idx:, :]    # 后 30%

        d = data_cpu.shape[1]
        #
        hybrid_cpu_index = faiss.IndexFlatL2(d)
        hybrid_cpu_index.add(data_cpu)

        res = faiss.StandardGpuResources()

        gpu_index_cpu_temp = faiss.IndexFlatL2(d)
        gpu_index_cpu_temp.add(data_gpu)

        hybrid_gpu_index = faiss.index_cpu_to_gpu(res, 0, gpu_index_cpu_temp)

        shard_index = faiss.IndexShards(d,
                                        threaded=False,     # 是否多线程
                                        successive_ids=True)  # 是否按顺序合并ID
        shard_index.add_shard(hybrid_cpu_index)
        shard_index.add_shard(hybrid_gpu_index)

        return shard_index

    def init_hybrid_index_ivf(self):
        """
        [优化 4] 创建一个真正混合的 IVF 索引，CPU 和 GPU 上都是 IVF。
        [优化 1] 使用 successive_ids=True 来解决索引管理问题。
        """

        res = faiss.StandardGpuResources()
        d = self.all_embeddings.shape[1]
        nlist = int(4 * np.sqrt(len(self.all_embeddings)))
        # 1. 创建两个独立的 CPU IVF 索引
        quantizer = faiss.IndexFlatL2(d)

        # CPU 部分的索引
        cpu_shard = faiss.IndexIVFFlat(
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        # GPU 部分的索引（临时在 CPU 上创建）
        gpu_shard_cpu_version = faiss.IndexIVFFlat(
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        # 2. 创建一个分片索引，并设置 ID 自动连续
        hybrid_index = faiss.IndexShards(
            d, threaded=False, successive_ids=True)

        # 3. 训练和添加数据
        split_idx = int(0.7 * len(self.all_embeddings))
        data_cpu = self.all_embeddings[:split_idx]
        data_gpu = self.all_embeddings[split_idx:]

        # 训练需要所有数据来获得一个好的全局聚类中心
        print("使用全部数据训练共享的量化器...")
        cpu_shard.train(self.all_embeddings)

        # 将训练好的量化器赋给 GPU 分片
        gpu_shard_cpu_version.quantizer = cpu_shard.quantizer
        gpu_shard_cpu_version.is_trained = True

        print("向 CPU 分片添加数据...")
        cpu_shard.add(data_cpu)
        cpu_shard.nprobe = 16

        print("向 GPU 分片添加数据...")
        gpu_shard_cpu_version.add(data_gpu)

        # 将 GPU 分片从 CPU 转换到 GPU
        gpu_shard = faiss.index_cpu_to_gpu(res, 0, gpu_shard_cpu_version)
        gpu_shard.nprobe = 16

        # 4. 将两个分片添加到主索引中
        hybrid_index.add_shard(cpu_shard)
        hybrid_index.add_shard(gpu_shard)

        return hybrid_index

    def cpu_search(self, cpu_search_index, query_vector, k=5):

        distances, indices = cpu_search_index.search(query_vector, k)
        self.print_search_results(distances, indices)
        # return distances, indices

    def gpu_search(self, gpu_search_index, query_vector, k=5):

        distances, indices = gpu_search_index.search(query_vector, k)
        self.print_search_results(distances, indices)
        # return distances, indices

    def hybrid_search(self, shard_index, query_vector, k=5):

        distances, indices = shard_index.search(query_vector, k)
        self.print_search_results(distances, indices)
        return distances, indices

    def print_search_results(self, distances, indices, k=5):
        for i in range(k):
            idx = indices[0][i]
            filepath = self.metadata_df.iloc[idx]['filepath']
            similarity = distances[0][i]
            print(
                f"  - 相似度: {similarity:.4f}, 路径: {os.path.basename(filepath)}")

    def llm_generation(self, query_text, retrieval_results, k=5):
        '''
        带完善用Qwen generation,API ,还不行，图片没发送进去。
        '''
        distances, indices = retrieval_results
        # 2. 从检索结果中提取上下文信息
        context_items = []
        for i in range(min(k, len(indices[0]))):
            idx = indices[0][i]
            if idx == -1:
                continue
            filepath = self.metadata_df.iloc[idx]['filepath']
            similarity = distances[0][i]
            # 我们只提取文件名，因为完整路径可能太长且包含不相关信息
            filename = os.path.basename(filepath)
            context_items.append(f"- 文件名: {filename} (相似度: {similarity:.4f})")

        context_str = "\n".join(context_items)
        # 3. 精心构建 Prompt
        # System Prompt 定义了模型的角色和行为准则
        system_prompt = "你是一个智能的图片内容分析助手。你的任务是基于用户的问题和系统提供的相关图片文件列表，对这些图片的主题、内容或共同点进行推断和总结，并用自然、流畅的中文回答用户。"

        # User Prompt 包含了用户的请求和我们提供的上下文
        user_prompt = f"""
                    用户的原始问题是："{query_text}"

                    根据这个问题，我为您检索到了以下最相关的图片文件列表：
                    {context_str}

                    请注意，你无法直接看到图片内容，但你可以根据文件名、路径和相似度得分进行高质量的推断。
                    请综合分析这些信息，回答用户的问题。请不要直接罗列文件名，而是对内容进行归纳总结。
                    """
        from dashscope import Generation
        # 4. 调用 DashScope API
        response = Generation.call(
            model='qwen-turbo',  # 使用 qwen-turbo 模型，性价比高
            system_prompt=system_prompt,
            prompt=user_prompt
        )

        return response.output.text
