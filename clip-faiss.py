
import concurrent.futures
import glob
import json
import os
import time

import clip
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm  # 用于显示漂亮的进度条


def convert_dataset_to_folder_format(
    annotations_file,
    images_dir,
    output_dir
):
    """
    将包含图片和JSON描述文件的数据集转换为 "图片-文本对" 文件夹格式。

    Args:
        annotations_file (str): 指向包含描述信息的 JSON 文件的路径。
        images_dir (str): 存放所有原始图片文件的目录。
        output_dir (str): 用于存放转换后数据的输出目录。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载描述文件
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设 JSON 结构是 {'annotations': [{'image_id': ..., 'caption': ...}], 'images': [{'id': ..., 'file_name': ...}]}
    # 这是 MS COCO 数据集的典型结构，很多其他数据集也类似

    # 创建一个从 image_id 到 file_name 的映射
    image_id_to_filename = {img['id']: img['file_name']
                            for img in data['images']}

    print(f"开始转换 {len(data['annotations'])} 条描述...")

    # 遍历所有描述
    for annotation in tqdm(data['annotations']):
        image_id = annotation['image_id']
        caption = annotation['caption']

        if image_id in image_id_to_filename:
            # 获取原始文件名和路径
            original_filename = image_id_to_filename[image_id]
            source_image_path = os.path.join(images_dir, original_filename)

            # 检查原始图片是否存在
            if not os.path.exists(source_image_path):
                continue

            # 定义输出文件的基本名 (不含扩展名)
            base_name = os.path.splitext(original_filename)[0]

            # 定义输出文本文件和图片文件的路径
            output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
            output_img_path = os.path.join(output_dir, original_filename)

            # 写入文本描述
            # 注意：一个图片可能有多个描述，这里选择覆盖写入或追加写入
            # 此处使用追加模式(a)，允许多个描述保存在一个文件中
            with open(output_txt_path, 'a', encoding='utf-8') as txt_file:
                txt_file.write(caption.strip() + '\n')

            # 复制图片文件 (如果尚未复制)
            if not os.path.exists(output_img_path):
                import shutil
                shutil.copy(source_image_path, output_img_path)

    print("转换完成！")

# --- 使用示例 ---
# 假设你已经下载了 MS COCO 数据集
# annotations_file = '/mnt/sdc/multi_model_data/data/coco-data/captions_train2017.json'
# images_dir = '/mnt/sdc/multi_model_data/data/coco-data/train2017'
# output_dir = '/mnt/sdc/multi_model_data/data/coco-data/train2017_formatted'

# convert_dataset_to_folder_format(annotations_file, images_dir, output_dir)


def process_embeddings():
    IMAGE_FOLDER = "/mnt/sdc/multi_model_data/data/coco-data/train2017_formatted"  # coco下载数据集的位置
    OUTPUT_FOLDER = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017_14L"  # 输出索引和元数据的文件夹
    # CLIP_MODEL_PATH = "/home/wangjingjing/SimpleRAG/Multi-Modal-RAG-Pipeline-on-Images-and-Text-Locally/embedding-weight/ViT-B-32.pt"               # 提前下载好的 CLIP 模型
    # 提前下载好的 CLIP 模型
    CLIP_MODEL_PATH = "/mnt/sdc/multi_model_data/weight/ViT-L-14.pt"
    BATCH_SIZE = 256                           # 批处理大小，根据您的 GPU 显存调整

    # --- 1. 初始化模型和设备 ---
    print(">>> 步骤 1: 初始化模型和设备...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告：未检测到 CUDA，将使用 CPU。这会非常慢。")

    # 加载 CLIP 模型和预处理器
    model, preprocess = clip.load(CLIP_MODEL_PATH, device=device, jit=False)
    print(f"模型 {CLIP_MODEL_PATH} 已加载到 {device}。")

    print("\n>>> 步骤 2: 生成图片 Embeddings...")
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    image_paths = list(glob.glob(f"{IMAGE_FOLDER}/**/*.jpg", recursive=True)) + \
        list(glob.glob(f"{IMAGE_FOLDER}/**/*.png", recursive=True))

    all_embeddings = []
    all_metadata = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="处理图片批次"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_images = []

        # 预处理图片
        for path in batch_paths:
            try:
                image = preprocess(Image.open(path)).unsqueeze(0)
                batch_images.append(image)
            except Exception as e:
                print(f"警告：无法处理图片 {path}，已跳过。错误: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.cat(batch_images).to(device)

        # 使用 CLIP 模型进行推理
        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)

        image_features = image_features.cpu().numpy().astype(np.float32)
        # L2 归一化，为余弦相似度搜索做准备
        faiss.normalize_L2(image_features)

        # 将向量移回 CPU 并添加到列表中
        all_embeddings.append(image_features)
        all_metadata.extend(batch_paths)  # 记录对应的文件路径

    # 将所有批次的向量合并成一个大的 numpy 数组
    all_embeddings = np.vstack(all_embeddings)
    np.save(OUTPUT_FOLDER+"/all_embeddings.npy", all_embeddings)

    metadata_df = pd.DataFrame({'filepath': all_metadata})
    metadata_output_path = os.path.join(
        OUTPUT_FOLDER, "all_metadata.feather")  # [修正] 文件扩展名
    metadata_df.to_feather(metadata_output_path)
    print("embedding finished ~ ~")


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


class EmbeddingMode():
    def __init__(self):
        model_path = "/home/wangjingjing/SimpleRAG/Multi-Modal-RAG-Pipeline-on-Images-and-Text-Locally/embedding-weight/ViT-B-32.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model, preprocess = clip.load(
            model_path, device=device, jit=False)

        self.model = embedding_model
        self.preprocess = preprocess
        self.device = device

    def encoding_query_text(self, query_text):
        tokenized_text = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            query_vector = self.model.encode_text(tokenized_text)
        query_vector = query_vector.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_vector)
        return query_vector

    def encoding_query_image(self, query_img_path):
        query_img = self.preprocess(Image.open(
            query_img_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_vector = self.model.encode_image(query_img)
        query_vector = query_vector.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_vector)
        return query_vector


EMBEDDING_FILE = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017/all_embeddings.npy"
METADATA_FILE = "/mnt/sdc/multi_model_data/data/index-data/coco_train_2017/all_metadata.feather"

searcher = FaissSearcher(EMBEDDING_FILE, METADATA_FILE)
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
