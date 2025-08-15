
import glob
import json
import os

import clip
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .encode_mode import EmbeddingMode


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

    model = EmbeddingMode(model_path=CLIP_MODEL_PATH)
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

        image_features = model.encoding_query_image(batch_tensor)

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
