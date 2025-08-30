import os
import torch
import clip
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import glob
import json
from tqdm import tqdm
from typing import Union, Sequence

from transformers import CLIPModel, CLIPProcessor
class EmbeddingMode():
    def __init__(self,model_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # embedding_model, preprocess = clip.load(
        #     "/mnt/sdc/multi_model_data/weight/ViT-L-14.pt", device=device, jit=False)
        # flash attention


        local_model_path = "/home/wangjingjing/clip-vit-large-patch14" # 使用你刚刚保存的路径

        embedding_model = CLIPModel.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        ).to(device)

        preprocess = CLIPProcessor.from_pretrained(local_model_path)
        self.model = embedding_model.eval()
        self.preprocess = preprocess
        self.device = device

    # ------- 工具函数 -------
    @staticmethod
    def _to_numpy_2d(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        x = x.astype(np.float32, copy=False)
        if x.ndim == 1:
            x = x[None, :]
        return x

    @staticmethod
    def _l2_normalize_inplace(mat: np.ndarray):
        # 对每一行做 L2 归一化，避免 faiss.normalize_L2 反复来回 CPU/GPU
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat /= norms

    def encoding_query_text(
        self,
        query_texts: Union[str, Sequence[str]],
    ) -> np.ndarray:

        if isinstance(query_texts, str):
            texts = [query_texts]
        elif isinstance(query_texts, Sequence):
            texts = list(query_texts)
        else:
            raise TypeError(f"query_texts 类型不支持: {type(query_texts)}")

        with torch.no_grad():

            inputs = self.preprocess(
                text=texts,
                return_tensors="pt",  # 返回 PyTorch 张量
                padding=True,         # 填充以匹配批次中的最长句子
      # 截断过长的句子
            )

            # 2. 将分词结果移动到正确的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            vecs = emb.detach().cpu().float().numpy().astype(np.float32, copy=False)

        return vecs
    def encoding_query_text_clip(
        self,
        query_texts: Union[str, Sequence[str]],
    ) -> np.ndarray:

        if isinstance(query_texts, str):
            texts = [query_texts]
        elif isinstance(query_texts, Sequence):
            # 简单检查：Sequence 但不是 str
            texts = list(query_texts)
        else:
            raise TypeError(f"query_texts 类型不支持: {type(query_texts)}")

        with torch.no_grad():

            tokens = clip.tokenize(texts).to(self.device)  # 批量 tokenize
            emb = self.model.encode_text(tokens)              # [bs, D]
            emb = emb / emb.norm(dim=1, keepdim=True)                       # L2 归一化（在 torch 里做）
            vecs=emb.detach().cpu().float().numpy().astype(np.float32, copy=False) 

        return vecs

    # ------- 单图或批量图片编码 -------
    def encoding_query_image(
        self,
        images: Union[str, Image.Image, Sequence[Union[str, Image.Image]]],
        chunk_size: int = 64,
    ) -> np.ndarray:
        """
        入参:
          - 单张: 路径或 PIL.Image
          - 批量: list/tuple[路径或 PIL.Image]
        出参:
          - np.ndarray, shape=[B, D], dtype=float32，L2 归一化
        """
        def _load_one(x: Union[str, Image.Image]) -> Image.Image:
            return x if isinstance(x, Image.Image) else Image.open(x)

        if isinstance(images, (str, Image.Image)):
            ims = [images]
        elif isinstance(images, Sequence):
            ims = list(images)
        else:
            raise TypeError(f"images 类型不支持: {type(images)}")

        feats = []
        with torch.no_grad():
            for start in range(0, len(ims), chunk_size):
                sub = ims[start:start + chunk_size]
                tensors = [self.preprocess(_load_one(p)) for p in sub]
                batch = torch.stack(tensors, dim=0).to(self.device)             # [bs, 3, H, W]
                emb: torch.Tensor = self.model.encode_image(batch)              # [bs, D]
                emb = emb / emb.norm(dim=1, keepdim=True)
                feats.append(emb.detach().cpu().float().numpy())

        vecs = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
        return vecs
