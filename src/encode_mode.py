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


class EmbeddingMode:
    def __init__(self, model_path="/mnt/sdc/multi_model_data/weight/ViT-L-14.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model, preprocess = clip.load(model_path, device=device, jit=False)

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
        query_img = (
            self.preprocess(Image.open(query_img_path)).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            query_vector = self.model.encode_image(query_img)
        query_vector = query_vector.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_vector)
        return query_vector
