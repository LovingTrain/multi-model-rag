# 文件名: stress_test_refactored.py
import os
import time
import argparse
from typing import List, Dict, Tuple, Union
import numpy as np
#import matplotlib.pyplot as plt
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import nvtx
import torch.nn.functional as F
# 导入 Pillow 库用于处理图片
try:
    from PIL import Image
except ImportError:
    print("错误: Pillow 库未安装。请运行 'pip install Pillow' 来安装。")
    exit(1)

# 假设您的原始模块在 src 文件夹下
from src.encode_mode import EmbeddingMode
from src.search_faiss import FaissSearcher



class NewEmbeddingComponent:

    def __init__(self, prompt_length: str):
        print(f"[组件] 正在初始化 EmbeddingComponent...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"30s watching time")
        # time.sleep(30)
        model_name = 'intfloat/e5-mistral-7b-instruct'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            "/home/judy/wjj/multi-model-rag/e5-mistral-7b-instruct",
            torch_dtype=torch.bfloat16,
            device_map=self.device  # 自动将模型加载到指定设备
        )
        self.model.eval() # 将模型设置为评估模式
        self.mode = 'e5-mistral'
        self.prompt_length= prompt_length

        print(f"[组件] EmbeddingComponent 初始化完成。")

    def last_token_pool(self,last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def expand_batch(self, batch_size):

        base_text = "This is a test sentence for generating a long prompt. "
        base_tokens = self.tokenizer.encode(base_text, add_special_tokens=False)
        if not base_tokens: raise ValueError("Tokenizer produced empty tokens.")
        
        repeated_tokens = []
        while len(repeated_tokens) < self.prompt_length:
            repeated_tokens.extend(base_tokens)
        
        final_tokens = repeated_tokens[:self.prompt_length]
        long_prompt_text = self.tokenizer.decode(final_tokens, skip_special_tokens=True)
        return [long_prompt_text] * batch_size

    def encode(self, input_texts: list) -> np.ndarray:
        with torch.no_grad():
            # 将文本列表分词，并移动到GPU
            batch_dict = self.tokenizer(
                input_texts, 
                max_length=4096,  # e5-mistral支持长达4096的上下文
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            vecs = F.normalize(embeddings, p=2, dim=1)   

        # 转换成 numpy 数组
        if hasattr(vecs, "detach"):
            vecs = vecs.detach().to(torch.float32).cpu().numpy()
        elif isinstance(vecs, list):
            vecs = np.asarray(vecs, dtype=np.float32)
        return vecs.astype(np.float32)

class SearchComponent:
    """
    只负责 Faiss GPU 索引的加载和搜索。
    """
    def __init__(self, embedding_path: str, metadata_path: str):
        print(f"[组件] 正在初始化 SearchComponent (仅限 GPU)...")
        self.searcher = FaissSearcher(
            embedding_path=embedding_path,
            metadata_path=metadata_path,
        )

        self.index = self.searcher.init_gpu_index()
        # print(f"30s watching time")
        # time.sleep(30)


        self.dim = int(self.searcher.all_embeddings.shape[1])
        print(f"[组件] SearchComponent (GPU) 初始化完成。")

    def search(self, query_vectors: np.ndarray):
        """执行 GPU 搜索"""
        return self.searcher.gpu_search(self.index, query_vectors)


# -------------------------------------------------------------------
# 2. 基准测试工具函数 (大部分保持不变)
# -------------------------------------------------------------------

def now() -> float:
    return time.perf_counter()

def maybe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def calculate_percentiles(arr: List[float], percentiles: List[int] = [90, 95, 99]) -> Dict[str, float]:
    if not arr: return {f"p{p}": float("nan") for p in percentiles}
    data = np.asarray(arr, dtype=float)
    percentile_values = np.percentile(data, percentiles)
    return {f"p{p}": val for p, val in zip(percentiles, percentile_values)}

# -------- 测量原语 --------
def measure_embedding_only(component: NewEmbeddingComponent, queries: list, query_type: str) -> float:
    maybe_cuda_sync()
    t0 = now()
    with nvtx.annotate(f"embedding {query_type} | {len(queries)}", color="red"):
        _ = component.encode(queries, query_type)
    maybe_cuda_sync()
    return now() - t0

def measure_search_only(component: SearchComponent, query_vectors: np.ndarray) -> float:
    maybe_cuda_sync()
    t0 = now()
    with nvtx.annotate(f"search | {len(query_vectors)}", color="green"):
        _ = component.search(query_vectors)
    maybe_cuda_sync()
    return now() - t0

def measure_end2end(embedding_comp: NewEmbeddingComponent, search_comp: SearchComponent, queries: list, query_type: str) -> float:
    maybe_cuda_sync()
    t0 = now()
    with nvtx.annotate(f"e2e embedding {query_type} | {len(queries)}", color="red"):
        query_vectors = embedding_comp.encode(queries)
    with nvtx.annotate(f"e2e search | {len(query_vectors)}", color="green"):
        _ = search_comp.search(query_vectors)
    maybe_cuda_sync()
    return now() - t0



def make_search_only_queries(
    dim: int,
    bsz: int,
    source: str = "gaussian",
    embedding_comp: NewEmbeddingComponent = None,
    text_seed: str = "a dog",
    image_seed_path: str = None
) -> np.ndarray:
    if source == "encode":
        if embedding_comp is None: raise ValueError("当 source='encode' 时，必须提供 EmbeddingComponent")
        
        # === 主要改动点 2: 允许用图片生成查询向量 ===
        if image_seed_path:
            # 使用图片种子生成向量
            emb = embedding_comp.encode([image_seed_path], query_type="image")
        else:
            # 使用文本种子生成向量
            emb = embedding_comp.encode([text_seed], query_type="text")
        return np.vstack([emb] * bsz)

    if source == "gaussian":
        q = np.random.randn(bsz, dim).astype(np.float32)
        norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        return (q / norm).astype(np.float32)
    raise ValueError(f"未知的 source: {source}")

def run_benchmark(
    target_component: Union[NewEmbeddingComponent, SearchComponent, Tuple[NewEmbeddingComponent, SearchComponent]],
    mode: str,
    base_queries: List[str], # 这里仍然是字符串列表，可以是文本或路径
    query_type: str,
    batch_sizes: List[int],
    repeats: int,
    warmup: int,
    search_only_source: str = "encode",
    image_seed_path: str = None ,# 传递图片种子路径
    dim=4096
) -> List[Dict]:
    results = []
    
    # dim = 768
    if mode == 'search': dim = target_component.dim
    elif mode == 'end2end': dim = target_component[1].dim

    for bsz in batch_sizes:
        if bsz == 0: continue
        totals = []
        

        if mode == "end2end":
            queries = target_component[0].expand_batch(bsz)
        if  mode == "embedding":
            queries = target_component.expand_batch(bsz)
        # 预热
        for _ in range(warmup):
            if mode == "end2end":
                measure_end2end(target_component[0], target_component[1], queries, query_type)
            elif mode == "embedding":
                measure_embedding_only(target_component, queries)
            else: # search
                qv = make_search_only_queries(dim, bsz, search_only_source, image_seed_path=image_seed_path)
                measure_search_only(target_component, qv)
        torch.cuda.empty_cache() 
        # 正式重复
        for _ in range(repeats):
            if mode == "end2end":
                total = measure_end2end(target_component[0], target_component[1], queries, query_type)
            elif mode == "embedding":
                total = measure_embedding_only(target_component, queries)
            else: # search
                embedding_comp_for_query_gen = target_component[0] if mode == 'end2end' else None
                qv = make_search_only_queries(dim, bsz, search_only_source, embedding_comp=embedding_comp_for_query_gen, image_seed_path=image_seed_path)
                total = measure_search_only(target_component, qv)
            totals.append(total)
        torch.cuda.empty_cache()    

        if not totals: continue

        percentile_results = calculate_percentiles(totals, [50,90, 95, 99])
        rec = {
            "mode": mode, "batch_size": bsz,
            "avg_total_s": float(np.mean(totals)),
            "avg_per_sample_s": float(np.mean(totals)) / bsz,
        }
        rec.update({f"{k}_total_s": v for k, v in percentile_results.items()})

        p_str = " | ".join([f"p{p}={v:.5f}s" for p, v in percentile_results.items()])
        print(
            f"[{mode:<9} bsz={bsz:<5d}] "
            f"Avg Total: {rec['avg_total_s']:.5f}s | "
            f"Avg Per-Sample: {rec['avg_per_sample_s']:.8f}s | "
            f"Percentiles: [ {p_str} ]"
        )
        results.append(rec)
    return results



# -------------------------------------------------------------------
# 3. 主程序入口
# -------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="RAG 延迟基准测试 (支持图片和文本)")
    
    # 文件路径
    ap.add_argument("--embedding_file", type=str, default="/home/judy/wjj/cocodataset/vector4096/all_embeddings.npy")
    ap.add_argument("--metadata_file", type=str, default="/home/judy/wjj/cocodataset/vector4096/all_metadata.feather")
    
    # 运行配置
    ap.add_argument("--mode", type=str, default="end2end", choices=["embedding", "search", "end2end"])
    ap.add_argument("--query_type", type=str, default="text", choices=["text", "image"])
    
    # === 主要改动点 3: 增加图片查询参数 ===
    ap.add_argument("--text_query", type=str, default="a dog playing on the beach", help="当 query_type='text' 时使用的基础查询文本。")
    ap.add_argument("--image_query", type=str, default="/home/judy/wjj/multi-model-rag/data/space.png", help="当 query_type='image' 时使用的基础查询图片路径。")
    # default=[1000, 3000,5000]+[i for i in range(10000,100000,10000)]
    ap.add_argument("--prompt_length", type=int, default=128)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default= [i for i in range(100,1000,100)])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="./outputs_simplified")
    ap.add_argument("--search_only_source", type=str, default="gaussian", choices=["encode", "gaussian"])
    ap.add_argument("--dim", type=int, default=4096)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # === 主要改动点 4: 根据 query_type 选择基础查询内容 ===
    if args.query_type == "text":
        base_queries = [args.text_query]
        print(f"测试模式: 文本查询, 内容: '{args.text_query}'")
    elif args.query_type == "image":
        base_queries = [args.image_query]
        print(f"测试模式: 图片查询, 路径: '{args.image_query}'")
        if not os.path.exists(args.image_query):
            print(f"错误: 找不到指定的图片文件: {args.image_query}")
            return
    else:
        # 这个分支理论上不会被执行，因为 argparse 的 choices 会拦截
        print(f"错误: 不支持的 query_type '{args.query_type}'")
        return

    # --- 按需初始化组件 ---
    embedding_comp = None
    search_comp = None
    target_component = None

    print("-" * 40)
    print(f"开始执行模式: {args.mode.upper()}")
    print("-" * 40)

    if args.mode in ["embedding", "end2end"]:
        embedding_comp = NewEmbeddingComponent(args.prompt_length)
    
    if args.mode in ["search", "end2end"]:
        if args.search_only_source == 'encode' and not embedding_comp:
            print("[注意] search-only 模式使用 'encode' 源，需要临时加载 embedding 模型...")
            embedding_comp = NewEmbeddingComponent(args.prompt_length)
        search_comp = SearchComponent(args.embedding_file, args.metadata_file)

    if args.mode == "embedding":
        target_component = embedding_comp
    elif args.mode == "search":
        target_component = search_comp
    elif args.mode == "end2end":
        target_component = (embedding_comp, search_comp)
    
    if target_component is None:
        raise ValueError(f"模式 '{args.mode}' 初始化失败，无有效组件。")
    
    # --- 运行基准测试 ---
    rows = run_benchmark(
        target_component=target_component,
        mode=args.mode,
        base_queries=base_queries,
        query_type=args.query_type,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        warmup=args.warmup,
          search_only_source=args.search_only_source,
        image_seed_path=args.image_query if args.query_type == 'image' else None,
        dim = args.dim

    )

    

if __name__ == "__main__":
    main() 