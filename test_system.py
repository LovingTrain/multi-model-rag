# 文件名: stress_test_nsys.py
import os
import time
import argparse
from typing import List, Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
import nvtx




from src.encode_mode import EmbeddingMode
from src.search_faiss import FaissSearcher

# -------------------------------------------------------------------
# 1. 核心组件定义 (保持不变)
# -------------------------------------------------------------------

class EmbeddingComponent:
    """
    只负责 Embedding 模型的加载和推理。
    """
    def __init__(self, model_path: str):
        print(f"[组件] 正在初始化 EmbeddingComponent...")
        self.model = EmbeddingMode(model_path=model_path)
        print(f"[组件] EmbeddingComponent 初始化完成。")

    def encode(self, queries: list, query_type: str) -> np.ndarray:
        """
        执行编码操作。
        对于图片，queries 列表包含的是图片文件路径。
        """
        if query_type == "text":
            vecs = self.model.encoding_query_text(queries)
        elif query_type == "image":
            for path in queries:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"图片文件未找到: {path}")
            image_batch = self.model.process_batch_images(queries)
            vecs = self.model.encoding_query_image(image_batch)
        else:
            raise ValueError(f"不支持的 query_type: {query_type}")

        if hasattr(vecs, "detach"):
            vecs = vecs.detach().cpu().numpy()
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
        self.dim = int(self.searcher.all_embeddings.shape[1])
        print(f"[组件] SearchComponent (GPU) 初始化完成。")

    def search(self, query_vectors: np.ndarray):
        """执行 GPU 搜索"""
        return self.searcher.gpu_search(self.index, query_vectors)


# -------------------------------------------------------------------
# 2. 基准测试工具函数 (已修改以支持 Nsys)
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

# -------- 测量原语 (=== 主要改动点 1: 增加 use_nvtx 参数 ===) --------
def measure_embedding_only(component: EmbeddingComponent, queries: list, query_type: str, use_nvtx: bool) -> float:
    maybe_cuda_sync()
    t0 = now()
    if use_nvtx:
        with nvtx.annotate(f"embedding_{query_type}_bsz{len(queries)}", color="red"):
            _ = component.encode(queries, query_type)
    else:
        _ = component.encode(queries, query_type)
    maybe_cuda_sync()
    return now() - t0

def measure_search_only(component: SearchComponent, query_vectors: np.ndarray, use_nvtx: bool) -> float:
    maybe_cuda_sync()
    t0 = now()
    if use_nvtx:
        with nvtx.annotate(f"search_bsz{len(query_vectors)}", color="green"):
            _ = component.search(query_vectors)
    else:
        _ = component.search(query_vectors)
    maybe_cuda_sync()
    return now() - t0

def measure_end2end(embedding_comp: EmbeddingComponent, search_comp: SearchComponent, queries: list, query_type: str, use_nvtx: bool) -> float:
    maybe_cuda_sync()
    t0 = now()
    # 在端到端测试中，为两个阶段分别打上 NVTX 标记
    if use_nvtx:
        with nvtx.annotate(f"e2e_embedding_{query_type}_bsz{len(queries)}", color="red"):
            query_vectors = embedding_comp.encode(queries, query_type)
        with nvtx.annotate(f"e2e_search_bsz{len(query_vectors)}", color="green"):
            _ = search_comp.search(query_vectors)
    else:
        query_vectors = embedding_comp.encode(queries, query_type)
        _ = search_comp.search(query_vectors)
    maybe_cuda_sync()
    return now() - t0

# -------- 辅助函数 (保持不变) --------
def expand_batch(base_samples: List, bsz: int) -> List:
    return base_samples * bsz

def make_search_only_queries(
    dim: int,
    bsz: int,
    source: str = "gaussian",
    embedding_comp: EmbeddingComponent = None,
    text_seed: str = "a dog",
    image_seed_path: str = None
) -> np.ndarray:
    if source == "encode":
        if embedding_comp is None: raise ValueError("当 source='encode' 时，必须提供 EmbeddingComponent")
        
        if image_seed_path:
            emb = embedding_comp.encode([image_seed_path], query_type="image")
        else:
            emb = embedding_comp.encode([text_seed], query_type="text")
        return np.vstack([emb] * bsz)

    if source == "gaussian":
        q = np.random.randn(bsz, dim).astype(np.float32)
        norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        return (q / norm).astype(np.float32)
    raise ValueError(f"未知的 source: {source}")

# -------- 基准测试主函数 (=== 主要改动点 2 & 3 ===) --------
def run_benchmark(
    target_component: Union[EmbeddingComponent, SearchComponent, Tuple[EmbeddingComponent, SearchComponent]],
    mode: str,
    base_queries: List[str],
    query_type: str,
    batch_sizes: List[int],
    repeats: int,
    warmup: int,
    search_only_source: str = "encode",
    image_seed_path: str = None
) -> List[Dict]:
    results = []
    
    dim = 768
    if mode == 'search': dim = target_component.dim
    elif mode == 'end2end': dim = target_component[1].dim

    for bsz in batch_sizes:
        # === 主要改动点 2: 为每个 batch size 添加顶层 NVTX 标记 ===
        with nvtx.annotate(f"Batch Size: {bsz}", color="purple"):
            if bsz == 0: continue
            totals = []
            
            if mode in ("end2end", "embedding"):
                queries = expand_batch(base_queries, bsz)

            # 预热: 调用测量函数时，use_nvtx=False
            print(f"[{mode:<9} bsz={bsz:<5d}] Warming up for {warmup} iterations...")
            for _ in range(warmup):
                if mode == "end2end":
                    measure_end2end(target_component[0], target_component[1], queries, query_type, use_nvtx=False)
                elif mode == "embedding":
                    measure_embedding_only(target_component, queries, query_type, use_nvtx=False)
                else: # search
                    qv = make_search_only_queries(dim, bsz, search_only_source, image_seed_path=image_seed_path)
                    measure_search_only(target_component, qv, use_nvtx=False)
                
            # 正式重复: 调用测量函数时，use_nvtx=True
            print(f"[{mode:<9} bsz={bsz:<5d}] Running {repeats} repetitions for measurement...")
            for i in range(repeats):
                # 为每一次重复也添加一个标记，方便在时间线上区分
                with nvtx.annotate(f"Repeat #{i+1}", color="blue"):
                    if mode == "end2end":
                        total = measure_end2end(target_component[0], target_component[1], queries, query_type, use_nvtx=True)
                    elif mode == "embedding":
                        total = measure_embedding_only(target_component, queries, query_type, use_nvtx=True)
                    else: # search
                        embedding_comp_for_query_gen = target_component[0] if mode == 'end2end' else None
                        qv = make_search_only_queries(dim, bsz, search_only_source, embedding_comp=embedding_comp_for_query_gen, image_seed_path=image_seed_path)
                        total = measure_search_only(target_component, qv, use_nvtx=True)
                    totals.append(total)
            


            if not totals: continue

            percentile_results = calculate_percentiles(totals, [90, 95, 99])
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
# 3. 主程序入口 (保持不变)
# -------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="RAG 延迟基准测试 (专为 Nsight Systems 剖析优化)")
    
    ap.add_argument("--embedding_file", type=str, default="/home/judy/wjj/cocodataset/vector/all_embeddings.npy")
    ap.add_argument("--metadata_file", type=str, default="/home/judy/wjj/cocodataset/vector/all_metadata.feather")
    ap.add_argument("--model_path", type=str, default="/home/judy/wjj/ViT-L-14.pt")
    
    ap.add_argument("--mode", type=str, default="end2end", choices=["embedding", "search", "end2end"])
    ap.add_argument("--query_type", type=str, default="text", choices=["text", "image"])
    
    ap.add_argument("--text_query", type=str, default="a dog playing on the beach", help="当 query_type='text' 时使用的基础查询文本。")
    ap.add_argument("--image_query", type=str, default="/home/wjj/multi-model-rag/data/space.png", help="当 query_type='image' 时使用的基础查询图片路径。")
    
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1000, 2000])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="./outputs_nsys")
    ap.add_argument("--search_only_source", type=str, default="gaussian", choices=["encode", "gaussian"])
    
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

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
        print(f"错误: 不支持的 query_type '{args.query_type}'")
        return

    embedding_comp = None
    search_comp = None
    target_component = None

    print("-" * 40)
    print(f"开始执行模式: {args.mode.upper()}")
    print("-" * 40)

    if args.mode in ["embedding", "end2end"]:
        embedding_comp = EmbeddingComponent(args.model_path)
    
    if args.mode in ["search", "end2end"]:
        if args.search_only_source == 'encode' and not embedding_comp:
            print("[注意] search-only 模式使用 'encode' 源，需要临时加载 embedding 模型...")
            embedding_comp = EmbeddingComponent(args.model_path)
        search_comp = SearchComponent(args.embedding_file, args.metadata_file)

    if args.mode == "embedding":
        target_component = embedding_comp
    elif args.mode == "search":
        target_component = search_comp
    elif args.mode == "end2end":
        target_component = (embedding_comp, search_comp)
    
    if target_component is None:
        raise ValueError(f"模式 '{args.mode}' 初始化失败，无有效组件。")
    
    rows = run_benchmark(
        target_component=target_component,
        mode=args.mode,
        base_queries=base_queries,
        query_type=args.query_type,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        warmup=args.warmup,
        search_only_source=args.search_only_source,
        image_seed_path=args.image_query if args.query_type == 'image' else None
    )



if __name__ == "__main__":
    main()


'''
sudo nsys profile \
    --trace=cuda,cudnn,cublas,nvtx,osrt  \
    -o outputs_nsys/text_emb \
    --force-overwrite true \
    --stats=true \
    --gpu-metrics-devices=0  \    
    $(which python) test_system.py \
        --batch_sizes 1000 4000  \
        --repeats 1 \
        --warmup 1 \
        --query_type text \
        --mode embedding 
        
8000 10000 12000 14000
'''