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
import threading
from tqdm import tqdm


from src.search_faiss import FaissSearcher
from utils import *


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



# -------- 测量原语 --------
def measure_embedding_only(component: NewEmbeddingComponent, queries: list) -> float:
    maybe_cuda_sync()
    t0 = now()
    _ = component.encode(queries)
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

def measure_fused_embed(embedding_comp: NewEmbeddingComponent, search_comp: SearchComponent, queries: list, query_type: str) -> float:
    maybe_cuda_sync()
    t0 = now()
    query_vectors = embedding_comp.encode(queries)
    _ = search_comp.search(query_vectors[0:1])
    maybe_cuda_sync()
    return now() - t0

def measure_fused_search(embedding_comp: NewEmbeddingComponent, search_comp: SearchComponent, queries: list, query_type: str) -> float:
    maybe_cuda_sync()
    t0 = now()
    query_vectors = embedding_comp.encode(queries)
    _ = search_comp.search(query_vectors[0:1])
    maybe_cuda_sync()
    return now() - t0


# ******************************************************************************
def background_search_task(
    search_comp: 'SearchComponent', 
    query_vectors: np.ndarray, 
    stop_event: threading.Event
):

    while not stop_event.is_set():
        # 使用 NVTX 来标记这个后台任务，方便在性能分析工具中查看
        with nvtx.annotate("bg_search", color="orange"):
            # 在后台持续执行搜索操作，以产生稳定的 GPU 负载
            _ = search_comp.search(query_vectors)

def background_embedding_task(
    embedding_comp: 'NewEmbeddingComponent', 
    queries: List[str], 
    stop_event: threading.Event
):

    while not stop_event.is_set():

         _ = embedding_comp.encode(queries)

def measure_embedding_with_concurrent_search(
    embedding_comp: 'NewEmbeddingComponent', 
    search_comp: 'SearchComponent', 
    queries: list
) -> float:

    stop_event = threading.Event()
    
    background_query_vectors = make_search_only_queries(dim=search_comp.dim, bsz=100)
    
    # 3. 启动后台线程
    # 我们创建一个新线程。`target` 是这个线程将要执行的函数。
    # `args` 是传递给目标函数的参数。
    search_thread = threading.Thread(
        target=background_search_task,
        args=(search_comp, background_query_vectors, stop_event)
    )
    # `start()` 会在新创建的线程中启动 `background_search_task` 函数的执行，
    # 使其与主线程并行运行。
    search_thread.start()
    
    # 4. 确保后台负载已激活
    # 我们短暂地暂停主线程。这是一个很实用的步骤，目的是给操作系统一些时间来调度
    # 新的线程，并让后台的第一个搜索操作有机会开始，从而确保当我们开始测量时，
    # GPU 已经处于负载状态。
    time.sleep(0.2) 
    
    # 5. 核心测量环节
    # 这是我们测量主任务延迟的关键部分。
    maybe_cuda_sync()  # 确保所有之前的 GPU 指令（比如搜索任务的启动）都已完成。
    t0 = now()         # 记录开始时间。
    
    _ = embedding_comp.encode(queries)
        
    maybe_cuda_sync()  # 确保 `encode` 操作在 GPU 上完全完成后，再记录结束时间。
    latency = now() - t0 # 计算总耗时。
    
    # 6. 清理工作
    # 通知后台线程它应该停止工作了。
    # `background_search_task` 函数中的 `while not stop_event.is_set()` 循环将会退出。
    stop_event.set()
    
    # 等待后台线程完全执行完毕。
    # 这对于干净的资源管理至关重要，并能确保后台任务不会“泄漏”到
    # 后续的测量中。
    search_thread.join()
    
    return latency

def measure_search_with_concurrent_embedding(
    embedding_comp: 'NewEmbeddingComponent', 
    search_comp: 'SearchComponent', 
    query_vectors: np.ndarray,
    embedding_batch=100,
) -> float:
    """
    测量在一个嵌入任务于后台并发运行时，搜索任务的延迟。
    此函数旨在模拟 GPU 资源争用（contention）的场景。

    这个函数模拟了“带有一个嵌入任务的搜索”场景。

    Args:
        embedding_comp: 用于后台任务（编码文本）的组件。
        search_comp: 用于主任务（向量搜索）的组件。
        query_vectors: 需要被搜索的输入向量。

    Returns:
        在并发负载下，搜索任务的延迟（单位：秒）。
    """
    # 1. 并发控制设置
    # 同样，我们使用 threading.Event 来作为后台线程的“停止信号”。
    stop_event = threading.Event()
    
    # 2. 准备后台负载
    # 这一次，后台任务是嵌入。我们需要准备一批文本数据来持续地
    # 输入给嵌入模型，以产生稳定的 GPU 负载。
    # 批处理大小（如 32）是根据典型嵌入任务的负载来选择的。
    background_text_queries = embedding_comp.expand_batch(batch_size=embedding_batch)
    
    # 3. 启动后台线程
    # 创建一个新线程，但这次的目标函数是 `background_embedding_task`，
    # 它会持续不断地执行嵌入操作。
    embedding_thread = threading.Thread(
        target=background_embedding_task,
        args=(embedding_comp, background_text_queries, stop_event)
    )
    # 启动后台嵌入线程，使其与主线程并行运行。
    embedding_thread.start()
    
    # 4. 确保后台负载已激活
    # 同样，暂停主线程一小段时间，以确保后台的嵌入任务已经开始运行，
    # 使得 GPU 在我们开始测量前就处于负载状态。
    time.sleep(0.2)
    
    # 5. 核心测量环节
    # 这是测量搜索任务在压力下延迟的关键部分。
    maybe_cuda_sync()  # 确保 GPU 空闲，准备好接收我们的搜索任务。
    t0 = now()         # 记录开始时间。
    
    # 使用 nvtx 进行性能分析可视化，颜色设为绿色以区分。
    with nvtx.annotate(f"search_with_embedding | {len(query_vectors)}", color="green"):
        # 执行主任务：进行向量搜索。
        # 这个搜索操作现在必须与后台线程中运行的 `encode` 操作
        # 竞争 GPU 资源。
        _ = search_comp.search(query_vectors)
        
    maybe_cuda_sync()  # 确保搜索任务在 GPU 上完全完成后，再记录结束时间。
    latency = now() - t0 # 计算总耗时。
    
    # 6. 清理工作
    # 测量完成，通知后台的嵌入线程停止。
    stop_event.set()
    

    embedding_thread.join()
    
    return latency


def make_search_only_queries(
    dim: int,
    bsz: int,

) -> np.ndarray:

    q = np.random.randn(bsz, dim).astype(np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    return (q / norm).astype(np.float32)


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
    dim=4096,
    with_embedding_size:int = 100
) -> List[Dict]:
    results = []
    
    # dim = 768
    if mode == 'search': dim = target_component.dim
    elif mode == 'end2end': dim = target_component[1].dim

    for bsz in batch_sizes:
        if bsz == 0: continue
        totals = []
        

        if mode == "end2end" or mode == "with_search"or mode == "with_embedding":
            queries = target_component[0].expand_batch(bsz)
        if  mode == "embedding":
            queries = target_component.expand_batch(bsz)
        # 预热
        for _ in range(warmup):
            if mode == "end2end":
                measure_end2end(target_component[0], target_component[1], queries, query_type)
            elif mode == "embedding":
                measure_embedding_only(target_component, queries)
            elif mode == "search":
                qv = make_search_only_queries(dim, bsz)
                measure_search_only(target_component, qv)
            elif mode == "with_search":
                measure_embedding_with_concurrent_search(target_component[0], target_component[1], queries)
            elif mode == "with_embedding":
                qv = make_search_only_queries(dim, bsz)
                measure_search_with_concurrent_embedding(target_component[0], target_component[1], qv,with_embedding_size)

        # 正式重复
        for _ in tqdm(range(repeats)):
            if mode == "end2end":
                total = measure_end2end(target_component[0], target_component[1], queries, query_type)
            elif mode == "embedding":
                total = measure_embedding_only(target_component, queries)
            elif mode == "search":
                embedding_comp_for_query_gen = target_component[0] if mode == 'end2end' else None
                qv = make_search_only_queries(dim, bsz)
                total = measure_search_only(target_component, qv)
            elif mode == "with_search":
                total = measure_embedding_with_concurrent_search(target_component[0], target_component[1], queries)
            
            elif mode == "with_embedding":
                qv = make_search_only_queries(dim, bsz)
                total =measure_search_with_concurrent_embedding(target_component[0], target_component[1], qv, with_embedding_size)            
            
            totals.append(total)


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
    ap.add_argument("--mode", type=str, default="end2end", choices=["embedding", "search", "end2end","with_search","with_embedding"])
    ap.add_argument("--query_type", type=str, default="text", choices=["text", "image"])
    
    # === 主要改动点 3: 增加图片查询参数 ===
    ap.add_argument("--text_query", type=str, default="a dog playing on the beach", help="当 query_type='text' 时使用的基础查询文本。")
    ap.add_argument("--image_query", type=str, default="/home/judy/wjj/multi-model-rag/data/space.png", help="当 query_type='image' 时使用的基础查询图片路径。")
    # default=[1000, 3000,5000]+[i for i in range(10000,100000,10000)]
    ap.add_argument("--prompt_length", type=int, default=128)

    ap.add_argument("--batch_sizes", type=int, nargs="+", default= [i for i in range(100,1000,100)])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="./outputs_simplified")
    ap.add_argument("--search_only_source", type=str, default="gaussian", choices=["encode", "gaussian"])
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--with_embedding_size", type=int, default=100)


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

    if args.mode in ["embedding", "end2end","with_search","with_embedding"]:
        embedding_comp = NewEmbeddingComponent(args.prompt_length)
    
    if args.mode in ["search", "end2end","with_search","with_embedding"]:
        if args.search_only_source == 'encode' and not embedding_comp:
            print("[注意] search-only 模式使用 'encode' 源，需要临时加载 embedding 模型...")
            embedding_comp = NewEmbeddingComponent(args.prompt_length)
        search_comp = SearchComponent(args.embedding_file, args.metadata_file)

    if args.mode == "embedding":
        target_component = embedding_comp
    elif args.mode == "search":
        target_component = search_comp
    elif args.mode == "end2end" or args.mode == "with_search"or args.mode == "with_embedding":
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
        dim = args.dim,
        with_embedding_size= args.with_embedding_size

    )

    

if __name__ == "__main__":
    main() 
