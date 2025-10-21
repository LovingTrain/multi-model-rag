# rag_bench.py
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import os
import time
import argparse
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch  # 仅用于 cuda 同步

from src.encode_mode import EmbeddingMode
from src.search_faiss import FaissSearcher
import nvtx


def now() -> float:
    return time.perf_counter()


def maybe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PipelineRAG:
    def __init__(
        self,
        embedding_file_path: str,
        metadata_file_path: str,
        embedding_model_path: str,
        search_type: str = "gpu",
        top_k: int = 10,
    ):
        self.embedding_model = EmbeddingMode(model_path=embedding_model_path)

        self.searcher = FaissSearcher(
            embedding_path=embedding_file_path,
            metadata_path=metadata_file_path,
        )
        self.search_type = search_type
        self.top_k = top_k

        self.index = self.searcher.init_gpu_index()

        print(f"[Init] search_type={search_type} 初始化完成。")

    # ====== 统一适配区：根据你的真实 API 改这两处 ======
    def _encode(self, queries, query_type: str) -> np.ndarray:
        if query_type == "text":
            vecs = self.embedding_model.encoding_query_text_all(queries)
        elif query_type == "image":
            vecs = self.embedding_model.encoding_query_image(queries)
        else:
            raise ValueError(f"不支持的 query_type: {query_type}")

        # if hasattr(vecs, "detach"):
        #     vecs = vecs.detach().cpu().numpy()
        # elif isinstance(vecs, list):
        #     vecs = np.asarray(vecs, dtype=np.float32)
        # return vecs.astype(np.float32)
        return vecs

    def _search(self, q: np.ndarray):
        if self.search_type == "cpu":
            return self.searcher.cpu_search(self.index, q)
        elif self.search_type == "gpu":
            return self.searcher.gpu_search(self.index, q)
        elif self.search_type == "hybrid":
            return self.searcher.hybrid_search(self.index, q)
        elif self.search_type == "hybrid_ivf":
            return self.searcher.hybrid_ivf_search(self.index, q)
        else:
            raise ValueError(f"不支持的 search_type: {self.search_type}")
    # =====================================================

    # -------- 三种测量原语 --------
    def measure_embedding_only(self, queries, query_type: str) -> float:
        t0 = now()
        embed = self.embedding_model.encoding_query_text(queries)
        return now() - t0

    def measure_search_only(self, query_vectors: np.ndarray) -> float:
        maybe_cuda_sync()
        t0 = now()
        _ = self._search(query_vectors)
        maybe_cuda_sync()
        return now() - t0

    def measure_end2end(self, queries, query_type: str) -> Tuple[float, float, float]:
        t0 = now()
        qv = self._encode(queries, query_type)
        _ = self._search(qv)
        t1 = now()
        return (t1 - t0)  # total, encode, search


def expand_batch(base_samples: List, bsz: int) -> List:

    return base_samples * bsz


def make_search_only_queries(
    pipeline: PipelineRAG,
    bsz: int,
    source: str = "encode",  # 'encode' | 'gaussian' | 'db_sample'
    text_seed: str = "a dog playing on the beach"
) -> np.ndarray:

    q = np.random.randn(bsz, dim).astype(np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    return (q / norm).astype(np.float32)

    # return np.vstack([emb] * bsz)


def parse_args():
    import argparse

    ap = argparse.ArgumentParser(description="RAG latency benchmark")

    # 数据 & 模型文件路径：现在都有默认值
    ap.add_argument("--embedding_file", type=str, default="/home/judy/wjj/cocodataset/vector/all_embeddings.npy")
    ap.add_argument("--metadata_file", type=str, default="/home/judy/wjj/cocodataset/vector/all_metadata.feather")
    ap.add_argument("--model_path", type=str, default="/home/judy/wjj/ViT-L-14.pt")

    # 运行配置
    ap.add_argument(
        "--search_type",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "hybrid", "hybrid_ivf"],
        help="Which FAISS index/search type to benchmark",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="end2end",
        choices=["end2end", "embedding", "search", "all"],
        help="Which mode to benchmark",
    )
    ap.add_argument(
        "--query_type",
        type=str,
        default="text",
        choices=["text", "image"],
        help="Type of query (text/image)",
    )
    ap.add_argument(
        "--base_query",
        type=str,
        default="a dog playing on the beach",
        help="Base query content, will be repeated for batch expansion",
    )
    ap.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1000,3000,5000]+[i for i in range(10000, 80000, 10000)],
        help="List of batch sizes to benchmark",
    )
    ap.add_argument("--repeats", type=int, default=1,
                    help="Number of repeats per batch")
    ap.add_argument("--warmup", type=int, default=0,
                    help="Warmup runs before measurement")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Top-K retrieved results")
    ap.add_argument("--outdir", type=str, default="./outputs",
                    help="Output directory")
    ap.add_argument(
        "--search_only_source",
        type=str,
        default="gaussian",
        choices=["encode", "gaussian", "db_sample"],
        help="Query vector source when benchmarking search-only",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    pipeline = PipelineRAG(
        embedding_file_path=args.embedding_file,
        metadata_file_path=args.metadata_file,
        embedding_model_path=args.model_path,
        search_type=args.search_type,
        top_k=args.top_k,
    )

    os.makedirs(args.outdir, exist_ok=True)
    base_queries = [args.base_query]

    prof_schedule = schedule(wait=1, warmup=1, active=2, repeat=1)
    total_profiling_steps = (1 + 1 + 2) * 1

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    mode = args.mode

    for bsz in args.batch_sizes:

        print(f"\n[INFO] Starting profiling for batch_size = {bsz}")
        prof_output_dir = os.path.join("/home/judy/wjj/multi-model-rag/outputs",f"{mode}_{args.query_type}", f"bsz_{bsz}")
        os.makedirs(prof_output_dir, exist_ok=True)

        if mode in ("end2end", "embedding"):
            queries = expand_batch(base_queries, bsz)
        else:
            qv = make_search_only_queries(
                pipeline, bsz, args.search_only_source)

        
        with profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(prof_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as prof:

            for step in range(total_profiling_steps):

                if mode == "end2end":
                    total = pipeline.measure_end2end(queries, args.query_type)
                elif mode == "embedding":
                    total = pipeline.measure_embedding_only(
                        queries, args.query_type)
                else:  # search
                    total = pipeline.measure_search_only(qv)
                prof.step()
                print(
                    f"  - Step {step + 1}/{total_profiling_steps} completed for bsz={bsz}.")

    print("\n[INFO] All profiling runs completed.")


if __name__ == "__main__":
    main()

# python tensorboard.py --mode search