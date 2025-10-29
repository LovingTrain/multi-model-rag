import os
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple, Union

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
