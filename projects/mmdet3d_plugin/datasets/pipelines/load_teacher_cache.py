import os
import pickle

import numpy as np
import torch

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadTeacherCache(object):
    """Load pre-computed BEVFusion teacher outputs from cache.

    The cache is a pickle file mapping sample tokens to teacher predictions.
    """

    def __init__(self, cache_path):
        print(f"Loading teacher cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            self.cache = pickle.load(f)
        print(f"Loaded {len(self.cache)} cached teacher outputs.")

    def __call__(self, results):
        token = results.get("scene_token", results.get("token", None))
        if token is None:
            # Fallback: try to construct key from available info
            token = results.get("sample_idx", None)

        if token is not None and token in self.cache:
            cached = self.cache[token]
            for key, val in cached.items():
                results[f"teacher_{key}"] = val.float().numpy() if isinstance(val, torch.Tensor) else val
        else:
            # If cache miss, fill with zeros (distillation will be skipped)
            results["teacher_dense_heatmap"] = np.zeros((9, 128, 128), dtype=np.float32)
            results["teacher_heatmap"] = np.zeros((9, 200), dtype=np.float32)
            results["teacher_center"] = np.zeros((2, 200), dtype=np.float32)
            results["teacher_height"] = np.zeros((1, 200), dtype=np.float32)
            results["teacher_dim"] = np.zeros((3, 200), dtype=np.float32)
            results["teacher_rot"] = np.zeros((2, 200), dtype=np.float32)
            results["teacher_vel"] = np.zeros((2, 200), dtype=np.float32)

        return results
