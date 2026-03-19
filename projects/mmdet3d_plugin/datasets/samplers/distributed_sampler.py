import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER

import pdb
import sys


class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace():
    ForkedPdb().set_trace(sys._getframe().f_back)


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        assert not self.shuffle

        timestamps = [
            x["timestamp"] for x in self.dataset.data_infos
        ]
        # Detect timestamp scale: nuScenes uses microseconds (>1e12),
        # B2D uses small integers.  Normalise to seconds for the gap check.
        ts_scale = 1e6 if timestamps and timestamps[0] > 1e12 else 1.0
        timestamps_sec = [t / ts_scale for t in timestamps]

        scene_tokens = [
            x.get("scene_token") for x in self.dataset.data_infos
        ]
        vehicle_idx = [
            x["lidar_path"].split("/")[-1][:4]
            if "lidar_path" in x
            else None
            for x in self.dataset.data_infos
        ]

        sequence_splits = []
        for i in range(len(timestamps)):
            if i == 0 or (
                abs(timestamps_sec[i] - timestamps_sec[i - 1]) > 4
                or vehicle_idx[i] != vehicle_idx[i - 1]
                or (scene_tokens[i] is not None and scene_tokens[i] != scene_tokens[i - 1])
            ):
                sequence_splits.append([i])
            else:
                sequence_splits[-1].append(i)

        indices = []
        perfix_sum = 0
        start = math.ceil(len(self.dataset) * self.rank / self.num_replicas)
        end = math.ceil(len(self.dataset) * (self.rank + 1) / self.num_replicas)
        for i in range(len(sequence_splits)):
            if perfix_sum >= end:
                break
            elif perfix_sum >= start:
                indices.extend(sequence_splits[i])
            perfix_sum += len(sequence_splits[i])

        self.num_samples = len(indices)
        return iter(indices)
