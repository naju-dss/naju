"""Shared utilities: seeding, config loading, parameter counting, timers."""
import os
import random
import time
import yaml
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: dict, overrides: dict) -> dict:
    """Override config values with non-None CLI args."""
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class CudaTimer:
    """Context manager measuring wall + GPU time and peak memory."""

    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == "cuda"

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.is_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
        self.peak_mem_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if self.is_cuda else 0.0
        )
