# utils/seed.py

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(
    seed: int,
    deterministic: bool = False,
    benchmark: bool = False,
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed:
            Global random seed.
        deterministic:
            If True, asks PyTorch to use deterministic algorithms where possible.
            This may reduce performance and may raise errors for unsupported ops.
        benchmark:
            Controls torch.backends.cudnn.benchmark.
            Usually False for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker seed function.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """
    Build a torch.Generator for DataLoader shuffling.

    Returns None if seed is None.
    """
    if seed is None:
        return None

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator