# data/dataset_sl.py

from __future__ import annotations

import json
import pickle
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from utils.paths import resolve_project_path
from utils.seed import build_torch_generator, seed_worker


class SatelliteExpertDataset(Dataset):
    """
    Satellite routing expert dataset.

    Supports two formats:

    1. Legacy single-file pickle:
        data/expert_data.pkl

    2. Chunked directory:
        outputs/datasets/expert_greedy_clean/
          metadata.json
          chunk_00000.pkl
          chunk_00001.pkl
          ...

    Each sample is expected to be:
        {
            "state": {
                "adjacency": np.ndarray,
                "node_features": np.ndarray,
                ...
            },
            "action": int
        }
    """

    def __init__(self, data_path: str | Path):
        super().__init__()

        self.data_path = resolve_project_path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_path} "
                f"(resolved: {self.data_path})"
            )

        self.data: Optional[List[Dict[str, Any]]] = None

        self.chunk_paths: List[Path] = []
        self.chunk_sizes: List[int] = []
        self.cumulative_sizes: List[int] = []
        self.total_samples = 0
        self.is_chunked = False

        self._cached_chunk_idx: Optional[int] = None
        self._cached_chunk_data: Optional[List[Dict[str, Any]]] = None

        if self.data_path.is_dir():
            self.is_chunked = True
            self._init_chunked_dataset(self.data_path)
            print(
                f"Loaded chunked expert dataset: {self.data_path} | "
                f"samples={self.total_samples} | chunks={len(self.chunk_paths)}"
            )
        else:
            self._init_legacy_dataset(self.data_path)
            assert self.data is not None
            print(
                f"Loaded legacy expert dataset: {self.data_path} | "
                f"samples={len(self.data)}"
            )

    def _init_legacy_dataset(self, data_path: Path) -> None:
        """
        Load legacy single pickle dataset.
        """
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            raise TypeError(
                f"Legacy dataset must be a list of samples, got: {type(data)}"
            )

        self.data = data
        self.total_samples = len(data)

    def _init_chunked_dataset(self, data_dir: Path) -> None:
        """
        Initialize chunked dataset metadata.
        """
        metadata_path = data_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            chunk_files = metadata.get("chunk_files", [])
            chunk_sizes = metadata.get("chunk_sizes", [])
        else:
            chunk_files = sorted(
                path.name
                for path in data_dir.iterdir()
                if path.name.startswith("chunk_") and path.suffix == ".pkl"
            )
            chunk_sizes = [None] * len(chunk_files)

        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk_*.pkl files found in dataset directory: {data_dir}"
            )

        if len(chunk_files) != len(chunk_sizes):
            raise ValueError(
                "metadata.json has inconsistent chunk_files and chunk_sizes lengths"
            )

        self.chunk_paths = [data_dir / file_name for file_name in chunk_files]

        self.chunk_sizes = []
        for idx, chunk_path in enumerate(self.chunk_paths):
            size = chunk_sizes[idx]

            if size is None:
                with open(chunk_path, "rb") as f:
                    size = len(pickle.load(f))

            self.chunk_sizes.append(int(size))

        total = 0
        self.cumulative_sizes = []

        for chunk_size in self.chunk_sizes:
            total += chunk_size
            self.cumulative_sizes.append(total)

        self.total_samples = total

    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        """
        Load one chunk with one-chunk cache.
        """
        if (
            self._cached_chunk_idx == chunk_idx
            and self._cached_chunk_data is not None
        ):
            return self._cached_chunk_data

        with open(self.chunk_paths[chunk_idx], "rb") as f:
            chunk_data = pickle.load(f)

        if not isinstance(chunk_data, list):
            raise TypeError(
                f"Chunk must be a list of samples: {self.chunk_paths[chunk_idx]}"
            )

        self._cached_chunk_idx = chunk_idx
        self._cached_chunk_data = chunk_data

        return chunk_data

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index out of range: {idx}")

        if self.data is not None:
            item = self.data[idx]
        else:
            chunk_idx = bisect_right(self.cumulative_sizes, idx)
            chunk_start = (
                0
                if chunk_idx == 0
                else self.cumulative_sizes[chunk_idx - 1]
            )
            chunk_data = self._load_chunk(chunk_idx)
            item = chunk_data[idx - chunk_start]

        state = item["state"]
        action = item["action"]

        adj = torch.tensor(state["adjacency"], dtype=torch.float32)
        node_features = torch.tensor(state["node_features"], dtype=torch.float32)
        target = torch.tensor(action, dtype=torch.long)

        return adj, node_features, target


class ChunkAwareBatchSampler(Sampler[List[int]]):
    """
    Batch sampler for chunked datasets.

    It shuffles chunk order first, then shuffles sample order inside each chunk.
    This reduces frequent cross-file loading when using chunked pickle datasets.
    """

    def __init__(
        self,
        dataset: SatelliteExpertDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if not dataset.is_chunked:
            raise ValueError("ChunkAwareBatchSampler requires a chunked dataset")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        chunk_indices = list(range(len(self.dataset.chunk_sizes)))

        if self.shuffle:
            random.shuffle(chunk_indices)

        for chunk_idx in chunk_indices:
            chunk_end = self.dataset.cumulative_sizes[chunk_idx]
            chunk_start = (
                0
                if chunk_idx == 0
                else self.dataset.cumulative_sizes[chunk_idx - 1]
            )

            indices = list(range(chunk_start, chunk_end))

            if self.shuffle:
                random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                yield batch

    def __len__(self) -> int:
        total_batches = 0

        for chunk_size in self.dataset.chunk_sizes:
            if self.drop_last:
                total_batches += chunk_size // self.batch_size
            else:
                total_batches += (chunk_size + self.batch_size - 1) // self.batch_size

        return total_batches


def get_dataloader(
    data_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Build DataLoader for expert dataset.
    """
    dataset = SatelliteExpertDataset(data_path)

    generator = build_torch_generator(seed)

    if dataset.is_chunked:
        batch_sampler = ChunkAwareBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=seed_worker if seed is not None else None,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=seed_worker if seed is not None else None,
    )