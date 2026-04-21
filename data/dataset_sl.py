import json
import os
import pickle
import random
from bisect import bisect_right

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_data_path(data_path):
    """优先解析到项目根目录，避免受当前启动目录影响。"""
    if os.path.isabs(data_path):
        return data_path

    cwd_candidate = os.path.join(os.getcwd(), data_path)
    root_candidate = os.path.join(PROJECT_ROOT, data_path)

    if os.path.exists(cwd_candidate):
        return cwd_candidate
    if os.path.exists(root_candidate):
        return root_candidate

    return root_candidate


class SatelliteExpertDataset(Dataset):
    """卫星路由专家数据集，兼容单文件和分块目录两种格式。"""

    def __init__(self, data_path="data/expert_data"):
        super().__init__()

        resolved_path = resolve_data_path(data_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"找不到数据集文件: {data_path} (解析后路径: {resolved_path})"
            )

        self.data_path = resolved_path
        self.data = None
        self.chunk_paths = []
        self.chunk_sizes = []
        self.cumulative_sizes = []
        self.is_chunked = False
        self._cached_chunk_idx = None
        self._cached_chunk_data = None

        if os.path.isdir(self.data_path):
            self.is_chunked = True
            self._init_chunked_dataset(self.data_path)
            print(f"成功加载分块数据集，共 {self.total_samples} 条状态-动作对")
        else:
            self._init_legacy_dataset(self.data_path)
            print(f"成功加载单文件数据集，共 {len(self.data)} 条状态-动作对")

    def _init_legacy_dataset(self, data_path):
        print(f"Loading legacy data from {data_path}...")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def _init_chunked_dataset(self, data_dir):
        metadata_path = os.path.join(data_dir, "metadata.json")
        metadata = None

        if os.path.exists(metadata_path):
            print(f"Loading chunked data from {data_dir}...")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        if metadata is not None:
            chunk_files = metadata.get("chunk_files", [])
            self.chunk_sizes = metadata.get("chunk_sizes", [])
        else:
            chunk_files = sorted(
                file_name
                for file_name in os.listdir(data_dir)
                if file_name.startswith("chunk_") and file_name.endswith(".pkl")
            )
            self.chunk_sizes = [None] * len(chunk_files)

        if not chunk_files:
            raise FileNotFoundError(f"在目录 {data_dir} 中没有找到任何 chunk 数据文件")

        self.chunk_paths = [os.path.join(data_dir, file_name) for file_name in chunk_files]

        if len(self.chunk_sizes) != len(self.chunk_paths):
            raise ValueError("metadata.json 中的 chunk_files 与 chunk_sizes 数量不一致")

        for idx, chunk_size in enumerate(self.chunk_sizes):
            if chunk_size is None:
                with open(self.chunk_paths[idx], "rb") as f:
                    self.chunk_sizes[idx] = len(pickle.load(f))

        total = 0
        for chunk_size in self.chunk_sizes:
            total += chunk_size
            self.cumulative_sizes.append(total)
        self.total_samples = total

    def _load_chunk(self, chunk_idx):
        if self._cached_chunk_idx == chunk_idx and self._cached_chunk_data is not None:
            return self._cached_chunk_data

        with open(self.chunk_paths[chunk_idx], "rb") as f:
            self._cached_chunk_data = pickle.load(f)
        self._cached_chunk_idx = chunk_idx
        return self._cached_chunk_data

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return self.total_samples

    def __getitem__(self, idx):
        if self.data is not None:
            item = self.data[idx]
        else:
            if idx < 0 or idx >= self.total_samples:
                raise IndexError(f"索引越界: {idx}")

            chunk_idx = bisect_right(self.cumulative_sizes, idx)
            chunk_start = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1]
            chunk_data = self._load_chunk(chunk_idx)
            item = chunk_data[idx - chunk_start]

        state = item["state"]
        action = item["action"]

        adj = torch.tensor(state["adjacency"], dtype=torch.float32)
        node_features = torch.tensor(state["node_features"], dtype=torch.float32)
        target = torch.tensor(action, dtype=torch.long)

        return adj, node_features, target


class ChunkAwareBatchSampler(Sampler):
    """按 chunk 采样，降低全局随机打乱时的跨文件读盘开销。"""

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        chunk_indices = list(range(len(self.dataset.chunk_sizes)))
        if self.shuffle:
            random.shuffle(chunk_indices)

        for chunk_idx in chunk_indices:
            chunk_end = self.dataset.cumulative_sizes[chunk_idx]
            chunk_start = 0 if chunk_idx == 0 else self.dataset.cumulative_sizes[chunk_idx - 1]
            indices = list(range(chunk_start, chunk_end))

            if self.shuffle:
                random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total_batches = 0
        for chunk_size in self.dataset.chunk_sizes:
            if self.drop_last:
                total_batches += chunk_size // self.batch_size
            else:
                total_batches += (chunk_size + self.batch_size - 1) // self.batch_size
        return total_batches


def get_dataloader(data_path="data/expert_data", batch_size=32, shuffle=True, num_workers=0):
    """获取训练用数据加载器。"""
    dataset = SatelliteExpertDataset(data_path)

    if dataset.is_chunked:
        batch_sampler = ChunkAwareBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
