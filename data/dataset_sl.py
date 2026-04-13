import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class SatelliteExpertDataset(Dataset):
    """ 卫星路由专家数据集读取生成的最优遍历轨迹，转换为 PyTorch 张量 """
    def __init__(self, data_path="data/expert_data.pkl"):
        super().__init__()
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据集文件: {data_path}")

        print(f"Loading data from {data_path}...")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        
        print(f"成功加载了 {len(self.data)} 条状态-动作对")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        state = item["state"]
        action = item["action"]

        # 1. 提取图结构：邻接矩阵 (Adjacency Matrix)
        # 形状: (N, N)，其中 N 是卫星总数
        adj = torch.tensor(state["adjacency"], dtype=torch.float32)

        # 2. 提取节点特征 (Node Features)
        # 形状: (N, 4)，包含轨面比例、索引比例、是否已访问、是否为当前节点
        node_features = torch.tensor(state["node_features"], dtype=torch.float32)

        # 3. 提取目标动作 (Target Action)
        # 动作是一个标量索引，代表下一跳的卫星 ID。作为分类任务的 Label，必须是 Long 类型
        target = torch.tensor(action, dtype=torch.long)

        return adj, node_features, target

def get_dataloader(data_path="data/expert_data.pkl", batch_size=32, shuffle=True, num_workers=0):
    """
    获取数据加载器，负责将数据打包成 Batch
    """
    dataset = SatelliteExpertDataset(data_path)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=False # 即使最后不够一个 batch_size 也保留
    )
    
    return dataloader