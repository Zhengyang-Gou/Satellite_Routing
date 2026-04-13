import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNLayer(nn.Module):
    """
    基础的稠密图卷积层
    """
    def __init__(self, in_features, out_features):
        super(DenseGCNLayer, self).__init__()
        # 对节点特征进行线性变换
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: 节点特征张量
        adj: 归一化后的邻接矩阵 
        """
        # 特征变换: X * W
        h = self.linear(x)  # 形状变为 (Batch_Size, N, out_features)
        
        # 消息传递: A * H
        out = torch.bmm(adj, h) 
        
        return out

class GNNEncoder(nn.Module):
    """图神经网络编码器：堆叠多层 GCN 来提取节点的高级特征表示"""
    def __init__(self, input_dim=4, hidden_dim=64):
        super(GNNEncoder, self).__init__()
        
        # 堆叠 3 层 GCN
        # 3 层意味着每个节点能看到距离自己 3 跳以内的网络拓扑信息
        self.conv1 = DenseGCNLayer(input_dim, hidden_dim)
        self.conv2 = DenseGCNLayer(hidden_dim, hidden_dim)
        self.conv3 = DenseGCNLayer(hidden_dim, hidden_dim)

    def _normalize_adj(self, adj):
        """
        对邻接矩阵进行归一化处理，防止度数大的节点特征爆炸，同时添加自环
        """
        batch_size, n_nodes, _ = adj.shape
        device = adj.device
        
        # 添加自环: A_hat = A + I
        # 确保节点在更新时不仅听取邻居的，也保留自己的特征
        eye = torch.eye(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_hat = adj + eye
        
        # 计算度矩阵并归一化
        # 统计每个节点有多少个连接
        degree = adj_hat.sum(dim=-1, keepdim=True)  # (Batch_Size, N, 1)
        
        # 归一化，避免多次相乘后数值过大
        adj_norm = adj_hat / degree
        
        return adj_norm

    def forward(self, adj, x):
        """
        输入:
            adj: (Batch_Size, N, N) 的邻接矩阵
            x: (Batch_Size, N, 4) 的节点特征
        输出:
            out: (Batch_Size, N, hidden_dim) 的节点高级表征
        """
        # 预处理邻接矩阵
        adj_norm = self._normalize_adj(adj)
        
        # 图卷积特征提取
        x = F.relu(self.conv1(x, adj_norm))
        x = F.relu(self.conv2(x, adj_norm))
        x = F.relu(self.conv3(x, adj_norm))
        
        return x
