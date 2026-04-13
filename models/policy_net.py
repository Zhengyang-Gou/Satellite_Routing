import sys
import os
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.gnn_encoder import GNNEncoder

class PolicyNet(nn.Module):
    """
    策略网络：基于 GNN 提取的特征，决定下一步路由动作
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super(PolicyNet, self).__init__()
        
        self.gnn = GNNEncoder(input_dim, hidden_dim)
        
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, adj, node_features):
        """
        adj: (Batch, N, N)
        node_features: (Batch, N, 4)
        """
        batch_size, n_nodes, _ = node_features.shape

        # 获得所有卫星的高级特征表示 -> (Batch, N, 64)
        node_embeddings = self.gnn(adj, node_features)

        # node_features[:, :, 3] 记录了哪颗卫星是当前节点 (1 为当前，0 为其他)
        curr_node_mask = node_features[:, :, 3].unsqueeze(-1)  # (Batch, N, 1)
        # 通过掩码相乘并按节点求和，把当前卫星的 64 维特征单独提取出来
        curr_embeddings = (node_embeddings * curr_node_mask).sum(dim=1, keepdim=True) # (Batch, 1, 64)

        # 把当前卫星特征扩展，并与所有卫星特征拼接在一起 -> (Batch, N, 128)
        curr_embeddings_expanded = curr_embeddings.expand(-1, n_nodes, -1)
        pair_features = torch.cat([curr_embeddings_expanded, node_embeddings], dim=-1)

        # 经过 MLP，得到当前卫星到每个节点的初步分数 -> (Batch, N, 1)
        raw_scores = self.score_mlp(pair_features).squeeze(-1) # 降维成 (Batch, N)

        # 找到当前所在的节点索引
        curr_node_indices = node_features[:, :, 3].argmax(dim=1) # (Batch,)
        
        # 从邻接矩阵中，抓取当前卫星与其他卫星的连接状态 (1表示有链路，0表示断开)
        valid_moves_mask = adj[torch.arange(batch_size), curr_node_indices, :]

        # 将没有链路相连的节点的打分强行设为极小值 
        # 这样在后续计算分类 Loss 时，这些不合法动作的概率就是 0
        masked_logits = raw_scores.masked_fill(valid_moves_mask == 0, -1e9)

        return masked_logits
