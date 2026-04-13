import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# 将项目根目录加入系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.dataset_sl import get_dataloader
from models.policy_net import PolicyNet

def train_supervised_learning(
    data_path="data/expert_data.pkl",
    save_path="models/sl_policy.pth",
    epochs=30,
    batch_size=16,
    lr=1e-3
):
    print("启动监督学习阶段")

    
    # 1. 自动选择计算设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"当前使用的计算设备: {device}\n")

    # 2. 加载数据集
    try:
        dataloader = get_dataloader(data_path=data_path, batch_size=batch_size)
    except FileNotFoundError:
        print("错误：找不到数据集")
        return

    # 3. 初始化模型、损失函数和优化器
    model = PolicyNet(input_dim=4, hidden_dim=64).to(device)
    
    # CrossEntropyLoss 会自动将输出的 Logits 进行 Softmax 并计算交叉熵
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器，负责根据误差调整 GNN 的参数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 开始训练循环
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (adj, features, target) in enumerate(dataloader):
            # 将张量推送到 GPU/CPU
            adj = adj.to(device)
            features = features.to(device)
            target = target.to(device)

            # --- 前向传播 ---
            # 清空上一轮的梯度
            optimizer.zero_grad()
            # 获取模型预测的 Logits
            logits = model(adj, features)
            
            # --- 计算误差 ---
            loss = criterion(logits, target)
            
            # --- 反向传播与参数更新 ---
            loss.backward()
            optimizer.step()

            # --- 统计准确率与 Loss ---
            total_loss += loss.item() * target.size(0)
            
            # 选出 Logits 最高的那个邻居作为最终决策
            predictions = logits.argmax(dim=-1)
            correct_predictions += (predictions == target).sum().item()
            total_samples += target.size(0)

        # 计算整个 Epoch 的平均指标
        avg_loss = total_loss / total_samples
        accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch [{epoch:02d}/{epochs}] | Loss: {avg_loss:.4f} | 准确率: {accuracy:5.2f}%")

        # 5. 保存表现最好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            # 记录一句小提示，表明模型有进步
            print(f"模型已进步，权重更新至 -> {save_path}")

    print("\n监督学习训练完成")

if __name__ == "__main__":
    # 如果你在生成数据时用的数据量比较大，可以适当调大 batch_size 和 epochs
    train_supervised_learning()