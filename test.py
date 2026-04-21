import sys
import os
import torch

# 将项目根目录加入系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from env.sat_env import SatelliteEnv
from models.policy_net import PolicyNet

def run_ai_inference(model_path="models/sl_policy.pth"):
    print("启动推理测试")

    env = SatelliteEnv(num_planes=6, sats_per_plane=10, failure_prob=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = PolicyNet(input_dim=4, hidden_dim=64).to(device)
    
    if not os.path.exists(model_path):
        print(f"找不到模型权重文件: {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式
    print(f"成功加载预训练模型: {model_path}\n")

    state = env.reset()
    done = False
    step_count = 0
    max_steps = env.num_satellites * 3 
    
    print(f"探测包初始位置: 卫星 {env.current_node}")

    with torch.no_grad(): # 推理阶段不需要计算梯度
        while not done and step_count < max_steps:
            adj_tensor = torch.tensor(state["adjacency"], dtype=torch.float32).unsqueeze(0).to(device)
            features_tensor = torch.tensor(state["node_features"], dtype=torch.float32).unsqueeze(0).to(device)
            
            logits = model(adj_tensor, features_tensor)
            
            # 过一遍 Softmax 变成概率
            probs = torch.softmax(logits, dim=-1)
    
            # 选出概率最高的那颗卫星
            action = probs.argmax(dim=-1).item()
            confidence = probs[0, action].item() * 100
            
            print(f" 步骤 {step_count+1:02d}: 发往卫星 {action:02d} ")

            state, reward, done, info = env.step(action)
            step_count += 1
            
    if done:
        print(f"测试成功 遍历了全部 {env.num_satellites} 颗卫星")
        print(f"消耗总时间/总延迟: {info['total_delay']:.4f} 秒")
        print(f"实际行走步数: {step_count} 步")
    else:
        print("测试失败 陷入了死循环或提前耗尽了最大步数")
        print(f"当前已访问: {len(env.visited)}/{env.num_satellites}")

if __name__ == "__main__":
    run_ai_inference()