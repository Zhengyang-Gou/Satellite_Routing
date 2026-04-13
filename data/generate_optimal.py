import sys
import os
import pickle
import networkx as nx

# 将项目根目录加入系统路径，以便找到 env 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from env.sat_env import SatelliteEnv

def greedy_expert_policy(env):
    """贪心策略，不再做全局规划，只看眼下离自己最近的未访问卫星
    """
    current = env.current_node
    unvisited = [n for n in range(env.num_satellites) if n not in env.visited]

    if not unvisited:
        return None  # 已经全部访问完，或遇到孤岛

    try:
        lengths = nx.single_source_dijkstra_path_length(env.graph, current, weight='delay')
    except Exception:
        return None

    # 找到距离最近的未访问节点作为目标
    closest_target = min(unvisited, key=lambda n: lengths.get(n, float('inf')))

    if lengths.get(closest_target, float('inf')) == float('inf'):
        return None

    # 算出前往这个最近目标的下一步怎么走
    try:
        path = nx.shortest_path(env.graph, source=current, target=closest_target, weight='delay')
        # path[0] 是 current，path[1] 是我们需要发送数据包的下一跳相邻卫星
        return path[1] if len(path) > 1 else None
    except nx.NetworkXNoPath:
        return None

def generate_dataset(num_episodes=500, save_path="data/expert_data.pkl"):
    """生成极速贪心遍历数据"""
    print(f" 开始使生成数据集")
    
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 初始化环境
    env = SatelliteEnv(num_planes=6, sats_per_plane=10, failure_prob=0.0, max_link_distance=10000e3)
    dataset = []
    successful_episodes = 0
    
    while successful_episodes < num_episodes:
        state = env.reset()
        episode_data = []
        done = False
        max_steps = env.num_satellites * 3 
        steps = 0
        
        while not done and steps < max_steps:
            # 调用贪心专家求动作
            action = greedy_expert_policy(env)
            if action is None:
                break 
                
            episode_data.append({
                "state": state,
                "action": action
            })
            
            state, reward, done, info = env.step(action)
            steps += 1
            
        # 如果走完了所有节点，就把这局的数据合并到总数据集中
        if done:
            dataset.extend(episode_data)
            successful_episodes += 1
            # 进度条打印频率可以调高一点，因为现在非常快
            if successful_episodes % 50 == 0:
                print(f"已完成 {successful_episodes}/{num_episodes} 个回合数据采集...")

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
        
    print(f"\n专家数据生成完毕！共收集了 {len(dataset)} 步高阶动作对，已保存至 {save_path}")

if __name__ == "__main__":
    # 速度变快了，可以直接生成 500 个回合的数据喂给模型
    generate_dataset(num_episodes=500)