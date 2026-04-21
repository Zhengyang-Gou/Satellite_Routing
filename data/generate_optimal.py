import sys
import os
import json
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

def _resolve_output_dir(save_path):
    """兼容旧的 pkl 路径配置，统一改为目录分块存储。"""
    normalized_path = os.path.normpath(save_path)
    if normalized_path.endswith(".pkl"):
        chunk_dir = os.path.splitext(normalized_path)[0]
        print(f"检测到旧版文件路径 {save_path}，将改为分块目录存储: {chunk_dir}")
        return chunk_dir
    return normalized_path

def _flush_chunk(chunk_data, chunk_index, output_dir):
    chunk_path = os.path.join(output_dir, f"chunk_{chunk_index:05d}.pkl")
    with open(chunk_path, "wb") as f:
        pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return chunk_path

def generate_dataset(num_episodes=500, save_path="data/expert_data", chunk_episode_size=100):
    """生成极速贪心遍历数据，并按回合分块写入磁盘。"""
    print("开始生成专家数据集")

    output_dir = _resolve_output_dir(save_path)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化环境
    env = SatelliteEnv(num_planes=6, sats_per_plane=10, failure_prob=0.0, max_link_distance=10000e3)
    current_chunk = []
    chunk_files = []
    chunk_sizes = []
    successful_episodes = 0
    total_samples = 0
    chunk_index = 0

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

        # 如果走完了所有节点，就把这局的数据写入当前分块
        if done:
            current_chunk.extend(episode_data)
            successful_episodes += 1

            if successful_episodes % chunk_episode_size == 0:
                chunk_path = _flush_chunk(current_chunk, chunk_index, output_dir)
                chunk_files.append(os.path.basename(chunk_path))
                chunk_sizes.append(len(current_chunk))
                total_samples += len(current_chunk)
                print(
                    f"已写入分块 {chunk_index + 1}: "
                    f"{successful_episodes}/{num_episodes} 个回合, {len(current_chunk)} 条样本"
                )
                current_chunk = []
                chunk_index += 1

            if successful_episodes % 50 == 0:
                print(f"已完成 {successful_episodes}/{num_episodes} 个回合数据采集...")

    if current_chunk:
        chunk_path = _flush_chunk(current_chunk, chunk_index, output_dir)
        chunk_files.append(os.path.basename(chunk_path))
        chunk_sizes.append(len(current_chunk))
        total_samples += len(current_chunk)
        print(
            f"已写入最终分块 {chunk_index + 1}: "
            f"{successful_episodes}/{num_episodes} 个回合, {len(current_chunk)} 条样本"
        )

    metadata = {
        "format": "chunked_expert_dataset",
        "num_episodes": successful_episodes,
        "total_samples": total_samples,
        "chunk_episode_size": chunk_episode_size,
        "chunk_files": chunk_files,
        "chunk_sizes": chunk_sizes,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"\n专家数据生成完毕！共收集了 {total_samples} 步高阶动作对，"
        f"已保存至目录 {output_dir}"
    )

if __name__ == "__main__":
    generate_dataset(num_episodes=5000)
