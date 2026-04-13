import numpy as np
import networkx as nx
import random
from env.orbit_dynamics import OrbitDynamics
from env.topology import TopologyBuilder

class SatelliteEnv:
    def __init__(
        self,
        num_planes=6,
        sats_per_plane=10,
        failure_prob=0.0,
        max_link_distance=10000e3,
        seed=42,
    ):
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.num_satellites = num_planes * sats_per_plane
        self.failure_prob = failure_prob

        random.seed(seed)
        np.random.seed(seed)

        # 初始化物理与拓扑引擎
        self.dynamics = OrbitDynamics(num_planes, sats_per_plane)
        self.topology = TopologyBuilder(num_planes, sats_per_plane, max_link_distance=max_link_distance)

        self.graph = None
        self.current_node = None
        self.visited = None
        self.time = 0
        self.total_delay = 0

    def _update_environment(self):
        """ 根据当前时间，更新物理位置和拓扑图 """
        positions = self.dynamics.compute_positions(self.time)
        dist_matrix = self.dynamics.compute_distance_matrix(positions)
        self.graph = self.topology.build_topology(dist_matrix)
        self._apply_failures()

    def _apply_failures(self):
        """ 根据故障概率，随机移除一些边 """
        if self.failure_prob == 0:
            return
        
        edges = list(self.graph.edges())
        edges_to_remove = [e for e in edges if random.random() < self.failure_prob]
        self.graph.remove_edges_from(edges_to_remove)

    def reset(self):
        """ 重置环境，随机选择一个起始节点 """
        self.time = 0
        self.total_delay = 0
        self._update_environment()

        self.current_node = random.randint(0, self.num_satellites - 1)
        self.visited = {self.current_node}  # 使用集合提升查找速度

        return self._get_state()

    def step(self, action):
        """ 执行动作，返回新的状态、奖励、是否结束以及额外信息 """
        if not self.graph.has_edge(self.current_node, action):
            raise ValueError(f"Invalid action: no link between {self.current_node} and {action}")

        # 获取当前边的物理延迟
        delay = self.graph[self.current_node][action]["delay"]

        # 状态更新
        self.total_delay += delay
        self.current_node = action
        self.visited.add(action)
        
        # 时间推移：真实世界中，探测包飞行需要消耗 delay 的时间
        self.time += delay
        
        # 物理环境随时间推进而刷新
        self._update_environment()

        done = len(self.visited) == self.num_satellites
        reward = -delay

        info = {
            "time": self.time,
            "total_delay": self.total_delay,
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """ 构建当前状态表示，包含邻接矩阵和节点特征 """
        adjacency = nx.to_numpy_array(self.graph)
        node_features = self._node_features()
        
        return {
            "adjacency": adjacency,
            "node_features": node_features,
            "current_node": self.current_node,
            "visited": list(self.visited),
            "time": self.time,
        }

    def _node_features(self):
        """ 构建节点特征矩阵，包含轨面比例、索引比例、是否已访问、是否为当前节点 """
        features = np.zeros((self.num_satellites, 4))
        for node in range(self.num_satellites):
            plane = node // self.sats_per_plane
            index = node % self.sats_per_plane
            
            features[node, 0] = plane / self.num_planes
            features[node, 1] = index / self.sats_per_plane
            
            features[node, 2] = 1.0 if node in self.visited else 0.0
            features[node, 3] = 1.0 if node == self.current_node else 0.0
            
        return features

    def render(self):
        print(f"Time: {self.time:.4f}s | Current: {self.current_node} | Visited: {len(self.visited)}/{self.num_satellites} | Delay: {self.total_delay:.4f}s")