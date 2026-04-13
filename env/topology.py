import networkx as nx

class TopologyBuilder:
    def __init__(self, num_planes, sats_per_plane, max_link_distance=2000e3, speed_of_light=3e8):
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.num_satellites = num_planes * sats_per_plane
        self.max_link_distance = max_link_distance
        self.c = speed_of_light

    def build_topology(self, dist_matrix):
        """ 传入距离矩阵，直接构建网络拓扑 """
        G = nx.Graph()
        G.add_nodes_from(range(self.num_satellites))

        edges_to_add = []

        for plane in range(self.num_planes):
            for i in range(self.sats_per_plane):
                node = plane * self.sats_per_plane + i

                # 同轨面 (Intra-plane)
                next_i = (i + 1) % self.sats_per_plane
                next_node = plane * self.sats_per_plane + next_i
                
                # 异轨面 (Inter-plane)
                next_plane = (plane + 1) % self.num_planes
                inter_node = next_plane * self.sats_per_plane + i

                # 检查距离并添加边
                d_intra = dist_matrix[node, next_node]
                if d_intra <= self.max_link_distance:
                    edges_to_add.append((node, next_node, {"delay": d_intra / self.c}))

                d_inter = dist_matrix[node, inter_node]
                if d_inter <= self.max_link_distance:
                    edges_to_add.append((node, inter_node, {"delay": d_inter / self.c}))

        G.add_edges_from(edges_to_add)
        return G