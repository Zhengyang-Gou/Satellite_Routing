import numpy as np

class OrbitDynamics:
    def __init__(
        self,
        num_planes,
        sats_per_plane,
        altitude=550e3,
        inclination_deg=53,
        earth_radius=6371e3,
        orbital_period=5400,
    ):
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.num_satellites = num_planes * sats_per_plane
        self.altitude = altitude
        self.radius = earth_radius + altitude
        self.inclination = np.radians(inclination_deg)
        self.orbital_period = orbital_period
        self.omega = 2 * np.pi / orbital_period
        
        self._init_phases()

    def _init_phases(self):
    
        p_indices = np.arange(self.num_planes)
        s_indices = np.arange(self.sats_per_plane)
        
        P, S = np.meshgrid(p_indices, s_indices, indexing='ij')
        
        # 计算每颗卫星的 RAAN 和相位
        self.raan_array = 2 * np.pi * P / self.num_planes
        self.phase_array = 2 * np.pi * S / self.sats_per_plane
        
        # 展平为 1D 数组，长度为 num_satellites
        self.raan_array = self.raan_array.flatten()
        self.phase_array = self.phase_array.flatten()

    def compute_positions(self, time):
        """ 计算每颗卫星在给定时间的三维位置 """
        theta = self.omega * time + self.phase_array
        r = self.radius

        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)

        # 倾角旋转
        x_inc = x_orb
        y_inc = y_orb * np.cos(self.inclination)
        z_inc = y_orb * np.sin(self.inclination)

        # RAAN 升交点赤经旋转
        x = x_inc * np.cos(self.raan_array) - y_inc * np.sin(self.raan_array)
        y = x_inc * np.sin(self.raan_array) + y_inc * np.cos(self.raan_array)
        z = z_inc

        # 组合成 (N, 3) 的矩阵
        return np.column_stack((x, y, z))

    def compute_distance_matrix(self, positions):
        """ 计算 NxN 距离矩阵 """
        # positions: (N, 3) -> 扩展维度进行相减
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        return dist_matrix