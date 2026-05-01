from __future__ import annotations

import numpy as np


class OrbitDynamics:
    """
    Simple circular-orbit satellite dynamics.

    This class computes:
        1. Satellite 3D positions at a given time.
        2. Pairwise satellite distance matrix.

    Assumptions:
        - All satellites have the same altitude.
        - All satellites have the same inclination.
        - Each orbital plane has evenly spaced satellites.
        - RAAN values are evenly distributed across orbital planes.
        - Motion is simplified as uniform circular motion.
    """

    def __init__(
        self,
        num_planes: int,
        sats_per_plane: int,
        altitude: float = 550e3,
        inclination_deg: float = 53.0,
        earth_radius: float = 6371e3,
        orbital_period: float = 5400.0,
    ):
        self.num_planes = int(num_planes)
        self.sats_per_plane = int(sats_per_plane)
        self.num_satellites = self.num_planes * self.sats_per_plane

        self.altitude = float(altitude)
        self.earth_radius = float(earth_radius)
        self.radius = self.earth_radius + self.altitude

        self.inclination_deg = float(inclination_deg)
        self.inclination = np.radians(self.inclination_deg)

        self.orbital_period = float(orbital_period)
        self.omega = 2.0 * np.pi / self.orbital_period

        self._validate_params()
        self._init_phases()

    def _validate_params(self) -> None:
        """
        Validate constructor parameters.
        """
        if self.num_planes <= 0:
            raise ValueError(f"num_planes must be positive, got: {self.num_planes}")

        if self.sats_per_plane <= 0:
            raise ValueError(
                f"sats_per_plane must be positive, got: {self.sats_per_plane}"
            )

        if self.altitude < 0:
            raise ValueError(f"altitude must be non-negative, got: {self.altitude}")

        if self.earth_radius <= 0:
            raise ValueError(
                f"earth_radius must be positive, got: {self.earth_radius}"
            )

        if self.orbital_period <= 0:
            raise ValueError(
                f"orbital_period must be positive, got: {self.orbital_period}"
            )

    def _init_phases(self) -> None:
        """
        Initialize RAAN and phase arrays for all satellites.

        Satellite id convention:
            node_id = plane * sats_per_plane + index
        """
        plane_indices = np.arange(self.num_planes)
        sat_indices = np.arange(self.sats_per_plane)

        planes, sats = np.meshgrid(
            plane_indices,
            sat_indices,
            indexing="ij",
        )

        self.raan_array = 2.0 * np.pi * planes / self.num_planes
        self.phase_array = 2.0 * np.pi * sats / self.sats_per_plane

        self.raan_array = self.raan_array.reshape(-1)
        self.phase_array = self.phase_array.reshape(-1)

    def compute_positions(self, time: float) -> np.ndarray:
        """
        Compute satellite 3D positions at given simulation time.

        Args:
            time:
                Simulation time in seconds.

        Returns:
            positions:
                Array with shape [num_satellites, 3].
        """
        time = float(time)

        theta = self.omega * time + self.phase_array
        radius = self.radius

        # Coordinates in orbital plane.
        x_orb = radius * np.cos(theta)
        y_orb = radius * np.sin(theta)

        # Inclination rotation.
        x_inc = x_orb
        y_inc = y_orb * np.cos(self.inclination)
        z_inc = y_orb * np.sin(self.inclination)

        # RAAN rotation.
        x = x_inc * np.cos(self.raan_array) - y_inc * np.sin(self.raan_array)
        y = x_inc * np.sin(self.raan_array) + y_inc * np.cos(self.raan_array)
        z = z_inc

        positions = np.column_stack((x, y, z))

        return positions

    def compute_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distance matrix.

        Args:
            positions:
                Array with shape [num_satellites, 3].

        Returns:
            dist_matrix:
                Array with shape [num_satellites, num_satellites].
        """
        positions = np.asarray(positions, dtype=float)

        expected_shape = (self.num_satellites, 3)

        if positions.shape != expected_shape:
            raise ValueError(
                f"positions must have shape {expected_shape}, got: {positions.shape}"
            )

        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        return dist_matrix