"""
Geometry and Computational Mesh for the Cold Storage Digital Twin.
Implements the domain Omega = [0, Lx] x [0, Ly] x [0, Lz] and a structured mesh.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class Mesh:
    """
    Represents a structured 3D mesh for the cold storage room.
    """
    Lx: float  # Room length in x (m)
    Ly: float  # Room length in y (m)
    Lz: float  # Room length in z (m)
    Nx: int    # Number of cells in x
    Ny: int    # Number of cells in y
    Nz: int    # Number of cells in z

    def __post_init__(self):
        # Cell dimensions
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.V_cell = self.dx * self.dy * self.dz

        # Face areas
        self.AE = self.AW = self.dy * self.dz
        self.AN = self.AS = self.dx * self.dz
        self.AT = self.AB = self.dx * self.dy

        # Total volume
        self.V_room = self.Lx * self.Ly * self.Lz

    def get_cell_center(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """
        Returns the center coordinates of cell (i, j, k).
        Indices i, j, k are 0-indexed.
        """
        x = (i + 0.5) * self.dx
        y = (j + 0.5) * self.dy
        z = (k + 0.5) * self.dz
        return x, y, z

    def get_cell_index(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Maps physical coordinates (x, y, z) to the nearest cell index (i, j, k).
        """
        i = int(x // self.dx)
        j = int(y // self.dy)
        k = int(z // self.dz)
        # Clamp to mesh boundaries
        i = max(0, min(i, self.Nx - 1))
        j = max(0, min(j, self.Ny - 1))
        k = max(0, min(k, self.Nz - 1))
        return i, j, k

    def validate(self) -> bool:
        """
        Verify that sum(V_ijk) = V_room
        """
        total_vol = self.Nx * self.Ny * self.Nz * self.V_cell
        return np.isclose(total_vol, self.V_room)
