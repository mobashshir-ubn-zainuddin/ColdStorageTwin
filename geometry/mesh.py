"""
Geometry and Computational Mesh for the Cold Storage Digital Twin.
Implements a structured Cartesian finite-volume mesh.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union

@dataclass
class Mesh:
    """
    Represents a structured 3D Cartesian mesh for a cold storage room.
    """
    Lx: float  # Total length in x [m]
    Ly: float  # Total length in y [m]
    Lz: float  # Total length in z [m]
    Nx: int    # Number of cells in x
    Ny: int    # Number of cells in y
    Nz: int    # Number of cells in z

    def __post_init__(self):
        # Cell widths
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.V_cell = self.dx * self.dy * self.dz

        # Face areas [m^2]
        self.A_E = self.A_W = self.dy * self.dz
        self.A_N = self.A_S = self.dx * self.dz
        self.A_T = self.A_B = self.dx * self.dy

        # Total Volume
        self.V_room = self.Lx * self.Ly * self.Lz

        # Face normals [Unit Vectors]
        self.normals = {
            'E': np.array([1, 0, 0]),
            'W': np.array([-1, 0, 0]),
            'N': np.array([0, 1, 0]),
            'S': np.array([0, -1, 0]),
            'T': np.array([0, 0, 1]),
            'B': np.array([0, 0, -1])
        }

        # Coordinate arrays for cell centers [m]
        self.X = np.meshgrid(
            np.linspace(self.dx/2, self.Lx - self.dx/2, self.Nx),
            np.linspace(self.dy/2, self.Ly - self.dy/2, self.Ny),
            np.linspace(self.dz/2, self.Lz - self.dz/2, self.Nz),
            indexing='ij'
        )
        # unpack the tuple from meshgrid
        self.X_coords, self.Y_coords, self.Z_coords = self.X

    def get_cell_center(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Return coordinates of cell center (i, j, k)."""
        return self.X_coords[i, j, k], self.Y_coords[i, j, k], self.Z_coords[i, j, k]

    def get_cell_index(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Maps physical coordinates (x, y, z) to cell index (i, j, k).
        Raises ValueError if coordinates are outside the domain.
        """
        if not (0 <= x < self.Lx and 0 <= y < self.Ly and 0 <= z < self.Lz):
            raise ValueError(f"Coordinates ({x}, {y}, {z}) are outside the domain [0, Lx]x[0, Ly]x[0, Lz]")

        i = int(x // self.dx)
        j = int(y // self.dy)
        k = int(z // self.dz)
        return i, j, k

    def get_neighbors(self, i: int, j: int, k: int) -> Dict[str, Optional[Tuple[int, int, int]]]:
        """
        Returns the six neighboring cell indices.
        Returns None if the neighbor is outside the boundary.
        """
        neighbors = {
            'E': (i+1, j, k) if i < self.Nx-1 else None,
            'W': (i-1, j, k) if i > 0 else None,
            'N': (i, j+1, k) if j < self.Ny-1 else None,
            'S': (i, j-1, k) if j > 0 else None,
            'T': (i, j, k+1) if k < self.Nz-1 else None,
            'B': (i, j, k-1) if k > 0 else None
        }
        return neighbors

    def is_boundary_cell(self, i: int, j: int, k: int) -> bool:
        """Check if cell is on any boundary."""
        return (i == 0 or i == self.Nx-1 or
                j == 0 or j == self.Ny-1 or
                k == 0 or k == self.Nz-1)

    def get_boundary_faces(self, i: int, j: int, k: int) -> List[str]:
        """Return a list of faces that are on the domain boundary."""
        faces = []
        if i == self.Nx-1: faces.append('E')
        if i == 0: faces.append('W')
        if j == self.Ny-1: faces.append('N')
        if j == 0: faces.append('S')
        if k == self.Nz-1: faces.append('T')
        if k == 0: faces.append('B')
        return faces

    def get_distance_to_source(self, i: int, j: int, k: int, source_pos: Tuple[float, float, float]) -> float:
        """
        Calculate distance from cell center to a source point.
        """
        cx, cy, cz = self.get_cell_center(i, j, k)
        sx, sy, sz = source_pos
        return np.sqrt((cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2)

    def validate(self) -> bool:
        """Verify volume conservation."""
        total_vol = self.Nx * self.Ny * self.Nz * self.V_cell
        return np.isclose(total_vol, self.V_room)
