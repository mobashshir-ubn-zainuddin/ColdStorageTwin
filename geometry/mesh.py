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

    # ------------------------------------------------------------------
    # Extended geometry utilities (Module 1, §8-§49)
    # ------------------------------------------------------------------

    @property
    def x_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.Lx, self.Nx + 1)

    @property
    def y_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.Ly, self.Ny + 1)

    @property
    def z_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.Lz, self.Nz + 1)

    @property
    def x_centers(self) -> np.ndarray:
        return (np.arange(self.Nx) + 0.5) * self.dx

    @property
    def y_centers(self) -> np.ndarray:
        return (np.arange(self.Ny) + 0.5) * self.dy

    @property
    def z_centers(self) -> np.ndarray:
        return (np.arange(self.Nz) + 0.5) * self.dz

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)

    @property
    def surface_area(self) -> float:
        return 2.0 * (self.Lx * self.Ly + self.Lx * self.Lz + self.Ly * self.Lz)

    @property
    def characteristic_size(self) -> float:
        return (self.dx * self.dy * self.dz) ** (1.0 / 3.0)

    @property
    def max_aspect_ratio(self) -> float:
        d = (self.dx, self.dy, self.dz)
        return max(d) / min(d)

    def locate(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Cell containing a point, clipped to the domain (plan §29)."""
        i = int(np.clip(np.floor(x / self.dx), 0, self.Nx - 1))
        j = int(np.clip(np.floor(y / self.dy), 0, self.Ny - 1))
        k = int(np.clip(np.floor(z / self.dz), 0, self.Nz - 1))
        return i, j, k

    def interpolate(self, field: np.ndarray, x: float, y: float, z: float) -> float:
        """
        Trilinear interpolation of a cell-centred field at (x, y, z) (plan §30-31).
        Points closer to a wall than half a cell use the nearest cell value in that
        direction. NaN cells (e.g. solid obstacles) are excluded from the weights.
        """
        def axis(q, d, n):
            s = np.clip(q / d - 0.5, 0.0, n - 1)
            i0 = int(np.floor(s))
            i1 = min(i0 + 1, n - 1)
            return i0, i1, s - i0

        i0, i1, xi = axis(x, self.dx, self.Nx)
        j0, j1, eta = axis(y, self.dy, self.Ny)
        k0, k1, zeta = axis(z, self.dz, self.Nz)
        total, wsum = 0.0, 0.0
        for a, wa in ((i0, 1 - xi), (i1, xi)):
            for b, wb in ((j0, 1 - eta), (j1, eta)):
                for c, wc in ((k0, 1 - zeta), (k1, zeta)):
                    w = wa * wb * wc
                    val = field[a, b, c]
                    if w > 0 and np.isfinite(val):
                        total += w * val
                        wsum += w
        return float(total / wsum) if wsum > 0 else float('nan')

    def gaussian_weights(self, position: Tuple[float, float, float], sigma: Optional[float] = None,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Discretely normalised Gaussian source distribution G(x) with sum(G)=1 over
        the (optionally masked) cells (plan §33, §41). Sources are therefore
        exactly conservative on the mesh.
        """
        sigma = sigma or max(self.dx, self.dy, self.dz)
        xs, ys, zs = position
        r2 = (self.X_coords - xs) ** 2 + (self.Y_coords - ys) ** 2 + (self.Z_coords - zs) ** 2
        g = np.exp(-r2 / (2.0 * sigma ** 2))
        g[r2 > (3.0 * sigma) ** 2] = 0.0
        if mask is not None:
            g = np.where(mask, g, 0.0)
        if g.sum() <= 0:
            g = np.zeros(self.shape)
            g[self.locate(xs, ys, zs)] = 1.0
        return g / g.sum()

    def box_mask(self, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
        """Cells whose centres lie inside an axis-aligned box (obstacles, zones)."""
        return ((self.X_coords >= x0) & (self.X_coords <= x1) &
                (self.Y_coords >= y0) & (self.Y_coords <= y1) &
                (self.Z_coords >= z0) & (self.Z_coords <= z1))

    # Boundary face patches -------------------------------------------------
    FACE_AXIS = {'W': (0, 0), 'E': (0, 1), 'S': (1, 0), 'N': (1, 1), 'B': (2, 0), 'T': (2, 1)}

    def face_plane_axes(self, face: str) -> Tuple[int, int]:
        """In-plane axes (width axis, height axis) of a boundary face."""
        normal = self.FACE_AXIS[face][0]
        if normal == 2:
            return 0, 1
        return (1, 2) if normal == 0 else (0, 2)

    def face_area(self, face: str) -> float:
        normal = self.FACE_AXIS[face][0]
        return (self.A_E, self.A_N, self.A_T)[normal]

    def boundary_patch(self, face: str, position: Tuple[float, float, float],
                       width: float, height: float) -> np.ndarray:
        """
        Boolean mask over the 2-D face grid of `face` selecting the face cells whose
        centres fall inside a width x height rectangle centred on `position`.
        Openings smaller than one face cell map to the nearest face cell.
        """
        a, b = self.face_plane_axes(face)
        centers = (self.x_centers, self.y_centers, self.z_centers)
        ca, cb = centers[a], centers[b]
        pa, pb = position[a], position[b]
        A, B = np.meshgrid(ca, cb, indexing='ij')
        mask = (np.abs(A - pa) <= width / 2 + 1e-9) & (np.abs(B - pb) <= height / 2 + 1e-9)
        if not mask.any():
            ia = int(np.argmin(np.abs(ca - pa)))
            ib = int(np.argmin(np.abs(cb - pb)))
            mask[ia, ib] = True
        return mask

    def summary(self) -> Dict[str, float]:
        return {
            'Lx': self.Lx, 'Ly': self.Ly, 'Lz': self.Lz,
            'Nx': self.Nx, 'Ny': self.Ny, 'Nz': self.Nz,
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'n_cells': self.Nx * self.Ny * self.Nz,
            'V_room': self.V_room, 'surface_area': self.surface_area,
            'h': self.characteristic_size, 'aspect_ratio_max': self.max_aspect_ratio,
            'volume_error_pct': abs(self.Nx * self.Ny * self.Nz * self.V_cell - self.V_room) / self.V_room * 100.0,
        }
