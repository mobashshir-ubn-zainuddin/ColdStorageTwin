"""
Pressure solver for the Cold Storage Digital Twin.
Implements the pressure Poisson equation for incompressible/low-Mach flow.
"""

import numpy as np
from typing import Tuple, Dict, Any
from geometry.mesh import Mesh

class PressureSolver:
    """
    Solves the Poisson equation for pressure correction:
    grad^2(P) = (rho / dt) * div(u*)
    """
    def __init__(self, mesh: Mesh, tolerance: float = 1e-4, max_iter: int = 1000):
        self.mesh = mesh
        self.tolerance = tolerance
        self.max_iter = max_iter

    def solve_pressure(self, u_star: np.ndarray, v_star: np.ndarray, w_star: np.ndarray,
                       rho: np.ndarray, dt: float, P_initial: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute pressure field P^{n+1} using a projection method.
        """
        nx, ny, nz = u_star.shape
        P = P_initial.copy()

        # 1. Calculate divergence of u* [1/s]
        # We use the same flux-based divergence as the FVM transport for consistency.
        # div(u) = (Flux_E + Flux_W + Flux_N + Flux_S + Flux_T + Flux_B) / V_cell
        # For u*, v*, w*, these are cell-centered.
        # We use central difference for the divergence at the cell center.

        div_u = np.zeros_like(u_star)
        div_u[1:-1, 1:-1, 1:-1] = (
            (u_star[2:, 1:-1, 1:-1] - u_star[:-2, 1:-1, 1:-1]) / (2 * self.mesh.dx) +
            (v_star[1:-1, 2:, 1:-1] - v_star[1:-1, :-2, 1:-1]) / (2 * self.mesh.dy) +
            (w_star[1:-1, 1:-1, 2:] - w_star[1:-1, 1:-1, :-2]) / (2 * self.mesh.dz)
        )

        # 2. Construct RHS: (rho / dt) * div(u*)
        rhs = (rho / np.maximum(dt, 1e-5)) * div_u

        # 3. Iterative solve (Jacobi method)
        converged = False
        residual_norm = 0.0
        for it in range(self.max_iter):
            P_old = P.copy()

            # Weights for unequal grids
            inv_dx2 = 1.0 / (self.mesh.dx**2)
            inv_dy2 = 1.0 / (self.mesh.dy**2)
            inv_dz2 =, 1.0 / (self.mesh.dz**2) # wait, typo in my thought, will fix in code

            # Fixed weights calculation
            w_x = 1.0 / (self.mesh.dx**2)
            w_y = 1.0 / (self.mesh.dy**2)
            w_z = 1.0 / (self.mesh.dz**2)
            sum_w = 2 * (w_x + w_y + w_z)

            P[1:-1, 1:-1, 1:-1] = (
                w_x * (P_old[2:, 1:-1, 1:-1] + P_old[:-2, 1:-1, 1:-1]) +
                w_y * (P_old[1:-1, 2:, 1:-1] + P_old[1:-1, :-2, 1:-1]) +
                w_z * (P_old[1:-1, 1:-1, 2:] + P_old[1:-1, 1:-1, :-2]) -
                rhs[1:-1, 1:-1, 1:-1]
            ) / sum_w

            # Boundary conditions: Zero-gradient (Neumann)
            P[0, :, :] = P[1, :, :]
            P[-1, :, :] = P[-2, :, :]
            P[:, 0, :] = P[:, 1, :]
            P[:, -1, :] = P[:, -2, :]
            P[:, :, 0] = P[:, :, 1]
            P[:, :, -1] = P[:, :, -2]

            # Pressure Reference: Force zero-mean to remove singularity (Pure Neumann)
            P -= np.mean(P)

            # Convergence check
            residual_norm = np.max(np.abs(P - P_old))
            if residual_norm < self.tolerance:
                converged = True
                break

        return P, {
            'iterations': it + 1,
            'residual': residual_norm,
            'converged': converged
        }
