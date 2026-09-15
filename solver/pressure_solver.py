"""
Pressure solver for the Cold Storage Digital Twin.
Implements the pressure Poisson equation for incompressible/low-Mach flow.
"""

import numpy as np
from geometry.mesh import Mesh
from simulation.state import SimulationState

class PressureSolver:
    """
    Solves grad^2(P) = (rho / dt) * div(u*)
    """
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def solve_pressure(self, u_star: np.ndarray, v_star: np.ndarray, w_star: np.ndarray,
                       rho: np.ndarray, dt: float, P_initial: np.ndarray) -> np.ndarray:
        """
        Compute pressure field P^{n+1} using the projection method.
        """
        nx, ny, nz = u_star.shape
        P = P_initial.copy()

        # Calculate divergence of u*
        div_u = np.zeros_like(u_star)
        # Central difference for divergence
        div_u[1:-1, 1:-1, 1:-1] = (
            (u_star[2:, 1:-1, 1:-1] - u_star[:-2, 1:-1, 1:-1]) / (2 * self.mesh.dx) +
            (v_star[1:-1, 2:, 1:-1] - v_star[1:-1, :-2, 1:-1]) / (2 * self.mesh.dy) +
            (w_star[1:-1, 1:-1, 2:] - w_star[1:-1, 1:-1, :-2]) / (2 * self.mesh.dz)
        )

        # RHS = (rho / dt) * div(u)
        rhs = (rho / np.maximum(dt, 1e-5)) * div_u

        # Solve Poisson equation using Jacobi iteration (baseline)
        # P_new = 1/6 * (P_E + P_W + P_N + P_S + P_T + P_B - dx^2 * rhs)
        for _ in range(100):
            P_old = P.copy()
            P[1:-1, 1:-1, 1:-1] = (1.0 / 6.0) * (
                P_old[2:, 1:-1, 1:-1] + P_old[:-2, 1:-1, 1:-1] +
                P_old[1:-1, 2:, 1:-1] + P_old[1:-1, :-2, 1:-1] +
                P_old[1:-1, 1:-1, 2:] + P_old[1:-1, 1:-1, :-2] -
                (self.mesh.dx**2) * rhs[1:-1, 1:-1, 1:-1]
            )
            # Boundary conditions: zero-gradient (Neumann)
            P[0, :, :] = P[1, :, :]
            P[-1, :, :] = P[-2, :, :]
            P[:, 0, :] = P[:, 1, :]
            P[:, -1, :] = P[:, -2, :]
            P[:, :, 0] = P[:, :, 1]
            P[:, :, -1] = P[:, :, -2]

            if np.max(np.abs(P - P_old)) < 1e-3:
                break

        return P
