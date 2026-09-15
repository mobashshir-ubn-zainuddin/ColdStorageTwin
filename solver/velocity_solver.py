"""
Velocity solver for the Cold Storage Digital Twin.
Implements momentum updates and projection corrections.
"""

import numpy as np
from geometry.mesh import Mesh
from simulation.state import SimulationState

class VelocitySolver:
    """
    Updates velocity fields based on momentum and pressure gradients.
    """
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def calculate_provisional_velocity(self, state: SimulationState, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute u*, v*, w* by solving the momentum equation without the pressure gradient.
        For the baseline, this just returns current velocity + sources.
        """
        # In a full solver, this would involve advection and diffusion.
        return state.u.copy(), state.v.copy(), state.w.copy()

    def project_velocity(self, u_star: np.ndarray, v_star: np.ndarray, w_star: np.ndarray,
                         P: np.ndarray, rho: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Correct the provisional velocity to be divergence-free.
        u^{n+1} = u* - (dt/rho) * grad(P)
        """
        nx, ny, nz = u_star.shape
        u_new = u_star.copy()
        v_new = v_star.copy()
        w_new = w_star.copy()

        # Gradient of P (Central Difference)
        # u_new = u* - (dt/rho) * (P[i+1] - P[i-1]) / (2*dx)

        # X-direction correction
        u_new[1:-1, :, :] = u_star[1:-1, :, :] - (dt / np.maximum(rho[1:-1, :, :], 1e-5)) * \
                            (P[2:, :, :] - P[:-2, :, :]) / (2 * self.mesh.dx)

        # Y-direction correction
        v_new[:, 1:-1, :] = v_star[:, 1:-1, :] - (dt / np.maximum(rho[:, 1:-1, :], 1e-5)) * \
                            (P[:, 2:, :] - P[:, :-2, :]) / (2 * self.mesh.dy)

        # Z-direction correction
        w_new[:, :, 1:-1] = w_star[:, :, 1:-1] - (dt / np.maximum(rho[:, :, 1:-1], 1e-5)) * \
                            (P[:, :, 2:] - P[:, :, :-2]) / (2 * self.mesh.dz)

        return u_new, v_new, w_new
