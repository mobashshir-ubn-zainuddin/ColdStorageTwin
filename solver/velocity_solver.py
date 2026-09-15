"""
Velocity solver for the Cold Storage Digital Twin.
Implements momentum transport and projection methods.
"""

import numpy as np
from typing import Tuple
from geometry.mesh import Mesh
from simulation.state import SimulationState
from solver.fv_solver import FVMTransport
from physics.airflow import calculate_momentum_sources, get_viscosity_field

class VelocitySolver:
    """
    Solves the momentum equations for a low-Mach incompressible approximation.
    """
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.transport = FVMTransport(mesh)

    def predict_velocity(self, state: SimulationState, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict provisional velocity u*, v*, w* using a momentum predictor.
        Solves: ∂(rho*u)/∂t + div(rho*u*u) = div(mu*grad(u)) + S_u
        """
        T, P, omega = state.T, state.P, state.omega
        u, v, w = state.u, state.v, state.w
        rho = state.get_derived('rho_ma')
        mu = get_viscosity_field(T, P, omega)

        # Momentum sources
        S_u, S_v, S_w = calculate_momentum_sources(rho, T, P, omega)

        # Solve for each component using the FVM transport framework
        # Momentum is transported as phi = u, v, w
        # Note: The transport framework is designed for scalar phi.
        # We treat each velocity component as a scalar transported by the velocity field.

        u_star, _ = self.transport.integrate_explicit(
            phi=u, rho=rho, u=u, v=v, w=w, Gamma=mu, S_phi=S_u, dt=dt, variable_name='u'
        )
        v_star, _ = self.transport.integrate_explicit(
            phi=v, rho=rho, u=u, v=v, w=w, Gamma=mu, S_phi=S_v, dt=dt, variable_name='v'
        )
        w_star, _ = self.transport.integrate_explicit(
            phi=w, rho=rho, u=u, v=v, w=w, Gamma=mu, S_phi=S_w, dt=dt, variable_name='w'
        )

        return u_star, v_star, w_star

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

        # Pressure gradient correction (Central Difference)
        # u_new = u* - (dt/rho) * (P[i+1] - P[i-1]) / (2*dx)

        # X-correction
        u_new[1:-1, :, :] = u_star[1:-1, :, :] - (dt / np.maximum(rho[1:-1, :, :], 1e-5)) * \
                            (P[2:, :, :] - P[:-2, :, :]) / (2 * self.mesh.dx)

        # Y-correction
        v_new[:, 1:-1, :] = v_star[:, 1:-1, :] - (dt / np.maximum(rho[:, 1:-1, :], 1e-5)) * \
                            (P[:, 2:, :] - P[:, :-2, :]) / (2 * self.mesh.dy)

        # Z-correction
        w_new[:, :, 1:-1] = w_star[:, :, 1:-1] - (dt / np.maximum(rho[:, :, 1:-1], 1e-5)) * \
                            (P[:, :, 2:] - P[:, :, :-2]) / (2 * self.mesh.dz)

        return u_new, v_new, w_new
