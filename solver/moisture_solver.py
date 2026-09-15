"""
Moisture solver for the Cold Storage Digital Twin.
Implements the FVM conservation equation for humidity ratio.
"""

import numpy as np
from typing import Tuple, Dict, Any
from geometry.mesh import Mesh
from simulation.state import SimulationState
from solver.fv_solver import FVMTransport
from physics.properties import moisture_diffusivity

class MoistureSolver:
    """
    Solves: ∂(rho_da * omega)/∂t + div(rho_da * u * omega) = div(rho_da * D_eff * grad(omega)) + S_omega
    """
    def __init__(self, mesh: Mesh, bc_handler=None):
        self.mesh = mesh
        self.transport = FVMTransport(mesh, bc_handler)

    def solve_step(self, state: SimulationState, dt: float,
                   S_omega: np.ndarray, S_cond: np.ndarray, S_evap: np.ndarray) -> np.ndarray:
        """
        Perform one time-step update of the humidity ratio field.
        """
        T, P, omega = state.T, state.P, state.omega
        u, v, w = state.u, state.v, state.w
        rho_da = state.get_derived('rho_da')
        D_eff = moisture_diffusivity(T, P, omega)

        # Total source = S_omega - S_cond + S_evap
        total_source = S_omega - S_cond + S_evap

        # Solve using FVM transport
        # phi = omega, rho = rho_da
        omega_new, _ = self.transport.integrate_explicit(
            phi=omega, rho=rho_da, u=u, v=v, w=w, Gamma=rho_da * D_eff, S_phi=total_source, dt=dt, variable_name='omega'
        )

        return omega_new
