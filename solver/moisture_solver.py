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
        self.bc_handler = bc_handler
        self.transport = FVMTransport(mesh, bc_handler)

    def solve_step(self, state: SimulationState, dt: float,
                   S_omega_vol: np.ndarray, S_cond_vol: np.ndarray, S_evap_vol: np.ndarray) -> np.ndarray:
        """
        Perform one time-step update of the humidity ratio field.

        Args:
            S_omega_vol: General moisture source [kg/(m³·s)]
            S_cond_vol: Condensation sink [kg/(m³·s)]
            S_evap_vol: Evaporation source [kg/(m³·s)]
        """
        T, P, omega = state.T, state.P, state.omega
        u, v, w = state.u, state.v, state.w
        rho_da = state.get_derived('rho_da')
        D_eff = moisture_diffusivity(T, P, omega)

        # Total volumetric source [kg/(m³·s)]
        S_total_vol = S_omega_vol - S_cond_vol + S_evap_vol

        # The conserved quantity is rho_da * omega.
        # In FVMTransport:
        # phi = omega
        # rho = rho_da
        # Gamma = rho_da * D_eff
        # S_phi = S_total_vol (volumetric source)

        omega_new, _ = self.transport.integrate_explicit(
            phi=omega, rho=rho_da, u=u, v=v, w=w, Gamma=rho_da * D_eff, S_phi=S_total_vol, dt=dt, variable_name='omega'
        )

        return omega_new
