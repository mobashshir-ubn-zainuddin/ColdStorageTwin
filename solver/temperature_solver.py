"""
Temperature solver for the Cold Storage Digital Twin.
Implements the FVM energy equation:
rho * cp * (dT/dt + u.grad(T)) = div(k*grad(T)) + S_Q + S_latent
"""

import numpy as np
from typing import Tuple, Dict, Any
from geometry.mesh import Mesh
from simulation.state import SimulationState
from solver.fv_solver import FVMTransport
from physics.properties import thermal_conductivity, cp_moist_air

class TemperatureSolver:
    """
    Solves the energy equation using FVM.
    """
    def __init__(self, mesh: Mesh, bc_handler=None):
        self.mesh = mesh
        self.bc_handler = bc_handler
        self.transport = FVMTransport(mesh, bc_handler)

    def solve_step(self, state: SimulationState, dt: float,
                   S_Q: np.ndarray, S_latent: np.ndarray) -> np.ndarray:
        """
        Perform one time-step update of the temperature field.

        Args:
            S_Q: Sensible heat source [W/m3]
            S_latent: Latent heat release [W/m3]
        """
        T, P, omega = state.T, state.P, state.omega
        u, v, w = state.u, state.v, state.w
        rho = state.get_derived('rho_ma')
        cp = cp_moist_air(T, omega)
        k = thermal_conductivity(T, P, omega)

        # The energy equation:
        # rho * cp * V * (T_new - T_old) / dt = -Sum(Fluxes) + (S_Q + S_latent) * V

        # To use FVMTransport.integrate_explicit:
        # phi = T
        # rho_eff = rho * cp
        # S_phi_vol = S_Q + S_latent
        # The formula in fv_solver is:
        # phi_new = phi - (dt / (rho_eff * V)) * (net_flux - S_phi_vol * V)
        # This is exactly what we need.

        rho_eff = rho * cp
        S_T_vol = S_Q + S_latent

        T_new, _ = self.transport.integrate_explicit(
            phi=T, rho=rho_eff, u=u, v=v, w=w, Gamma=k, S_phi=S_T_vol, dt=dt, variable_name='T'
        )

        return T_new
