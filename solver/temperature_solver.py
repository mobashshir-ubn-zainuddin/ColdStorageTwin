"""
Temperature solver for the Cold Storage Digital Twin.
Implements the FVM energy equation.
"""

import numpy as np
from typing import Tuple, Dict, Any
from geometry.mesh import Mesh
from simulation.state import SimulationState
from solver.fv_solver import FVMTransport
from physics.properties import thermal_conductivity, cp_moist_air

class TemperatureSolver:
    """
    Solves: rho*cp * (dT/dt + u.grad(T)) = div(k*grad(T)) + S_Q + S_latent
    """
    def __init__(self, mesh: Mesh, bc_handler=None):
        self.mesh = mesh
        self.transport = FVMTransport(mesh, bc_handler)

    def solve_step(self, state: SimulationState, dt: float,
                   S_Q: np.ndarray, S_latent: np.ndarray) -> np.ndarray:
        """
        Perform one time-step update of the temperature field.
        """
        T, P, omega = state.T, state.P, state.omega
        u, v, w = state.u, state.v, state.w
        rho = state.get_derived('rho_ma')
        cp = cp_moist_air(T, omega)
        k = thermal_conductivity(T, P, omega)

        # Note: The energy equation is usually solved for enthalpy or temperature.
        # For simplicity, we solve for T and include rho*cp in the transient term.
        # This means the 'rho' in FVMTransport needs to be 'rho * cp'.

        rho_eff = rho * cp

        # Solve using FVM transport
        # phi = T
        # S_phi = (S_Q + S_latent) / (rho * cp)
        S_T = (S_Q + S_latent) / np.maximum(rho_eff, 1e-5)

        T_new, _ = self.transport.integrate_explicit(
            phi=T, rho=rho_eff, u=u, v=v, w=w, Gamma=k, S_phi=S_T, dt=dt, variable_name='T'
        )

        return T_new
