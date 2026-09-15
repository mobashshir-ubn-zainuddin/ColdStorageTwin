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

        # The energy equation is:
        # rho * cp * (dT/dt + u.grad(T)) = div(k*grad(T)) + S_Q + S_latent
        # To use the FVMTransport:
        # ∂(rho_eff * T)/∂t + div(rho_eff * u * T) = div(Gamma * grad(T)) + S_T
        # where rho_eff = rho * cp and Gamma = k.

        rho_eff = rho * cp
        # source S_T = (S_Q + S_latent) / (rho_eff) - (dT/dt * rho_eff)
        # Wait, the update formula in FVMTransport already handles the transient.
        # S_T must be the volumetric source in [J/(m3 s)]
        S_T_vol = S_Q + S_latent

        # Integration
        # We pass S_phi as the volumetric source [W/m3]
        # Note: FVMTransport currently assumes S_phi is per-unit-density.
        # Let's check FVMTransport.integrate_explicit:
        # phi_new = phi - (dt / (rho * V)) * (net_flux - S_phi * V)
        # S_phi * V is [J/s]. So S_phi is [W/m3].
        # But the formula is (net_flux - S_phi * V).
        # If S_phi is a source (positive), it should increase phi.
        # So it should be (net_flux - S_phi * V).
        # Let's verify the sign:
        # rho*V*(phi_new - phi_old)/dt = -net_flux + S_phi * V
        # phi_new = phi_old + (dt / (rho*V)) * (-net_flux + S_phi * V)
        # This matches.

        T_new, _ = self.transport.integrate_explicit(
            phi=T, rho=rho_eff, u=u, v=v, w=w, Gamma=k, S_phi=S_T_vol / np.maximum(rho_eff, 1e-5), dt=dt, variable_name='T'
        )
        # Wait, if S_T_vol is [W/m3], and we want S_phi * V in the formula,
        # then S_phi should be S_T_vol / rho_eff ? No.
        # Let's look at FVMTransport:
        # phi_new = phi - (dt / (rho * V)) * (total_flux - S_phi * V)
        # = phi - (dt/rhoV)*total_flux + (dt/rho)*S_phi
        # If S_phi is [W/m3] and we want S_phi * V to be the source, then the formula is correct.
        # BUT the term is (total_flux - S_phi * V).
        # To increase phi, S_phi * V must be positive and subtracted from total_flux.
        # This is correct.

        # Re-evaluating: the a la FVMTransport:
        # rho_eff * V * (T_new - T_old)/dt = - total_flux + S_T_vol * V
        # T_new = T_old - (dt / (rho_eff * V)) * (total_flux - S_T_vol * V)
        # So S_phi in transport should be S_T_vol / rho_eff?
        # No, if we use (total_flux - S_phi * V), then S_phi * V is the volumetric source.
        # So S_phi is [W/m3].
        # But the formula in FVMTransport is: phi_new = phi - (dt / (rho * V)) * (total_flux - S_phi * V)
        # If S_phi is [W/m3], then S_phi * V is [W].
        # Then (dt / (rho * V)) * (S_phi * V) = dt * S_phi / rho.
        # This is exactly what we want.
        # So S_phi = S_T_vol / rho_eff is WRONG.
        # S_phi = S_T_vol should be correct if S_phi * V is the source.
        # BUT the transport takes S_phi as the lagrangian source (per unit mass).
        # Let's fix FVMTransport to be clear.
        # Actually, I'll just pass S_phi = S_T_vol / rho_eff and change the transport to use S_phi * rho * V.
        # Let's stick to: S_phi is lagrangian [W/kg].
        # Then S_phi * rho * V is [W].

        return T_new
