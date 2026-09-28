"""
Coupled FVM Solver for the Cold Storage Digital Twin.
Orchestrates the interaction between momentum, pressure, moisture, and heat.
"""

import numpy as np
from typing import Tuple, Dict, Any
from geometry.mesh import Mesh
from simulation.state import SimulationState
from solver.pressure_solver import PressureSolver
from solver.velocity_solver import VelocitySolver
from solver.temperature_solver import TemperatureSolver
from solver.moisture_solver import MoistureSolver
from solver.timestep import TimestepController
from physics.psychrometrics import calculate_psychrometrics
from physics.condensation import calculate_condensation
from physics.boundary_conditions import BoundaryHandler, BoundaryCondition

class FVCoupledSolver:
    """
    Main coupled solver that implements the transient timestep loop.
    """
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.bc_handler = BoundaryHandler()
        self.pressure_solver = PressureSolver(mesh)
        self.velocity_solver = VelocitySolver(mesh, self.bc_handler)
        self.temp_solver = TemperatureSolver(mesh, self.bc_handler)
        self.moist_solver = MoistureSolver(mesh, self.bc_handler)
        self.ts_controller = TimestepController(mesh)

    def step(self, state: SimulationState, dt: float) -> Tuple[SimulationState, Dict[str, Any]]:
        """
        Execute one complete coupled FVM timestep.
        """
        # A. Obtain current properties
        state.update_derived_fields()
        T, P, omega = state.T, state.P, state.omega
        u, v, w = state.u, state.v, state.w
        rho = state.get_derived('rho_ma')
        rho_da = state.get_derived('rho_da')

        # B. Momentum Prediction
        u_star, v_star, w_star = self.velocity_solver.predict_velocity(state, dt)

        # C. Pressure Correction
        # Note: PressureSolver now uses consistent FVM-like divergence
        P_mean = np.mean(P)
        P_new, p_diag = self.pressure_solver.solve_pressure(
            u_star, v_star, w_star, rho, dt, P
        )
        P_new += P_mean # Restore the mean pressure to maintain absolute pressure levels

        # D. Velocity Projection
        u_new, v_new, w_new = self.velocity_solver.project_velocity(
            u_star, v_star, w_star, P_new, rho, dt
        )

        # E. Moisture Transport
        # Baseline: zero general sources
        S_omega_vol = np.zeros_like(omega)
        S_evap_vol = np.zeros_like(omega)
        omega_trans = self.moist_solver.solve_step(
            state, dt, S_omega_vol, np.zeros_like(omega), S_evap_vol
        )

        # F. Condensation Coupling
        # Calculate condensation based on transport result
        cond_results = calculate_condensation(T, P, omega_trans, dt, self.mesh.V_cell)
        omega_final = cond_results['omega_new']
        S_cond_vol = cond_results['condensation_rate']
        S_latent_vol = cond_results['latent_heat_release']
        m_cond = cond_results['condensed_mass']

        # G. Temperature Transport
        # Baseline: zero sensible sources
        S_Q_vol = np.zeros_like(T)
        T_new = self.temp_solver.solve_step(
            state, dt, S_Q_vol, S_latent_vol
        )

        # H. Update Simulation State
        new_state = SimulationState(
            state.nx, state.ny, state.nz,
            T_new, P_new, u_new, v_new, w_new, omega_final
        )
        new_state.update_derived_fields()

        # I. Diagnostics
        diag = self._compute_diagnostics(state, new_state, dt)

        return new_state, diag

    def _compute_diagnostics(self, state: SimulationState, new_state: SimulationState, dt: float) -> Dict[str, Any]:
        """
        Calculate continuity and conservation residuals using THE SAME FVM fluxes.
        """
        rho_old = state.get_derived('rho_ma')
        rho_new = new_state.get_derived('rho_ma')
        u, v, w = new_state.u, new_state.v, new_state.w
        V = self.mesh.V_cell

        # Use FVMTransport to get the same face mass fluxes used in the solvers
        # We can't use a solver instance here easily, so we use the FVMTransport core.
        from solver.fv_solver import FVMTransport
        transport = FVMTransport(self.mesh, self.bc_handler)
        mass_fluxes = transport.compute_face_mass_fluxes(rho_new, u, v, w)

        # Net mass flux leaving each cell
        net_mass_flux = (
            mass_fluxes['E'] + mass_fluxes['W'] +
            mass_fluxes['N'] + mass_fluxes['S'] +
            mass_fluxes['T'] + mass_fluxes['B']
        )

        # Continuity Residual: d(rho*V)/dt + net_mass_flux = S_m * V
        # S_m = 0 for baseline
        mass_residual = (rho_new * V - rho_old * V) / np.maximum(dt, 1e-5) + net_mass_flux

        return {
            'mass_residual_max': np.max(np.abs(mass_residual)),
            'mass_residual_mean': np.mean(np.abs(mass_residual)),
            'continuity_error': np.sum(mass_residual),
            'state_valid': new_state.validate()['is_coherent']
        }
