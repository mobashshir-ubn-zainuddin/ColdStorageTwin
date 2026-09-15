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

    def step(self, state: SimulationState, dt: float) -> Tuple[SimulationState, Dict[str, Any]]:
        """
        Execute one complete coupled FVM timestep.
        """
        # A. Obtain current properties
        state.update_derived_fields()
        T, P, omega = state.T, state.P, state.omega
        u, v, w = state.u, state.v, state.w
        rho = state.get_derived('rho_ma')

        # B. Momentum Prediction
        u_star, v_star, w_star = self.velocity_solver.predict_velocity(state, dt)

        # C. Pressure Correction
        P_new, p_diag = self.pressure_solver.solve_pressure(
            u_star, v_star, w_star, rho, dt, P
        )

        # D. Velocity Projection
        u_new, v_new, w_new = self.velocity_solver.project_velocity(
            u_star, v_star, w_star, P_new, rho, dt
        )

        # E. Moisture Transport
        # Baseline: zero general sources
        S_omega_vol = np.zeros_like(omega)
        S_evap_vol = np.zeros_like(omega)
        # First pass transport
        omega_trans = self.moist_solver.solve_step(
            state, dt, S_omega_vol, np.zeros_like(omega), S_evap_vol
        )

        # F. Condensation Coupling
        # Calculate condensation based on transport result
        cond_results = calculate_condensation(T, P, omega_trans, dt, self.mesh.V_cell)
        omega_final = cond_results['omega_new']
        S_cond_vol = cond_results['condensation_rate']
        S_latent_vol = cond_results['latent_heat_release']

        # G. Temperature Transport
        # Use latent heat from condensation
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
        Calculate continuity and conservation residuals.
        """
        rho_old = state.get_derived('rho_ma')
        rho_new = new_state.get_derived('rho_ma')
        u, v, w = new_state.u, new_state.v, new_state.w
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz

        # Continuity: d(rho)/dt + div(rho*u) = 0
        # divergence calculation consistent with pressure solver
        div_rho_u = np.zeros_like(rho_new)
        div_rho_u[1:-1, 1:-1, 1:-1] = (
            (rho_new[2:, 1:-1, 1:-1]*u[2:, 1:-1, 1:-1] - rho_new[:-2, 1:-1, 1:-1]*u[:-2, 1:-1, 1:-1]) / (2*dx) +
            (rho_new[1:-1, 2:, 1:-1]*v[1:-1, 2:, 1:-1] - rho_new[1:-1, :-2, 1:-1]*v[1:-1, :-2, 1:-1]) / (2*dy) +
            (rho_new[1:-1, 1:-1, 2:]*w[1:-1, 1:-1, 2:] - rho_new[1:-1, 1:-1, :-2]*w[1:-1, 1:-1, :-2]) / (2*dz)
        )

        mass_residual = (rho_new - rho_old) / np.maximum(dt, 1e-5) + div_rho_u

        return {
            'mass_residual_max': np.max(np.abs(mass_residual)),
            'mass_residual_mean': np.mean(np.abs(mass_residual)),
            'continuity_error': np.sum(mass_residual) * self.mesh.V_cell
        }
