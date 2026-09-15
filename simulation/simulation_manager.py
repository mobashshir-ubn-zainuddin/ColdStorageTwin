"""
Simulation Manager for coordinating the execution of the digital twin.
Supports both FDM baseline and FVM coupled solvers.
"""

from geometry.mesh import Mesh
from simulation.state import SimulationState
from physics.psychrometrics import calculate_psychrometrics
from solver.fdm_solver import create_solver_from_params, FiniteDifference3DSolver
from solver.fv_coupled_solver import FVCoupledSolver
import numpy as np

class SimulationManager:
    """
    Coordinates the setup, execution, and data retrieval of the simulation.
    """
    def __init__(self, mesh: Mesh, initial_conditions: dict):
        self.mesh = mesh
        self.ic = initial_conditions
        self.state = self._initialize_state()
        self.current_time = 0.0
        self.history = []

    def _initialize_state(self) -> SimulationState:
        """
        Initialize the state variables based on initial conditions.
        """
        nx, ny, nz = self.mesh.Nx, self.mesh.Ny, self.mesh.Nz

        def get_field(name, default):
            val = self.ic.get(name, default)
            if isinstance(val, (int, float)):
                return np.full((nx, ny, nz), float(val))
            return np.asanyarray(val)

        T = get_field('T_initial', -20.0)
        P = get_field('P_initial', 101325.0)
        u = get_field('u_initial', 0.0)
        v = get_field('v_initial', 0.0)
        w = get_field('w_initial', 0.0)
        omega = get_field('omega_initial', 0.002)

        state = SimulationState(nx, ny, nz, T, P, u, v, w, omega)
        state.update_derived_fields()
        return state

    def run_fdm_step(self, solver: FiniteDifference3DSolver):
        """
        Execute one time step using the FDM baseline solver.
        """
        # The FDM solver manages its own internal state update
        solver.step()
        self.state = solver.state # Sync state back
        self.current_time += solver.config.dt

    def run_fvm_step(self, solver: FVCoupledSolver, dt: float):
        """
        Execute one time step using the Coupled FVM solver.
        """
        new_state, diag = solver.step(self.state, dt)
        self.state = new_state
        self.current_time += dt
        return diag

    def run(self, total_time: float, dt: float, solver_type: str = "fdm", solver_instance=None):
        """
        Run simulation until total_time is reached.
        """
        steps = int(total_time / dt)

        if solver_type == "fdm":
            if solver_instance is None:
                # This is a simplified call; normally config would be passed
                # For now we assume the caller provides the solver or we use defaults
                raise ValueError("FDM solver instance required for run()")
            for _ in range(steps):
                self.run_fdm_step(solver_instance)

        elif solver_type == "fvm":
            if solver_instance is None:
                solver_instance = FVCoupledSolver(self.mesh)
            for _ in range(steps):
                self.run_fvm_step(solver_instance, dt)

        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
