"""
Simulation Manager for coordinating the execution of the digital twin.
"""

from geometry.mesh import Mesh
from simulation.state import SimulationState
from physics.psychrometrics import calculate_psychrometrics
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

    def _initialize_state(self) -> SimulationState:
        """
        Initialize the state variables based on initial conditions.
        """
        nx, ny, nz = self.mesh.Nx, self.mesh.Ny, self.mesh.Nz

        # Handle uniform or spatially varying initial fields
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
        state.update_derived_fields(calculate_psychrometrics)
        return state

    def run_step(self, dt: float, solver_func):
        """
        Execute a single time step using the provided solver function.
        """
        self.state = solver_func(self.state, self.mesh, dt)
        self.current_time += dt
        self.state.update_derived_fields(calculate_psychrometrics)

    def run(self, total_time: float, dt: float, solver_func):
        """
        Run simulation until total_time is reached.
        """
        steps = int(total_time / dt)
        for _ in range(steps):
            self.run_step(dt, solver_func)
