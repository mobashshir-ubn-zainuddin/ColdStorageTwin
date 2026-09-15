"""
3D Finite Difference Method Solver for Heat and Moisture Transfer in Cold Storage.
Refactored to use modular physics modules.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from geometry.mesh import Mesh
from simulation.state import SimulationState
from physics.psychrometrics import calculate_psychrometrics
from physics.properties import thermal_diffusivity, moisture_diffusivity
from physics.condensation import calculate_condensation
from physics.moisture import update_moisture_diffusion
from physics.heat_transfer import update_temperature

@dataclass
class ColdStorageConfig:
    """Configuration for cold storage simulation with moisture"""
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 10.0
    nx: int = 10
    ny: int = 10
    nz: int = 10
    k: float = 0.024 # W/mK (Air)
    rho: float = 1.2 # kg/m3 (Air)
    Cp: float = 1005.0 # J/kgK (Air)
    alpha: Optional[float] = None
    Lv: float = 2.26e6 # J/kg
    Dm: float = 2.2e-5 # m2/s
    T_initial: float = -20.0
    T_wall: float = -17.0
    T_inlet: float = -25.0
    W_initial: float = 0.002
    W_wall: float = 0.000
    W_inlet: float = 0.001
    RH_ambient: float = 0.2
    time_steps: int = 10
    dt: float = 100.0

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.k / (self.rho * self.Cp)

class FiniteDifference3DSolver:
    """
    Refactored Explicit Finite Difference Method (EFDM) solver.
    Uses physics modules for transport and phase change.
    """
    def __init__(self, config: ColdStorageConfig):
        self.config = config
        self.mesh = Mesh(config.Lx, config.Ly, config.Lz, config.nx, config.ny, config.nz)

        # Initialize state
        T = np.full((config.nx, config.ny, config.nz), config.T_initial, dtype=np.float64)
        W = np.full((config.nx, config.ny, config.nz), config.W_initial, dtype=np.float64)
        P = np.full((config.nx, config.ny, config.nz), 101325.0, dtype=np.float64)
        u = np.zeros((config.nx, config.ny, config.nz))
        v = np.zeros((config.nx, config.ny, config.nz))
        w = np.zeros((config.nx, config.ny, config.nz))

        # Inlet boundary conditions
        T[:, :, 0] = config.T_inlet
        W[:, :, 0] = config.W_inlet

        self.state = SimulationState(config.nx, config.ny, config.nz, T, P, u, v, w, W)
        self.state.update_derived_fields()

        # History for visualization
        self.history = [self.state.T.copy()]
        self.moisture_history = [self.state.omega.copy()]
        self.time_history = [0.0]
        self.condensation_rate = np.zeros_like(self.state.T)

    def get_stability_info(self) -> Dict[str, float]:
        # Find max local diffusivity for stability check
        alpha_field = thermal_diffusivity(self.state.T, self.state.P, self.state.omega)
        D_field = moisture_diffusivity(self.state.T, self.state.P, self.state.omega)

        max_alpha = np.max(alpha_field)
        max_D = np.max(D_field)

        # Stability check: alpha * dt * (1/dx^2 + 1/dy^2 + 1/dz^2) <= 1/2
        # For cubic mesh dx=dy=dz, this is 3 * alpha * dt / dx^2 <= 1/2 => alpha * dt / dx^2 <= 1/6
        rT = (max_alpha * self.config.dt) / (self.mesh.dx**2)
        rW = (max_D * self.config.dt) / (self.mesh.dx**2)

        threshold = 1.0 / 6.0
        return {
            'rT': float(rT), 'rW': float(rW), 'threshold': float(threshold),
            'is_stable': bool(rT <= threshold and rW <= threshold),
            'margin_T': float(threshold - rT), 'margin_W': float(threshold - rW),
        }

    def step(self):
        # 1. Update Moisture (Diffusion)
        # Source term S_omega = 0 for baseline
        S_omega = np.zeros_like(self.state.omega)
        omega_new = update_moisture_diffusion(
            self.state.T, self.state.P, self.state.omega,
            self.state.u, self.state.v, self.state.w,
            self.config.dt, self.mesh.dx, self.mesh.dy, self.mesh.dz,
            S_omega
        )

        # 2. Handle Condensation / Phase Change
        cond_results = calculate_condensation(
            self.state.T, self.state.P, omega_new,
            self.config.dt, self.mesh.V_cell
        )
        omega_final = cond_results['omega_new']
        S_latent = cond_results['latent_heat_release']
        self.condensation_rate = cond_results['condensation_rate']

        # 3. Update Temperature (Diffusion + Latent Heat)
        # Source term S_Q = 0 for baseline
        S_Q = np.zeros_like(self.state.T)
        T_new = update_temperature(
            self.state.T, self.state.P, omega_final,
            self.state.u, self.state.v, self.state.w,
            self.config.dt, self.mesh.dx, self.mesh.dy, self.mesh.dz,
            S_Q, S_latent
        )

        # 4. Boundaries (Dirichlet)
        T_new[0, :, :], T_new[-1, :, :], T_new[:, 0, :], T_new[:, -1, :], T_new[:, :, -1] = \
            self.config.T_wall, self.config.T_wall, self.config.T_wall, self.config.T_wall, self.config.T_wall
        T_new[:, :, 0] = self.config.T_inlet

        W_new = omega_final.copy()
        W_new[0, :, :], W_new[-1, :, :], W_new[:, 0, :], W_new[:, -1, :], W_new[:, :, -1] = \
            self.config.W_wall, self.config.W_wall, self.config.W_wall, self.config.W_wall, self.config.W_wall
        W_new[:, :, 0] = self.config.W_inlet

        # Update state
        self.state.T, self.state.omega = T_new, W_new
        self.state.update_derived_fields()

        # Store history
        self.history.append(self.state.T.copy())
        self.moisture_history.append(self.state.omega.copy())
        self.time_history.append(self.time_history[-1] + self.config.dt)

    def solve(self):
        for _ in range(self.config.time_steps):
            self.step()

    def get_statistics(self) -> Dict[str, float]:
        return {
            'min_temp': float(np.min(self.state.T)), 'max_temp': float(np.max(self.state.T)),
            'mean_temp': float(np.mean(self.state.T)), 'std_temp': float(np.std(self.state.T)),
            'min_moisture': float(np.min(self.state.omega)), 'max_moisture': float(np.max(self.state.omega)),
            'mean_moisture': float(np.mean(self.state.omega)), 'std_moisture': float(np.std(self.state.omega)),
            'total_condensation': float(np.sum(self.condensation_rate)),
            'max_condensation': float(np.max(self.condensation_rate)),
        }

    def get_temperature_field(self) -> np.ndarray: return self.state.T.copy()
    def get_moisture_field(self) -> np.ndarray: return self.state.omega.copy()

def create_solver_from_params(**params) -> FiniteDifference3DSolver:
    config = ColdStorageConfig(
        Lx=params.get('Lx', 10.0), Ly=params.get('Ly', 10.0), Lz=params.get('Lz', 10.0),
        nx=params.get('nx', 10), ny=params.get('ny', 10), nz=params.get('nz', 10),
        k=params.get('k', 0.024), rho=params.get('rho', 1.2), Cp=params.get('Cp', 1005.0),
        alpha=params.get('alpha'), Lv=params.get('Lv', 2.26e6), Dm=params.get('Dm', 2.2e-5),
        T_initial=params.get('T_initial', -20.0), T_wall=params.get('T_wall', -17.0), T_inlet=params.get('T_inlet', -25.0),
        W_initial=params.get('W_initial', 0.002), W_wall=params.get('W_wall', 0.000), W_inlet=params.get('W_inlet', 0.001),
        RH_ambient=params.get('RH_ambient', 0.2), time_steps=params.get('time_steps', 10), dt=params.get('dt', 100.0)
    )
    return FiniteDifference3DSolver(config)
