"""
3D Finite Difference Method Solver for Heat and Moisture Transfer in Cold Storage.
Refactored to use modular SimulationState and Mesh.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from geometry.mesh import Mesh
from simulation.state import SimulationState
from physics.psychrometrics import calculate_psychrometrics

@dataclass
class ColdStorageConfig:
    """Configuration for cold storage simulation with moisture"""
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 10.0
    nx: int = 10
    ny: int = 10
    nz: int = 10
    k: float = 0.5
    rho: float = 1000.0
    Cp: float = 4200.0
    alpha: Optional[float] = None
    Lv: float = 2.26e6
    Dm: float = 1e-6
    T_initial: float = -20.0
    T_wall: float = -20.0
    T_inlet: float = -25.0
    W_initial: float = 0.01
    W_wall: float = 0.005
    W_inlet: float = 0.008
    RH_ambient: float = 0.8
    time_steps: int = 10
    dt: float = 100.0

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.k / (self.rho * self.Cp)

class FiniteDifference3DSolver:
    """
    Refactored Explicit Finite Difference Method (EFDM) solver.
    """
    def __init__(self, config: ColdStorageConfig):
        self.config = config
        self.mesh = Mesh(config.Lx, config.Ly, config.Lz, config.nx, config.ny, config.nz)

        # Stability parameters
        self.rT = self.config.alpha * config.dt / (self.mesh.dx ** 2)
        self.rW = config.Dm * config.dt / (self.mesh.dx ** 2)

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
        self.state.update_derived_fields(calculate_psychrometrics)

        # History for visualization
        self.history = [self.state.T.copy()]
        self.moisture_history = [self.state.omega.copy()]
        self.time_history = [0.0]
        self.condensation_rate = np.zeros_like(self.state.T)

    def is_stable(self) -> bool:
        return self.rT <= 1.0/6.0 and self.rW <= 1.0/6.0

    def get_stability_info(self) -> Dict[str, float]:
        threshold = 1.0 / 6.0
        return {
            'rT': self.rT, 'rW': self.rW, 'threshold': threshold,
            'is_stable': self.is_stable(),
            'margin_T': threshold - self.rT, 'margin_W': threshold - self.rW,
        }

    def calculate_saturation_moisture(self, T: np.ndarray) -> np.ndarray:
        T_ref, W_sat_ref = 20.0, 0.02
        W_sat = W_sat_ref * np.exp(0.05 * (T - T_ref))
        return np.clip(W_sat, 0.001, 0.1)

    def apply_condensation(self, T: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        W_sat = self.calculate_saturation_moisture(T)
        condensing = W > W_sat
        excess = np.where(condensing, W - W_sat, 0.0)
        W_new = np.where(condensing, W_sat, W)
        latent_factor = self.config.Lv / (self.config.rho * self.config.Cp)
        T_new = T + latent_factor * excess
        return T_new, W_new, excess

    def step(self):
        if not self.is_stable():
            raise ValueError("Unstable configuration")

        T_old, W_old = self.state.T.copy(), self.state.omega.copy()
        T_new, W_new = T_old.copy(), W_old.copy()

        # Moisture diffusion
        W_new[1:-1, 1:-1, 1:-1] = (
            W_old[1:-1, 1:-1, 1:-1] + self.rW * (
                W_old[2:, 1:-1, 1:-1] + W_old[:-2, 1:-1, 1:-1] +
                W_old[1:-1, 2:, 1:-1] + W_old[1:-1, :-2, 1:-1] +
                W_old[1:-1, 1:-1, 2:] + W_old[1:-1, 1:-1, :-2] - 6 * W_old[1:-1, 1:-1, 1:-1]
            )
        )

        # Temperature diffusion + latent heat
        latent_factor = self.config.Lv / (self.config.rho * self.config.Cp)
        T_new[1:-1, 1:-1, 1:-1] = (
            T_old[1:-1, 1:-1, 1:-1] + self.rT * (
                T_old[2:, 1:-1, 1:-1] + T_old[:-2, 1:-1, 1:-1] +
                T_old[1:-1, 2:, 1:-1] + T_old[1:-1, :-2, 1:-1] +
                T_old[1:-1, 1:-1, 2:] + T_old[1:-1, 1:-1, :-2] - 6 * T_old[1:-1, 1:-1, 1:-1]
            ) + latent_factor * (W_new[1:-1, 1:-1, 1:-1] - W_old[1:-1, 1:-1, 1:-1])
        )

        T_new, W_new, self.condensation_rate = self.apply_condensation(T_new, W_new)

        # Boundaries
        T_new[0, :, :], T_new[-1, :, :], T_new[:, 0, :], T_new[:, -1, :], T_new[:, :, -1] = self.config.T_wall, self.config.T_wall, self.config.T_wall, self.config.T_wall, self.config.T_wall
        T_new[:, :, 0] = self.config.T_inlet
        W_new[0, :, :], W_new[-1, :, :], W_new[:, 0, :], W_new[:, -1, :], W_new[:, :, -1] = self.config.W_wall, self.config.W_wall, self.config.W_wall, self.config.W_wall, self.config.W_wall
        W_new[:, :, 0] = self.config.W_inlet

        self.state.T, self.state.omega = T_new, W_new
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
    # Handle the mapping between app.py params and ColdStorageConfig
    config = ColdStorageConfig(
        Lx=params.get('Lx', 10.0), Ly=params.get('Ly', 10.0), Lz=params.get('Lz', 10.0),
        nx=params.get('nx', 10), ny=params.get('ny', 10), nz=params.get('nz', 10),
        k=params.get('k', 0.5), rho=params.get('rho', 1000.0), Cp=params.get('Cp', 4200.0),
        alpha=params.get('alpha'), Lv=params.get('Lv', 2.26e6), Dm=params.get('Dm', 1e-6),
        T_initial=params.get('T_initial', -20.0), T_wall=params.get('T_wall', -20.0), T_inlet=params.get('T_inlet', -25.0),
        W_initial=params.get('W_initial', 0.01), W_wall=params.get('W_wall', 0.005), W_inlet=params.get('W_inlet', 0.008),
        RH_ambient=params.get('RH_ambient', 0.8), time_steps=params.get('time_steps', 10), dt=params.get('dt', 100.0)
    )
    return FiniteDifference3DSolver(config)
