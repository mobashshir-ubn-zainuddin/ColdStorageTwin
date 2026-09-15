"""
Simulation State Management for the Cold Storage Digital Twin.
Stores the primary thermodynamic and kinematic variables for every cell.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationState:
    """
    Holds the state of the simulation at a given time t.
    State is defined on cell centers.
    """
    # Mesh dimensions
    nx: int
    ny: int
    nz: int

    # Primary Variables (as 3D numpy arrays)
    T: np.ndarray      # Temperature (°C)
    P: np.ndarray      # Absolute Pressure (Pa)
    u: np.ndarray      # Velocity X (m/s)
    v: np.ndarray      # Velocity Y (m/s)
    w: np.ndarray      # Velocity Z (m/s)
    omega: np.ndarray  # Humidity Ratio (kg/kg dry air)

    # Derived Fields (cached for efficiency)
    derived: Dict[str, np.ndarray] = None

    def __post_init__(self):
        if self.derived is None:
            self.derived = {}

    def update_derived_fields(self, psych_engine_func):
        """
        Recalculate all derived psychrometric and thermodynamic properties.

        Args:
            psych_engine_func: Function that takes (T, P, omega) and returns derived properties.
        """
        # Calculate for the whole field
        props = psych_engine_func(self.T, self.P, self.omega)
        for key, value in props.items():
            self.derived[key] = value

    def get_point_state(self, i: int, j: int, k: int) -> Dict[str, Any]:
        """
        Get the complete state at cell (i, j, k).
        """
        state = {
            'T': self.T[i, j, k],
            'P': self.P[i, j, k],
            'u': self.u[i, j, k],
            'v': self.v[i, j, k],
            'w': self.w[i, j, k],
            'omega': self.omega[i, j, k]
        }
        if self.derived:
            for key, field in self.derived.items():
                state[key] = field[i, j, k]
        return state
