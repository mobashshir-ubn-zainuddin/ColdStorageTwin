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

    def validate(self) -> Dict[str, Any]:
        """
        Validate the physical coherence of the state.
        Returns a dictionary of validation results.
        """
        props = calculate_psychrometrics(self.T, self.P, self.omega)

        # 1. T > absolute zero (-273.15 °C)
        t_valid = np.all(self.T > -273.15)

        # 2. P > 0
        p_valid = np.all(self.P > 0)

        # 3. pv > 0 and pv < P
        pv = props['pv']
        pv_valid = np.all((pv > 0) & (pv < self.P))

        # 4. omega >= 0
        omega_valid = np.all(self.omega >= 0)

        # 5. rho > 0
        rho_valid = np.all(props['rho_ma'] > 0)

        # 6. Detect supersaturation (RH > 100%)
        rh = props['RH']
        supersaturated = np.any(rh > 100.0)

        return {
            't_valid': t_valid,
            'p_valid': p_valid,
            'pv_valid': pv_valid,
            'omega_valid': omega_valid,
            'rho_valid': rho_valid,
            'supersaturated': supersaturated,
            'is_coherent': t_valid and p_valid and pv_valid and omega_valid and rho_valid
        }

    def get_point_state(self, i: int, j: int, k: int) -> Dict[str, Any]:
