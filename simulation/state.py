"""
Simulation State Management for the Cold Storage Digital Twin.
Represents the primary state and derives thermodynamic properties.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Union
from physics.psychrometrics import calculate_psychrometrics

@dataclass
class SimulationState:
    """
    Holds the primary state variables and derived properties.
    Primary State Q = [T, P, u, v, w, omega]
    """
    # Mesh dimensions
    nx: int
    ny: int
    nz: int

    # Primary Variables (3D numpy arrays)
    T: np.ndarray      # Temperature [°C]
    P: np.ndarray      # Absolute Pressure [Pa]
    u: np.ndarray      # Velocity X [m/s]
    v: np.ndarray      # Velocity Y [m/s]
    w: np.ndarray      # Velocity Z [m/s]
    omega: np.ndarray  # Humidity Ratio [kg water / kg dry air]

    # Derived properties cache
    derived: Dict[str, np.ndarray] = None

    def __post_init__(self):
        if self.derived is None:
            self.derived = {}

    def update_derived_fields(self):
        """
        Recalculate all derived psychrometric properties based on current primary state.
        """
        props = calculate_psychrometrics(self.T, self.P, self.omega)
        self.derived.update(props)

    def get_derived(self, name: str) -> np.ndarray:
        """
        Returns a specific derived property field.
        """
        if name not in self.derived:
            # Try to update if missing
            self.update_derived_fields()
            if name not in self.derived:
                raise KeyError(f"Derived property '{name}' not available.")
        return self.derived[name]

    def validate(self) -> Dict[str, Any]:
        """
        Perform strong physical validation of the current state.
        Returns a report on coherence.
        """
        # Basic shape check
        shapes = [self.T.shape, self.P.shape, self.u.shape, self.v.shape, self.w.shape, self.omega.shape]
        shape_ok = all(s == (self.nx, self.ny, self.nz) for s in shapes)

        # Finite check
        finite_ok = np.all(np.isfinite(self.T)) and np.all(np.isfinite(self.P)) and np.all(np.isfinite(self.omega))

        # Physical limits
        t_ok = np.all(self.T > -273.15)
        p_ok = np.all(self.P > 0)
        omega_ok = np.all(self.omega >= 0)

        # Derived limits
        props = calculate_psychrometrics(self.T, self.P, self.omega)
        pv = props['pv']
        pv_ok = np.all((pv >= 0) & (pv < self.P))
        rho_ok = np.all(props['rho_ma'] > 0)

        # Supersaturation detection
        rh = props['RH']
        supersaturated = np.any(rh > 100.0)

        # Invalid omega_s detection
        omega_s = props['omega_s']
        omega_s_ok = np.all(np.isfinite(omega_s))

        is_coherent = shape_ok and finite_ok and t_ok and p_ok and omega_ok and pv_ok and rho_ok and omega_s_ok

        return {
            'is_coherent': is_coherent,
            'details': {
                'shape_ok': shape_ok,
                'finite_ok': finite_ok,
                't_ok': t_ok,
                'p_ok': p_ok,
                'omega_ok': omega_ok,
                'pv_ok': pv_ok,
                'rho_ok': rho_ok,
                'omega_s_ok': omega_s_ok,
                'supersaturated': supersaturated
            }
        }

    def get_local_state(self, i: int, j: int, k: int) -> Dict[str, Any]:
        """
        Return the complete primary and derived state at cell (i, j, k).
        """
        state = {
            'T': self.T[i, j, k],
            'P': self.P[i, j, k],
            'u': self.u[i, j, k],
            'v': self.v[i, j, k],
            'w': self.w[i, j, k],
            'omega': self.omega[i, j, k]
        }
        # Add derived fields
        for name, field in self.derived.items():
            state[name] = field[i, j, k]
        return state
