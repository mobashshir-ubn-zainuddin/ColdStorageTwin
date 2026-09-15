"""
Airflow physics for the Cold Storage Digital Twin.
Handles momentum-related sources and physical properties.
"""

import numpy as np
from typing import Tuple, Union
from .properties import dynamic_viscosity

def calculate_momentum_sources(rho: np.ndarray, T: np.ndarray, P: np.ndarray, omega: np.ndarray,
                                g_accel: float = 9.81) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate momentum source terms S_u, S_v, S_w [kg/(m³·s²)].
    Includes buoyancy effects: S_buoyancy = rho * g * (T - T_ref) / T_ref
    """
    # For now, we implement a simple buoyancy source in the Z direction.
    # T_ref = 288.15 K
    T_ref = 288.15
    Tk = T + 273.15

    # Buoyancy: source proportional to temperature difference
    # S_w = rho * g * (Tk - T_ref) / T_ref
    S_w = rho * g_accel * (Tk - T_ref) / T_ref

    S_u = np.zeros_like(rho)
    S_v = np.zeros_like(rho)

    return S_u, S_v, S_w

def get_viscosity_field(T: np.ndarray, P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Returns the dynamic viscosity field mu [Pa·s].
    """
    return dynamic_viscosity(T, P, omega)
