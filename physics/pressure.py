"""
Pressure physics for the Cold Storage Digital Twin.
Handles pressure-related definitions and source terms.
"""

import numpy as np
from typing import Union

def calculate_gauge_pressure(absolute_pressure: Union[float, np.ndarray],
                             ambient_pressure: float = 101325.0) -> Union[float, np.ndarray]:
    """
    Calculate gauge pressure: P_gauge = P_abs - P_ambient [Pa].
    """
    return absolute_pressure - ambient_pressure

def pressure_source_term(P: np.ndarray, S_m: np.ndarray, rho: np.ndarray, dt: float) -> np.ndarray:
    """
    Source term for the pressure Poisson equation:
    S_P = (rho / dt) * div(u)
    """
    # This is typically implemented in the solver.
    return np.zeros_like(P)
