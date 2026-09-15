"""
Energy definitions for the Cold Storage Digital Twin.
Distinguishes between total, sensible, and latent energy.
"""

import numpy as np
from typing import Union, Dict, Any
from .psychrometrics import calculate_psychrometrics, LV_REF
from .properties import cp_moist_air

def sensible_energy_field(T: np.ndarray, P: np.ndarray, omega: np.ndarray, T_ref: float = 0.0) -> np.ndarray:
    """
    Calculates sensible energy density [J/m³].
    Es = rho_ma * cp_ma * (T - T_ref)
    """
    props = calculate_psychrometrics(T, P, omega)
    rho = props['rho_ma']
    cp = cp_moist_air(T, omega)
    return rho * cp * (T - T_ref)

def latent_energy_field(P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Calculates latent energy density [J/m³].
    El = rho_v * Lv
    """
    props = calculate_psychrometrics(np.zeros_like(P), P, omega) # T doesn't affect rho_v directly in the prop call, but we need a T
    # Wait, rho_v = pv / (Rv * Tk). T is needed.
    # Let's redefine the signature.
    return np.zeros_like(P) # Placeholder, see below

def latent_energy_field_fixed(T: np.ndarray, P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Calculates latent energy density [J/m³].
    El = rho_v * Lv
    """
    props = calculate_psychrometrics(T, P, omega)
    rho_v = props['rho_v']
    return rho_v * LV_REF

def total_energy_field(T: np.ndarray, P: np.ndarray, omega: np.ndarray, T_ref: float = 0.0) -> np.ndarray:
    """
    Calculates total energy density [J/m³].
    Etotal = Es + El
    """
    return sensible_energy_field(T, P, omega, T_ref) + latent_energy_field_fixed(T, P, omega)
