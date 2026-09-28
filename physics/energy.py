"""
Energy definitions for the Cold Storage Digital Twin.
Distinguishes between total, sensible, and latent energy.

Reference State:
T_ref = 0 °C
P_ref = 101325 Pa
"""

import numpy as np
from typing import Union, Dict, Any
from .psychrometrics import calculate_psychrometrics, LV_REF, L_VAP0
from .properties import cp_moist_air

T_REF = 0.0  # Reference temperature [°C]

def sensible_energy_density(T: np.ndarray, P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Calculates sensible energy density [J/m³].
    e_s = rho_ma * cp_ma * (T - T_ref)
    """
    props = calculate_psychrometrics(T, P, omega)
    rho_ma = props['rho_ma']
    cp = cp_moist_air(T, omega)
    return rho_ma * cp * (T - T_REF)

def latent_energy_density(T: np.ndarray, P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Calculates latent energy density [J/m³].
    e_l = rho_v * Lv
    """
    props = calculate_psychrometrics(T, P, omega)
    rho_v = props['rho_v']
    return rho_v * LV_REF

def total_energy_density(T: np.ndarray, P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Calculates total energy density [J/m³].
    e_total = e_s + e_l
    """
    return sensible_energy_density(T, P, omega) + latent_energy_density(T, P, omega)

def sensible_energy(mass_ma: np.ndarray, T: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Calculates total sensible energy [J].
    E_s = m_ma * cp_ma * (T - T_ref)
    """
    cp = cp_moist_air(T, omega)
    return mass_ma * cp * (T - T_REF)

def latent_energy(mass_v: np.ndarray) -> np.ndarray:
    """
    Calculates total latent energy [J].
    E_l = m_v * Lv
    """
    return mass_v * LV_REF

def total_energy(mass_ma: np.ndarray, mass_v: np.ndarray, T: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Calculates total energy [J].
    E_total = E_s + E_l
    """
    return sensible_energy(mass_ma, T, omega) + latent_energy(mass_v)


def sensible_energy_field(T: np.ndarray, P: np.ndarray, omega: np.ndarray, T_ref: float = T_REF) -> np.ndarray:
    """Sensible energy density rho_ma * cp_ma * (T - T_ref) [J/m³] with an explicit reference temperature."""
    props = calculate_psychrometrics(T, P, omega)
    return props['rho_ma'] * cp_moist_air(T, omega) * (np.asarray(T) - T_ref)


def latent_energy_field(T: np.ndarray, P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Latent energy density rho_da * omega * L [J/m³] (plan Module 4 §26)."""
    props = calculate_psychrometrics(T, P, omega)
    return props['rho_da'] * np.asarray(omega) * L_VAP0


def total_energy_field(T: np.ndarray, P: np.ndarray, omega: np.ndarray, T_ref: float = T_REF) -> np.ndarray:
    """Total energy density: sensible + latent [J/m³]."""
    return sensible_energy_field(T, P, omega, T_ref) + latent_energy_field(T, P, omega)
