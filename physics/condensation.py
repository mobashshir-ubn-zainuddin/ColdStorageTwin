"""
Condensation and evaporation physics for the Cold Storage Digital Twin.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .psychrometrics import saturation_pressure, LV_REF

def calculate_condensation(T: np.ndarray, P: np.ndarray, omega: np.ndarray, dt: float, V_cell: float) -> Dict[str, np.ndarray]:
    """
    Implements equilibrium condensation based on saturation.

    Returns:
        Dict containing:
        - condensation_rate: [kg/(m³·s)]
        - condensed_mass: [kg] per cell
        - latent_heat_release: [W/m³]
    """
    # Saturation humidity ratio
    pws = saturation_pressure(T)
    omega_s = (0.621945 * pws) / (P - pws)

    # Equilibrium condensation: omega_new = min(omega, omega_s)
    # Excess moisture
    delta_omega = np.maximum(0.0, omega - omega_s)

    # rho_da = pda / (Rda * Tk)
    # Need T and P for rho_da
    R_DA = 287.058
    Tk = T + 273.15
    pv = (omega * P) / (0.621945 + omega)
    pda = P - pv
    rho_da = pda / (R_DA * Tk)

    # Mass of condensed water per cell [kg]
    # m_cond = rho_da * delta_omega * V_cell
    condensed_mass = rho_da * delta_omega * V_cell

    # Condensation rate [kg/(m³·s)]
    condensation_rate = (rho_da * delta_omega) / dt

    # Latent heat release [W/m³] = Qdot = (m_cond / (V_cell * dt)) * Lv
    latent_heat_release = condensation_rate * LV_REF

    return {
        'condensation_rate': condensation_rate,
        'condensed_mass': condensed_mass,
        'latent_heat_release': latent_heat_release,
        'omega_new': np.minimum(omega, omega_s)
    }

def calculate_evaporation(T: np.ndarray, P: np.ndarray, omega: np.ndarray,
                         liquid_inventory: np.ndarray,
                         k_evap: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates evaporation rate given liquid water availability.
    S_evap = rho_da * k_evap * max(0, omega_s - omega)
    """
    pws = saturation_pressure(T)
    omega_s = (0.621945 * pws) / (P - pws)

    R_DA = 287.058
    Tk = T + 273.15
    pv = (omega * P) / (0.621945 + omega)
    pda = P - pv
    rho_da = pda / (R_DA * Tk)

    # Potential evaporation rate [kg/(m³·s)]
    S_evap_pot = rho_da * k_evap * np.maximum(0.0, omega_s - omega)

    # Limit by available liquid water: m_evap = min(S_evap * V_cell * dt, liquid_inventory)
    # This function returns the rate.
    # The caller handles the integration.

    return S_evap_pot, omega_s
