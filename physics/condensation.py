"""
Condensation and evaporation physics for the Cold Storage Digital Twin.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .psychrometrics import saturation_pressure, LV_REF

def calculate_condensation(T: np.ndarray, P: np.ndarray, omega: np.ndarray,
                          dt: float, V_cell: float) -> Dict[str, np.ndarray]:
    """
    Implements equilibrium condensation based on saturation.

    Logic:
    pws = pws(T)
    omega_s = 0.621945 * pws / (P - pws)
    delta_omega_cond = max(0, omega - omega_s)

    Args:
        T: Temperature [°C]
        P: Absolute pressure [Pa]
        omega: Humidity ratio [kg/kg dry air]
        dt: Timestep [s]
        V_cell: Cell volume [m³]

    Returns:
        Dict containing:
        - 'omega_new': Updated humidity ratio [kg/kg dry air]
        - 'condensation_rate': Rate of condensation [kg/(m³·s)]
        - 'condensed_mass': Mass of condensed water per cell [kg]
        - 'latent_heat_release': Latent heat released [W/m³]
    """
    # Saturation humidity ratio
    pws = saturation_pressure(T)
    # Guard against P <= pws
    omega_s = (0.621945 * pws) / np.maximum(P - pws, 1e-5)

    # Condensation amount
    delta_omega = np.maximum(0.0, omega - omega_s)
    omega_new = omega - delta_omega

    # rho_da = pda / (Rda * Tk)
    R_DA = 287.058
    Tk = T + 273.15
    pv = (omega * P) / (0.621945 + omega)
    pda = P - pv
    rho_da = pda / (R_DA * Tk)

    # m_cond = rho_da * delta_omega * V_cell [kg]
    condensed_mass = rho_da * delta_omega * V_cell

    # condensation_rate = rho_da * delta_omega / dt [kg/(m³·s)]
    condensation_rate = (rho_da * delta_omega) / np.maximum(dt, 1e-5)

    # latent_heat_release = condensation_rate * Lv [W/m³]
    latent_heat_release = condensation_rate * LV_REF

    return {
        'omega_new': omega_new,
        'condensation_rate': condensation_rate,
        'condensed_mass': condensed_mass,
        'latent_heat_release': latent_heat_release
    }

def calculate_evaporation(T: np.ndarray, P: np.ndarray, omega: np.ndarray,
                           liquid_inventory: np.ndarray,
                           k_evap: float = 1e-5,
                           dt: float = 1.0,
                           V_cell: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optional evaporation mechanism.
    S_evap = rho_da * k_evap * max(0, omega_s - omega)

    Args:
        liquid_inventory: Liquid water mass per cell [kg]

    Returns:
        Tuple of (evaporation_rate [kg/(m³·s)], omega_new [kg/kg dry air])
    """
    pws = saturation_pressure(T)
    omega_s = (0.621945 * pws) / np.maximum(P - pws, 1e-5)

    R_DA = 287.058
    Tk = T + 273.15
    pv = (omega * P) / (0.621945 + omega)
    pda = P - pv
    rho_da = pda / (R_DA * Tk)

    # Potential evaporation rate
    S_evap_pot = rho_da * k_evap * np.maximum(0.0, omega_s - omega)

    # Mass available for evaporation
    m_evap_pot = S_evap_pot * V_cell * dt
    m_evap_actual = np.minimum(m_evap_pot, liquid_inventory)

    # Actual rate
    S_evap_actual = m_evap_actual / (V_cell * np.maximum(dt, 1e-5))

    # Update omega
    omega_new = omega + (S_evap_actual * dt / rho_da)

    return S_evap_actual, omega_new
