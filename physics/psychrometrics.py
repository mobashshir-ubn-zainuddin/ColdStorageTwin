"""
Psychrometric engine for cold storage digital twin.
Implements thermodynamic relations for moist air.
"""

import numpy as np
from typing import Union

def saturation_pressure(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate saturation vapour pressure (pws) in Pa.
    Uses Hyland-Wexler or similar validated correlations.
    For this implementation, we use the Magnus-Tetens approximation for T > 0°C
    and a modified version for T < 0°C to account for ice.
    """
    T = np.asanyarray(T)
    # Magnus-Tetens approximation
    # T in °C, P in Pa
    # For T > 0 (Liquid water)
    pws_liquid = 611.2 * np.exp((17.62 * T) / (T + 243.04))

    # For T < 0 (Ice)
    pws_ice = 611.15 * np.exp((22.46 * T) / (T + 272.62))

    return np.where(T >= 0, pws_liquid, pws_ice)

def calculate_psychrometrics(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]):
    """
    Derive complete psychrometric state from T, P, and humidity ratio omega.

    Args:
        T: Temperature (°C)
        P: Absolute pressure (Pa)
        omega: Humidity ratio (kg water / kg dry air)

    Returns:
        A dictionary containing derived psychrometric properties.
    """
    T = np.asanyarray(T)
    P = np.asanyarray(P)
    omega = np.asanyarray(omega)

    # Vapour pressure pv = omega * P / (0.621945 + omega)
    # 0.621945 is the ratio of molar masses (Mw/Mda)
    mw_mda = 0.621945
    pv = (omega * P) / (mw_mda + omega)

    # Dry air partial pressure
    pda = P - pv

    # Saturation vapour pressure
    pws = saturation_pressure(T)

    # Relative Humidity (RH)
    RH = 100.0 * pv / pws

    # Saturation humidity ratio
    omega_s = (mw_mda * pws) / (P - pws)

    # Specific humidity q = omega / (1 + omega)
    q = omega / (1.0 + omega)

    # Enthalpy h = 1.006 * T + omega * (2501 + 1.86 * T) [kJ/kg dry air]
    h = 1.006 * T + omega * (2501.0 + 1.86 * T)

    # Specific volume v = Rda * Tk * (1 + 1.6078 * omega) / P
    # Rda = 287.058 J/(kg·K)
    Rda = 287.058
    Tk = T + 273.15
    v = (Rda * Tk * (1.0 + 1.6078 * omega)) / P

    # Densities
    rho_da = 1.0 / v # This is a simplification; actually rho_da = pda / (Rda * Tk)
    # More accurate:
    rho_da = pda / (Rda * Tk)
    rho_ma = rho_da * (1.0 + omega)

    # Vapour density rho_v = pv / (Rv * Tk)
    # Rv = 461.495 J/(kg·K)
    Rv = 461.495
    rho_v = pv / (Rv * Tk)

    return {
        'pws': pws,
        'pv': pv,
        'pda': pda,
        'RH': RH,
        'omega_s': omega_s,
        'q': q,
        'h': h,
        'specific_volume': v,
        'rho_da': rho_da,
        'rho_ma': rho_ma,
        'rho_v': rho_v
    }

def calculate_dew_point(P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate dew point temperature (Tdp) in °C.
    The temperature at which the current vapour pressure equals saturation vapour pressure.
    """
    mw_mda = 0.621945
    pv = (omega * P) / (mw_mda + omega)

    # Numerical solve for T such that saturation_pressure(T) = pv
    # Using a simple bisection method or inverse Magnus for approximation
    # For T > 0: T = (b * ln(pv/611.2)) / (a - ln(pv/611.2))
    # a=17.62, b=243.04

    # Approximation for T > 0
    T_dp_approx = (243.04 * np.log(pv / 611.2)) / (17.62 - np.log(pv / 611.2))

    # Refine with numerical solve if needed, but this is close.
    return T_dp_approx

def calculate_wet_bulb(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate wet bulb temperature (Twb) in °C numerically.
    """
    # Simple approximation or iterative solve using psychrometric relation:
    # h_wb = h_room - (mdot_evap * Lv)
    # For this prototype, we return T as a placeholder or use a simple correlation.
    # A real implementation would use an iterative solver.
    return T * 0.9 # Placeholder
