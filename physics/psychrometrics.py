"""
Psychrometric engine for cold storage digital twin.
Implements thermodynamic relations for moist air.

Primary state variables:
    T: Temperature [°C]
    P: Absolute Pressure [Pa]
    omega: Humidity Ratio [kg water / kg dry air]
"""

import numpy as np
from typing import Union, Dict, Any
from scipy import optimize

# Physical Constants
R_DA = 287.058  # Specific gas constant for dry air [J/(kg·K)]
R_V = 461.495   # Specific gas constant for water vapour [J/(kg·K)]
MW_RATIO = 0.621945  # Molar mass ratio Mw/Mda
LV_REF = 2.26e6  # Latent heat of vaporization [J/kg]

def saturation_pressure(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate saturation vapour pressure (pws) in Pa.
    Uses the Magnus-Tetens approximation for liquid water (T > 0°C)
    and a modified version for ice (T < 0°C).

    Args:
        T: Temperature in °C.
    Returns:
        pws in Pascals (Pa).
    """
    T = np.asanyarray(T)

    # Water (T > 0 °C)
    pws_water = 611.2 * np.exp((17.62 * T) / (T + 243.04))

    # Ice (T < 0 °C)
    pws_ice = 611.15 * np.exp((22.46 * T) / (T + 272.62))

    return np.where(T >= 0, pws_water, pws_ice)

def calculate_psychrometrics(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Dict[str, Any]:
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

    # Vapour pressure: pv = omega * P / (mw_ratio + omega)
    pv = (omega * P) / (MW_RATIO + omega)

    # Dry air partial pressure
    pda = P - pv

    # Saturation vapour pressure
    pws = saturation_pressure(T)

    # Relative Humidity (RH) [0-100%]
    # Guard against pws=0
    RH = 100.0 * pv / np.maximum(pws, 1e-5)

    # Saturation humidity ratio
    # Guard against P <= pws
    omega_s = (MW_RATIO * pws) / np.maximum(P - pws, 1e-5)

    # Specific humidity q = omega / (1 + omega)
    q = omega / (1.0 + omega)

    # Enthalpy: h = 1.006 * T + omega * (2501 + 1.86 * T) [kJ/kg dry air]
    h = 1.006 * T + omega * (2501.0 + 1.86 * T)

    # Specific volume: v = Rda * Tk * (1 + 1.6078 * omega) / P [m3/kg dry air]
    Tk = T + 273.15
    v = (R_DA * Tk * (1.0 + 1.6078 * omega)) / np.maximum(P, 1e-5)

    # Densities
    rho_da = pda / (R_DA * Tk)
    rho_ma = rho_da * (1.0 + omega)
    rho_v = pv / (R_V * Tk)

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
    Solves pws(Tdp) - pv = 0 numerically.
    """
    P = np.asanyarray(P)
    omega = np.asanyarray(omega)
    pv = (omega * P) / (MW_RATIO + omega)

    def solve_tdp(pv_val):
        if pv_val <= 0: return -273.15
        def objective(T):
            return saturation_pressure(T) - pv_val
        try:
            return optimize.brentq(objective, -100.0, 100.0)
        except (ValueError, RuntimeError):
            return np.nan

    if pv.ndim == 0:
        return solve_tdp(pv)

    return np.array([solve_tdp(p) for p in pv.flatten()]).reshape(pv.shape)

def calculate_wet_bulb(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate wet bulb temperature (Twb) in °C numerically.
    """
    T = np.asanyarray(T)
    P = np.asanyarray(P)
    omega = np.asanyarray(omega)

    def solve_twb(T_val, P_val, omega_val):
        def objective(Twb):
            # Saturation humidity ratio at Twb
            pws_twb = saturation_pressure(Twb)
            omega_s_twb = (MW_RATIO * pws_twb) / np.maximum(P_val - pws_twb, 1e-5)

            # Psychrometric relation:
            # omega_s(Twb) = omega + ( (P * (T - Twb)) / (Lv * (Twb + 273.15)) )
            # based on the heat transfer balance at the wet bulb.
            return omega_s_twb - (omega_val + (P_val * (T_val - Twb)) / (LV_REF * (Twb + 273.15)))

        try:
            T_dp = calculate_dew_point(P_val, omega_val)
            # Twb is between dew point and dry bulb
            return optimize.brentq(objective, T_dp if not np.isnan(T_dp) else -100.0, T_val)
        except (ValueError, RuntimeError):
            return T_val

    if T.ndim == 0:
        return solve_twb(T, P, omega)

    res = []
    it = np.nditer([T, P, omega])
    for t, p, o in it:
        res.append(solve_twb(t, p, o))

    return np.array(res).reshape(T.shape)
