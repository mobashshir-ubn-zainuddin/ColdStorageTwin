"""
Psychrometric engine for cold storage digital twin (Module 3).
Implements thermodynamic relations for moist air.

Primary state variables:
    T: Temperature [°C]
    P: Absolute Pressure [Pa]
    omega: Humidity Ratio [kg water / kg dry air]

Saturation pressure uses the Hyland-Wexler formulation (ASHRAE Handbook of
Fundamentals, 2017, Ch. 1, Eq. 5-6): over ice below 0 °C and over liquid water
at or above 0 °C. Every function is vectorised so it can be applied to whole
3-D fields at every time step.
"""

import numpy as np
from typing import Union, Dict, Any

ArrayLike = Union[float, np.ndarray]

# Physical Constants
R_DA = 287.058  # Specific gas constant for dry air [J/(kg·K)]
R_V = 461.495   # Specific gas constant for water vapour [J/(kg·K)]
MW_RATIO = 0.621945  # Molar mass ratio Mw/Mda
LV_REF = 2.26e6  # Latent heat of vaporization used by the legacy FDM model [J/kg]

# Latent heats at 0 °C [J/kg]
L_VAP0 = 2.501e6   # vapour -> liquid
L_FUS = 0.3337e6   # liquid -> ice
L_SUB0 = L_VAP0 + L_FUS  # vapour -> ice (deposition)

# Specific heats [J/(kg·K)]
CP_DA = 1006.0
CP_V = 1860.0

# Hyland-Wexler coefficients (T in K, pws in Pa)
_ICE = (-5.6745359e3, 6.3925247, -9.6778430e-3, 6.2215701e-7,
        2.0747825e-9, -9.4840240e-13, 4.1635019)
_WATER = (-5.8002206e3, 1.3914993, -4.8640239e-2, 4.1764768e-5,
          -1.4452093e-8, 6.5459673)


def _ln_pws(T: np.ndarray) -> np.ndarray:
    Tk = np.asarray(T, dtype=float) + 273.15
    c1, c2, c3, c4, c5, c6, c7 = _ICE
    ln_ice = c1 / Tk + c2 + c3 * Tk + c4 * Tk**2 + c5 * Tk**3 + c6 * Tk**4 + c7 * np.log(Tk)
    c8, c9, c10, c11, c12, c13 = _WATER
    ln_water = c8 / Tk + c9 + c10 * Tk + c11 * Tk**2 + c12 * Tk**3 + c13 * np.log(Tk)
    return np.where(np.asarray(T) >= 0.0, ln_water, ln_ice)


def saturation_pressure(T: ArrayLike) -> ArrayLike:
    """
    Saturation vapour pressure pws(T) [Pa], over ice for T < 0 °C and over
    liquid water for T >= 0 °C (Hyland-Wexler).
    """
    T = np.asarray(T, dtype=float)
    out = np.exp(_ln_pws(np.clip(T, -100.0, 200.0)))
    return out if out.ndim else float(out)


def saturation_pressure_water(T: ArrayLike) -> ArrayLike:
    """Saturation pressure over liquid water (also valid for supercooled water) [Pa]."""
    Tk = np.asarray(T, dtype=float) + 273.15
    c8, c9, c10, c11, c12, c13 = _WATER
    out = np.exp(c8 / Tk + c9 + c10 * Tk + c11 * Tk**2 + c12 * Tk**3 + c13 * np.log(Tk))
    return out if out.ndim else float(out)


def saturation_pressure_ice(T: ArrayLike) -> ArrayLike:
    """Saturation pressure over ice [Pa]."""
    Tk = np.asarray(T, dtype=float) + 273.15
    c1, c2, c3, c4, c5, c6, c7 = _ICE
    out = np.exp(c1 / Tk + c2 + c3 * Tk + c4 * Tk**2 + c5 * Tk**3 + c6 * Tk**4 + c7 * np.log(Tk))
    return out if out.ndim else float(out)


def latent_heat(T: ArrayLike) -> ArrayLike:
    """Latent heat released when vapour changes phase at T: deposition below 0 °C, condensation above."""
    return np.where(np.asarray(T) < 0.0, L_SUB0, L_VAP0)


# --------------------------------------------------------------------------
# Conversions between moisture specifications
# --------------------------------------------------------------------------

def vapour_pressure_from_omega(P: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """pv = omega * P / (0.621945 + omega)"""
    return (np.asarray(omega) * np.asarray(P)) / (MW_RATIO + np.asarray(omega))


def humidity_ratio_from_vapour_pressure(P: ArrayLike, pv: ArrayLike) -> ArrayLike:
    """omega = 0.621945 * pv / (P - pv)"""
    pv = np.asarray(pv, dtype=float)
    return MW_RATIO * pv / np.maximum(np.asarray(P) - pv, 1e-6)


def humidity_ratio_from_rh(T: ArrayLike, P: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """RH in percent [0-100]."""
    pv = np.asarray(RH, dtype=float) / 100.0 * saturation_pressure(T)
    return humidity_ratio_from_vapour_pressure(P, pv)


def humidity_ratio_from_dew_point(P: ArrayLike, T_dp: ArrayLike) -> ArrayLike:
    return humidity_ratio_from_vapour_pressure(P, saturation_pressure(T_dp))


def humidity_ratio_from_specific_humidity(q: ArrayLike) -> ArrayLike:
    q = np.asarray(q, dtype=float)
    return q / (1.0 - q)


def rh_from_humidity_ratio(T: ArrayLike, P: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """RH in percent."""
    return 100.0 * vapour_pressure_from_omega(P, omega) / np.maximum(saturation_pressure(T), 1e-9)


def saturation_humidity_ratio(T: ArrayLike, P: ArrayLike) -> ArrayLike:
    pws = saturation_pressure(T)
    return MW_RATIO * pws / np.maximum(np.asarray(P) - pws, 1e-6)


def humidity_ratio_from_spec(T: float, P: float, mode: str, value: float) -> float:
    """
    Convert any supported moisture specification to omega (plan §58).
    mode: 'rh' [%], 'omega' [kg/kg], 'dew_point' [°C], 'vapour_pressure' [Pa],
          'specific_humidity' [kg/kg], 'absolute_humidity' (vapour density) [kg/m³]
    """
    mode = (mode or 'rh').lower()
    if mode == 'rh':
        return float(humidity_ratio_from_rh(T, P, value))
    if mode == 'omega':
        return float(value)
    if mode == 'dew_point':
        return float(humidity_ratio_from_dew_point(P, value))
    if mode == 'vapour_pressure':
        return float(humidity_ratio_from_vapour_pressure(P, value))
    if mode == 'specific_humidity':
        return float(humidity_ratio_from_specific_humidity(value))
    if mode == 'absolute_humidity':
        pv = value * R_V * (T + 273.15)
        return float(humidity_ratio_from_vapour_pressure(P, pv))
    raise ValueError(f"Unknown moisture specification mode '{mode}'")


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------

def enthalpy(T: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """Moist-air enthalpy h = 1.006 T + omega (2501 + 1.86 T) [kJ/kg dry air]."""
    T = np.asarray(T)
    return 1.006 * T + np.asarray(omega) * (2501.0 + 1.86 * T)


def specific_volume(T: ArrayLike, P: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """v = R_da T_K (1 + 1.6078 omega) / P  [m³/kg dry air]"""
    return R_DA * (np.asarray(T) + 273.15) * (1.0 + 1.6078 * np.asarray(omega)) / np.maximum(P, 1e-6)


def dry_air_density(T: ArrayLike, P: ArrayLike, omega: ArrayLike) -> ArrayLike:
    return 1.0 / specific_volume(T, P, omega)


def moist_air_density(T: ArrayLike, P: ArrayLike, omega: ArrayLike) -> ArrayLike:
    return dry_air_density(T, P, omega) * (1.0 + np.asarray(omega))


def calculate_psychrometrics(T: ArrayLike, P: ArrayLike, omega: ArrayLike) -> Dict[str, Any]:
    """
    Derive the (fast, non-iterative) psychrometric state from T, P, and omega.
    For dew point and wet bulb see complete_state().
    """
    T = np.asanyarray(T, dtype=float)
    P = np.asanyarray(P, dtype=float)
    omega = np.asanyarray(omega, dtype=float)

    pv = (omega * P) / (MW_RATIO + omega)
    pda = P - pv
    pws = saturation_pressure(T)
    RH = 100.0 * pv / np.maximum(pws, 1e-9)
    omega_s = (MW_RATIO * pws) / np.maximum(P - pws, 1e-6)
    q = omega / (1.0 + omega)
    h = 1.006 * T + omega * (2501.0 + 1.86 * T)
    Tk = T + 273.15
    v = (R_DA * Tk * (1.0 + 1.6078 * omega)) / np.maximum(P, 1e-6)
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


def calculate_dew_point(P: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """
    Dew/frost point temperature [°C], solving pws(Tdp) = pv by vectorised Newton
    iteration on ln(pws). Returns NaN where there is no vapour.
    """
    P = np.asarray(P, dtype=float)
    omega = np.asarray(omega, dtype=float)
    pv = np.broadcast_to((omega * P) / (MW_RATIO + omega), np.broadcast(P, omega).shape).astype(float)
    valid = pv > 1e-6
    ln_pv = np.log(np.where(valid, pv, 1.0))
    # Magnus inversion as initial guess
    g = ln_pv - np.log(611.2)
    T = 243.04 * g / (17.62 - g)
    T = np.clip(T, -95.0, 95.0)
    h = 1e-4
    for _ in range(25):
        f = _ln_pws(T) - ln_pv
        dfdT = (_ln_pws(T + h) - _ln_pws(T - h)) / (2 * h)
        step = f / dfdT
        T = np.clip(T - step, -100.0, 100.0)
        if np.all(np.abs(step) < 1e-8):
            break
    T = np.where(valid, T, np.nan)
    return T if T.ndim else float(T)


def calculate_wet_bulb(T: ArrayLike, P: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """
    Thermodynamic wet-bulb (ice-bulb below 0 °C) temperature [°C] from the ASHRAE
    psychrometric relation (Handbook of Fundamentals Ch.1, Eq. 33/35), solved by
    vectorised bisection between -100 °C and the dry-bulb temperature.
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    omega = np.asarray(omega, dtype=float)
    shape = np.broadcast(T, P, omega).shape
    T_b, P_b, W = (np.broadcast_to(a, shape).astype(float) for a in (T, P, omega))

    def omega_from_twb(tw):
        ws = saturation_humidity_ratio(tw, P_b)
        water = ((2501.0 - 2.326 * tw) * ws - 1.006 * (T_b - tw)) / (2501.0 + 1.86 * T_b - 4.186 * tw)
        ice = ((2830.0 - 0.24 * tw) * ws - 1.006 * (T_b - tw)) / (2830.0 + 1.86 * T_b - 2.1 * tw)
        return np.where(tw >= 0.0, water, ice)

    lo = np.full(shape, -100.0)
    hi = T_b.copy()
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        too_wet = omega_from_twb(mid) > W
        hi = np.where(too_wet, mid, hi)
        lo = np.where(too_wet, lo, mid)
    Twb = 0.5 * (lo + hi)
    # Saturated/supersaturated air: wet bulb equals dry bulb
    Twb = np.where(W >= saturation_humidity_ratio(T_b, P_b), T_b, Twb)
    return Twb if Twb.ndim else float(Twb)


def complete_state(T: ArrayLike, P: ArrayLike, omega: ArrayLike,
                   P_ref: float = 101325.0) -> Dict[str, Any]:
    """
    Complete psychrometric state operator P = F(T, P, omega) (plan §68).
    Includes dew point, wet bulb, gauge pressure, saturation state flags,
    moisture deficit and the condensation risk index.
    """
    props = calculate_psychrometrics(T, P, omega)
    omega_arr = np.asarray(omega, dtype=float)
    props['T'] = np.asarray(T, dtype=float)
    props['P'] = np.asarray(P, dtype=float)
    props['P_gauge'] = np.asarray(P, dtype=float) - P_ref
    props['omega'] = omega_arr
    shape = np.broadcast(np.asarray(T), np.asarray(P), omega_arr).shape
    props['T_dp'] = np.broadcast_to(calculate_dew_point(P, omega), shape).copy() if shape else calculate_dew_point(P, omega)
    props['T_wb'] = calculate_wet_bulb(T, P, omega)
    props['moisture_deficit'] = props['omega_s'] - omega_arr
    props['CRI'] = (omega_arr - props['omega_s']) / np.maximum(props['omega_s'], 1e-12)
    props['saturation_state'] = np.where(props['RH'] > 100.0 + 1e-6, 2,
                                         np.where(props['RH'] >= 100.0 - 1e-6, 1, 0))
    return props


def validate_state(T: ArrayLike, P: ArrayLike, omega: ArrayLike) -> Dict[str, bool]:
    """Physical validity checks of plan §62."""
    props = calculate_psychrometrics(T, P, omega)
    return {
        'T_K_positive': bool(np.all(np.asarray(T) + 273.15 > 0)),
        'P_positive': bool(np.all(np.asarray(P) > 0)),
        'pv_valid': bool(np.all((props['pv'] >= 0) & (props['pv'] < np.asarray(P)))),
        'omega_nonnegative': bool(np.all(np.asarray(omega) >= 0)),
        'q_valid': bool(np.all((props['q'] >= 0) & (props['q'] < 1))),
        'v_positive': bool(np.all(props['specific_volume'] > 0)),
        'rho_positive': bool(np.all(props['rho_ma'] > 0)),
        'consistency_P': bool(np.allclose(props['pda'] + props['pv'], P)),
    }


def mix_streams(m_da1: float, T1: float, w1: float, m_da2: float, T2: float, w2: float) -> Dict[str, float]:
    """Adiabatic mixing of two moist-air streams (plan §56-57)."""
    m = m_da1 + m_da2
    w = (m_da1 * w1 + m_da2 * w2) / m
    h = (m_da1 * enthalpy(T1, w1) + m_da2 * enthalpy(T2, w2)) / m
    T = (h - 2501.0 * w) / (1.006 + 1.86 * w)
    return {'T': float(T), 'omega': float(w), 'h': float(h)}
