"""
Thermophysical property layer for moist air.
Provides transport and thermodynamic coefficients based on state.
"""

import numpy as np
from typing import Union, Dict, Any
from .psychrometrics import calculate_psychrometrics

def cp_dry_air(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Specific heat of dry air [J/(kg·K)].
    """
    # Approximation for cold storage range
    return 1006.0

def cp_water_vapour(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Specific heat of water vapour [J/(kg·K)].
    """
    return 1860.0

def cp_moist_air(T: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Specific heat of moist air [J/(kg·K)].
    cp_ma = cp_da + omega * cp_v
    """
    return cp_dry_air(T) + omega * cp_water_vapour(T)

def thermal_conductivity(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Thermal conductivity of moist air [W/(m·K)].
    Weighted average of dry air and water vapour.
    """
    # Approximate values
    k_da = 0.026
    k_v = 0.018
    return (1.0 / (1.0 + omega)) * k_da + (omega / (1.0 + omega)) * k_v

def dynamic_viscosity(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Dynamic viscosity of moist air [Pa·s].
    Uses Sutherland's Law for dry air component.
    """
    T = np.asanyarray(T)
    Tk = T + 273.15
    mu0 = 1.716e-5
    T0 = 273.11
    S = 110.56
    mu_da = mu0 * (Tk / T0)**1.5 * (T0 + S) / (Tk + S)
    mu_v = 1.2e-5
    return (1.0 / (1.0 + omega)) * mu_da + (omega / (1.0 + omega)) * mu_v

def kinematic_viscosity(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Kinematic viscosity nu = mu / rho_ma [m²/s].
    """
    props = calculate_psychrometrics(T, P, omega)
    rho_ma = props['rho_ma']
    mu = dynamic_viscosity(T, P, omega)
    return mu / np.maximum(rho_ma, 1e-5)

def moisture_diffusivity(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Diffusivity of water vapour in air [m²/s].
    """
    T = np.asanyarray(T)
    Tk = T + 273.15
    D_ref = 2.42e-5
    T_ref = 293.15
    return D_ref * (Tk / T_ref)**1.75

def thermal_diffusivity(T: Union[float, np.ndarray], P: Union[float, np.ndarray], omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Thermal diffusivity alpha = k / (rho_ma * cp_ma) [m²/s].
    """
    props = calculate_psychrometrics(T, P, omega)
    rho_ma = props['rho_ma']
    cp = cp_moist_air(T, omega)
    k = thermal_conductivity(T, P, omega)
    return k / np.maximum(rho_ma * cp, 1e-5)
