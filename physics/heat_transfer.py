"""
Heat transfer physics for the Cold Storage Digital Twin.
Implements the energy equation.
"""

import numpy as np
from .properties import thermal_conductivity, cp_moist_air

def update_temperature(T: np.ndarray, P: np.ndarray, omega: np.ndarray,
                       u: np.ndarray, v: np.ndarray, w: np.ndarray,
                       dt: float, dx: float, dy: float, dz: float,
                       S_Q: np.ndarray, S_latent: np.ndarray) -> np.ndarray:
    """
    Update temperature field using a diffusion-only baseline.
    Equation: rho * cp * ∂T/∂t = ∇ · (k * ∇T) + S_Q + S_latent

    Current implementation assumes u=v=w=0.
    """
    k = thermal_conductivity(T, P, omega)
    cp = cp_moist_air(T, omega)

    # rho_da calculation
    R_DA = 287.058
    Tk = T + 273.15
    pv = (omega * P) / (0.621945 + omega)
    pda = P - pv
    rho_da = pda / (R_DA * Tk)
    rho_ma = rho_da * (1.0 + omega)

    T_new = T.copy()

    # coeff = (k * dt) / (rho * cp)
    coeff = (k * dt) / (rho_ma * cp)

    # Central difference for Laplacian
    T_new[1:-1, 1:-1, 1:-1] = (
        T[1:-1, 1:-1, 1:-1] +
        coeff[1:-1, 1:-1, 1:-1] * (
            (T[2:, 1:-1, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / dx**2 +
            (T[1:-1, 2:, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / dy**2 +
            (T[1:-1, 1:-1, 2:] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / dz**2
        ) +
        (S_Q[1:-1, 1:-1, 1:-1] * dt / (rho_ma[1:-1, 1:-1, 1:-1] * cp[1:-1, 1:-1, 1:-1])) +
        (S_latent[1:-1, 1:-1, 1:-1] * dt / (rho_ma[1:-1, 1:-1, 1:-1] * cp[1:-1, 1:-1, 1:-1]))
    )

    return T_new
