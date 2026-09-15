"""
Moisture transport physics for the Cold Storage Digital Twin.
Implements moisture diffusion and prepares for advection.
"""

import numpy as np
from typing import Tuple
from .properties import moisture_diffusivity

def update_moisture_diffusion(T: np.ndarray, P: np.ndarray, omega: np.ndarray,
                              u: np.ndarray, v: np.ndarray, w: np.ndarray,
                              dt: float, dx: float, dy: float, dz: float,
                              S_omega: np.ndarray) -> np.ndarray:
    """
    Update humidity ratio field using a diffusion-only baseline.
    Equation: ∂(rho_da * omega)/∂t = ∇ · (rho_da * D_eff * ∇omega) + S_omega

    Current implementation assumes u=v=w=0.
    """
    D_eff = moisture_diffusivity(T, P, omega)

    # We solve for omega. In a simplified FDM:
    # omega_new = omega + dt * ( (1/rho_da) * ∇ · (rho_da * D_eff * ∇omega) + S_omega/rho_da )

    # For the current baseline, we assume rho_da is slowly varying across cells
    # and use a central difference for the Laplacian.

    # rho_da calculation
    R_DA = 287.058
    Tk = T + 273.15
    pv = (omega * P) / (0.621945 + omega)
    pda = P - pv
    rho_da = pda / (R_DA * Tk)

    omega_new = omega.copy()

    # Interior cells
    # Use a simplified FDM for ∇ · (rho_da * D_eff * ∇omega)
    # This is a placeholder for the full FVM implementation in next stages.
    # We use a constant-coefficient approximation locally.

    # Precompute coefficients
    coeff = (D_eff * dt)

    # Central difference for Laplacian (assuming constant rho_da locally)
    # This matches the baseline FDM prototype but uses dynamic D_eff
    omega_new[1:-1, 1:-1, 1:-1] = (
        omega[1:-1, 1:-1, 1:-1] +
        coeff[1:-1, 1:-1, 1:-1] * (
            (omega[2:, 1:-1, 1:-1] - 2*omega[1:-1, 1:-1, 1:-1] + omega[:-2, 1:-1, 1:-1]) / dx**2 +
            (omega[1:-1, 2:, 1:-1] - 2*omega[1:-1, 1:-1, 1:-1] + omega[1:-1, :-2, 1:-1]) / dy**2 +
            (omega[1:-1, 1:-1, 2:] - 2*omega[1:-1, 1:-1, 1:-1] + omega[1:-1, 1:-1, :-2]) / dz**2
        ) +
        (S_omega[1:-1, 1:-1, 1:-1] * dt / rho_da[1:-1, 1:-1, 1:-1])
    )

    return omega_new
