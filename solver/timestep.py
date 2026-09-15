"""
Timestep control for the Cold Storage Digital Twin.
Calculates stability restrictions based on diffusion and advection (CFL).
"""

import numpy as np
from typing import Dict, Any, Tuple
from geometry.mesh import Mesh

class TimestepController:
    """
    Manages the simulation timestep to ensure numerical stability.
    """
    def __init__(self, mesh: Mesh, safety_factor: float = 0.9):
        self.mesh = mesh
        self.safety_factor = safety_factor

    def calculate_dt(self,
                     rho: np.ndarray,
                     u: np.ndarray,
                     v: np.ndarray,
                     w: np.ndarray,
                     alpha: np.ndarray,
                     D_eff: np.ndarray) -> Dict[str, Any]:
        """
        Calculate the maximum allowable timestep dt.
        """
        # 1. Thermal Diffusion Limit: alpha * dt * (1/dx^2 + 1/dy^2 + 1/dz^2) <= 1/2
        inv_dist_sum = (1.0 / self.mesh.dx**2 +
                       1.0 / self.mesh.dy**2 +
                       1.0 / self.mesh.dz**2)
        max_alpha = np.max(alpha)
        dt_thermal = 0.5 / (max_alpha * inv_dist_sum)

        # 2. Moisture Diffusion Limit: D_eff * dt * (1/dx^2 + 1/dy^2 + 1/dz^2) <= 1/2
        max_D = np.max(D_eff)
        dt_moisture = 0.5 / (max_D * inv_dist_sum)

        # 3. CFL (Advection) Limit: dt <= min(dx/u, dy/v, dz/w)
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        w_max = np.max(np.abs(w))

        dt_cfl_x = self.mesh.dx / u_max if u_max > 1e-8 else float('inf')
        dt_cfl_y = self.mesh.dy / v_max if v_max > 1e-8 else float('inf')
        dt_cfl_z = self.mesh.dz / w_max if w_max > 1e-8 else float('inf')
        dt_cfl = min(dt_cfl_x, dt_cfl_y, dt_cfl_z)

        # Selected dt
        dt_selected = self.safety_factor * min(dt_thermal, dt_moisture, dt_cfl)

        limits = {
            'thermal': dt_thermal,
            'moisture': dt_moisture,
            'cfl': dt_cfl
        }
        limiting_mechanism = min(limits, key=limits.get)

        return {
            'dt_selected': dt_selected,
            'dt_thermal': dt_thermal,
            'dt_moisture': dt_moisture,
            'dt_cfl': dt_cfl,
            'limiting_mechanism': limiting_mechanism,
            'limits': limits
        }
