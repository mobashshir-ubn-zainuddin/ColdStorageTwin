"""
Boundary Condition (BC) layer for the Cold Storage Digital Twin.
Provides a generic interface for applying Dirichlet, Neumann, and Robin BCs.
"""

import numpy as np
from typing import Union, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BoundaryCondition:
    """
    Base class for boundary conditions.
    """
    type: str  # 'dirichlet', 'neumann', 'robin'
    value: Union[float, np.ndarray]  # Boundary value
    # For Robin: value is phi_inf, and we need a transfer coefficient h
    h: Optional[float] = None

class BoundaryHandler:
    """
    Handles the application of boundary conditions to scalar fields.
    """
    def __init__(self):
        # Store BCs by face and variable
        # Format: { 'variable': { 'face': BoundaryCondition } }
        self.bcs: Dict[str, Dict[str, BoundaryCondition]] = {}

    def set_bc(self, variable: str, face: str, bc: BoundaryCondition):
        """Set a boundary condition for a specific variable and face."""
        if variable not in self.bcs:
            self.bcs[variable] = {}
        self.bcs[variable][face] = bc

    def apply_flux(self, variable: str, face: str, phi_p: Union[float, np.ndarray], phi_n: Optional[Union[float, np.ndarray]],
                   gamma: Union[float, np.ndarray], dx: float, rho: Union[float, np.ndarray], vel_n: Union[float, np.ndarray], area: float) -> Union[float, np.ndarray]:
        """
        Calculate the total flux across a boundary face.
        Flux = Convective + Diffusive. Vectorized implementation.
        """
        bc = self.bcs.get(variable, {}).get(face)

        # Default: zero-gradient (adiabatic/impermeable)
        if bc is None:
            return rho * vel_n * phi_p * area

        # Convective flux (Upwind)
        # If vel_n > 0: fluid enters domain (phi_b)
        # If vel_n <= 0: fluid leaves domain (phi_p)
        phi_b = bc.value
        convective = np.where(vel_n > 0, rho * vel_n * phi_b * area, rho * vel_n * phi_p * area)

        # Diffusive flux: -gamma * (dphi/dn) * area
        if bc.type == 'dirichlet':
            diffusive = -gamma * (phi_b - phi_p) / (dx / 2.0) * area
        elif bc.type == 'neumann':
            diffusive = -gamma * phi_b * area
        elif bc.type == 'robin':
            h = bc.h if bc.h is not None else 0.0
            diffusive = h * (phi_b - phi_p) * area
        else:
            diffusive = 0.0

        return convective + diffusive

    def get_boundary_value(self, variable: str, face: str) -> Optional[float]:
        """Return the boundary value for Dirichlet conditions."""
        bc = self.bcs.get(variable, {}).get(face)
        return bc.value if bc and bc.type == 'dirichlet' else None
