"""
Module 11 — Wall, floor and ceiling heat transfer.

Each of the six room surfaces is a thermal boundary. For an insulated panel:

    1/U = 1/h_i + Σ L_j/k_j + Σ R_contact + 1/h_o
    q'' = U (T_eq - T_air)          [W/m², positive into the room]

with T_eq the sol-air temperature T_out + α_s I_s / h_o (plan §11.12) and an
optional diurnal variation of T_out. A 'fixed' surface holds the legacy Dirichlet
wall temperature through the inside film (q'' = h_i (T_wall - T_air)).

The inside surface temperature T_si = T_air + q''/h_i feeds the surface
condensation / frost model (plan §6.15-6.18).
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class WallSpec:
    mode: str = 'panel'                    # panel | fixed | adiabatic
    T_out: float = 30.0                    # outside air / ground temperature [°C]
    T_out_amplitude: float = 0.0           # diurnal swing amplitude [K]
    layers: List[Tuple[float, float]] = field(default_factory=lambda: [(0.15, 0.022)])  # (L [m], k [W/mK])
    contact_resistance: float = 0.0        # Σ R_c [m²K/W]
    h_in: float = 8.0                      # inside film coefficient [W/m²K]
    h_out: float = 25.0                    # outside film coefficient [W/m²K]
    solar_absorptivity: float = 0.0
    irradiance: float = 0.0                # [W/m²]
    T_surface: float = -17.0               # for mode = 'fixed'

    @classmethod
    def from_dict(cls, d: Dict[str, Any], defaults: 'WallSpec' = None) -> 'WallSpec':
        base = defaults or cls()
        layers = d.get('layers')
        if layers is not None:
            layers = [(float(l['L']), float(l['k'])) if isinstance(l, dict) else (float(l[0]), float(l[1]))
                      for l in layers]
        return cls(mode=d.get('mode', base.mode), T_out=float(d.get('T_out', base.T_out)),
                   T_out_amplitude=float(d.get('T_out_amplitude', base.T_out_amplitude)),
                   layers=layers if layers is not None else list(base.layers),
                   contact_resistance=float(d.get('contact_resistance', base.contact_resistance)),
                   h_in=float(d.get('h_in', base.h_in)), h_out=float(d.get('h_out', base.h_out)),
                   solar_absorptivity=float(d.get('solar_absorptivity', base.solar_absorptivity)),
                   irradiance=float(d.get('irradiance', base.irradiance)),
                   T_surface=float(d.get('T_surface', base.T_surface)))

    @property
    def R_total(self) -> float:
        return 1.0 / self.h_in + sum(L / k for L, k in self.layers) + self.contact_resistance + 1.0 / self.h_out

    @property
    def U(self) -> float:
        """Overall heat-transfer coefficient air-to-air [W/m²K]."""
        if self.mode == 'adiabatic':
            return 0.0
        if self.mode == 'fixed':
            return self.h_in
        return 1.0 / self.R_total

    def T_equivalent(self, t: float) -> float:
        """Driving temperature: sol-air temperature (panel) or wall surface (fixed)."""
        if self.mode == 'fixed':
            return self.T_surface
        T_out = self.T_out + self.T_out_amplitude * math.sin(2 * math.pi * t / 86400.0)
        return T_out + self.solar_absorptivity * self.irradiance / self.h_out

    def heat_flux(self, t: float, T_air):
        """q'' into the room [W/m²] for the adjacent air temperature(s)."""
        return self.U * (self.T_equivalent(t) - T_air)

    def inner_surface_temperature(self, t: float, T_air):
        return T_air + self.heat_flux(t, T_air) / self.h_in


DEFAULT_WALLS = {
    'W': WallSpec(), 'E': WallSpec(), 'S': WallSpec(), 'N': WallSpec(),
    'T': WallSpec(),
    'B': WallSpec(T_out=10.0, layers=[(0.2, 1.4), (0.15, 0.03)], h_out=1e6),  # slab on heated ground
}


def lewis_mass_transfer_coefficient(h_c: float, rho: float, cp: float = 1006.0, Le: float = 0.85) -> float:
    """Chilton-Colburn analogy h_m = h / (ρ cp Le^(2/3)) [m/s] (plan §6.18)."""
    return h_c / (rho * cp * Le ** (2.0 / 3.0))
