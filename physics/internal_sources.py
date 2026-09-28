"""
Module 12 — Internal heat (and moisture) sources, and stored-product blocks.

Heat sources are volumetric S_Q = Q · G(x) with G a discretely normalised
Gaussian or a uniform box distribution, so ∫ S_Q dV = Q exactly (plan §12.3).

Product blocks (pallets/racks) occupy solid cells that block the airflow and
exchange heat with the neighbouring air through their exposed faces:
    Q = h A (T_product - T_air)
They release respiration heat q_resp(T) = q_ref · Q10^(T/10) [W/kg] and may lose
moisture by transpiration (plan §12.8-12.10).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .schedules import Schedule


@dataclass
class HeatSource:
    name: str
    kind: str = 'equipment'         # equipment | lighting | motor | people | forklift | custom
    position: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    sigma: Optional[float] = None
    box: Optional[List[float]] = None   # [x0, x1, y0, y1, z0, z1] uniform distribution
    power_W: float = 500.0
    fraction_to_room: float = 1.0       # lighting / motor losses reaching the air
    n_people: int = 0
    moisture_kg_s: float = 0.0
    schedule: Schedule = field(default_factory=Schedule)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], idx: int = 0) -> 'HeatSource':
        kind = d.get('kind', 'equipment')
        n_people = int(d.get('n_people', 0) or 0)
        moisture = d.get('moisture_kg_s')
        if moisture in (None, ''):
            # ≈ 50 g/h of vapour per worker in a cold room
            moisture = 1.4e-5 * n_people if kind == 'people' else 0.0
        return cls(name=d.get('name') or f'Heat source {idx + 1}', kind=kind,
                   position=[float(v) for v in d.get('position', [1, 1, 1])],
                   sigma=float(d['sigma']) if d.get('sigma') not in (None, '') else None,
                   box=[float(v) for v in d['box']] if d.get('box') else None,
                   power_W=float(d.get('power_W', 0.0) or 0.0),
                   fraction_to_room=float(d.get('fraction_to_room', 1.0)),
                   n_people=n_people, moisture_kg_s=float(moisture),
                   schedule=Schedule.from_dict(d.get('schedule')))

    def heat_rate(self, t: float, T_room: float) -> float:
        """Heat released into the air [W]."""
        f = self.schedule.factor(t)
        if f == 0.0:
            return 0.0
        if self.kind == 'people':
            # ASHRAE Refrigeration: q = 272 - 6 t  [W/person], t in °C
            return f * self.n_people * max(0.0, 272.0 - 6.0 * T_room)
        return f * self.power_W * self.fraction_to_room

    def moisture_rate(self, t: float) -> float:
        return self.moisture_kg_s * self.schedule.factor(t)


@dataclass
class ProductBlock:
    name: str
    box: List[float]                      # [x0, x1, y0, y1, z0, z1]
    T_initial: float = -10.0
    bulk_density: float = 500.0           # product mass per block volume [kg/m³]
    cp: float = 2000.0                    # [J/kgK] (frozen food ≈ 1800-2100)
    h_surface: float = 8.0                # [W/m²K]
    respiration_ref_W_kg: float = 0.0     # respiration heat at 0 °C [W/kg]
    Q10: float = 2.5
    transpiration_kg_kg_s: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any], idx: int = 0) -> 'ProductBlock':
        return cls(name=d.get('name') or f'Product {idx + 1}',
                   box=[float(v) for v in d['box']],
                   T_initial=float(d.get('T_initial', -10.0)),
                   bulk_density=float(d.get('bulk_density', 500.0)),
                   cp=float(d.get('cp', 2000.0)), h_surface=float(d.get('h_surface', 8.0)),
                   respiration_ref_W_kg=float(d.get('respiration_ref_W_kg', 0.0)),
                   Q10=float(d.get('Q10', 2.5)),
                   transpiration_kg_kg_s=float(d.get('transpiration_kg_kg_s', 0.0)))

    def respiration(self, T):
        """Respiration heat generation [W/kg] at product temperature T."""
        return self.respiration_ref_W_kg * self.Q10 ** (T / 10.0)
