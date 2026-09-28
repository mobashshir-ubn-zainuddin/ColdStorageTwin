"""
Module 8 — Air injection model.

An injection source delivers moist air of prescribed state into the cold room,
either through a real opening on a boundary face (preferred, plan §9.41) or as a
sub-grid point/plume source at an interior location (Gaussian distribution).

The user specifies one quantity from each group (plan §58) and the remaining
dependent properties are derived:
    flow     : velocity | volumetric | mass | pressure
    moisture : rh | omega | dew_point | vapour_pressure | specific_humidity | absolute_humidity
    thermal  : temperature | sensible_heat   (plus optional extra heater power)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .psychrometrics import (humidity_ratio_from_spec, moist_air_density, enthalpy,
                             saturation_humidity_ratio, L_VAP0, CP_DA)
from .schedules import Schedule

G = 9.81
FACES = ('W', 'E', 'S', 'N', 'B', 'T')


@dataclass
class OpeningGeometry:
    """Location of a source: a boundary-face patch or an interior point."""
    face: str = 'interior'                 # W/E/S/N/B/T or 'interior'
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width: float = 0.5                     # along first in-plane axis [m]
    height: float = 0.5                    # along second in-plane axis [m]
    area: Optional[float] = None           # effective free area [m²]
    direction: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])  # interior jets
    sigma: Optional[float] = None          # interior source spread [m]

    @property
    def is_boundary(self) -> bool:
        return self.face in FACES

    @property
    def effective_area(self) -> float:
        return float(self.area) if self.area else self.width * self.height

    @property
    def z_mid(self) -> float:
        return float(self.position[2])

    def unit_direction(self):
        d = [float(c) for c in self.direction]
        n = math.sqrt(sum(c * c for c in d)) or 1.0
        return [c / n for c in d]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OpeningGeometry':
        return cls(face=d.get('face', 'interior') or 'interior',
                   position=[float(v) for v in d.get('position', [0, 0, 0])],
                   width=float(d.get('width', 0.5)), height=float(d.get('height', 0.5)),
                   area=float(d['area']) if d.get('area') not in (None, '') else None,
                   direction=[float(v) for v in d.get('direction', [1, 0, 0])],
                   sigma=float(d['sigma']) if d.get('sigma') not in (None, '') else None)


def orifice_mass_flow(delta_p: float, area: float, rho_up: float, Cd: float = 0.65,
                      n: float = 0.5, dp_ref: float = 4.0) -> float:
    """
    Power-law opening flow (orifice for n = 0.5, crack/ELA for n ≈ 0.65):
        m = sign(ΔP) · ρ · Cd A sqrt(2/ρ) · ΔP_ref^(0.5-n) · |ΔP|^n
    Positive ΔP gives a positive (inward) flow. For n = 0.5 this reduces to
    Cd A sqrt(2 ρ |ΔP|).
    """
    if area <= 0 or delta_p == 0:
        return 0.0
    C = Cd * area * math.sqrt(2.0 / rho_up) * dp_ref ** (0.5 - n)
    return math.copysign(rho_up * C * abs(delta_p) ** n, delta_p)


@dataclass
class InjectionSource:
    name: str
    geometry: OpeningGeometry
    flow_mode: str = 'volumetric'      # velocity | volumetric | mass | pressure
    flow_value: float = 0.1            # m/s | m³/s | kg/s | (unused for pressure)
    P_supply: Optional[float] = None   # absolute supply pressure [Pa]
    Cd: float = 0.65
    thermal_mode: str = 'temperature'  # temperature | sensible_heat
    thermal_value: float = -20.0       # °C | W relative to room air
    heat_W: float = 0.0                # extra heater power added to the stream [W]
    moisture_mode: str = 'rh'
    moisture_value: float = 80.0
    schedule: Schedule = field(default_factory=Schedule)

    kind = 'injection'

    @classmethod
    def from_dict(cls, d: Dict[str, Any], idx: int = 0) -> 'InjectionSource':
        flow = d.get('flow', {})
        thermal = d.get('thermal', {})
        moist = d.get('moisture', {})
        return cls(
            name=d.get('name') or f'Injection {idx + 1}',
            geometry=OpeningGeometry.from_dict(d),
            flow_mode=flow.get('mode', d.get('flow_mode', 'volumetric')),
            flow_value=float(flow.get('value', d.get('flow_value', 0.1))),
            P_supply=float(d['P_supply']) if d.get('P_supply') not in (None, '') else None,
            Cd=float(d.get('Cd', 0.65)),
            thermal_mode=thermal.get('mode', 'temperature'),
            thermal_value=float(thermal.get('value', d.get('T', -20.0))),
            heat_W=float(d.get('heat_W', 0.0) or 0.0),
            moisture_mode=moist.get('mode', 'rh'),
            moisture_value=float(moist.get('value', d.get('RH', 80.0))),
            schedule=Schedule.from_dict(d.get('schedule')),
        )

    @property
    def pressure_dependent(self) -> bool:
        return self.flow_mode == 'pressure'

    def supply_state(self, t: float, room_T: float, room_omega: float, room_P: float) -> Dict[str, float]:
        """Inflowing air state (T_in, omega_in, rho_in, P_in) at time t."""
        P_in = self.P_supply if self.P_supply else room_P
        T_in = self.thermal_value if self.thermal_mode == 'temperature' else room_T
        omega_in = humidity_ratio_from_spec(T_in, P_in, self.moisture_mode, self.moisture_value)
        rho_in = float(moist_air_density(T_in, P_in, omega_in))
        # Heat-specified streams: T_in = T_room + Q / (m_da cp) (plan §30). Iterate
        # because the flow rate depends on density, which depends on T_in.
        needs_heat = self.thermal_mode == 'sensible_heat' or self.heat_W
        for _ in range(3 if needs_heat else 0):
            m = self.fixed_mass_flow(rho_in) if not self.pressure_dependent else None
            if not m:
                break
            m_da = m / (1.0 + omega_in)
            Q = (self.thermal_value if self.thermal_mode == 'sensible_heat' else 0.0) + self.heat_W
            base = room_T if self.thermal_mode == 'sensible_heat' else self.thermal_value
            T_in = base + Q / (m_da * CP_DA)
            if self.moisture_mode == 'rh':
                omega_in = humidity_ratio_from_spec(T_in, P_in, 'rh', self.moisture_value)
            rho_in = float(moist_air_density(T_in, P_in, omega_in))
        return {'T': T_in, 'omega': omega_in, 'rho': rho_in, 'P': P_in}

    def fixed_mass_flow(self, rho_in: float) -> float:
        """Moist-air mass flow [kg/s] for flow modes that do not depend on room pressure."""
        A = self.geometry.effective_area
        if self.flow_mode == 'velocity':
            return rho_in * A * self.flow_value
        if self.flow_mode == 'volumetric':
            return rho_in * self.flow_value
        if self.flow_mode == 'mass':
            return self.flow_value
        return 0.0

    def mass_flow(self, t: float, P_room_z: float, rho_room: float, supply: Dict[str, float]) -> float:
        """Moist-air mass flow into the room [kg/s]; negative means backflow."""
        f = self.schedule.factor(t)
        if f == 0.0:
            return 0.0
        if self.flow_mode == 'pressure':
            P_src = self.P_supply if self.P_supply else P_room_z
            dP = P_src - P_room_z
            rho_up = supply['rho'] if dP >= 0 else rho_room
            return f * orifice_mass_flow(dP, self.geometry.effective_area, rho_up, self.Cd)
        return f * self.fixed_mass_flow(supply['rho'])

    def jet_velocity(self, mdot: float, rho: float) -> float:
        """True discharge velocity through the physical opening [m/s]."""
        return mdot / max(rho * self.geometry.effective_area, 1e-12)

    def energy_rates(self, mdot: float, supply: Dict[str, float], room_T: float, room_omega: float) -> Dict[str, float]:
        """Sensible/latent/total heat carried relative to room air [W] (plan §30-31)."""
        m_da = mdot / (1.0 + supply['omega'])
        Q_s = m_da * CP_DA * (supply['T'] - room_T)
        Q_l = m_da * (supply['omega'] - room_omega) * L_VAP0
        Q_t = m_da * (enthalpy(supply['T'], supply['omega']) - enthalpy(room_T, room_omega)) * 1000.0
        return {'Q_sensible': Q_s, 'Q_latent': Q_l, 'Q_total': float(Q_t)}

    def inlet_supersaturated(self, supply: Dict[str, float]) -> bool:
        return supply['omega'] > float(saturation_humidity_ratio(supply['T'], supply['P']))
