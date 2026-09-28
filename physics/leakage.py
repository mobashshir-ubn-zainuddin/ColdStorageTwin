"""
Module 9 — Air leakage / infiltration, and Module 10 — door-opening events.

Leakage is pressure driven: ΔP = P_out(z) - P_in(z), including the hydrostatic
variation of both air columns so that openings at different heights feel the
stack effect. Flow direction follows the sign of ΔP (plan §9.1, §9.6).

A door is a transient large opening. Besides the net pressure-driven flow, a
cold-room door carries a buoyancy-driven two-way exchange (warm air in at the
top, cold air out at the bottom) with zero net mass. That exchange is modelled
with the Gosney-Olama correlation used in the ASHRAE Refrigeration Handbook:

    V_ex = 0.221 A sqrt(g H (1 - ρ_light/ρ_heavy)) F_m
    F_m  = [2 / (1 + (ρ_heavy/ρ_light)^(1/3))]^1.5

reduced by (1 - E) for a strip curtain / air curtain of effectiveness E.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any

from .psychrometrics import humidity_ratio_from_spec, moist_air_density, enthalpy, L_VAP0, CP_DA
from .injection import OpeningGeometry, orifice_mass_flow, G
from .schedules import Schedule


@dataclass
class OutsideAir:
    T: float = 30.0
    moisture_mode: str = 'rh'
    moisture_value: float = 60.0
    P: float = 101325.0   # absolute pressure outside at floor level (z = 0) [Pa]

    @classmethod
    def from_dict(cls, d: Dict[str, Any], defaults: 'OutsideAir' = None) -> 'OutsideAir':
        base = defaults or cls()
        m = d.get('moisture', {})
        return cls(T=float(d.get('T_out', base.T)),
                   moisture_mode=m.get('mode', 'rh' if 'RH_out' in d else base.moisture_mode),
                   moisture_value=float(m.get('value', d.get('RH_out', base.moisture_value))),
                   P=float(d.get('P_out', base.P)))

    def state(self) -> Dict[str, float]:
        omega = humidity_ratio_from_spec(self.T, self.P, self.moisture_mode, self.moisture_value)
        return {'T': self.T, 'omega': omega, 'P': self.P,
                'rho': float(moist_air_density(self.T, self.P, omega))}

    def pressure_at(self, z: float, rho: float) -> float:
        return self.P - rho * G * z


@dataclass
class LeakageOpening:
    name: str
    geometry: OpeningGeometry
    outside: OutsideAir = field(default_factory=OutsideAir)
    Cd: float = 0.6
    exponent: float = 0.5
    schedule: Schedule = field(default_factory=Schedule)

    kind = 'leakage'
    pressure_dependent = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any], idx: int = 0, ambient: OutsideAir = None) -> 'LeakageOpening':
        return cls(name=d.get('name') or f'Leakage {idx + 1}',
                   geometry=OpeningGeometry.from_dict(d),
                   outside=OutsideAir.from_dict(d, ambient),
                   Cd=float(d.get('Cd', 0.6)), exponent=float(d.get('exponent', 0.5)),
                   schedule=Schedule.from_dict(d.get('schedule')))

    def open_area(self, t: float) -> float:
        return self.geometry.effective_area * self.schedule.factor(t)

    def supply_state(self, t: float, *_args) -> Dict[str, float]:
        return self.outside.state()

    def delta_p(self, P_room_floor: float, rho_room: float, out: Dict[str, float]) -> float:
        z = self.geometry.z_mid
        return self.outside.pressure_at(z, out['rho']) - (P_room_floor - rho_room * G * z)

    def mass_flow(self, t: float, P_room_floor: float, rho_room: float, out: Dict[str, float]) -> float:
        """Moist-air mass flow into the room [kg/s]; negative = exfiltration."""
        A = self.open_area(t)
        if A <= 0:
            return 0.0
        dP = self.delta_p(P_room_floor, rho_room, out)
        rho_up = out['rho'] if dP >= 0 else rho_room
        return orifice_mass_flow(dP, A, rho_up, self.Cd, self.exponent)

    def energy_rates(self, mdot: float, out: Dict[str, float], room_T: float, room_omega: float) -> Dict[str, float]:
        if mdot <= 0:
            return {'Q_sensible': 0.0, 'Q_latent': 0.0, 'Q_total': 0.0}
        m_da = mdot / (1.0 + out['omega'])
        return {'Q_sensible': m_da * CP_DA * (out['T'] - room_T),
                'Q_latent': m_da * (out['omega'] - room_omega) * L_VAP0,
                'Q_total': float(m_da * (enthalpy(out['T'], out['omega']) - enthalpy(room_T, room_omega)) * 1000.0)}


@dataclass
class DoorEvent(LeakageOpening):
    """A door modelled as a scheduled large opening (A_open(t) = f_open · W · H)."""
    open_fraction: float = 1.0
    curtain_effectiveness: float = 0.0
    stack_exchange: bool = True

    kind = 'door'

    @classmethod
    def from_dict(cls, d: Dict[str, Any], idx: int = 0, ambient: OutsideAir = None) -> 'DoorEvent':
        sched = dict(d.get('schedule') or {})
        if 't_open' in d:
            sched.setdefault('t_start', d['t_open'])
        if 't_close' in d:
            sched.setdefault('t_end', d['t_close'])
        geom = OpeningGeometry.from_dict(d)
        geom.area = None  # door area is always width x height
        return cls(name=d.get('name') or f'Door {idx + 1}', geometry=geom,
                   outside=OutsideAir.from_dict(d, ambient), Cd=float(d.get('Cd', 0.6)),
                   exponent=0.5, schedule=Schedule.from_dict(sched),
                   open_fraction=float(d.get('open_fraction', 1.0)),
                   curtain_effectiveness=float(d.get('curtain_effectiveness', 0.0)),
                   stack_exchange=bool(d.get('stack_exchange', True)))

    def open_area(self, t: float) -> float:
        return self.geometry.width * self.geometry.height * self.open_fraction * self.schedule.factor(t)

    def exchange_flow(self, t: float, rho_room: float, out: Dict[str, float]) -> Dict[str, float]:
        """
        Buoyancy-driven two-way volumetric exchange [m³/s] through the open door.
        Returns the exchange rate and whether the inflow enters at the top.
        """
        A = self.open_area(t)
        if A <= 0 or not self.stack_exchange:
            return {'V_ex': 0.0, 'inflow_top': True}
        rho_o = out['rho']
        heavy, light = max(rho_room, rho_o), min(rho_room, rho_o)
        if heavy <= light:
            return {'V_ex': 0.0, 'inflow_top': True}
        H = self.geometry.height
        Fm = (2.0 / (1.0 + (heavy / light) ** (1.0 / 3.0))) ** 1.5
        V = 0.221 * A * math.sqrt(G * H * (1.0 - light / heavy)) * Fm
        V *= (1.0 - self.curtain_effectiveness)
        # Warm (light) outside air enters at the top when the room is the denser side.
        return {'V_ex': V, 'inflow_top': rho_room >= rho_o}


def envelope_leakage(name: str, face: str, position, ela_m2: float, outside: OutsideAir) -> LeakageOpening:
    """Background envelope leakage given as an effective leakage area (ELA at 4 Pa, n = 0.65)."""
    geom = OpeningGeometry(face=face, position=list(position), width=0.1, height=0.1, area=ela_m2)
    return LeakageOpening(name=name, geometry=geom, outside=outside, Cd=1.0, exponent=0.65)
