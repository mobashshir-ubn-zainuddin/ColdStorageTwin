"""
Refrigeration unit cooler (evaporator) — the heat sink of the cold store.

A unit cooler recirculates room air: it draws air through a return opening,
passes it over the coil, and discharges it through a supply opening. The net
mass exchange with the room is zero. When the coil is on:

    T_sup   = T_coil_leaving                           (air leaves near coil temperature)
    ω_sup   = min(ω_ret, ω_s(T_sup))                   (dehumidification -> coil frost)
    Q_coil  = m_da [ (h_ret - h_sup) ] + m_w L_f        (sensible + latent + freezing)

If Q_coil exceeds the installed capacity, the coil duty is scaled down and the
leaving air is correspondingly warmer and wetter. A thermostat with deadband
switches the coil on/off based on the return air temperature (plan §6.3
'thermostat feedback loop'). Fan heat is added to the supply air.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .psychrometrics import saturation_humidity_ratio, moist_air_density, CP_DA, L_VAP0, L_FUS
from .injection import OpeningGeometry


@dataclass
class CoolingUnit:
    name: str
    supply: OpeningGeometry
    ret: OpeningGeometry
    airflow_m3_s: float = 1.0
    setpoint: float = -20.0
    deadband: float = 1.0
    T_coil_leaving: float = -28.0
    capacity_W: float = 10000.0
    fan_power_W: float = 300.0
    fan_mode: str = 'continuous'       # continuous | cycling
    control: str = 'thermostat'        # thermostat | always_on | off
    coil_on: bool = True

    kind = 'cooling_unit'

    @classmethod
    def from_dict(cls, d: Dict[str, Any], idx: int = 0) -> 'CoolingUnit':
        return cls(name=d.get('name') or f'Unit cooler {idx + 1}',
                   supply=OpeningGeometry.from_dict(d.get('supply', {})),
                   ret=OpeningGeometry.from_dict(d.get('return', {})),
                   airflow_m3_s=float(d.get('airflow_m3_s', 1.0)),
                   setpoint=float(d.get('setpoint', -20.0)), deadband=float(d.get('deadband', 1.0)),
                   T_coil_leaving=float(d.get('T_coil_leaving', -28.0)),
                   capacity_W=float(d.get('capacity_W', 10000.0)),
                   fan_power_W=float(d.get('fan_power_W', 300.0)),
                   fan_mode=d.get('fan_mode', 'continuous'), control=d.get('control', 'thermostat'))

    def update_control(self, T_return: float) -> None:
        if self.control == 'always_on':
            self.coil_on = True
        elif self.control == 'off':
            self.coil_on = False
        elif T_return > self.setpoint + self.deadband / 2:
            self.coil_on = True
        elif T_return < self.setpoint - self.deadband / 2:
            self.coil_on = False

    def fan_running(self) -> bool:
        return self.control != 'off' and (self.fan_mode == 'continuous' or self.coil_on)

    def process(self, T_ret: float, omega_ret: float, P: float) -> Dict[str, float]:
        """Supply-air state and coil duty for the current return-air state."""
        if not self.fan_running():
            return {'running': False, 'coil_on': False, 'T': T_ret, 'omega': omega_ret, 'm_da': 0.0,
                    'V': 0.0, 'Q_coil': 0.0, 'Q_sensible': 0.0, 'Q_latent': 0.0, 'water_removal': 0.0}
        rho = float(moist_air_density(T_ret, P, omega_ret))
        m_da = rho * self.airflow_m3_s / (1.0 + omega_ret)
        T_sup, w_sup = T_ret, omega_ret
        Q_s = Q_l = m_w = 0.0
        if self.coil_on and T_ret > self.T_coil_leaving:
            T_sup = self.T_coil_leaving
            w_sup = min(omega_ret, float(saturation_humidity_ratio(T_sup, P)))
            m_w = m_da * (omega_ret - w_sup)
            Q_s = m_da * CP_DA * (T_ret - T_sup)
            Q_l = m_w * (L_VAP0 + (L_FUS if T_sup < 0 else 0.0))
            Q = Q_s + Q_l
            if Q > self.capacity_W > 0:
                f = self.capacity_W / Q
                T_sup = T_ret - f * (T_ret - T_sup)
                w_sup = omega_ret - f * (omega_ret - w_sup)
                Q_s, Q_l, m_w = f * Q_s, f * Q_l, f * m_w
        T_sup += self.fan_power_W / (m_da * CP_DA)
        return {'running': True, 'coil_on': self.coil_on, 'T': T_sup, 'omega': w_sup, 'm_da': m_da,
                'V': self.airflow_m3_s, 'Q_coil': Q_s + Q_l, 'Q_sensible': Q_s, 'Q_latent': Q_l,
                'water_removal': m_w}
