"""
Time schedules for transient sources (injection, leakage, doors, heat loads).
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Schedule:
    """
    Activity window [t_start, t_end) with an optional repeating on/off cycle
    and optional sinusoidal modulation of the magnitude:

        active(t) = t_start <= t < t_end  and  ((t - t_start) mod period) < on_duration
        factor(t) = 1 + amplitude * sin(2 pi (t - t_start) / mod_period)
    """
    t_start: float = 0.0
    t_end: Optional[float] = None
    period: Optional[float] = None
    on_duration: Optional[float] = None
    amplitude: float = 0.0
    mod_period: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'Schedule':
        d = d or {}
        def num(key, default=None):
            v = d.get(key, default)
            return None if v in (None, '') else float(v)
        return cls(t_start=num('t_start', 0.0) or 0.0, t_end=num('t_end'), period=num('period'),
                   on_duration=num('on_duration'), amplitude=num('amplitude', 0.0) or 0.0,
                   mod_period=num('mod_period'))

    def is_active(self, t: float) -> bool:
        if t < self.t_start:
            return False
        if self.t_end is not None and t >= self.t_end:
            return False
        if self.period and self.on_duration is not None:
            return ((t - self.t_start) % self.period) < self.on_duration
        return True

    def factor(self, t: float) -> float:
        if not self.is_active(t):
            return 0.0
        if self.amplitude and self.mod_period:
            return max(0.0, 1.0 + self.amplitude * math.sin(2 * math.pi * (t - self.t_start) / self.mod_period))
        return 1.0

    def next_event_after(self, t: float) -> Optional[float]:
        """Next switching time after t (used to land time steps on events)."""
        candidates = []
        if t < self.t_start:
            candidates.append(self.t_start)
        if self.t_end is not None and t < self.t_end:
            candidates.append(self.t_end)
        if self.period and self.on_duration is not None and t >= self.t_start:
            n = math.floor((t - self.t_start) / self.period)
            for base in (n, n + 1):
                for off in (0.0, self.on_duration):
                    te = self.t_start + base * self.period + off
                    if te > t + 1e-9 and (self.t_end is None or te <= self.t_end):
                        candidates.append(te)
        return min(candidates) if candidates else None
