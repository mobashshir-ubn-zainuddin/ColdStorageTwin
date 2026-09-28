"""
Time-dependent result storage Q(x_i, y_j, z_k, t_n) and post-processing
(plan §42-§53): snapshots of the primary state, derived psychrometric fields,
point queries with full state tables and time histories, planar slices,
room-level statistics, performance indicators and conservation reports.
"""

from typing import Dict, Any, List, Optional

import numpy as np

from physics.psychrometrics import (complete_state, calculate_psychrometrics, CP_DA, L_VAP0,
                                    saturation_pressure, MW_RATIO)
from physics.injection import G

# Field catalogue shown in the dashboard (plan §43)
FIELDS: Dict[str, Dict[str, str]] = {
    'T': {'label': 'Temperature', 'unit': '°C'},
    'P': {'label': 'Absolute pressure', 'unit': 'Pa'},
    'P_gauge': {'label': 'Pressure vs. outside (same height)', 'unit': 'Pa'},
    'p_dyn': {'label': 'Dynamic pressure', 'unit': 'Pa'},
    'RH': {'label': 'Relative humidity', 'unit': '%'},
    'omega': {'label': 'Humidity ratio', 'unit': 'kg/kg'},
    'q': {'label': 'Specific humidity', 'unit': 'kg/kg'},
    'pv': {'label': 'Vapour pressure', 'unit': 'Pa'},
    'pda': {'label': 'Dry-air pressure', 'unit': 'Pa'},
    'pws': {'label': 'Saturation pressure', 'unit': 'Pa'},
    'omega_s': {'label': 'Saturation humidity ratio', 'unit': 'kg/kg'},
    'T_dp': {'label': 'Dew/frost point', 'unit': '°C'},
    'T_wb': {'label': 'Wet-bulb temperature', 'unit': '°C'},
    'dT_dp': {'label': 'Condensation margin T − T_dp', 'unit': 'K'},
    'h': {'label': 'Enthalpy', 'unit': 'kJ/kg'},
    'rho_ma': {'label': 'Moist-air density', 'unit': 'kg/m³'},
    'rho_da': {'label': 'Dry-air density', 'unit': 'kg/m³'},
    'rho_v': {'label': 'Vapour density', 'unit': 'kg/m³'},
    'specific_volume': {'label': 'Specific volume', 'unit': 'm³/kg'},
    'V': {'label': 'Velocity magnitude', 'unit': 'm/s'},
    'u': {'label': 'Velocity X', 'unit': 'm/s'},
    'v': {'label': 'Velocity Y', 'unit': 'm/s'},
    'w': {'label': 'Velocity Z', 'unit': 'm/s'},
    'e_sensible': {'label': 'Sensible energy density', 'unit': 'kJ/m³'},
    'e_latent': {'label': 'Latent energy density', 'unit': 'kJ/m³'},
    'e_total': {'label': 'Total energy density', 'unit': 'kJ/m³'},
    'cond_rate': {'label': 'Condensation rate', 'unit': 'g/(m³·h)'},
    'condensate': {'label': 'Deposited water/frost', 'unit': 'g/m³'},
    'CRI': {'label': 'Condensation risk index', 'unit': '−'},
    'nu_t': {'label': 'Eddy viscosity', 'unit': 'm²/s'},
}

# Quantities offered for point histories
HISTORY_KEYS = ['T', 'P', 'P_gauge', 'RH', 'omega', 'h', 'T_dp', 'T_wb', 'V', 'rho_ma', 'pv',
                'e_sensible', 'e_latent', 'e_total', 'cond_rate']


class ResultStore:
    def __init__(self, solver):
        m = solver.mesh
        self.mesh = m
        self.fluid = solver.fluid.copy()
        self.P_ref = float(solver.sc.ambient.P)
        self.rho_out = float(solver.sc.ambient.state()['rho'])
        self.snapshots: List[Dict[str, Any]] = []
        self.series: List[Dict[str, Any]] = []
        self.wall_time = 0.0
        self.T_target = float(solver.num.get('T_target', -18.0))
        self.cooling_time: Optional[float] = None
        self._derived_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_snapshot(self, s) -> None:
        uc, vc, wc = s.cell_velocities()
        f32 = lambda a: np.asarray(a, dtype=np.float32).copy()
        self.snapshots.append({
            't': float(s.t), 'P_room': float(s.P_room), 'rho_bar': float(s.rho_bar),
            'T': f32(s.T), 'omega': f32(s.omega), 'u': f32(uc), 'v': f32(vc), 'w': f32(wc),
            'p_dyn': f32(s.phi * s.rho_bar), 'cond_rate': f32(s.cond_rate),
            'condensate': f32(s.liquid + s.ice), 'nu_t': f32(s.nu_t),
            'units': [dict(u) for u in s.last.get('units', [])],
            'sources': [dict(x) for x in s.last.get('sources', [])],
        })

    def record_series(self, s) -> None:
        fl = s.fluid
        T = s.T[fl]
        props = calculate_psychrometrics(T, s.P_room, s.omega[fl])
        RH = props['RH']
        uc, vc, wc = s.cell_velocities()
        V = np.sqrt(uc ** 2 + vc ** 2 + wc ** 2)[fl]
        cons = s.conservation()
        Tm = float(T.mean())
        if self.cooling_time is None and Tm <= self.T_target:
            self.cooling_time = float(s.t)
        units = s.last.get('units', [])
        wall_frost = sum(float(s.wall_ice[f].sum() + s.wall_liquid[f].sum()) for f in s.wall_ice)
        self.series.append({
            't': float(s.t), 'dt': float(s.dt_last),
            'T_mean': Tm, 'T_min': float(T.min()), 'T_max': float(T.max()), 'T_std': float(T.std()),
            'RH_mean': float(RH.mean()), 'RH_max': float(RH.max()), 'omega_mean': float(s.omega[fl].mean()),
            'P_room': float(s.P_room), 'P_gauge': float(s.P_room - self.P_ref),
            'dp_max': float(np.abs(s.phi[fl] * s.rho_bar).max()),
            'V_mean': float(V.mean()), 'V_max': float(V.max()),
            'T_product': float(s.T[s.solid].mean()) if s.solid.any() else None,
            'Q_walls': float(s.last.get('Q_walls', 0.0)), 'Q_internal': float(s.last.get('Q_internal', 0.0)),
            'Q_coil': float(s.last.get('Q_coil', 0.0)), 'Q_infiltration': float(s.last.get('Q_infiltration', 0.0)),
            'Q_respiration': float(s.last.get('Q_respiration', 0.0)),
            'coil_on': [bool(u['coil_on']) for u in units],
            'cond_rate_bulk': float(s.last.get('cond_bulk_rate', 0.0)),
            'cond_rate_wall': float(s.last.get('cond_wall_rate', 0.0)),
            'condensate_bulk': float(s.liquid.sum() + s.ice.sum()), 'frost_wall': wall_frost,
            'coil_water': float(s.cum['coil_water']),
            'injected_mass': float(s.cum['injected_mass']), 'leak_in_mass': float(s.cum['leak_in_mass']),
            'leak_out_mass': float(s.cum['leak_out_mass']), 'door_in_mass': float(s.cum['door_in_mass']),
            'door_out_mass': float(s.cum['door_out_mass']),
            'E_walls': float(s.cum['Q_walls']), 'E_internal': float(s.cum['Q_internal']),
            'E_coil': float(s.cum['Q_coil']), 'E_infiltration': float(s.cum['Q_infiltration']),
            'mass_error_pct': float(cons['dry_air_error_pct']), 'water_error_pct': float(cons['water_error_pct']),
            'energy_error_pct': float(cons['energy_error_pct']),
            'divergence': s.divergence_residual(),
            'sources': [{k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                         for k, v in x.items()} for x in s.last.get('sources', [])],
        })

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------
    def pressure_field(self, snap: Dict[str, Any]) -> np.ndarray:
        """P_abs(x,t) = P_room(floor) − ρ̄ g z + p_dyn."""
        rho = snap['rho_bar'] * (1.0 + snap['omega'].mean())
        return snap['P_room'] - rho * G * self.mesh.Z_coords + snap['p_dyn']

    def gauge(self, P, z):
        """Pressure relative to the outside air at the same height (drives leakage)."""
        return P - (self.P_ref - self.rho_out * G * z)

    def derived(self, index: int) -> Dict[str, np.ndarray]:
        if index in self._derived_cache:
            return self._derived_cache[index]
        snap = self.snapshots[index]
        T = snap['T'].astype(float)
        w = snap['omega'].astype(float)
        P = self.pressure_field(snap)
        st = complete_state(T, P, w, P_ref=self.P_ref)
        out = {k: st[k] for k in ('RH', 'q', 'pv', 'pda', 'pws', 'omega_s', 'T_dp', 'T_wb', 'h', 'rho_ma',
                                  'rho_da', 'rho_v', 'specific_volume', 'CRI', 'P_gauge')}
        out['T'], out['omega'], out['P'] = T, w, P
        out['P_gauge'] = self.gauge(P, self.mesh.Z_coords)
        out['p_dyn'] = snap['p_dyn'].astype(float)
        out['dT_dp'] = T - st['T_dp']
        u, v, ww = (snap[k].astype(float) for k in ('u', 'v', 'w'))
        out['u'], out['v'], out['w'] = u, v, ww
        out['V'] = np.sqrt(u ** 2 + v ** 2 + ww ** 2)
        out['e_sensible'] = st['rho_ma'] * CP_DA * T / 1000.0
        out['e_latent'] = st['rho_da'] * w * L_VAP0 / 1000.0
        out['e_total'] = out['e_sensible'] + out['e_latent']
        out['cond_rate'] = snap['cond_rate'].astype(float) / self.mesh.V_cell * 1000.0 * 3600.0
        out['condensate'] = snap['condensate'].astype(float) / self.mesh.V_cell * 1000.0
        out['nu_t'] = snap['nu_t'].astype(float)
        solid = ~self.fluid
        for k, arr in out.items():
            if k != 'T' and isinstance(arr, np.ndarray) and arr.shape == T.shape:
                arr[solid] = np.nan
        if len(self._derived_cache) > 8:
            self._derived_cache.pop(next(iter(self._derived_cache)))
        self._derived_cache[index] = out
        return out

    def field(self, name: str, index: int) -> np.ndarray:
        if name not in FIELDS:
            raise KeyError(f'Unknown field {name}')
        return self.derived(index)[name]

    # ------------------------------------------------------------------
    # Point queries (plan §45-46, §51)
    # ------------------------------------------------------------------
    def _point_primary(self, snap, x, y, z) -> Dict[str, float]:
        m = self.mesh
        vals = {k: m.interpolate(np.where(self.fluid, snap[k], np.nan).astype(float), x, y, z)
                for k in ('T', 'omega', 'u', 'v', 'w', 'p_dyn', 'cond_rate', 'condensate')}
        rho = snap['rho_bar'] * (1.0 + snap['omega'].mean())
        vals['P'] = snap['P_room'] - rho * G * z + vals['p_dyn']
        return vals

    def point_state(self, x: float, y: float, z: float, index: int) -> Dict[str, Any]:
        m = self.mesh
        i, j, k = m.locate(x, y, z)
        snap = self.snapshots[index]
        if not self.fluid[i, j, k]:
            return {'inside_product': True, 'cell': [i, j, k], 't': snap['t'],
                    'T_product': float(snap['T'][i, j, k])}
        p = self._point_primary(snap, x, y, z)
        st = complete_state(p['T'], p['P'], p['omega'], P_ref=self.P_ref)
        V = float(np.sqrt(p['u'] ** 2 + p['v'] ** 2 + p['w'] ** 2))
        A = m.characteristic_size ** 2
        rows = [
            ('Temperature', p['T'], '°C'), ('Absolute pressure', p['P'], 'Pa'),
            ('Gauge pressure (vs. outside at same height)', self.gauge(p['P'], z), 'Pa'), ('Relative humidity', st['RH'], '%'),
            ('Humidity ratio', p['omega'], 'kg/kg'), ('Saturation humidity ratio', st['omega_s'], 'kg/kg'),
            ('Specific humidity', st['q'], 'kg/kg'), ('Vapour pressure', st['pv'], 'Pa'),
            ('Dry-air pressure', st['pda'], 'Pa'), ('Saturation pressure', st['pws'], 'Pa'),
            ('Dew/frost point', st['T_dp'], '°C'), ('Wet-bulb temperature', st['T_wb'], '°C'),
            ('Enthalpy', st['h'], 'kJ/kg'), ('Specific volume', st['specific_volume'], 'm³/kg'),
            ('Moist-air density', st['rho_ma'], 'kg/m³'), ('Dry-air density', st['rho_da'], 'kg/m³'),
            ('Vapour density', st['rho_v'], 'kg/m³'),
            ('Velocity magnitude', V, 'm/s'), ('Velocity u, v, w', f"{p['u']:.3f}, {p['v']:.3f}, {p['w']:.3f}", 'm/s'),
            ('Mass flux ρ|V|', st['rho_ma'] * V, 'kg/(m²·s)'), ('Volumetric flow through cell face', V * A, 'm³/s'),
            ('Sensible energy density', st['rho_ma'] * CP_DA * p['T'] / 1000.0, 'kJ/m³'),
            ('Latent energy density', st['rho_da'] * p['omega'] * L_VAP0 / 1000.0, 'kJ/m³'),
            ('Total energy density', (st['rho_ma'] * CP_DA * p['T'] + st['rho_da'] * p['omega'] * L_VAP0) / 1000.0, 'kJ/m³'),
            ('Condensation rate', p['cond_rate'] / m.V_cell * 3.6e6, 'g/(m³·h)'),
            ('Deposited water/frost', p['condensate'] / m.V_cell * 1000.0, 'g/m³'),
            ('Condensation risk index', st['CRI'], '−'),
        ]
        return {'inside_product': False, 'cell': [i, j, k], 't': snap['t'], 'position': [x, y, z],
                'saturation_state': ['unsaturated', 'saturated', 'supersaturated'][int(st['saturation_state'])],
                'rows': [{'name': n, 'value': (float(v) if not isinstance(v, str) else v), 'unit': u} for n, v, u in rows]}

    def point_history(self, x: float, y: float, z: float) -> Dict[str, Any]:
        out = {k: [] for k in HISTORY_KEYS}
        out['t'] = []
        m = self.mesh
        for snap in self.snapshots:
            p = self._point_primary(snap, x, y, z)
            props = calculate_psychrometrics(p['T'], p['P'], p['omega'])
            st = complete_state(p['T'], p['P'], p['omega'], P_ref=self.P_ref)
            out['t'].append(snap['t'])
            out['T'].append(p['T']); out['P'].append(p['P']); out['P_gauge'].append(self.gauge(p['P'], z))
            out['RH'].append(float(props['RH'])); out['omega'].append(p['omega']); out['h'].append(float(props['h']))
            out['T_dp'].append(float(st['T_dp'])); out['T_wb'].append(float(st['T_wb']))
            out['V'].append(float(np.sqrt(p['u'] ** 2 + p['v'] ** 2 + p['w'] ** 2)))
            out['rho_ma'].append(float(props['rho_ma'])); out['pv'].append(float(props['pv']))
            es = float(props['rho_ma']) * CP_DA * p['T'] / 1000.0
            el = float(props['rho_da']) * p['omega'] * L_VAP0 / 1000.0
            out['e_sensible'].append(es); out['e_latent'].append(el); out['e_total'].append(es + el)
            out['cond_rate'].append(p['cond_rate'] / m.V_cell * 3.6e6)
        return {k: [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in vals] for k, vals in out.items()}

    # ------------------------------------------------------------------
    # Slices (plan §47-48)
    # ------------------------------------------------------------------
    def slice(self, name: str, index: int, plane: str, pos: float) -> Dict[str, Any]:
        m = self.mesh
        data = self.field(name, index)
        d = self.derived(index)
        if plane == 'xy':
            k = int(np.clip(pos / m.dz, 0, m.Nz - 1))
            arr, a, b = data[:, :, k], m.x_centers, m.y_centers
            ua, ub = d['u'][:, :, k], d['v'][:, :, k]
            coord = m.z_centers[k]
        elif plane == 'xz':
            j = int(np.clip(pos / m.dy, 0, m.Ny - 1))
            arr, a, b = data[:, j, :], m.x_centers, m.z_centers
            ua, ub = d['u'][:, j, :], d['w'][:, j, :]
            coord = m.y_centers[j]
        else:
            i = int(np.clip(pos / m.dx, 0, m.Nx - 1))
            arr, a, b = data[i, :, :], m.y_centers, m.z_centers
            ua, ub = d['v'][i, :, :], d['w'][i, :, :]
            coord = m.x_centers[i]
        return {'plane': plane, 'coord': float(coord), 'a': a.tolist(), 'b': b.tolist(),
                'values': _clean(arr.T), 'ua': _clean(ua.T), 'ub': _clean(ub.T)}

    # ------------------------------------------------------------------
    # Summary / analytics (plan §52-54)
    # ------------------------------------------------------------------
    def indicators(self) -> Dict[str, Any]:
        s = self.series
        if not s:
            return {}
        last = s[-1]
        return {
            'cooling_time': self.cooling_time, 'T_target': self.T_target,
            'T_max_overall': max(r['T_max'] for r in s), 'T_min_overall': min(r['T_min'] for r in s),
            'RH_max_overall': max(r['RH_max'] for r in s),
            'dP_max': max(max(abs(r['P_gauge']) for r in s), max(r['dp_max'] for r in s)),
            'total_condensation_kg': last['condensate_bulk'] + last['frost_wall'],
            'coil_water_kg': last['coil_water'],
            'Q_in_walls_kJ': last['E_walls'] / 1000.0, 'Q_in_internal_kJ': last['E_internal'] / 1000.0,
            'Q_in_infiltration_kJ': last['E_infiltration'] / 1000.0, 'Q_removed_coil_kJ': last['E_coil'] / 1000.0,
            'leak_in_mass_kg': last['leak_in_mass'] + last['door_in_mass'],
            'leak_out_mass_kg': last['leak_out_mass'] + last['door_out_mass'],
            'injected_mass_kg': last['injected_mass'],
            'mass_error_pct': last['mass_error_pct'], 'water_error_pct': last['water_error_pct'],
            'energy_error_pct': last['energy_error_pct'],
            'max_abs_mass_error_pct': max(abs(r['mass_error_pct']) for r in s),
            'max_abs_water_error_pct': max(abs(r['water_error_pct']) for r in s),
            'max_abs_energy_error_pct': max(abs(r['energy_error_pct']) for r in s),
        }


def _clean(arr: np.ndarray):
    a = np.asarray(arr, dtype=float)
    return [[None if not np.isfinite(v) else round(float(v), 7) for v in row] for row in a]
