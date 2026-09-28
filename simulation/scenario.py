"""
Scenario definition for the full cold-storage numerical model.

A scenario is a plain JSON-serialisable dict (what the dashboard sends); this
module fills defaults, validates it, and builds the physics objects.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from geometry.mesh import Mesh
from physics.injection import InjectionSource, FACES
from physics.leakage import LeakageOpening, DoorEvent, OutsideAir, envelope_leakage
from physics.walls import WallSpec
from physics.internal_sources import HeatSource, ProductBlock
from physics.cooling_unit import CoolingUnit
from physics.psychrometrics import humidity_ratio_from_spec, saturation_humidity_ratio


DEFAULT_SCENARIO: Dict[str, Any] = {
    'name': 'Frozen-food store: pull-down with door opening',
    'geometry': {'Lx': 10.0, 'Ly': 8.0, 'Lz': 4.0, 'Nx': 20, 'Ny': 16, 'Nz': 8},
    'initial': {
        'T': -18.0, 'P': 101325.0,
        'moisture': {'mode': 'rh', 'value': 85.0},
        'T_gradient': None,            # {'T_bottom': .., 'T_top': ..}
        'zones': [],                   # [{'box': [x0,x1,y0,y1,z0,z1], 'T': .., 'RH': ..}]
    },
    'ambient': {'T_out': 30.0, 'RH_out': 60.0, 'P_out': 101325.0},
    'walls': {
        'default': {'mode': 'panel', 'layers': [{'L': 0.15, 'k': 0.022}], 'h_in': 8.0, 'h_out': 25.0},
        'B': {'T_out': 10.0, 'layers': [{'L': 0.2, 'k': 1.4}, {'L': 0.15, 'k': 0.03}], 'h_out': 1e6},
        'T': {'solar_absorptivity': 0.6, 'irradiance': 400.0},
    },
    'envelope_leakage': {'ela_per_m2': 0.5e-4},
    'injections': [],
    'leakages': [],
    'doors': [
        {'name': 'Loading door', 'face': 'W', 'position': [0.0, 4.0, 1.25], 'width': 2.0, 'height': 2.5,
         't_open': 600.0, 't_close': 630.0, 'curtain_effectiveness': 0.6},
    ],
    'heat_sources': [
        {'name': 'Forklift', 'kind': 'forklift', 'position': [3.0, 4.0, 0.75], 'power_W': 3000.0,
         'schedule': {'t_start': 600.0, 't_end': 900.0}},
        {'name': 'Lighting', 'kind': 'lighting', 'box': [0.0, 10.0, 0.0, 8.0, 3.5, 4.0], 'power_W': 400.0},
    ],
    'products': [
        {'name': 'Pallet row', 'box': [3.0, 7.0, 1.0, 2.0, 0.0, 1.5], 'T_initial': -12.0,
         'bulk_density': 450.0, 'cp': 1900.0, 'h_surface': 6.0},
    ],
    'cooling_units': [
        {'name': 'Unit cooler', 'airflow_m3_s': 2.0, 'setpoint': -20.0, 'deadband': 1.0,
         'T_coil_leaving': -27.0, 'capacity_W': 12000.0, 'fan_power_W': 400.0,
         'supply': {'face': 'E', 'position': [10.0, 4.0, 3.5], 'width': 2.0, 'height': 0.5},
         'return': {'face': 'E', 'position': [10.0, 4.0, 2.5], 'width': 2.0, 'height': 0.5}},
    ],
    'numerics': {
        't_end': 1200.0, 'dt_mode': 'auto', 'dt': 0.5, 'dt_max': 2.0, 'cfl': 0.8,
        'output_interval': 60.0,
        'turbulence': {'model': 'smagorinsky', 'Cs': 0.17, 'nu_t_min': 1e-4},
        'buoyancy': True, 'k_cond': 1.0, 'k_evap': 0.05, 'surface_condensation': True,
        'T_target': -20.0,
    },
}

# Plan §59 "Example Complete Simulation" as a ready-made preset.
PLAN_EXAMPLE_SCENARIO: Dict[str, Any] = copy.deepcopy(DEFAULT_SCENARIO)
PLAN_EXAMPLE_SCENARIO.update({
    'name': 'Plan §59 example: injection + door leakage',
    'initial': {'T': -18.0, 'P': 101325.0, 'moisture': {'mode': 'rh', 'value': 85.0}, 'zones': []},
    'ambient': {'T_out': 30.0, 'RH_out': 60.0, 'P_out': 101500.0},
    'injections': [
        {'name': 'Injection 1', 'face': 'interior', 'position': [2.0, 4.0, 3.0], 'direction': [1, 0, 0],
         'area': 0.1, 'P_supply': 103000.0,
         'flow': {'mode': 'volumetric', 'value': 0.5},
         'thermal': {'mode': 'temperature', 'value': -5.0},
         'moisture': {'mode': 'rh', 'value': 70.0}},
    ],
    'leakages': [
        {'name': 'Pressure relief vent', 'face': 'T', 'position': [9.0, 1.0, 4.0],
         'width': 0.5, 'height': 0.5, 'area': 0.1, 'Cd': 0.6},
    ],
    'doors': [
        {'name': 'Door', 'face': 'W', 'position': [0.0, 4.0, 1.0], 'width': 1.5, 'height': 2.0,
         't_open': 600.0, 't_close': 660.0},
    ],
    'heat_sources': [], 'products': [],
})
PLAN_EXAMPLE_SCENARIO['numerics'] = dict(DEFAULT_SCENARIO['numerics'], t_end=1200.0)

PRESETS = {'default': DEFAULT_SCENARIO, 'plan_example': PLAN_EXAMPLE_SCENARIO}

MAX_CELLS = 60000


def _merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Scenario:
    raw: Dict[str, Any]
    mesh: Mesh
    T0: float
    P0: float
    omega0: float
    initial: Dict[str, Any]
    ambient: OutsideAir
    walls: Dict[str, WallSpec]
    injections: List[InjectionSource] = field(default_factory=list)
    leakages: List[LeakageOpening] = field(default_factory=list)
    doors: List[DoorEvent] = field(default_factory=list)
    heat_sources: List[HeatSource] = field(default_factory=list)
    products: List[ProductBlock] = field(default_factory=list)
    cooling_units: List[CoolingUnit] = field(default_factory=list)
    numerics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, fill_defaults: bool = True) -> 'Scenario':
        base = DEFAULT_SCENARIO if fill_defaults else {k: {} for k in ('geometry', 'initial', 'ambient', 'walls', 'numerics')}
        # Lists replace (not merge with) the defaults so users can remove sources.
        d = _merge(base, data or {})
        warnings: List[str] = []

        g = d['geometry']
        Lx, Ly, Lz = float(g['Lx']), float(g['Ly']), float(g['Lz'])
        if 'cell_size' in g and g['cell_size']:
            h = float(g['cell_size'])
            Nx, Ny, Nz = (max(2, int(-(-L // h))) for L in (Lx, Ly, Lz))
        else:
            Nx, Ny, Nz = int(g['Nx']), int(g['Ny']), int(g['Nz'])
        if min(Lx, Ly, Lz) <= 0:
            raise ValueError('Room dimensions must be positive.')
        if min(Nx, Ny, Nz) < 2:
            raise ValueError('At least 2 cells are required in every direction.')
        if Nx * Ny * Nz > MAX_CELLS:
            raise ValueError(f'Mesh has {Nx * Ny * Nz} cells; the web solver is limited to {MAX_CELLS}.')
        mesh = Mesh(Lx, Ly, Lz, Nx, Ny, Nz)
        if mesh.max_aspect_ratio > 4:
            warnings.append(f'Cell aspect ratio {mesh.max_aspect_ratio:.1f} is high; keep cells close to cubic.')

        ic = d['initial']
        T0, P0 = float(ic.get('T', -18.0)), float(ic.get('P', 101325.0))
        m = ic.get('moisture', {'mode': 'rh', 'value': 85.0})
        omega0 = humidity_ratio_from_spec(T0, P0, m.get('mode', 'rh'), float(m.get('value', 85.0)))
        if omega0 > float(saturation_humidity_ratio(T0, P0)) * (1 + 1e-9):
            warnings.append('Initial state is supersaturated; excess vapour will condense in the first steps.')
        if omega0 < 0:
            raise ValueError('Initial humidity must be non-negative.')

        amb = d.get('ambient', {})
        ambient = OutsideAir(T=float(amb.get('T_out', 30.0)), moisture_mode='rh',
                             moisture_value=float(amb.get('RH_out', 60.0)), P=float(amb.get('P_out', 101325.0)))

        wall_cfg = d.get('walls', {})
        # Every surface starts from the 'default' spec (outside = ambient air) and
        # applies its own overrides, e.g. the floor sits on ground at 10 °C.
        default_wall = WallSpec.from_dict(wall_cfg.get('default', {}), WallSpec(T_out=ambient.T))
        walls = {f: WallSpec.from_dict(wall_cfg.get(f, {}), default_wall) for f in FACES}

        injections = [InjectionSource.from_dict(x, i) for i, x in enumerate(d.get('injections') or [])]
        leakages = [LeakageOpening.from_dict(x, i, ambient) for i, x in enumerate(d.get('leakages') or [])]
        doors = [DoorEvent.from_dict(x, i, ambient) for i, x in enumerate(d.get('doors') or [])]
        heat_sources = [HeatSource.from_dict(x, i) for i, x in enumerate(d.get('heat_sources') or [])]
        products = [ProductBlock.from_dict(x, i) for i, x in enumerate(d.get('products') or [])]
        units = [CoolingUnit.from_dict(x, i) for i, x in enumerate(d.get('cooling_units') or [])]

        # Background envelope leakage: split between a low and a high crack so
        # the stack effect acts on the envelope.
        env = d.get('envelope_leakage') or {}
        ela = float(env.get('ela_per_m2', 0.0) or 0.0) * mesh.surface_area
        if ela > 0:
            leakages.append(envelope_leakage('Envelope leakage (low)', 'S', [Lx / 2, 0.0, 0.25 * Lz], ela / 2, ambient))
            leakages.append(envelope_leakage('Envelope leakage (high)', 'N', [Lx / 2, Ly, 0.75 * Lz], ela / 2, ambient))

        for src in injections + leakages + doors:
            _check_position(src.geometry.position, mesh, src.name, warnings)
        for hs in heat_sources:
            _check_position(hs.position, mesh, hs.name, warnings)

        num = d.get('numerics', {})
        if float(num.get('t_end', 0)) <= 0:
            raise ValueError('Simulation end time must be positive.')

        return cls(raw=d, mesh=mesh, T0=T0, P0=P0, omega0=omega0, initial=ic, ambient=ambient, walls=walls,
                   injections=injections, leakages=leakages, doors=doors, heat_sources=heat_sources,
                   products=products, cooling_units=units, numerics=num, warnings=warnings)


def _check_position(pos, mesh: Mesh, name: str, warnings: List[str]) -> None:
    x, y, z = pos
    if not (0 <= x <= mesh.Lx and 0 <= y <= mesh.Ly and 0 <= z <= mesh.Lz):
        warnings.append(f"'{name}' position {tuple(pos)} lies outside the room and was clipped to the nearest cell.")
