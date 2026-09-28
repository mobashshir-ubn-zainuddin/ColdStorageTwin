"""
Conservation verification (plan §54, Phase 10): runs a scenario with every
source/boundary mechanism active and reports the worst dry-air, water and energy
ledger errors and the worst discrete divergence residual over the run.
"""

from typing import Dict, Any, Optional

from simulation.scenario import Scenario, DEFAULT_SCENARIO
from simulation.results import ResultStore
from solver.coupled_solver import ColdRoomSolver


def conservation_audit(overrides: Optional[Dict[str, Any]] = None, t_end: float = 120.0) -> Dict[str, Any]:
    cfg = dict(overrides or {})
    num = dict(DEFAULT_SCENARIO['numerics'], **cfg.get('numerics', {}))
    num['t_end'] = t_end
    num['output_interval'] = t_end / 4
    cfg['numerics'] = num
    s = ColdRoomSolver(Scenario.from_dict(cfg))
    rec = ResultStore(s)
    s.run(recorder=rec)
    return {
        'max_mass_error_pct': max(abs(r['mass_error_pct']) for r in rec.series),
        'max_water_error_pct': max(abs(r['water_error_pct']) for r in rec.series),
        'max_energy_error_pct': max(abs(r['energy_error_pct']) for r in rec.series),
        'max_divergence': max(r['divergence'] for r in rec.series),
        'steps': s.step_count, 'final': s.conservation(),
    }
