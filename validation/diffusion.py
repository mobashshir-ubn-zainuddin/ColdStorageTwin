"""
Verification against analytical solutions (plan Phase 10, steps 1-2).

A sealed, adiabatic box with still air and an initial cosine mode

    φ(x, 0) = φ0 + A cos(π x / L)

decays exactly as A exp(-Γ π² t / L²) for both temperature (Γ = α) and
humidity ratio (Γ = D). The solver runs with buoyancy and turbulence switched
off so it reduces to pure diffusion; the numerical decay is compared with the
analytic decay.
"""

import math
from typing import Dict, Any

import numpy as np

from simulation.scenario import Scenario
from solver.coupled_solver import ColdRoomSolver, K_AIR, D_VAPOUR
from physics.psychrometrics import CP_DA


def _diffusion_box(nx: int, L: float = 1.0) -> Dict[str, Any]:
    return {
        'geometry': {'Lx': L, 'Ly': 0.25, 'Lz': 0.25, 'Nx': nx, 'Ny': 2, 'Nz': 2},
        'initial': {'T': -10.0, 'P': 101325.0, 'moisture': {'mode': 'omega', 'value': 0.0008}},
        'walls': {'default': {'mode': 'adiabatic'}, 'B': {'mode': 'adiabatic'}, 'T': {'mode': 'adiabatic'}},
        'envelope_leakage': {'ela_per_m2': 0.0},
        'injections': [], 'leakages': [], 'doors': [], 'heat_sources': [], 'products': [], 'cooling_units': [],
        'numerics': {'t_end': 600.0, 'dt_mode': 'auto', 'dt_max': 1.0, 'cfl': 0.5, 'output_interval': 600.0,
                     'turbulence': {'model': 'laminar'}, 'buoyancy': False, 'surface_condensation': False},
    }


def cosine_decay(nx: int = 16, t_end: float = 600.0, amplitude_T: float = 2.0,
                 amplitude_w: float = 1e-4) -> Dict[str, Any]:
    cfg = _diffusion_box(nx)
    cfg['numerics']['t_end'] = t_end
    sc = Scenario.from_dict(cfg, fill_defaults=True)
    s = ColdRoomSolver(sc)
    L = s.mesh.Lx
    mode = np.cos(math.pi * s.mesh.X_coords / L)
    s.T = s.T + amplitude_T * mode
    s.omega = s.omega + amplitude_w * mode
    # re-baseline the ledgers after perturbing the initial state
    s.W0, s.E0 = s._total_water(), s._total_energy()
    T0, w0 = float(s.T.mean()), float(s.omega.mean())
    s.run()
    proj = lambda f, f0: float(((f - f0) * mode).sum() / (mode * mode).sum())
    alpha = K_AIR / (s.rho_bar * CP_DA)
    exact_T = amplitude_T * math.exp(-alpha * math.pi ** 2 * s.t / L ** 2)
    exact_w = amplitude_w * math.exp(-D_VAPOUR * math.pi ** 2 * s.t / L ** 2)
    num_T, num_w = proj(s.T, T0), proj(s.omega, w0)
    return {'nx': nx, 'dx': s.mesh.dx, 't': s.t, 'alpha': alpha, 'D': D_VAPOUR,
            'amplitude_T_numeric': num_T, 'amplitude_T_exact': exact_T,
            'error_T': abs(num_T - exact_T) / amplitude_T,
            'amplitude_w_numeric': num_w, 'amplitude_w_exact': exact_w,
            'error_w': abs(num_w - exact_w) / amplitude_w,
            'max_velocity': float(max(np.abs(s.u).max(), np.abs(s.v).max(), np.abs(s.w).max())),
            'conservation': s.conservation()}
