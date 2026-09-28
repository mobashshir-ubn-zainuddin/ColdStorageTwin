"""
Grid-convergence study (plan Phase 10 and proposal §6.1).

Runs the analytical diffusion problem on successively refined meshes
(refinement ratio r = 2) and reports the observed order of accuracy
p = ln(e_coarse / e_fine) / ln(r). The finite-volume scheme is second order in
space for diffusion, so p should approach 2 (time error is kept small by Δt).
"""

import math
from typing import List, Dict, Any

from .diffusion import cosine_decay


def grid_convergence(levels=(8, 16, 32), t_end: float = 600.0) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = [cosine_decay(nx, t_end) for nx in levels]
    orders = []
    for a, b in zip(runs[:-1], runs[1:]):
        r = a['dx'] / b['dx']
        orders.append({'from_nx': a['nx'], 'to_nx': b['nx'],
                       'p_T': math.log(a['error_T'] / b['error_T']) / math.log(r) if b['error_T'] > 0 else float('nan'),
                       'p_w': math.log(a['error_w'] / b['error_w']) / math.log(r) if b['error_w'] > 0 else float('nan')})
    return {'runs': runs, 'observed_order': orders}
