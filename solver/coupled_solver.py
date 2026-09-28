"""
Fully coupled transient 3-D finite-volume solver for a cold-storage room
(plan Modules 4-12, Phase 8 "full numerical coupling").

Discretisation
--------------
* Structured Cartesian finite volumes. Scalars (T, ω, pressure) live at cell
  centres; velocity components live on cell faces (staggered MAC grid). The
  discrete divergence used by the projection is exactly the one used by scalar
  transport, so mass, moisture and energy are conserved to round-off.
* Momentum: explicit upwind advection + eddy-viscosity diffusion (Smagorinsky or
  constant ν_t) + Boussinesq buoyancy from the local moist-air density
  ρ(T, P, ω), including the humidity effect. Chorin projection with a sparse
  LU-factorised pressure Poisson operator (obstacles supported).
* Room pressure: a lumped node P_room(t) solved implicitly each step from the
  dry-air mass balance  (M(P) - M_old)/Δt = Σ ṁ_da,i(P), with M(P) from the ideal
  gas law, as in multizone airflow models. Every opening's flow (orifice/crack power
  laws with hydrostatic stack terms) is evaluated at that pressure, so
  infiltration/exfiltration direction follows ΔP automatically. A sealed room
  pressurises or depressurises (e.g. while cooling down).
* Low-Mach continuity: in-room air uses the spatially uniform dry-air density
  ρ̄(t) = M_da/V; ∇·u = q_src - (dρ̄/dt)/ρ̄. Conservative upwind transport of ρ̄ω
  and ρ̄ c_p T with inflow boundary values at openings.
* Phase change: bulk supersaturation relaxes to saturation (condensation to
  liquid above 0 °C, deposition to ice below), deposited water can evaporate or
  sublimate, and wall-surface condensation/frost uses the Lewis analogy.
  Latent heat is added to the energy equation exactly once.
* Boundary/source mechanisms: injection (face or interior Gaussian jet), leakage,
  doors with Gosney-Olama stack exchange, wall U-value/sol-air heat flux, internal
  heat & moisture sources, product blocks (thermal mass + respiration), and
  thermostat-controlled unit coolers.
"""

import math
import time as _time
from typing import Dict, Any, List, Optional, Callable

import numpy as np
from scipy import optimize, sparse
from scipy.sparse.linalg import splu
from scipy import ndimage

from geometry.mesh import Mesh
from physics.psychrometrics import (saturation_pressure, saturation_humidity_ratio, moist_air_density,
                                    R_DA, R_V, CP_DA, L_VAP0, L_FUS, L_SUB0)
from physics.injection import InjectionSource, G, FACES
from physics.leakage import LeakageOpening, DoorEvent
from physics.walls import lewis_mass_transfer_coefficient
from simulation.scenario import Scenario

K_AIR = 0.024          # thermal conductivity of air [W/mK]
NU_AIR = 1.2e-5        # kinematic viscosity [m²/s]
D_VAPOUR = 2.2e-5      # vapour diffusivity in air [m²/s]
PR_T, SC_T = 0.85, 0.7  # turbulent Prandtl / Schmidt numbers

# Face -> (velocity array name, axis, index side, inward sign)
FACE_INFO = {'W': ('u', 0, 0, +1), 'E': ('u', 0, -1, -1), 'S': ('v', 1, 0, +1),
             'N': ('v', 1, -1, -1), 'B': ('w', 2, 0, +1), 'T': ('w', 2, -1, -1)}


def _cell_slice(face: str):
    """Index expression selecting the 2-D layer of cells adjacent to a boundary face."""
    _, axis, side, _ = FACE_INFO[face]
    idx = [slice(None)] * 3
    idx[axis] = side
    return tuple(idx)


def _direction(mdot: float, V_ex: float = 0.0) -> str:
    net = 'into room' if mdot > 1e-9 else ('out of room' if mdot < -1e-9 else 'none')
    if V_ex > 0:
        return 'two-way exchange' + ('' if net == 'none' else ', net ' + net.split()[0])
    return net


class BoundaryPatch:
    """A set of boundary-face cells belonging to one opening."""

    def __init__(self, mesh: Mesh, face: str, position, width: float, height: float):
        self.face = face
        self.mask = mesh.boundary_patch(face, position, width, height)
        self.n = int(self.mask.sum())
        self.face_area = mesh.face_area(face)
        self.area = self.n * self.face_area
        a, b = mesh.face_plane_axes(face)
        centers = (mesh.x_centers, mesh.y_centers, mesh.z_centers)
        # Height (z) of every face cell, needed for door splitting
        if b == 2:
            Z = np.broadcast_to(centers[2][None, :], self.mask.shape)
        else:
            Z = np.full(self.mask.shape, 0.0 if face == 'B' else mesh.Lz)
        self.z = Z


class ColdRoomSolver:
    """Transient coupled solver. Construct from a Scenario, then call run()."""

    def __init__(self, scenario: Scenario):
        self.sc = scenario
        m = self.mesh = scenario.mesh
        self.nx, self.ny, self.nz = m.Nx, m.Ny, m.Nz
        self.shape = (m.Nx, m.Ny, m.Nz)
        self.V = m.V_cell
        num = scenario.numerics
        self.num = num
        self.g = G if num.get('buoyancy', True) else 0.0
        turb = num.get('turbulence', {}) or {}
        self.turb_model = turb.get('model', 'smagorinsky')
        self.Cs = float(turb.get('Cs', 0.17))
        self.nu_t_min = float(turb.get('nu_t_min', 1e-4))
        self.nu_t_const = float(turb.get('nu_t', 1e-3))
        self.k_cond = float(num.get('k_cond', 1.0))
        self.k_evap = float(num.get('k_evap', 0.05))
        self.surface_condensation = bool(num.get('surface_condensation', True))
        self.cfl = float(num.get('cfl', 0.7))

        # ---- geometry: fluid / product cells --------------------------------
        self.fluid = np.ones(self.shape, dtype=bool)
        self._init_products()
        self.solid = ~self.fluid
        self.V_fluid = self.fluid.sum() * self.V
        if self.V_fluid <= 0:
            raise ValueError('Product blocks fill the whole room.')
        f = self.fluid
        self.open_x = f[:-1] & f[1:]
        self.open_y = f[:, :-1] & f[:, 1:]
        self.open_z = f[:, :, :-1] & f[:, :, 1:]

        # ---- state -----------------------------------------------------------
        self.t = 0.0
        self.T = np.full(self.shape, scenario.T0, dtype=float)
        self.omega = np.full(self.shape, scenario.omega0, dtype=float)
        self._apply_initial_distribution()
        self.T[self.solid] = self.T_prod_init[self.solid]
        self.omega[self.solid] = 0.0
        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))
        self.phi = np.zeros(self.shape)           # kinematic dynamic pressure [m²/s²]
        self.P_room = scenario.P0                  # absolute pressure at floor level [Pa]
        Tm, wm = self._room_means()
        self.M_da = self._dry_air_mass(self.P_room, Tm, wm)
        self.rho_bar = self.M_da / self.V_fluid
        self.liquid = np.zeros(self.shape)         # deposited liquid water per cell [kg]
        self.ice = np.zeros(self.shape)            # deposited ice/frost per cell [kg]
        self.cond_rate = np.zeros(self.shape)      # bulk phase-change rate (+cond, -evap) [kg/s per cell]
        self.nu_t = np.full(self.shape, self.nu_t_min)

        # ---- boundary openings, sources, walls ------------------------------
        self._init_openings()
        self._init_walls()
        self._init_heat_sources()
        self._build_poisson()

        # ---- ledgers ---------------------------------------------------------
        self.cum = {k: 0.0 for k in ('dry_air_in', 'water_in', 'energy_in', 'injected_mass', 'leak_in_mass',
                                     'leak_out_mass', 'door_in_mass', 'door_out_mass', 'Q_walls', 'Q_internal',
                                     'Q_coil', 'Q_infiltration', 'condensed_bulk', 'condensed_wall',
                                     'coil_water', 'Q_respiration')}
        self.M_da0 = self.M_da
        self.W0 = self._total_water()
        self.E0 = self._total_energy()
        self.last = {}          # last-step rates for reporting
        self.stability = {}
        self.step_count = 0
        self.dt_last = 0.0
        self.aborted = None

    # =====================================================================
    # Initialisation helpers
    # =====================================================================
    def _init_products(self):
        m = self.mesh
        self.T_prod_init = np.zeros(self.shape)
        self.prod_mass = np.zeros(self.shape)
        self.prod_cp = np.ones(self.shape)
        self.prod_h = np.zeros(self.shape)
        self.prod_qref = np.zeros(self.shape)
        self.prod_Q10 = np.ones(self.shape)
        self.prod_transp = np.zeros(self.shape)
        for p in self.sc.products:
            mask = m.box_mask(*p.box)
            if not mask.any():
                self.sc.warnings.append(f"Product block '{p.name}' is smaller than a cell and was ignored.")
                continue
            self.fluid[mask] = False
            self.T_prod_init[mask] = p.T_initial
            self.prod_mass[mask] = p.bulk_density * self.V
            self.prod_cp[mask] = p.cp
            self.prod_h[mask] = p.h_surface
            self.prod_qref[mask] = p.respiration_ref_W_kg
            self.prod_Q10[mask] = p.Q10
            self.prod_transp[mask] = p.transpiration_kg_kg_s

    def _apply_initial_distribution(self):
        ic, m, sc = self.sc.initial, self.mesh, self.sc
        from physics.psychrometrics import humidity_ratio_from_spec
        grad = ic.get('T_gradient')
        if grad:
            Tb, Tt = float(grad['T_bottom']), float(grad['T_top'])
            self.T = Tb + (Tt - Tb) * m.Z_coords / m.Lz
            mo = ic.get('moisture', {'mode': 'rh', 'value': 85.0})
            if mo.get('mode', 'rh') == 'rh':
                from physics.psychrometrics import humidity_ratio_from_rh
                self.omega = humidity_ratio_from_rh(self.T, sc.P0, float(mo['value']))
        for zone in ic.get('zones') or []:
            mask = m.box_mask(*[float(v) for v in zone['box']])
            Tz = float(zone.get('T', sc.T0))
            self.T[mask] = Tz
            if 'RH' in zone:
                self.omega[mask] = humidity_ratio_from_spec(Tz, sc.P0, 'rh', float(zone['RH']))
            elif 'omega' in zone:
                self.omega[mask] = float(zone['omega'])

    def _init_openings(self):
        m = self.mesh
        self.elements: List[Dict[str, Any]] = []
        interior = []
        for src in self.sc.injections:
            if src.geometry.is_boundary:
                patch = BoundaryPatch(m, src.geometry.face, src.geometry.position, src.geometry.width,
                                      src.geometry.height)
                self.elements.append({'obj': src, 'kind': 'injection', 'patch': patch})
            else:
                wts = m.gaussian_weights(tuple(src.geometry.position), src.geometry.sigma, self.fluid)
                interior.append({'obj': src, 'kind': 'injection', 'weights': wts})
        for lk in self.sc.leakages:
            patch = BoundaryPatch(m, lk.geometry.face, lk.geometry.position, lk.geometry.width, lk.geometry.height)
            self.elements.append({'obj': lk, 'kind': 'leakage', 'patch': patch})
        for dr in self.sc.doors:
            patch = BoundaryPatch(m, dr.geometry.face, dr.geometry.position, dr.geometry.width, dr.geometry.height)
            self.elements.append({'obj': dr, 'kind': 'door', 'patch': patch})
        self.units = []
        for cu in self.sc.cooling_units:
            entry = {'obj': cu}
            for key, geom in (('supply', cu.supply), ('return', cu.ret)):
                if geom.is_boundary:
                    entry[key + '_patch'] = BoundaryPatch(m, geom.face, geom.position, geom.width, geom.height)
                else:
                    entry[key + '_weights'] = m.gaussian_weights(tuple(geom.position), geom.sigma, self.fluid)
            self.units.append(entry)
        self.interior_injections = interior
        # Openings attached to product cells cannot carry flow
        for el in self.elements:
            cells = self.fluid[_cell_slice(el['patch'].face)]
            el['patch'].mask &= cells
            el['patch'].n = int(el['patch'].mask.sum())
            el['patch'].area = el['patch'].n * el['patch'].face_area

    def _init_walls(self):
        # Wall heat transfer acts on every boundary face not occupied by an opening
        self.wall_active = {}
        for f in FACES:
            _, axis, _, _ = FACE_INFO[f]
            shape2d = tuple(s for i, s in enumerate(self.shape) if i != axis)
            self.wall_active[f] = np.ones(shape2d, dtype=bool)
        for el in self.elements:
            if el['kind'] in ('door', 'injection'):
                self.wall_active[el['patch'].face] &= ~el['patch'].mask
        self.wall_ice = {f: np.zeros_like(self.wall_active[f], dtype=float) for f in FACES}
        self.wall_liquid = {f: np.zeros_like(self.wall_active[f], dtype=float) for f in FACES}

    def _init_heat_sources(self):
        m = self.mesh
        self.hs_weights = []
        for hs in self.sc.heat_sources:
            if hs.box:
                mask = m.box_mask(*hs.box) & self.fluid
                if not mask.any():
                    mask = np.zeros(self.shape, bool)
                    mask[m.locate(*hs.position)] = True
                w = mask.astype(float) / mask.sum()
            else:
                w = m.gaussian_weights(tuple(hs.position), hs.sigma, self.fluid)
            self.hs_weights.append(w)

    def _build_poisson(self):
        """Sparse Laplacian with homogeneous Neumann conditions on walls/obstacles."""
        nx, ny, nz = self.shape
        N = nx * ny * nz
        idx = np.arange(N).reshape(self.shape)
        rows, cols, vals = [], [], []
        diag = np.zeros(N)
        for open_mask, a, b, d in ((self.open_x, idx[:-1], idx[1:], self.mesh.dx),
                                   (self.open_y, idx[:, :-1], idx[:, 1:], self.mesh.dy),
                                   (self.open_z, idx[:, :, :-1], idx[:, :, 1:], self.mesh.dz)):
            c = 1.0 / d ** 2
            p, q = a[open_mask], b[open_mask]
            rows += [p, q]
            cols += [q, p]
            vals += [np.full(p.size, c), np.full(q.size, c)]
            np.add.at(diag, p, -c)
            np.add.at(diag, q, -c)
        labels, n_comp = ndimage.label(self.fluid)
        self.components = [(labels == c).ravel() for c in range(1, n_comp + 1)]
        self.pins = [int(np.flatnonzero(cm)[0]) for cm in self.components]
        fixed = np.zeros(N, dtype=bool)
        fixed[~self.fluid.ravel()] = True
        fixed[self.pins] = True
        rows = np.concatenate(rows) if rows else np.array([], int)
        cols = np.concatenate(cols) if cols else np.array([], int)
        vals = np.concatenate(vals) if vals else np.array([])
        keep = ~fixed[rows]
        diag[fixed] = 1.0
        L = sparse.coo_matrix((np.concatenate([vals[keep], diag]),
                               (np.concatenate([rows[keep], np.arange(N)]),
                                np.concatenate([cols[keep], np.arange(N)]))), shape=(N, N)).tocsc()
        self.poisson_lu = splu(L)
        self.poisson_fixed = fixed

    # =====================================================================
    # Room-level helpers
    # =====================================================================
    def _room_means(self):
        return float(self.T[self.fluid].mean()), float(self.omega[self.fluid].mean())

    def _dry_air_mass(self, P, Tm, wm):
        return P * self.V_fluid / (R_DA * (Tm + 273.15) * (1.0 + 1.6078 * wm))

    def _total_water(self):
        wall = sum(self.wall_ice[f].sum() + self.wall_liquid[f].sum() for f in FACES)
        return float(self.rho_bar * self.V * self.omega[self.fluid].sum() + self.liquid.sum() + self.ice.sum() + wall)

    def _total_energy(self):
        air = self.rho_bar * self.V * (CP_DA * self.T[self.fluid].sum() + L_VAP0 * self.omega[self.fluid].sum())
        ice = self.ice.sum() + sum(self.wall_ice[f].sum() for f in FACES)
        prod = float((self.prod_mass * self.prod_cp * self.T)[self.solid].sum())
        return float(air - L_FUS * ice + prod)

    def _adjacent_state(self, patch: BoundaryPatch):
        sl = _cell_slice(patch.face)
        T = self.T[sl][patch.mask]
        w = self.omega[sl][patch.mask]
        return (float(T.mean()), float(w.mean())) if T.size else self._room_means()

    # =====================================================================
    # Lumped room pressure and opening flows
    # =====================================================================
    def _resolve_openings(self, dt: float) -> Dict[str, Any]:
        """Solve the implicit dry-air balance for P_room and evaluate every opening flow."""
        t = self.t + dt
        Tm, wm = self._room_means()
        rho_room = float(moist_air_density(Tm, self.P_room, wm))
        fixed_mda = 0.0
        entries = []
        for el in self.elements:
            obj = el['obj']
            Tadj, wadj = self._adjacent_state(el['patch'])
            state = obj.supply_state(t, Tadj, wadj, self.P_room)
            entries.append((el, state, Tadj, wadj))
        for src in self.interior_injections:
            obj = src['obj']
            state = obj.supply_state(t, Tm, wm, self.P_room)
            src['state'] = state
            if not obj.pressure_dependent:
                src['mdot'] = obj.mass_flow(t, self.P_room, rho_room, state)
                fixed_mda += src['mdot'] / (1 + (state['omega'] if src['mdot'] > 0 else wm))

        def flows(P):
            out = []
            for el, state, Tadj, wadj in entries:
                obj = el['obj']
                if el['kind'] == 'injection':
                    Pz = P - rho_room * G * obj.geometry.z_mid
                    mdot = obj.mass_flow(t, Pz, rho_room, state)
                else:
                    mdot = obj.mass_flow(t, P, rho_room, state)
                out.append(mdot / (1.0 + (state['omega'] if mdot > 0 else wadj)))
            for src in self.interior_injections:
                obj = src['obj']
                if obj.pressure_dependent:
                    Pz = P - rho_room * G * obj.geometry.z_mid
                    src['mdot'] = obj.mass_flow(t, Pz, rho_room, src['state'])
            return out

        def residual(P):
            mda = sum(flows(P)) + fixed_mda
            for src in self.interior_injections:
                if src['obj'].pressure_dependent:
                    mda += src['mdot'] / (1 + src['state']['omega'])
            return (self._dry_air_mass(P, Tm, wm) - self.M_da) / dt - mda

        P0 = self.P_room
        lo, hi = P0 - 2000.0, P0 + 2000.0
        rlo, rhi = residual(lo), residual(hi)
        expand = 0
        while rlo * rhi > 0 and expand < 12:
            lo, hi = lo - 4000.0 * 2 ** expand, hi + 4000.0 * 2 ** expand
            lo = max(lo, 1000.0)
            rlo, rhi = residual(lo), residual(hi)
            expand += 1
        P_new = optimize.brentq(residual, lo, hi, xtol=1e-6) if rlo * rhi <= 0 else (lo if abs(rlo) < abs(rhi) else hi)
        mda_list = flows(P_new)
        total_mda = sum(mda_list) + fixed_mda + sum(
            s['mdot'] / (1 + s['state']['omega']) for s in self.interior_injections if s['obj'].pressure_dependent)
        info = []
        for (el, state, Tadj, wadj), mda in zip(entries, mda_list):
            obj = el['obj']
            mdot = mda * (1.0 + (state['omega'] if mda > 0 else wadj))
            rec = {'el': el, 'state': state, 'mda': mda, 'mdot': mdot, 'T_adj': Tadj, 'w_adj': wadj}
            if el['kind'] == 'door':
                rec['exchange'] = obj.exchange_flow(t, rho_room, state)
                rec['dP'] = obj.delta_p(P_new, rho_room, state)
            elif el['kind'] == 'leakage':
                rec['dP'] = obj.delta_p(P_new, rho_room, state)
            else:
                rec['dP'] = (obj.P_supply - (P_new - rho_room * G * obj.geometry.z_mid)) if obj.P_supply else 0.0
            info.append(rec)
        return {'P_new': P_new, 'records': info, 'total_mda': total_mda, 'rho_room': rho_room,
                'M_new': self.M_da + dt * total_mda}

    # =====================================================================
    # Boundary velocities and scalar inflow values
    # =====================================================================
    def _boundary_conditions(self, op: Dict[str, Any], units_state: List[Dict[str, Any]]):
        """Set boundary-face normal velocities and the inflow values of T and ω."""
        rb = self.rho_bar
        vel = {'u': self.u, 'v': self.v, 'w': self.w}
        self.bnd_T, self.bnd_w, self.bnd_jet = {}, {}, {}
        for f in FACES:
            name, axis, side, sign = FACE_INFO[f]
            arr = vel[name]
            idx = [slice(None)] * 3
            idx[axis] = side
            arr[tuple(idx)] = 0.0
            shape2d = self.wall_active[f].shape
            self.bnd_T[f] = np.zeros(shape2d)
            self.bnd_w[f] = np.zeros(shape2d)
            self.bnd_jet[f] = np.zeros(shape2d)
        self.exchange_sources = []

        def apply(patch: BoundaryPatch, V_in: np.ndarray, T_in, w_in, jet=None):
            name, axis, side, sign = FACE_INFO[patch.face]
            arr = vel[name]
            idx = [slice(None)] * 3
            idx[axis] = side
            layer = arr[tuple(idx)]
            layer[patch.mask] += sign * V_in / patch.face_area
            self.bnd_T[patch.face][patch.mask] = T_in
            self.bnd_w[patch.face][patch.mask] = w_in
            if jet is not None:
                self.bnd_jet[patch.face][patch.mask] = jet

        for rec in op['records']:
            el, st = rec['el'], rec['state']
            patch = el['patch']
            if patch.n == 0:
                continue
            V_net = rec['mda'] / rb                      # inward volumetric flow in room metric
            per = np.full(patch.n, V_net / patch.n)
            jet = None
            if el['kind'] == 'injection' and V_net > 0:
                jet = el['obj'].jet_velocity(rec['mdot'], st['rho'])
            if el['kind'] == 'door':
                ex = rec['exchange']['V_ex']
                if ex > 0:
                    z = patch.z[patch.mask]
                    zmid = 0.5 * (z.min() + z.max())
                    top, bot = z > zmid + 1e-9, z < zmid - 1e-9
                    if top.any() and bot.any():
                        inflow = top if rec['exchange']['inflow_top'] else bot
                        outflow = bot if rec['exchange']['inflow_top'] else top
                        per = per + np.where(inflow, ex / inflow.sum(), 0.0) - np.where(outflow, ex / outflow.sum(), 0.0)
                    else:
                        # Door spans a single cell row: represent the two-way exchange as a
                        # zero-net-mass source/sink pair in the adjacent cells.
                        self.exchange_sources.append({'patch': patch, 'V': ex, 'T': st['T'], 'w': st['omega']})
            apply(patch, per, st['T'], st['omega'], jet)
            rec['V_in'] = V_net
        for us in units_state:
            if not us['proc']['running']:
                continue
            Vd = us['proc']['m_da'] / rb
            us['V_room'] = Vd
            if 'supply_patch' in us['entry']:
                p = us['entry']['supply_patch']
                apply(p, np.full(p.n, Vd / p.n), us['proc']['T'], us['proc']['omega'])
            if 'return_patch' in us['entry']:
                p = us['entry']['return_patch']
                apply(p, np.full(p.n, -Vd / p.n), 0.0, 0.0)

    # =====================================================================
    # Momentum, projection
    # =====================================================================
    @staticmethod
    def _momentum_rhs(q, cb, cc, nu_c, da, db, dc):
        """Upwind advection + diffusion of a staggered component in canonical axis order."""
        qi = q[1:-1]
        Ua = qi
        Ub = 0.5 * (cb[:-1] + cb[1:])
        Uc = 0.5 * (cc[:-1] + cc[1:])
        nu = 0.5 * (nu_c[:-1] + nu_c[1:])
        dm = (q[1:-1] - q[:-2]) / da
        dp = (q[2:] - q[1:-1]) / da
        adv = np.where(Ua > 0, Ua * dm, Ua * dp)
        lap = (q[2:] - 2 * qi + q[:-2]) / da ** 2
        # no-slip ghosts on the tangential walls
        gb = np.concatenate([-qi[:, :1], qi, -qi[:, -1:]], axis=1)
        adv += np.where(Ub > 0, Ub * (gb[:, 1:-1] - gb[:, :-2]) / db, Ub * (gb[:, 2:] - gb[:, 1:-1]) / db)
        lap += (gb[:, 2:] - 2 * qi + gb[:, :-2]) / db ** 2
        gc = np.concatenate([-qi[:, :, :1], qi, -qi[:, :, -1:]], axis=2)
        adv += np.where(Uc > 0, Uc * (gc[:, :, 1:-1] - gc[:, :, :-2]) / dc, Uc * (gc[:, :, 2:] - gc[:, :, 1:-1]) / dc)
        lap += (gc[:, :, 2:] - 2 * qi + gc[:, :, :-2]) / dc ** 2
        return -adv + nu * lap

    def cell_velocities(self):
        return (0.5 * (self.u[:-1] + self.u[1:]), 0.5 * (self.v[:, :-1] + self.v[:, 1:]),
                0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:]))

    def _update_turbulence(self, uc, vc, wc):
        if self.turb_model == 'constant':
            self.nu_t = np.full(self.shape, max(self.nu_t_const, self.nu_t_min))
            return
        if self.turb_model == 'laminar':
            self.nu_t = np.zeros(self.shape)
            return
        m = self.mesh
        gu = np.gradient(uc, m.dx, m.dy, m.dz)
        gv = np.gradient(vc, m.dx, m.dy, m.dz)
        gw = np.gradient(wc, m.dx, m.dy, m.dz)
        S2 = 2 * (gu[0] ** 2 + gv[1] ** 2 + gw[2] ** 2) + (gu[1] + gv[0]) ** 2 + (gu[2] + gw[0]) ** 2 + (gv[2] + gw[1]) ** 2
        delta = m.characteristic_size
        self.nu_t = np.maximum((self.Cs * delta) ** 2 * np.sqrt(S2), self.nu_t_min)
        self.nu_t[self.solid] = 0.0

    def _momentum_and_projection(self, dt: float, q_src: np.ndarray, q_dil: float, jet_accel):
        m = self.mesh
        uc, vc, wc = self.cell_velocities()
        self._update_turbulence(uc, vc, wc)
        nu = NU_AIR + self.nu_t

        ru = self._momentum_rhs(self.u, vc, wc, nu, m.dx, m.dy, m.dz)
        rv = self._momentum_rhs(self.v.transpose(1, 0, 2), uc.transpose(1, 0, 2), wc.transpose(1, 0, 2),
                                nu.transpose(1, 0, 2), m.dy, m.dx, m.dz).transpose(1, 0, 2)
        rw = self._momentum_rhs(self.w.transpose(2, 0, 1), uc.transpose(2, 0, 1), vc.transpose(2, 0, 1),
                                nu.transpose(2, 0, 1), m.dz, m.dx, m.dy).transpose(1, 2, 0)

        # Boussinesq buoyancy from the local moist-air density
        if self.g:
            rho = moist_air_density(self.T, self.P_room, self.omega)
            rho_ref = float(rho[self.fluid].mean())
            rw += -self.g * (0.5 * (rho[:, :, :-1] + rho[:, :, 1:]) - rho_ref) / rho_ref

        ax, ay, az = jet_accel
        u_s, v_s, w_s = self.u.copy(), self.v.copy(), self.w.copy()
        u_s[1:-1] += dt * (ru + 0.5 * (ax[:-1] + ax[1:]))
        v_s[:, 1:-1] += dt * (rv + 0.5 * (ay[:, :-1] + ay[:, 1:]))
        w_s[:, :, 1:-1] += dt * (rw + 0.5 * (az[:, :, :-1] + az[:, :, 1:]))
        self._add_inlet_momentum(u_s, v_s, w_s, dt)
        u_s[1:-1][~self.open_x] = 0.0
        v_s[:, 1:-1][~self.open_y] = 0.0
        w_s[:, :, 1:-1][~self.open_z] = 0.0

        # Pressure Poisson: ∇²φ = (∇·u* - q_src - q_dil) / Δt
        div = ((u_s[1:] - u_s[:-1]) / m.dx + (v_s[:, 1:] - v_s[:, :-1]) / m.dy +
               (w_s[:, :, 1:] - w_s[:, :, :-1]) / m.dz)
        self._div_target = q_src + q_dil
        rhs = ((div - q_src - q_dil) / dt).ravel()
        rhs[~self.fluid.ravel()] = 0.0
        for comp in self.components:
            rhs[comp] -= rhs[comp].mean()
        rhs[self.poisson_fixed] = 0.0
        phi = self.poisson_lu.solve(rhs).reshape(self.shape)
        phi[self.solid] = 0.0
        self.phi = phi
        u_s[1:-1] -= dt * np.where(self.open_x, (phi[1:] - phi[:-1]) / m.dx, 0.0)
        v_s[:, 1:-1] -= dt * np.where(self.open_y, (phi[:, 1:] - phi[:, :-1]) / m.dy, 0.0)
        w_s[:, :, 1:-1] -= dt * np.where(self.open_z, (phi[:, :, 1:] - phi[:, :, :-1]) / m.dz, 0.0)
        self.u, self.v, self.w = u_s, v_s, w_s

    def _add_inlet_momentum(self, u_s, v_s, w_s, dt):
        """
        Openings smaller than a face cell are represented by a lower face velocity
        with the same volume flow; the missing jet momentum ṁ(U_jet - U_face) is
        added to the first interior face so jets penetrate realistically.
        """
        for f in FACES:
            jet = self.bnd_jet[f]
            if not jet.any():
                continue
            name, axis, side, sign = FACE_INFO[f]
            arr = {'u': u_s, 'v': v_s, 'w': w_s}[name]
            d = (self.mesh.dx, self.mesh.dy, self.mesh.dz)[axis]
            idx_b = [slice(None)] * 3
            idx_b[axis] = side
            U_face = sign * arr[tuple(idx_b)]
            idx_i = [slice(None)] * 3
            idx_i[axis] = 1 if side == 0 else -2
            add = np.where(jet > U_face, U_face * (jet - U_face) / d, 0.0) * dt
            arr[tuple(idx_i)] += sign * add

    # =====================================================================
    # Scalar transport
    # =====================================================================
    def _advective_divergence(self, phi: np.ndarray, bnd: Dict[str, np.ndarray]):
        """Per-cell net outflow Σ V̇_f φ_f (upwind) and the net boundary inflow of φ."""
        m = self.mesh
        net_in = 0.0
        out = np.zeros(self.shape)
        for name, axis, d in (('u', 0, None), ('v', 1, None), ('w', 2, None)):
            vel = getattr(self, name)
            area = (m.A_E, m.A_N, m.A_T)[axis]
            F = vel * area
            ph = np.moveaxis(phi, axis, 0)
            Fm = np.moveaxis(F, axis, 0)
            face_val = np.empty_like(Fm)
            face_val[1:-1] = np.where(Fm[1:-1] > 0, ph[:-1], ph[1:])
            f_lo, f_hi = ('W', 'E') if axis == 0 else (('S', 'N') if axis == 1 else ('B', 'T'))
            face_val[0] = np.where(Fm[0] > 0, bnd[f_lo], ph[0])
            face_val[-1] = np.where(Fm[-1] < 0, bnd[f_hi], ph[-1])
            flux = Fm * face_val
            net_in += float(flux[0].sum() - flux[-1].sum())
            out += np.moveaxis(flux[1:] - flux[:-1], 0, axis)
        return out, net_in

    def _diffusion(self, phi: np.ndarray, gamma: np.ndarray):
        """Per-cell net diffusive inflow Σ Γ_f A_f (φ_N - φ_P)/d over open interior faces [φ·m³/s]."""
        m = self.mesh
        res = np.zeros(self.shape)
        for axis, open_mask, d, area in ((0, self.open_x, m.dx, m.A_E), (1, self.open_y, m.dy, m.A_N),
                                         (2, self.open_z, m.dz, m.A_T)):
            ph = np.moveaxis(phi, axis, 0)
            gm = np.moveaxis(gamma, axis, 0)
            om = np.moveaxis(open_mask, axis, 0)
            gf = 2 * gm[:-1] * gm[1:] / np.maximum(gm[:-1] + gm[1:], 1e-30)
            flux = np.where(om, gf * area * (ph[1:] - ph[:-1]) / d, 0.0)
            r = np.zeros_like(ph)
            r[:-1] += flux
            r[1:] -= flux
            res += np.moveaxis(r, 0, axis)
        return res

    # =====================================================================
    # Source terms
    # =====================================================================
    def _wall_heat(self, t: float):
        """Wall heat flow into boundary cells [W per cell] plus inner surface temperatures."""
        S = np.zeros(self.shape)
        Q_total = 0.0
        surf = {}
        for f in FACES:
            spec = self.sc.walls[f]
            if spec.U <= 0:
                continue
            sl = _cell_slice(f)
            T_air = self.T[sl]
            q = spec.heat_flux(t, T_air) * self.wall_active[f]
            Q = q * self.mesh.face_area(f)
            layer = S[sl]
            layer += Q
            Q_total += float(Q.sum())
            surf[f] = spec.inner_surface_temperature(t, T_air)
        return S, Q_total, surf

    def _product_exchange(self):
        """Air <-> product heat exchange across exposed product faces [W per cell] (+ into cell)."""
        S = np.zeros(self.shape)
        if not self.solid.any():
            return S
        m = self.mesh
        for axis, area in ((0, m.A_E), (1, m.A_N), (2, m.A_T)):
            T = np.moveaxis(self.T, axis, 0)
            sol = np.moveaxis(self.solid, axis, 0)
            h = np.moveaxis(self.prod_h, axis, 0)
            hf = np.maximum(h[:-1], h[1:])
            pair = sol[:-1] ^ sol[1:]
            q = np.where(pair, hf * area * (T[1:] - T[:-1]), 0.0)  # flow from cell i+1 to cell i
            r = np.zeros_like(T)
            r[:-1] += q
            r[1:] -= q
            S += np.moveaxis(r, 0, axis)
        return S

    # =====================================================================
    # Time step control
    # =====================================================================
    def stable_dt(self, op_estimate: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        m = self.mesh
        inv2 = 1 / m.dx ** 2 + 1 / m.dy ** 2 + 1 / m.dz ** 2
        umax, vmax, wmax = (float(np.abs(a).max()) for a in (self.u, self.v, self.w))
        # include the flows the openings are about to impose
        if op_estimate:
            for rec in op_estimate['records']:
                p = rec['el']['patch']
                if p.n:
                    Vf = abs(rec['mda']) / self.rho_bar / p.area
                    if rec['el']['kind'] == 'door':
                        Vf += 2 * rec['exchange']['V_ex'] / p.area
                    axis = FACE_INFO[p.face][1]
                    if axis == 0:
                        umax = max(umax, Vf)
                    elif axis == 1:
                        vmax = max(vmax, Vf)
                    else:
                        wmax = max(wmax, Vf)
        for us in self.units:
            cu = us['obj']
            for key in ('supply_patch', 'return_patch'):
                if key in us:
                    p = us[key]
                    Vf = cu.airflow_m3_s * 1.1 / max(p.area, 1e-9)
                    axis = FACE_INFO[p.face][1]
                    if axis == 0:
                        umax = max(umax, Vf)
                    elif axis == 1:
                        vmax = max(vmax, Vf)
                    else:
                        wmax = max(wmax, Vf)
        # Per-cell Courant sum from the current face velocities, compared with the
        # face speeds the openings are about to impose
        au, av, aw = np.abs(self.u), np.abs(self.v), np.abs(self.w)
        per_cell = (np.maximum(au[:-1], au[1:]) / m.dx + np.maximum(av[:, :-1], av[:, 1:]) / m.dy +
                    np.maximum(aw[:, :, :-1], aw[:, :, 1:]) / m.dz)
        imposed = max(umax / m.dx, vmax / m.dy, wmax / m.dz) * 1.5
        adv = max(float(per_cell.max()), imposed)
        gam = float((NU_AIR + self.nu_t).max()) * max(1.0, 1 / PR_T, 1 / SC_T) + max(K_AIR / (self.rho_bar * CP_DA), D_VAPOUR)
        diff = 2 * gam * inv2
        dt_adv_diff = self.cfl / max(adv + diff, 1e-12)
        dt_cfl = self.cfl / adv if adv > 0 else float('inf')
        dt_diff = 0.5 / (gam * inv2)
        # stiff wall / product exchange limits
        cap = self.rho_bar * CP_DA * self.V
        Umax = max((self.sc.walls[f].U for f in FACES), default=0.0)
        dt_wall = 0.5 * cap / max(3 * Umax * max(m.A_E, m.A_N, m.A_T), 1e-12)
        hmax = float(self.prod_h.max()) if self.solid.any() else 0.0
        dt_prod = 0.5 * cap / max(6 * hmax * max(m.A_E, m.A_N, m.A_T), 1e-12) if hmax else float('inf')
        dt = min(dt_adv_diff, dt_wall, dt_prod)
        alpha = K_AIR / (self.rho_bar * CP_DA) + float(self.nu_t.max()) / PR_T
        D = D_VAPOUR + float(self.nu_t.max()) / SC_T
        return {'dt': dt, 'dt_cfl': dt_cfl, 'dt_diffusion': dt_diff, 'dt_wall': dt_wall, 'dt_product': dt_prod,
                'u_max': max(umax, vmax, wmax), 'alpha_eff': alpha, 'D_eff': D, 'inv_d2': inv2}

    # =====================================================================
    # One time step
    # =====================================================================
    def step(self, dt: float) -> Dict[str, Any]:
        t_new = self.t + dt
        m = self.mesh
        rho_n = self.rho_bar

        # 1. Unit coolers: control and coil process on the current return air
        units_state = []
        for entry in self.units:
            cu = entry['obj']
            if 'return_patch' in entry:
                Tr, wr = self._adjacent_state(entry['return_patch'])
            else:
                wts = entry['return_weights']
                Tr, wr = float((wts * self.T).sum()), float((wts * self.omega).sum())
            cu.update_control(Tr)
            proc = cu.process(Tr, wr, self.P_room)
            units_state.append({'entry': entry, 'proc': proc, 'T_ret': Tr, 'w_ret': wr})

        # 2. Lumped room pressure & opening flows (implicit in time)
        op = self._resolve_openings(dt)
        self.P_room = op['P_new']
        M_new = op['M_new']
        self._boundary_conditions(op, units_state)

        # 3. Interior (point) sources: volumetric q_src and their scalar values
        q_src = np.zeros(self.shape)
        src_T = np.zeros(self.shape)   # Σ q φ_src for inflowing sources
        src_w = np.zeros(self.shape)
        ax = np.zeros(self.shape); ay = np.zeros(self.shape); az = np.zeros(self.shape)
        interior_in_mass = 0.0
        for src in self.interior_injections:
            mdot = src.get('mdot', 0.0)
            if mdot == 0.0:
                continue
            st = src['state']
            mda = mdot / (1 + (st['omega'] if mdot > 0 else self._room_means()[1]))
            q = src['weights'] * (mda / rho_n) / self.V
            q_src += q
            if mdot > 0:
                src_T += q * st['T']
                src_w += q * st['omega']
                U = src['obj'].jet_velocity(mdot, st['rho'])
                d = src['obj'].geometry.unit_direction()
                acc = src['weights'] * mdot * U / (rho_n * self.V)
                ax += acc * d[0]; ay += acc * d[1]; az += acc * d[2]
            else:
                src_T += q * self.T
                src_w += q * self.omega
            interior_in_mass += mda
        for us in units_state:
            entry, proc = us['entry'], us['proc']
            if not proc['running']:
                continue
            Vd = proc['m_da'] / rho_n
            if 'supply_weights' in entry:
                q = entry['supply_weights'] * Vd / self.V
                q_src += q; src_T += q * proc['T']; src_w += q * proc['omega']
                d = entry['obj'].supply.unit_direction()
                area = entry['obj'].supply.effective_area
                U = Vd / max(area, 1e-9)
                acc = entry['supply_weights'] * proc['m_da'] * U / (rho_n * self.V)
                ax += acc * d[0]; ay += acc * d[1]; az += acc * d[2]
            if 'return_weights' in entry:
                q = entry['return_weights'] * Vd / self.V
                q_src -= q; src_T -= q * self.T; src_w -= q * self.omega
        for ex in self.exchange_sources:
            sl = _cell_slice(ex['patch'].face)
            layer_q = np.zeros(self.shape)
            lq = layer_q[sl]
            lq[ex['patch'].mask] = ex['V'] / ex['patch'].n / self.V
            src_T += layer_q * (ex['T'] - self.T)
            src_w += layer_q * (ex['w'] - self.omega)

        # Uniform dilatation that makes the in-room flow consistent with dρ̄/dt
        net_in_bnd = self._net_boundary_inflow()
        # ∇·u = q_src + q_dil in every fluid cell, and Σ V ∇·u = -(net boundary inflow)
        q_dil = (-net_in_bnd - q_src[self.fluid].sum() * self.V) / self.V_fluid
        q_dil_field = np.where(self.fluid, q_dil, 0.0)

        # 4. Momentum predictor + projection
        self._momentum_and_projection(dt, q_src, q_dil_field, (ax, ay, az))

        # 5. Scalar transport (conservative, consistent with the discrete continuity)
        rho_np1 = M_new / self.V_fluid
        S_T = np.zeros(self.shape)
        S_w = np.zeros(self.shape)
        Sw_heat, Q_walls, T_surface = self._wall_heat(t_new)
        S_T += Sw_heat
        Q_internal, m_internal = 0.0, 0.0
        Tm, wm = self._room_means()
        for hs, wts in zip(self.sc.heat_sources, self.hs_weights):
            Q = hs.heat_rate(t_new, Tm)
            mv = hs.moisture_rate(t_new)
            S_T += Q * wts
            S_w += mv * wts
            Q_internal += Q
            m_internal += mv
        S_prod = self._product_exchange()
        S_T += np.where(self.fluid, S_prod, 0.0)

        adv_T, in_T = self._advective_divergence(self.T, self.bnd_T)
        adv_w, in_w = self._advective_divergence(self.omega, self.bnd_w)
        gam_T = K_AIR / (rho_n * CP_DA) + self.nu_t / PR_T
        gam_w = D_VAPOUR + self.nu_t / SC_T
        gam_T = np.where(self.fluid, gam_T, 0.0)
        gam_w = np.where(self.fluid, gam_w, 0.0)
        dif_T = self._diffusion(self.T, gam_T)
        dif_w = self._diffusion(self.omega, gam_w)

        T_new = (rho_n * self.T * self.V + dt * rho_n * (-adv_T + dif_T + src_T * self.V)
                 + dt * S_T / CP_DA) / (rho_np1 * self.V)
        w_new = (rho_n * self.omega * self.V + dt * rho_n * (-adv_w + dif_w + src_w * self.V)
                 + dt * S_w) / (rho_np1 * self.V)

        # 6. Products: lumped thermal mass per cell, respiration, transpiration
        if self.solid.any():
            resp = self.prod_mass * self.prod_qref * self.prod_Q10 ** (self.T / 10.0)
            transp = self.prod_mass * self.prod_transp
            dE = dt * (S_prod + Sw_heat + resp - transp * L_VAP0)
            T_new[self.solid] = (self.T + dE / np.maximum(self.prod_mass * self.prod_cp, 1e-9))[self.solid]
            w_new[self.solid] = 0.0
            Q_resp = float(resp[self.solid].sum())
            self._distribute_transpiration(transp, w_new, T_new, rho_np1, dt)
            m_transp = float(transp[self.solid].sum())
        else:
            Q_resp, m_transp = 0.0, 0.0

        self.T, self.omega = T_new, np.maximum(w_new, 0.0)
        self.rho_bar = rho_np1
        self.M_da = M_new

        # 7. Phase change: bulk condensation/deposition, evaporation/sublimation, wall frost
        m_bulk = self._bulk_phase_change(dt)
        m_wall = self._wall_condensation(dt, t_new, T_surface) if self.surface_condensation else 0.0

        # 8. Ledgers
        water_in = rho_n * (in_w + src_w[self.fluid].sum() * self.V) + m_internal + m_transp
        energy_in = (rho_n * CP_DA * (in_T + src_T[self.fluid].sum() * self.V) + rho_n * L_VAP0 *
                     (in_w + src_w[self.fluid].sum() * self.V) + Q_walls + Q_internal + Q_resp
                     + L_VAP0 * m_internal)
        self.cum['dry_air_in'] += dt * op['total_mda']
        self.cum['water_in'] += dt * water_in
        self.cum['energy_in'] += dt * energy_in
        self.cum['Q_walls'] += dt * Q_walls
        self.cum['Q_internal'] += dt * Q_internal
        self.cum['Q_respiration'] += dt * Q_resp
        self.cum['condensed_bulk'] += m_bulk
        self.cum['condensed_wall'] += m_wall
        self._record_flows(op, units_state, dt)

        self.t = t_new
        self.step_count += 1
        self.dt_last = dt
        self.last.update({'Q_walls': Q_walls, 'Q_internal': Q_internal, 'Q_respiration': Q_resp,
                          'cond_bulk_rate': m_bulk / dt, 'cond_wall_rate': m_wall / dt, 'q_dil': q_dil})
        return self.last

    def _net_boundary_inflow(self) -> float:
        m = self.mesh
        return float((self.u[0].sum() - self.u[-1].sum()) * m.A_E + (self.v[:, 0].sum() - self.v[:, -1].sum()) * m.A_N
                     + (self.w[:, :, 0].sum() - self.w[:, :, -1].sum()) * m.A_T)

    def _distribute_transpiration(self, transp, w_new, T_new, rho, dt):
        """Moisture lost by products enters the neighbouring air cells."""
        total = np.zeros(self.shape)
        count = np.zeros(self.shape)
        for axis in range(3):
            sol = np.moveaxis(self.solid, axis, 0)
            tr = np.moveaxis(transp, axis, 0)
            n = np.zeros(sol.shape)
            n[:-1] += sol[:-1] & ~sol[1:]
            n[1:] += sol[1:] & ~sol[:-1]
            count += np.moveaxis(n, 0, axis)
        if not transp.any():
            return
        for axis in range(3):
            sol = np.moveaxis(self.solid, axis, 0)
            tr = np.moveaxis(transp / np.maximum(np.where(self.solid, count, 1), 1), axis, 0)
            r = np.zeros(sol.shape)
            r[1:] += np.where(sol[:-1] & ~sol[1:], tr[:-1], 0.0)
            r[:-1] += np.where(sol[1:] & ~sol[:-1], tr[1:], 0.0)
            total += np.moveaxis(r, 0, axis)
        w_new += dt * total / (rho * self.V)

    def _bulk_phase_change(self, dt: float) -> float:
        """Relaxation of supersaturation (condensation/deposition) and evaporation of deposits."""
        P = self.P_room
        ws = saturation_humidity_ratio(self.T, P)
        m_air = self.rho_bar * self.V
        excess = np.where(self.fluid, self.omega - ws, 0.0)
        frac_c = min(1.0, self.k_cond * dt)
        d_cond = np.where(excess > 0, excess * frac_c, 0.0)
        # Evaporation / sublimation of deposited water where subsaturated
        frac_e = min(1.0, self.k_evap * dt)
        avail = (self.liquid + self.ice) / m_air
        d_evap = np.where((excess < 0) & (avail > 0), np.minimum(-excess * frac_e, avail), 0.0)
        mc = d_cond * m_air
        me = d_evap * m_air
        freezing = self.T < 0.0
        # deposit
        self.ice += np.where(freezing, mc, 0.0)
        self.liquid += np.where(freezing, 0.0, mc)
        # remove evaporated water: liquid first, then ice
        from_liq = np.minimum(me, self.liquid)
        from_ice = me - from_liq
        self.liquid -= from_liq
        self.ice = np.maximum(self.ice - from_ice, 0.0)
        L_c = np.where(freezing, L_SUB0, L_VAP0)
        heat = mc * L_c - from_liq * L_VAP0 - from_ice * L_SUB0      # [J]
        self.omega += d_evap - d_cond
        self.T += heat / (m_air * CP_DA)
        self.cond_rate = (mc - me) / dt
        return float(mc.sum() - me.sum())

    def _wall_condensation(self, dt: float, t: float, T_surface: Dict[str, np.ndarray]) -> float:
        """Surface condensation/frost on walls below the dew point (Lewis analogy)."""
        total = 0.0
        m_air = self.rho_bar * self.V
        for f, Tsi in T_surface.items():
            spec = self.sc.walls[f]
            sl = _cell_slice(f)
            T_air, w_air = self.T[sl], self.omega[sl]
            active = self.wall_active[f] & self.fluid[sl]
            pv = w_air * self.P_room / (0.621945 + w_air)
            rho_v_air = pv / (R_V * (T_air + 273.15))
            rho_v_s = saturation_pressure(Tsi) / (R_V * (Tsi + 273.15))
            hm = lewis_mass_transfer_coefficient(spec.h_in, self.rho_bar)
            flux = hm * (rho_v_air - rho_v_s) * self.mesh.face_area(f) * dt      # kg per face cell
            dep = np.where(active & (flux > 0), np.minimum(flux, w_air * m_air), 0.0)
            inv = self.wall_ice[f] + self.wall_liquid[f]
            sub = np.where(active & (flux < 0), np.minimum(-flux, inv), 0.0)
            icy = Tsi < 0.0
            self.wall_ice[f] += np.where(icy, dep, 0.0)
            self.wall_liquid[f] += np.where(icy, 0.0, dep)
            from_liq = np.minimum(sub, self.wall_liquid[f])
            from_ice = sub - from_liq
            self.wall_liquid[f] -= from_liq
            self.wall_ice[f] = np.maximum(self.wall_ice[f] - from_ice, 0.0)
            heat = dep * np.where(icy, L_SUB0, L_VAP0) - from_liq * L_VAP0 - from_ice * L_SUB0
            layer_w = self.omega[sl]
            layer_w += (sub - dep) / m_air
            layer_T = self.T[sl]
            layer_T += heat / (m_air * CP_DA)
            total += float(dep.sum() - sub.sum())
        return total

    def _record_flows(self, op, units_state, dt):
        """Per-source diagnostics (plan §49-50) and cumulative mass/heat."""
        rb = self.rho_bar
        sources = []
        Tm, wm = self._room_means()
        Q_inf = 0.0
        for rec in op['records']:
            obj, kind = rec['el']['obj'], rec['el']['kind']
            st = rec['state']
            mdot = rec['mdot']
            V_ex = rec.get('exchange', {}).get('V_ex', 0.0) if kind == 'door' else 0.0
            if kind == 'injection':
                e = obj.energy_rates(max(mdot, 0.0), st, rec['T_adj'], rec['w_adj']) if mdot > 0 else \
                    {'Q_sensible': 0.0, 'Q_latent': 0.0, 'Q_total': 0.0}
                self.cum['injected_mass'] += dt * max(mdot, 0.0)
            else:
                m_in = max(mdot, 0.0) + V_ex * st['rho']
                e = obj.energy_rates(m_in, st, rec['T_adj'], rec['w_adj'])
                Q_inf += e['Q_total']
                key = 'door' if kind == 'door' else 'leak'
                self.cum[key + '_in_mass'] += dt * m_in
                self.cum[key + '_out_mass'] += dt * (max(-mdot, 0.0) + V_ex * rb)
            sources.append({'name': obj.name, 'kind': kind, 'mdot': mdot, 'V': mdot / st['rho'] if mdot > 0 else mdot / rb,
                            'V_exchange': V_ex, 'dP': rec.get('dP', 0.0),
                            'direction': _direction(mdot, V_ex),
                            'T': st['T'], 'omega': st['omega'], 'active': bool(abs(mdot) > 1e-12 or V_ex > 0), **e})
        for src in self.interior_injections:
            obj, st, mdot = src['obj'], src['state'], src.get('mdot', 0.0)
            e = obj.energy_rates(mdot, st, Tm, wm) if mdot > 0 else {'Q_sensible': 0.0, 'Q_latent': 0.0, 'Q_total': 0.0}
            self.cum['injected_mass'] += dt * max(mdot, 0.0)
            sources.append({'name': obj.name, 'kind': 'injection', 'mdot': mdot, 'V': mdot / st['rho'], 'V_exchange': 0.0,
                            'dP': 0.0, 'direction': 'into room' if mdot > 0 else 'none', 'T': st['T'],
                            'omega': st['omega'], 'active': mdot > 0, **e})
        units = []
        Q_coil = 0.0
        for us in units_state:
            p = us['proc']
            Q_coil += p['Q_coil']
            self.cum['coil_water'] += dt * p['water_removal']
            units.append({'name': us['entry']['obj'].name, 'coil_on': p['coil_on'], 'running': p['running'],
                          'T_return': us['T_ret'], 'T_supply': p['T'], 'Q_coil': p['Q_coil'],
                          'Q_sensible': p['Q_sensible'], 'Q_latent': p['Q_latent'], 'water_removal': p['water_removal'],
                          'V': p['V']})
        self.cum['Q_coil'] += dt * Q_coil
        self.cum['Q_infiltration'] += dt * Q_inf
        self.last.update({'sources': sources, 'units': units, 'Q_coil': Q_coil, 'Q_infiltration': Q_inf})

    # =====================================================================
    # Diagnostics
    # =====================================================================
    def conservation(self) -> Dict[str, float]:
        W = self._total_water()
        E = self._total_energy()
        M_expected = self.M_da0 + self.cum['dry_air_in']
        W_expected = self.W0 + self.cum['water_in']
        E_expected = self.E0 + self.cum['energy_in']
        ref_W = max(abs(self.W0), 1e-9)
        ref_E = max(abs(self.E0), self.rho_bar * self.V_fluid * CP_DA * 1.0)
        return {
            'dry_air_mass': self.M_da, 'dry_air_error_pct': 100 * (self.M_da - M_expected) / M_expected,
            'air_volume_check_pct': 100 * (self.rho_bar * self.V_fluid - self.M_da) / self.M_da,
            'water_total': W, 'water_error_kg': W - W_expected, 'water_error_pct': 100 * (W - W_expected) / ref_W,
            'energy_total': E, 'energy_error_J': E - E_expected, 'energy_error_pct': 100 * (E - E_expected) / ref_E,
        }

    def divergence_residual(self) -> float:
        m = self.mesh
        div = ((self.u[1:] - self.u[:-1]) / m.dx + (self.v[:, 1:] - self.v[:, :-1]) / m.dy +
               (self.w[:, :, 1:] - self.w[:, :, :-1]) / m.dz)
        target = getattr(self, '_div_target', np.zeros(self.shape))
        return float(np.abs(div - target)[self.fluid].max()) if self.fluid.any() else 0.0

    def next_event_time(self) -> Optional[float]:
        times = []
        for obj in list(self.sc.injections) + list(self.sc.leakages) + list(self.sc.doors) + list(self.sc.heat_sources):
            te = obj.schedule.next_event_after(self.t)
            if te is not None:
                times.append(te)
        return min(times) if times else None

    # =====================================================================
    # Driver
    # =====================================================================
    def run(self, recorder=None, progress: Optional[Callable[[float, str], None]] = None,
            cancel: Optional[Callable[[], bool]] = None) -> None:
        num = self.num
        t_end = float(num.get('t_end', 600.0))
        dt_mode = num.get('dt_mode', 'auto')
        dt_user = float(num.get('dt', 1.0))
        dt_max = float(num.get('dt_max', 5.0))
        out_int = float(num.get('output_interval', max(t_end / 50, 1.0)))
        next_out = out_int
        if recorder:
            recorder.record_snapshot(self)
            recorder.record_series(self)
        wall0 = _time.time()
        series_int = max(out_int / 5.0, 1e-9)
        next_series = series_int
        while self.t < t_end - 1e-9:
            if cancel and cancel():
                self.aborted = 'cancelled'
                break
            op_est = self._estimate_openings()
            lim = self.stable_dt(op_est)
            if dt_mode == 'fixed':
                dt = dt_user
                if dt > lim['dt'] * 1.0001:
                    self.aborted = (f"Fixed time step Δt = {dt:g} s violates the stability limit "
                                    f"Δt_max = {lim['dt']:.4g} s (CFL + diffusion) at t = {self.t:.1f} s.")
                    break
            else:
                dt = min(lim['dt'], dt_max)
            self.stability = lim
            # land exactly on outputs and scheduled events
            for te in (next_out, t_end, self.next_event_time()):
                if te is not None and self.t < te < self.t + dt:
                    dt = te - self.t
            dt = max(dt, 1e-6)
            self.step(dt)
            if not np.all(np.isfinite(self.T[self.fluid])):
                self.aborted = f'Numerical divergence detected at t = {self.t:.2f} s.'
                break
            if recorder and self.t >= next_series - 1e-9:
                recorder.record_series(self)
                next_series += series_int
            if recorder and self.t >= next_out - 1e-9:
                recorder.record_snapshot(self)
                next_out += out_int
            if progress and self.step_count % 20 == 0:
                progress(self.t / t_end, f't = {self.t:.0f} s of {t_end:.0f} s  (Δt = {dt:.3g} s)')
        if recorder:
            if not recorder.snapshots or recorder.snapshots[-1]['t'] < self.t - 1e-9:
                recorder.record_snapshot(self)
            recorder.record_series(self)
            recorder.wall_time = _time.time() - wall0
        if progress:
            progress(1.0, self.aborted or 'Finished')

    def _estimate_openings(self):
        """Cheap estimate of opening flows at the current pressure (for Δt selection)."""
        try:
            saved = (self.P_room, self.M_da)
            est = self._resolve_openings(max(self.dt_last, 0.1))
            self.P_room, self.M_da = saved
            return est
        except Exception:
            return None
