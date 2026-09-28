# 3D Numerical Digital Twin for Cold Storage

A Flask web application that simulates the complete thermo-hygrometric state of a cold-storage room.
The quantities are temperature, pressure, airflow, humidity, energy, condensation and frost.
They are computed at every cell of a 3-D mesh and every saved time.

Deployed on https://coldstoragetwin.onrender.com/

## Two models

| Page | Model | Physics |
|---|---|---|
| `/dashboard` | Legacy explicit FDM (BTP proposal, Sections 3–5) | Coupled heat + moisture diffusion, Dirichlet walls |
| `/twin` | **Full coupled finite-volume model** (Implementation plan, Modules 1–12) | Airflow, pressure, buoyancy, heat, moisture, condensation/frost, injection, leakage, doors, walls, internal loads, products, unit coolers |

## Running the website locally

Requirements: Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment (first time only)
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / cmd
# source .venv/bin/activate     # macOS / Linux / Git Bash

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python app.py
```

Then open http://localhost:10000 (the full model is at http://localhost:10000/twin).
Set `PORT=5000` to use a different port, and `FLASK_DEBUG=1` for auto-reload while developing.

### Production

Simulation jobs run in background threads and are held in memory, so the server must be a **single process**:

```bash
pip install gunicorn
gunicorn -w 1 --threads 8 --timeout 120 -b 0.0.0.0:$PORT app:app
```

### Tests and verification

```bash
python run_tests.py                                           # whole test suite
python -c "from validation.convergence import grid_convergence as g; print(g()['observed_order'])"
```

## The full numerical model (`/twin`)

**State.** The model advances the state `Q = [T, P, u, v, w, ω]` of every cell. The psychrometric engine derives everything else from it: RH, p_v, p_da, p_ws, ω_s, q, T_dp, T_wb, h, v, ρ, ρ_da, ρ_v, and sensible/latent/total energy.

**Numerics** (`solver/coupled_solver.py`)
- Staggered (MAC) Cartesian finite volumes. The projection's discrete divergence is exactly the one used by scalar transport, so mass, water and energy close to machine precision.
- Momentum uses explicit upwind advection with Smagorinsky (or constant/laminar) eddy viscosity. Boussinesq buoyancy comes from the local moist-air density ρ(T, P, ω).
- Pressure projection uses a sparse LU-factorised Poisson operator. Product blocks act as flow obstacles.
- Room pressure is a lumped node, solved implicitly each step from the dry-air mass balance. Leakage, doors and pressure-driven injection use orifice/crack power laws with hydrostatic stack terms.
- Transport of ρ̄ω and ρ̄c_pT is conservative, with inflow values at openings. Latent heat from condensation, deposition, evaporation and wall frost is added exactly once.
- Δt adapts to the CFL + diffusion limit. Fixed-Δt runs stop with a diagnostic if they become unstable.

**Sources and boundaries** (`physics/`)
- **Injection** (`injection.py`): on a boundary face or as an interior jet. Flow is given by velocity, volumetric flow, mass flow or supply pressure. Moisture is given by RH, ω, dew point, vapour pressure or specific humidity. Heat is given by temperature or sensible heat. Each injection has a schedule.
- **Leakage and doors** (`leakage.py`): flow direction follows ΔP. Doors add the Gosney–Olama buoyancy exchange, with strip-curtain effectiveness.
- **Walls** (`walls.py`): multilayer U-value with a sol-air temperature, an optional diurnal swing, a floor on ground, and fixed or adiabatic options.
- **Internal sources and products** (`internal_sources.py`): internal heat and moisture sources (people, lights, forklifts, …). Product blocks have thermal mass, respiration (Q10) and transpiration.
- **Unit coolers** (`cooling_unit.py`): recirculating air, a thermostat with deadband, a capacity limit, dehumidification into coil frost, and fan heat.

**Verification** (`validation/`)
- **Analytic diffusion:** a cosine-mode decay test gives observed order p ≈ 2.0–2.2 for both T and ω.
- **Conservation audit:** all mechanisms active; dry-air, water and energy errors ~10⁻¹² %.
- **Unit tests:** `tests/test_full_model.py` covers the psychrometrics (against ASHRAE values), mesh, every source model, solver behaviour and the API.

**Dashboard sections**
- **A. Setup:** geometry and mesh, initial conditions, walls and outside air, numerics, and raw JSON.
- **B. Sources:** add injections, leakages, doors, heat sources, products and unit coolers.
- **C. Fields:** any of 30 fields as a 2-D slice or 3-D volume, with a time slider/animation, velocity vectors, source markers and a per-source flow/heat table.
- **D. Point analysis:** the complete psychrometric state at any point, its time history, and its psychrometric-chart trajectory.
- **E. Analytics:** temperature, RH, heat flows, room pressure, condensate/frost and conservation-error time series, plus stability numbers and CSV export.

### REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/twin/defaults` | Default scenario, presets, field catalogue |
| POST | `/api/twin/run` | Start a run with `{"scenario": {...}}` → `job_id` |
| GET | `/api/twin/jobs/<id>` | Status and progress |
| GET | `/api/twin/jobs/<id>/summary` | Mesh, times, sources, time series, indicators |
| GET | `/api/twin/jobs/<id>/field?name=RH&t=5` | Full 3-D field at snapshot `t` |
| GET | `/api/twin/jobs/<id>/slice?name=T&plane=xz&pos=4&t=5` | 2-D slice with in-plane velocities |
| GET | `/api/twin/jobs/<id>/point?x=&y=&z=&t=` | Complete state table and history at a point |
| GET | `/api/twin/jobs/<id>/export.csv` | Room-level time series |
| POST | `/api/twin/psychrometrics` | Standalone psychrometric calculator |
| POST | `/api/simulate` | Legacy FDM model |

## Project structure

```
app.py                      Flask app (pages + legacy API), registers api/routes.py
api/routes.py               REST API of the full model
geometry/mesh.py            Cartesian mesh, patches, interpolation, Gaussian sources
physics/                    psychrometrics, injection, leakage/doors, walls, internal sources, unit cooler, schedules
solver/coupled_solver.py    Full coupled transient FV solver (MAC grid)
solver/fdm_solver.py        Legacy explicit FDM solver used by /dashboard
simulation/                 scenario definition, result store & post-processing, background jobs
validation/                 analytic diffusion, grid convergence, conservation audit
templates/twin.html         Full-model dashboard;  templates/dashboard.html legacy dashboard
tests/                      unittest suite (python run_tests.py)
```

## Known limitations

- Turbulence is a zero-equation (Smagorinsky) model. Jets smaller than a mesh cell are represented with their exact volume flow and a momentum correction, not resolved.
- In-room air uses a spatially uniform dry-air density (low-Mach/Boussinesq); buoyancy uses the local density.
- Deposited condensate/frost is stationary. Existing liquid does not later freeze.
- Validation against measured cold-room data (proposal §6.2) is still to be done.

## License

Educational and research use.
