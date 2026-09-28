"""
REST API for the full numerical digital twin (/api/twin/...).

    GET  /api/twin/defaults                 default scenario + presets + field catalogue
    POST /api/twin/run                      start a simulation  -> {job_id}
    GET  /api/twin/jobs/<id>                status / progress
    POST /api/twin/jobs/<id>/cancel         stop a running job
    GET  /api/twin/jobs/<id>/summary        grid, times, sources, analytics time series, indicators
    GET  /api/twin/jobs/<id>/field          ?name=T&t=<snapshot index>   full 3-D field
    GET  /api/twin/jobs/<id>/slice          ?name=&t=&plane=xy|xz|yz&pos=
    GET  /api/twin/jobs/<id>/point          ?x=&y=&z=&t=   complete psychrometric state + history
    GET  /api/twin/jobs/<id>/export.csv     room-level time series as CSV
    POST /api/twin/psychrometrics           standalone psychrometric calculator
"""

import csv
import io
import math

import numpy as np
from flask import Blueprint, jsonify, request, Response

from simulation.jobs import manager
from simulation.results import FIELDS, HISTORY_KEYS
from simulation.scenario import DEFAULT_SCENARIO, PRESETS
from physics.psychrometrics import complete_state, humidity_ratio_from_spec

twin_api = Blueprint('twin_api', __name__, url_prefix='/api/twin')


def _num(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return _num(obj)


def _job_or_404(job_id, need_results=True):
    job = manager.get(job_id)
    if job is None:
        return None, (jsonify({'error': 'Unknown or expired job id'}), 404)
    if need_results and (job.store is None or not job.store.snapshots):
        return None, (jsonify({'error': 'Results not available yet', 'status': job.status}), 409)
    return job, None


def _snapshot_index(job):
    n = len(job.store.snapshots)
    idx = int(request.args.get('t', n - 1))
    return max(0, min(idx, n - 1))


@twin_api.get('/defaults')
def defaults():
    return jsonify({'scenario': DEFAULT_SCENARIO, 'presets': PRESETS,
                    'fields': FIELDS, 'history_keys': HISTORY_KEYS})


@twin_api.post('/run')
def run():
    data = request.get_json(silent=True) or {}
    try:
        job = manager.submit(data.get('scenario', data))
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    return jsonify({'success': True, 'job_id': job.id, 'warnings': job.warnings})


@twin_api.get('/jobs/<job_id>')
def job_status(job_id):
    job, err = _job_or_404(job_id, need_results=False)
    return err or jsonify(_clean(job.to_dict()))


@twin_api.post('/jobs/<job_id>/cancel')
def job_cancel(job_id):
    return jsonify({'cancelled': manager.cancel(job_id)})


@twin_api.get('/jobs/<job_id>/summary')
def summary(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    st, sv = job.store, job.solver
    m = st.mesh
    sc = sv.sc

    def geom(g):
        return {'face': g.face, 'position': g.position, 'width': g.width, 'height': g.height,
                'area': g.effective_area, 'direction': g.unit_direction()}

    sources = []
    for x in sc.injections:
        sources.append({'kind': 'injection', 'name': x.name, **geom(x.geometry),
                        'spec': {'flow': f'{x.flow_mode} = {x.flow_value:g}', 'T': x.thermal_value,
                                 'moisture': f'{x.moisture_mode} = {x.moisture_value:g}',
                                 'P_supply': x.P_supply, 'heat_W': x.heat_W}})
    for x in sc.leakages:
        sources.append({'kind': 'leakage', 'name': x.name, **geom(x.geometry),
                        'spec': {'T_out': x.outside.T, 'RH_out': x.outside.moisture_value, 'P_out': x.outside.P,
                                 'Cd': x.Cd, 'n': x.exponent}})
    for x in sc.doors:
        sources.append({'kind': 'door', 'name': x.name, **geom(x.geometry),
                        'spec': {'T_out': x.outside.T, 'RH_out': x.outside.moisture_value, 'P_out': x.outside.P,
                                 'open': [x.schedule.t_start, x.schedule.t_end],
                                 'curtain_effectiveness': x.curtain_effectiveness}})
    for x in sc.heat_sources:
        sources.append({'kind': 'heat', 'name': x.name, 'position': x.position, 'box': x.box,
                        'spec': {'kind': x.kind, 'power_W': x.power_W, 'n_people': x.n_people,
                                 'window': [x.schedule.t_start, x.schedule.t_end]}})
    for x in sc.cooling_units:
        sources.append({'kind': 'cooler', 'name': x.name, 'supply': geom(x.supply), 'return': geom(x.ret),
                        'spec': {'airflow_m3_s': x.airflow_m3_s, 'setpoint': x.setpoint,
                                 'capacity_W': x.capacity_W, 'T_coil_leaving': x.T_coil_leaving}})
    products = [{'name': p.name, 'box': p.box, 'T_initial': p.T_initial} for p in sc.products]

    return jsonify(_clean({
        'name': sc.raw.get('name'),
        'mesh': m.summary(),
        'x': m.x_centers, 'y': m.y_centers, 'z': m.z_centers,
        'fluid': st.fluid.ravel().astype(int).tolist(),
        'times': [s['t'] for s in st.snapshots],
        'series': st.series, 'indicators': st.indicators(),
        'stability': {k: v for k, v in sv.stability.items()},
        'walls': {f: {'mode': w.mode, 'U': w.U, 'T_out': w.T_out} for f, w in sc.walls.items()},
        'sources': sources, 'products': products,
        'wall_time': st.wall_time, 'steps': sv.step_count, 'warnings': job.warnings,
        'status': job.status, 'fields': FIELDS,
    }))


@twin_api.get('/jobs/<job_id>/field')
def field(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    name = request.args.get('name', 'T')
    if name not in FIELDS:
        return jsonify({'error': f'Unknown field {name}'}), 400
    idx = _snapshot_index(job)
    data = job.store.field(name, idx)
    finite = data[np.isfinite(data)]
    return jsonify({'name': name, 't': job.store.snapshots[idx]['t'], 'index': idx,
                    'shape': list(data.shape),
                    'min': float(finite.min()) if finite.size else None,
                    'max': float(finite.max()) if finite.size else None,
                    # C-order (x slowest, z fastest), NaN -> null
                    'values': [None if not math.isfinite(v) else round(float(v), 6) for v in data.ravel()],
                    **FIELDS[name]})


@twin_api.get('/jobs/<job_id>/slice')
def slice_(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    name = request.args.get('name', 'T')
    if name not in FIELDS:
        return jsonify({'error': f'Unknown field {name}'}), 400
    idx = _snapshot_index(job)
    plane = request.args.get('plane', 'xy')
    pos = float(request.args.get('pos', 1.0))
    out = job.store.slice(name, idx, plane, pos)
    out.update({'t': job.store.snapshots[idx]['t'], 'name': name, **FIELDS[name]})
    return jsonify(out)


@twin_api.get('/jobs/<job_id>/point')
def point(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    try:
        x, y, z = (float(request.args[k]) for k in ('x', 'y', 'z'))
    except (KeyError, ValueError):
        return jsonify({'error': 'x, y and z are required numbers'}), 400
    m = job.store.mesh
    x, y, z = min(max(x, 0.0), m.Lx), min(max(y, 0.0), m.Ly), min(max(z, 0.0), m.Lz)
    idx = _snapshot_index(job)
    return jsonify(_clean({'state': job.store.point_state(x, y, z, idx),
                           'history': job.store.point_history(x, y, z)}))


@twin_api.get('/jobs/<job_id>/export.csv')
def export_csv(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    series = job.store.series
    keys = [k for k in series[0].keys() if k not in ('sources', 'coil_on')]
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(keys)
    for r in series:
        wr.writerow([r[k] for k in keys])
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=coldstore_{job_id}.csv'})


@twin_api.post('/psychrometrics')
def psychrometrics():
    """Phase-1 deliverable: standalone psychrometric calculator."""
    d = request.get_json(silent=True) or {}
    try:
        T = float(d.get('T', 20.0))
        P = float(d.get('P', 101325.0))
        omega = humidity_ratio_from_spec(T, P, d.get('mode', 'rh'), float(d.get('value', 50.0)))
        st = complete_state(T, P, omega, P_ref=float(d.get('P_ref', 101325.0)))
    except (ValueError, TypeError) as exc:
        return jsonify({'error': str(exc)}), 400
    keys = ['T', 'P', 'P_gauge', 'omega', 'RH', 'pv', 'pda', 'pws', 'omega_s', 'q', 'T_dp', 'T_wb', 'h',
            'specific_volume', 'rho_da', 'rho_ma', 'rho_v', 'moisture_deficit', 'CRI']
    return jsonify({k: _num(np.asarray(st[k]).item()) for k in keys})
