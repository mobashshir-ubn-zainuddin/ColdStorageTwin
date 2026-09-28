"""
Tests for the full coupled numerical model (Modules 1-12, Phase 10 verification).
"""

import math
import time
import unittest

import numpy as np

from geometry.mesh import Mesh
from physics.psychrometrics import (saturation_pressure, humidity_ratio_from_spec, calculate_dew_point,
                                    calculate_wet_bulb, complete_state, validate_state, mix_streams,
                                    saturation_humidity_ratio)
from physics.injection import InjectionSource, orifice_mass_flow
from physics.leakage import LeakageOpening, DoorEvent, OutsideAir
from physics.walls import WallSpec
from physics.cooling_unit import CoolingUnit
from physics.schedules import Schedule
from simulation.scenario import Scenario, DEFAULT_SCENARIO, PLAN_EXAMPLE_SCENARIO
from simulation.results import ResultStore
from solver.coupled_solver import ColdRoomSolver


def closed_box(**num):
    """Sealed, still, adiabatic room with no sources."""
    numerics = {'t_end': 20.0, 'dt_max': 1.0, 'output_interval': 10.0, 'turbulence': {'model': 'laminar'}}
    numerics.update(num)
    return {
        'geometry': {'Lx': 4.0, 'Ly': 3.0, 'Lz': 2.0, 'Nx': 8, 'Ny': 6, 'Nz': 4},
        'initial': {'T': -18.0, 'P': 101325.0, 'moisture': {'mode': 'rh', 'value': 80.0}},
        'walls': {'default': {'mode': 'adiabatic'}, 'B': {'mode': 'adiabatic'}, 'T': {'mode': 'adiabatic'}},
        'envelope_leakage': {'ela_per_m2': 0.0},
        'injections': [], 'leakages': [], 'doors': [], 'heat_sources': [], 'products': [], 'cooling_units': [],
        'numerics': numerics,
    }


class TestPsychrometricEngine(unittest.TestCase):
    def test_ashrae_reference_values(self):
        # ASHRAE Handbook of Fundamentals Table 3 values
        self.assertAlmostEqual(saturation_pressure(20.0), 2339.3, delta=1.0)
        self.assertAlmostEqual(saturation_pressure(-20.0), 103.26, delta=0.05)
        self.assertAlmostEqual(saturation_pressure(0.0), 611.2, delta=0.5)
        w = humidity_ratio_from_spec(30.0, 101325.0, 'rh', 50.0)
        self.assertAlmostEqual(calculate_wet_bulb(30.0, 101325.0, w), 22.0, delta=0.1)
        self.assertAlmostEqual(calculate_dew_point(101325.0, w), 18.45, delta=0.05)

    def test_moisture_spec_round_trip(self):
        T, P = -18.0, 101325.0
        w = humidity_ratio_from_spec(T, P, 'rh', 85.0)
        tdp = calculate_dew_point(P, w)
        self.assertAlmostEqual(humidity_ratio_from_spec(T, P, 'dew_point', tdp), w, places=9)
        st = complete_state(T, P, w)
        self.assertAlmostEqual(float(st['RH']), 85.0, places=6)
        self.assertTrue(float(st['T_wb']) <= T and float(st['T_wb']) >= float(tdp))
        self.assertTrue(all(validate_state(T, P, w).values()))

    def test_vectorised_fields(self):
        T = np.linspace(-30, 20, 24).reshape(2, 3, 4)
        st = complete_state(T, 101325.0, 0.0005)
        self.assertEqual(st['T_dp'].shape, T.shape)
        self.assertEqual(st['T_wb'].shape, T.shape)

    def test_adiabatic_mixing(self):
        m = mix_streams(1.0, 30.0, 0.015, 1.0, -20.0, 0.0005)
        self.assertAlmostEqual(m['omega'], 0.00775, places=6)
        self.assertTrue(-20 < m['T'] < 30)


class TestMesh(unittest.TestCase):
    def test_mesh_geometry(self):
        m = Mesh(10, 8, 4, 20, 16, 8)
        self.assertTrue(m.validate())
        self.assertAlmostEqual(m.summary()['volume_error_pct'], 0.0)
        self.assertEqual(m.locate(10.0, 8.0, 4.0), (19, 15, 7))
        self.assertAlmostEqual(m.gaussian_weights((5, 4, 2)).sum(), 1.0)
        patch = m.boundary_patch('W', (0, 4, 1.25), 2.0, 2.5)
        self.assertEqual(patch.sum(), 4 * 5)
        self.assertEqual(m.boundary_patch('T', (5, 4, 4), 0.01, 0.01).sum(), 1)

    def test_trilinear_interpolation_is_exact_for_linear_fields(self):
        m = Mesh(4, 4, 4, 8, 8, 8)
        f = 2 * m.X_coords + 3 * m.Y_coords - m.Z_coords
        self.assertAlmostEqual(m.interpolate(f, 1.3, 2.1, 0.9), 2 * 1.3 + 3 * 2.1 - 0.9, places=10)


class TestSourcePhysics(unittest.TestCase):
    def test_injection_flow_modes(self):
        base = {'face': 'interior', 'position': [1, 1, 1], 'area': 0.05,
                'thermal': {'mode': 'temperature', 'value': -5.0}, 'moisture': {'mode': 'rh', 'value': 70.0}}
        vel = InjectionSource.from_dict({**base, 'flow': {'mode': 'velocity', 'value': 3.0}})
        st = vel.supply_state(0, -18, 0.0005, 101325)
        self.assertAlmostEqual(vel.mass_flow(0, 101325, 1.3, st), st['rho'] * 0.15, places=9)
        vol = InjectionSource.from_dict({**base, 'flow': {'mode': 'volumetric', 'value': 0.15}})
        self.assertAlmostEqual(vol.mass_flow(0, 101325, 1.3, st), st['rho'] * 0.15, places=9)
        prs = InjectionSource.from_dict({**base, 'flow': {'mode': 'pressure'}, 'P_supply': 101425.0, 'Cd': 0.65})
        self.assertAlmostEqual(prs.mass_flow(0, 101325, 1.3, st), 0.65 * 0.05 * math.sqrt(2 * st['rho'] * 100), places=9)
        self.assertLess(prs.mass_flow(0, 101500, 1.3, st), 0.0)  # backflow when room pressure is higher

    def test_sensible_heat_specified_stream(self):
        src = InjectionSource.from_dict({'face': 'interior', 'position': [1, 1, 1], 'flow': {'mode': 'mass', 'value': 0.5},
                                         'thermal': {'mode': 'sensible_heat', 'value': 2000.0},
                                         'moisture': {'mode': 'omega', 'value': 0.001}})
        st = src.supply_state(0, -18.0, 0.0005, 101325)
        self.assertAlmostEqual(st['T'], -18.0 + 2000.0 / (0.5 / 1.001 * 1006.0), places=6)

    def test_leakage_direction_follows_pressure(self):
        lk = LeakageOpening.from_dict({'face': 'W', 'position': [0, 1, 0.0], 'area': 0.01},
                                      ambient=OutsideAir(T=30, moisture_value=60, P=101325.0))
        out = lk.outside.state()
        self.assertGreater(lk.mass_flow(0, 101300.0, 1.3, out), 0)   # outside higher -> infiltration
        self.assertLess(lk.mass_flow(0, 101350.0, 1.3, out), 0)      # room higher -> exfiltration
        self.assertAlmostEqual(orifice_mass_flow(4.0, 1.0, 1.2, 0.6, 0.65), orifice_mass_flow(4.0, 1.0, 1.2, 0.6, 0.5))

    def test_door_stack_exchange(self):
        door = DoorEvent.from_dict({'face': 'W', 'position': [0, 2, 1], 'width': 2.0, 'height': 2.5, 't_open': 10, 't_close': 70},
                                   ambient=OutsideAir(T=30, moisture_value=60))
        out = door.outside.state()
        self.assertEqual(door.exchange_flow(5, 1.38, out)['V_ex'], 0.0)
        ex = door.exchange_flow(20, 1.38, out)
        # Gosney-Olama for a 5 m² door, −18 °C room / 30 °C outside: ≈ 2 m³/s, warm air in at the top
        self.assertTrue(1.5 < ex['V_ex'] < 2.5)
        self.assertTrue(ex['inflow_top'])
        door.curtain_effectiveness = 0.8
        self.assertAlmostEqual(door.exchange_flow(20, 1.38, out)['V_ex'], 0.2 * ex['V_ex'])

    def test_wall_u_value(self):
        w = WallSpec(layers=[(0.15, 0.022)], h_in=8, h_out=25)
        self.assertAlmostEqual(w.U, 1 / (1 / 8 + 0.15 / 0.022 + 1 / 25))
        self.assertGreater(w.heat_flux(0, -20.0), 0)
        self.assertEqual(WallSpec(mode='adiabatic').U, 0.0)

    def test_cooling_unit_capacity_limit(self):
        cu = CoolingUnit.from_dict({'airflow_m3_s': 2.0, 'T_coil_leaving': -30, 'capacity_W': 5000, 'fan_power_W': 0,
                                    'control': 'always_on', 'supply': {'face': 'E', 'position': [1, 1, 1]},
                                    'return': {'face': 'E', 'position': [1, 1, 0.5]}})
        cu.update_control(-10)
        p = cu.process(-10.0, 0.0015, 101325.0)
        self.assertAlmostEqual(p['Q_coil'], 5000.0, places=6)
        self.assertGreater(p['T'], -30.0)
        self.assertGreater(p['water_removal'], 0.0)

    def test_schedule(self):
        s = Schedule(t_start=100, t_end=1000, period=300, on_duration=60)
        self.assertFalse(s.is_active(50))
        self.assertTrue(s.is_active(130))
        self.assertFalse(s.is_active(200))
        self.assertTrue(s.is_active(410))
        self.assertAlmostEqual(s.next_event_after(130), 160)


class TestCoupledSolver(unittest.TestCase):
    def test_quiescent_isothermal_room_stays_at_rest(self):
        s = ColdRoomSolver(Scenario.from_dict(closed_box()))
        s.run()
        self.assertLess(max(np.abs(s.u).max(), np.abs(s.v).max(), np.abs(s.w).max()), 1e-10)
        self.assertAlmostEqual(float(s.T.std()), 0.0, places=10)
        self.assertAlmostEqual(s.P_room, 101325.0, delta=1e-6)

    def test_projection_is_divergence_free_with_buoyancy(self):
        cfg = closed_box(t_end=30.0)
        # A warm block next to cold air: horizontal density contrast drives a gravity current
        cfg['initial']['zones'] = [{'box': [0.0, 1.5, 0.0, 3.0, 0.0, 2.0], 'T': -8.0}]
        s = ColdRoomSolver(Scenario.from_dict(cfg))
        s.run()
        self.assertGreater(np.abs(s.w).max(), 1e-4)       # convection develops
        self.assertLess(s.divergence_residual(), 1e-10)

    def test_conservation_with_all_mechanisms(self):
        cfg = dict(DEFAULT_SCENARIO)
        cfg = {**cfg, 'doors': [dict(DEFAULT_SCENARIO['doors'][0], t_open=20.0, t_close=50.0)],
               'injections': PLAN_EXAMPLE_SCENARIO['injections'],
               'heat_sources': [{'name': 'Worker', 'kind': 'people', 'n_people': 2, 'position': [4, 4, 1]}],
               'numerics': dict(DEFAULT_SCENARIO['numerics'], t_end=80.0, output_interval=20.0)}
        cfg['leakages'] = PLAN_EXAMPLE_SCENARIO['leakages']
        s = ColdRoomSolver(Scenario.from_dict(cfg))
        rec = ResultStore(s)
        s.run(recorder=rec)
        self.assertIsNone(s.aborted)
        for r in rec.series:
            self.assertLess(abs(r['mass_error_pct']), 1e-8)
            self.assertLess(abs(r['water_error_pct']), 1e-8)
            self.assertLess(abs(r['energy_error_pct']), 1e-8)
            self.assertLess(r['divergence'], 1e-8)
        self.assertGreater(rec.series[-1]['injected_mass'], 0)
        self.assertGreater(rec.series[-1]['door_in_mass'], 0)

    def test_sealed_room_pressurises_with_injection(self):
        cfg = closed_box(t_end=10.0)
        cfg['injections'] = [{'face': 'W', 'position': [0, 1.5, 1], 'width': 0.5, 'height': 0.5,
                              'flow': {'mode': 'mass', 'value': 0.01},
                              'thermal': {'mode': 'temperature', 'value': -18.0}, 'moisture': {'mode': 'rh', 'value': 80.0}}]
        s = ColdRoomSolver(Scenario.from_dict(cfg))
        M0 = s.M_da
        s.run()
        w_in = humidity_ratio_from_spec(-18.0, 101325.0, 'rh', 80.0)
        self.assertAlmostEqual(s.M_da - M0, 0.01 / (1 + w_in) * 10.0, places=6)
        # ideal gas: ΔP ≈ ΔM R T / V
        self.assertAlmostEqual(s.P_room - 101325.0, (s.M_da - M0) / M0 * 101325.0, delta=0.5)

    def test_door_warms_room_and_infiltrates_moisture(self):
        cfg = closed_box(t_end=60.0, buoyancy=True, turbulence={'model': 'smagorinsky'})
        cfg['doors'] = [{'name': 'Door', 'face': 'W', 'position': [0, 1.5, 0.75], 'width': 1.0, 'height': 1.5,
                         't_open': 0.0, 't_close': 120.0}]
        cfg['ambient'] = {'T_out': 30.0, 'RH_out': 60.0, 'P_out': 101325.0}
        s = ColdRoomSolver(Scenario.from_dict(cfg))
        w0 = s.omega.mean()
        s.run()
        self.assertGreater(s.T[s.fluid].mean(), -18.0 + 1.0)
        self.assertGreater(s.omega.mean() * s.rho_bar * s.V_fluid + s.liquid.sum() + s.ice.sum(), w0 * s.rho_bar * s.V_fluid)
        # warm air enters through the upper half of the door, cold air leaves through the lower half
        top = s.u[0][:, 2]
        bottom = s.u[0][:, 0]
        self.assertGreater(top.max(), 0)
        self.assertLess(bottom.min(), 0)

    def test_supersaturation_condenses_and_releases_latent_heat(self):
        cfg = closed_box(t_end=5.0, k_cond=10.0)
        cfg['initial']['moisture'] = {'mode': 'rh', 'value': 150.0}
        s = ColdRoomSolver(Scenario.from_dict(cfg))
        T0 = s.T.mean()
        s.run()
        ws = saturation_humidity_ratio(s.T, s.P_room)
        self.assertLess(float((s.omega / ws).max()), 1.001)
        self.assertGreater(s.ice.sum(), 0)
        self.assertGreater(s.T.mean(), T0)                   # latent heat of deposition
        self.assertLess(abs(s.conservation()['energy_error_pct']), 1e-9)

    def test_products_and_cooling_unit(self):
        cfg = closed_box(t_end=60.0, turbulence={'model': 'smagorinsky'})
        cfg['products'] = [{'name': 'Warm pallet', 'box': [1.0, 2.0, 1.0, 2.0, 0.0, 1.0], 'T_initial': 5.0,
                            'bulk_density': 400, 'cp': 3500, 'h_surface': 8, 'respiration_ref_W_kg': 0.02}]
        cfg['cooling_units'] = [{'name': 'UC', 'airflow_m3_s': 0.5, 'setpoint': -20, 'T_coil_leaving': -28,
                                 'capacity_W': 3000, 'supply': {'face': 'E', 'position': [4, 1.5, 1.75], 'width': 1, 'height': 0.5},
                                 'return': {'face': 'E', 'position': [4, 1.5, 0.75], 'width': 1, 'height': 0.5}}]
        s = ColdRoomSolver(Scenario.from_dict(cfg))
        Tp0 = s.T[s.solid].mean()
        s.run()
        self.assertLess(s.T[s.solid].mean(), Tp0)            # product cools
        self.assertGreater(s.cum['Q_coil'], 0)
        self.assertLess(abs(s.conservation()['energy_error_pct']), 1e-9)

    def test_fixed_timestep_rejects_unstable_step(self):
        cfg = closed_box(t_end=10.0, dt_mode='fixed', dt=5000.0)
        s = ColdRoomSolver(Scenario.from_dict(cfg))
        s.run()
        self.assertIn('stability limit', s.aborted or '')


class TestVerification(unittest.TestCase):
    def test_second_order_grid_convergence(self):
        from validation.convergence import grid_convergence
        res = grid_convergence(levels=(8, 16), t_end=300.0)
        order = res['observed_order'][0]
        self.assertGreater(order['p_T'], 1.8)
        self.assertGreater(order['p_w'], 1.8)
        self.assertLess(res['runs'][-1]['error_T'], 1e-3)


class TestApi(unittest.TestCase):
    def test_full_api_round_trip(self):
        from app import app
        c = app.test_client()
        self.assertEqual(c.get('/twin').status_code, 200)
        sc = c.get('/api/twin/defaults').get_json()['scenario']
        sc['numerics'].update(t_end=30.0, output_interval=15.0)
        job = c.post('/api/twin/run', json={'scenario': sc}).get_json()
        self.assertTrue(job['success'])
        for _ in range(240):
            st = c.get(f"/api/twin/jobs/{job['job_id']}").get_json()
            if st['status'] in ('finished', 'failed'):
                break
            time.sleep(0.25)
        self.assertEqual(st['status'], 'finished')
        jid = job['job_id']
        summ = c.get(f'/api/twin/jobs/{jid}/summary').get_json()
        self.assertEqual(len(summ['times']), 3)
        f = c.get(f'/api/twin/jobs/{jid}/field?name=RH&t=1').get_json()
        self.assertEqual(len(f['values']), summ['mesh']['n_cells'])
        p = c.get(f'/api/twin/jobs/{jid}/point?x=5&y=4&z=3&t=2').get_json()
        self.assertGreater(len(p['state']['rows']), 20)
        self.assertEqual(c.get(f'/api/twin/jobs/{jid}/slice?name=T&plane=yz&pos=2').status_code, 200)
        self.assertEqual(c.post('/api/twin/run', json={'scenario': {'geometry': {'Nx': 1}}}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
