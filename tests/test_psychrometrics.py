"""
Tests for the psychrometric engine.
"""

import unittest
import numpy as np
from physics.psychrometrics import calculate_psychrometrics, saturation_pressure, calculate_dew_point, calculate_wet_bulb

class TestPsychrometrics(unittest.TestCase):
    def test_saturation_pressure(self):
        # Test water (T > 0)
        T_water = 20.0
        pws_water = saturation_pressure(T_water)
        self.assertTrue(2300 < pws_water < 2400)

        # Test ice (T < 0)
        T_ice = -20.0
        pws_ice = saturation_pressure(T_ice)
        self.assertTrue(100 < pws_ice < 110)

    def test_psychrometric_forward(self):
        T = 20.0
        P = 101325.0
        omega = 0.01
        props = calculate_psychrometrics(T, P, omega)

        expected_keys = ['pws', 'pv', 'pda', 'RH', 'omega_s', 'q', 'h', 'specific_volume', 'rho_da', 'rho_ma', 'rho_v']
        for key in expected_keys:
            self.assertIn(key, props)

        self.assertGreater(props['rho_ma'], props['rho_da'])
        self.assertAlmostEqual(P, props['pda'] + props['pv'], places=2)

    def test_dew_point_inversion(self):
        P = 101325.0
        omega = 0.01
        Tdp = calculate_dew_point(P, omega)
        pv = (omega * P) / (0.621945 + omega)
        self.assertAlmostEqual(saturation_pressure(Tdp), pv, delta=0.1)

    def test_wet_bulb_limits(self):
        P = 101325.0
        T = 20.0
        pws = saturation_pressure(T)
        omega_s = (0.621945 * pws) / (P - pws)
        Twb = calculate_wet_bulb(T, P, omega_s)
        self.assertAlmostEqual(T, Twb, delta=0.5)

        omega_dry = 0.001
        Twb_dry = calculate_wet_bulb(T, P, omega_dry)
        self.assertLess(Twb_dry, T)

if __name__ == '__main__':
    unittest.main()
