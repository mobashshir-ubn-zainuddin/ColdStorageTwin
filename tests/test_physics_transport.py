"""
Tests for condensation and moisture modules.
"""

import unittest
import numpy as np
from physics.condensation import calculate_condensation
from physics.moisture import update_moisture_diffusion

class TestPhysicsTransport(unittest.TestCase):
    def test_condensation_logic(self):
        T = np.full((3,3,3), -10.0)
        P = np.full((3,3,3), 101325.0)
        omega = np.full((3,3,3), 0.05)
        dt = 1.0
        V_cell = 1.0

        results = calculate_condensation(T, P, omega, dt, V_cell)

        self.assertTrue(np.all(results['omega_new'] < omega))
        self.assertTrue(np.all(results['condensed_mass'] > 0))
        self.assertTrue(np.all(results['latent_heat_release'] > 0))

    def test_moisture_diffusion(self):
        T = np.full((5,5,5), 20.0)
        P = np.full((5,5,5), 101325.0)
        omega = np.zeros((5,5,5))
        omega[2,2,2] = 0.1
        u = np.zeros((5,5,5))
        v = np.zeros((5,5,5))
        w = np.zeros((5,5,5))
        S_omega = np.zeros((5,5,5))

        omega_new = update_moisture_diffusion(T, P, omega, u, v, w, 0.1, 1.0, 1.0, 1.0, S_omega)

        self.assertLess(omega_new[2,2,2], 0.1)
        self.assertGreater(omega_new[1,2,2], 0)

if __name__ == '__main__':
    unittest.main()
