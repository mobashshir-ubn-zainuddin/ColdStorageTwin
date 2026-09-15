"""
Tests for properties and energy modules.
"""

import unittest
import numpy as np
from physics.properties import thermal_diffusivity
from physics.energy import sensible_energy_field, total_energy_field

class TestPropertiesEnergy(unittest.TestCase):
    def test_properties(self):
        T, P, omega = 20.0, 101325.0, 0.01
        alpha = thermal_diffusivity(T, P, omega)
        self.assertTrue(1e-5 < alpha < 3e-5)

    def test_energy(self):
        T = np.array([20.0])
        P = np.array([101325.0])
        omega = np.array([0.01])

        Es = sensible_energy_field(T, P, omega, T_ref=0.0)
        Et = total_energy_field(T, P, omega, T_ref=0.0)
        self.assertTrue(np.all(Et > Es))

if __name__ == '__main__':
    unittest.main()
