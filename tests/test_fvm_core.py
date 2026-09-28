"""
Validation tests for the FVM Core and Coupled Solver.
Focuses on conservation laws and consistency of the FVM framework.
"""

import unittest
import numpy as np
from geometry.mesh import Mesh
from simulation.state import SimulationState
from solver.fv_coupled_solver import FVCoupledSolver

class TestFVMCore(unittest.TestCase):
    def setUp(self):
        # Small mesh for fast validation
        self.nx, self.ny, self.nz = 4, 4, 4
        self.mesh = Mesh(4.0, 4.0, 4.0, self.nx, self.ny, self.nz)
        self.solver = FVCoupledSolver(self.mesh)

        # Initial state: Uniform conditions
        T = np.full((self.nx, self.ny, self.nz), -20.0)
        P = np.full((self.nx, self.ny, self.nz), 101325.0)
        u = np.zeros((self.nx, self.ny, self.nz))
        v = np.zeros((self.nx, self.ny, self.nz))
        w = np.zeros((self.nx, self.ny, self.nz))
        omega = np.full((self.nx, self.ny, self.nz), 0.0001)

        self.state = SimulationState(self.nx, self.ny, self.nz, T, P, u, v, w, omega)
        self.state.update_derived_fields()

    def test_mass_conservation_closed_box(self):
        """
        In a closed box with no injection/leakage and zero boundary velocities,
        the total mass should remain constant.
        """
        dt = 0.1

        # Get initial total mass
        rho_initial = self.state.get_derived('rho_ma')
        total_mass_initial = np.sum(rho_initial * self.mesh.V_cell)
        print(f"Initial total mass: {total_mass_initial}")

        # Step the solver
        new_state, diag = self.solver.step(self.state, dt)

        # Get final total mass
        new_state.update_derived_fields()
        rho_final = new_state.get_derived('rho_ma')
        total_mass_final = np.sum(rho_final * self.mesh.V_cell)
        print(f"Final total mass: {total_mass_final}")
        print(f"Diagnostics: {diag}")

        # Check conservation (within floating point precision)
        self.assertAlmostEqual(total_mass_initial, total_mass_final, places=7)
        self.assertLess(diag['mass_residual_max'], 1e-5, "Continuity residual too high")

    def test_energy_conservation_closed_box(self):
        """
        In a closed box with no sources and adiabatic boundaries,
        the total energy should remain constant.
        """
        dt = 0.1

        # Total energy ~ sum(rho * V * cp * T)
        rho = self.state.get_derived('rho_ma')
        omega = self.state.omega
        # Simplified: Use a constant cp for the conservation check if possible,
        # or compute it exactly.
        from physics.properties import cp_moist_air
        cp = cp_moist_air(self.state.T, omega)
        total_energy_initial = np.sum(rho * self.mesh.V_cell * cp * self.state.T)

        new_state, _ = self.solver.step(self.state, dt)

        new_rho = new_state.get_derived('rho_ma')
        new_cp = cp_moist_air(new_state.T, new_state.omega)
        total_energy_final = np.sum(new_rho * self.mesh.V_cell * new_cp * new_state.T)

        # Note: This is a first-order test. We expect conservation in a closed system.
        self.assertAlmostEqual(total_energy_initial, total_energy_final, places=5)

    def test_stability_limit(self):
        """
        Verify that the solver doesn't explode with a small, safe timestep.
        """
        dt = 0.01
        try:
            for _ in range(5):
                self.state, _ = self.solver.step(self.state, dt)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Solver crashed with safe timestep: {e}")

if __name__ == '__main__':
    unittest.main()
