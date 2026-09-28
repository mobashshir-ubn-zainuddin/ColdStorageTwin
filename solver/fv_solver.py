"""
Finite Volume Method (FVM) Transport Framework.
Implements a generic structured-Cartesian scalar transport equation:
∂(rho*phi)/∂t + div(rho*u*phi) = div(Gamma*grad(phi)) + S_phi

Numerical Convention:
- All fluxes are defined as LEAVING the cell (i, j, k).
- Internal faces: Flux(P -> N) = -Flux(N -> P).
- S_phi is a volumetric source [Quantity / (m^3 s)].
- Convection: First-order Upwind.
- Diffusion: Second-order Central Difference.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union
from geometry.mesh import Mesh

class FVMTransport:
    """
    Core FVM transport logic for structured Cartesian grids.
    """
    def __init__(self, mesh: Mesh, bc_handler=None):
        self.mesh = mesh
        self.bc_handler = bc_handler

    def compute_face_mass_fluxes(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute mass fluxes across all six faces for the entire domain.
        Flux = rho_f * u_n_f * Area [kg/s]
        """
        nx, ny, nz = rho.shape
        fluxes = {face: np.zeros_like(rho) for face in ['E', 'W', 'N', 'S', 'T', 'B']}

        # --- X-Direction (East-West) ---
        # Face E of cell i is Face W of cell i+1
        rho_f_E = 0.5 * (rho[:-1, :, :] + rho[1:, :, :])
        u_f_E = 0.5 * (u[:-1, :, :] + u[1:, :, :])
        F_E = rho_f_E * u_f_E * self.mesh.A_E
        fluxes['E'][:-1, :, :] = F_E
        fluxes['W'][1:, :, :] = -F_E

        # --- Y-Direction (North-South) ---
        rho_f_N = 0.5 * (rho[:, :-1, :] + rho[:, 1:, :])
        v_f_N = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        F_N = rho_f_N * v_f_N * self.mesh.A_N
        fluxes['N'][:, :-1, :] = F_N
        fluxes['S'][:, 1:, :] = -F_N

        # --- Z-Direction (Top-Bottom) ---
        rho_f_T = 0.5 * (rho[:, :, :-1] + rho[:, :, 1:])
        w_f_T = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        F_T = rho_f_T * w_f_T * self.mesh.A_T
        fluxes['T'][:, :, :-1] = F_T
        fluxes['B'][:, :, 1:] = -F_T

        # Note: Boundary faces are currently zero. They are handled separately in integrate_explicit
        # or via the BC handler for scalars. For mass flux, we'll add boundary terms here.
        if self.bc_handler:
            # This is a simplification. In a full FVM, boundary mass flux is a separate
            # physical input (e.g. injection/leakage).
            pass

        return fluxes

    def integrate_explicit(self,
                           phi: np.ndarray,
                           rho: np.ndarray,
                           u: np.ndarray,
                           v: np.ndarray,
                           w: np.ndarray,
                           Gamma: np.ndarray,
                           S_phi: np.ndarray,
                           dt: float,
                           variable_name: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform one explicit FVM update step.
        Equation: rho * V * (phi_new - phi_old) / dt = - Sum(Fluxes) + S_phi * V

        Args:
            S_phi: Volumetric source [Quantity / (m^3 s)]
        """
        nx, ny, nz = phi.shape
        V = self.mesh.V_cell

        # 1. Compute face mass fluxes
        mass_fluxes = self.compute_face_mass_fluxes(rho, u, v, w)

        # 2. Compute total scalar fluxes (Convective + Diffusive)
        total_flux = np.zeros_like(phi)
        phi_fluxes = {face: np.zeros_like(phi) for face in ['E', 'W', 'N', 'S', 'T', 'B']}

        # --- X-DIRECTION ---
        # Internal East faces
        F_E = mass_fluxes['E'][:-1, :, :]
        phi_f_E = np.where(F_E >= 0, phi[:-1, :, :], phi[1:, :, :])
        diff_E = -Gamma[:-1, :, :] * (phi[1:, :, :] - phi[:-1, :, :]) / self.mesh.dx * self.mesh.A_E

        flux_E_val = F_E * phi_f_E + diff_E
        phi_fluxes['E'][:-1, :, :] = flux_E_val
        phi_fluxes['W'][1:, :, :] = -flux_E_val

        # --- Y-DIRECTION ---
        F_N = mass_fluxes['N'][:, :-1, :]
        phi_f_N = np.where(F_N >= 0, phi[:, :-1, :], phi[:, 1:, :])
        diff_N = -Gamma[:, :-1, :] * (phi[:, 1:, :], - phi[:, :-1, :]) / self.mesh.dy * self.mesh.A_N # Typo fix: minus sign
        # wait, a typo: phi[:, 1:, :] - phi[:, :-1, :]
        # Let's rewrite carefully.

        # Re-doing diffusion for all to be safe
        # X
        # (handled above)

        # Y
        diff_N = -Gamma[:, :-1, :] * (phi[:, 1:, :] - phi[:, :-1, :]) / self.mesh.dy * self.mesh.A_N
        flux_N_val = F_N * phi_f_N + diff_N
        phi_fluxes['N'][:, :-1, :] = flux_N_val
        phi_fluxes['S'][:, 1:, :] = -flux_N_val

        # Z
        F_T = mass_fluxes['T'][:, :, :-1]
        phi_f_T = np.where(F_T >= 0, phi[:, :, :-1], phi[:, :, 1:])
        diff_T = -Gamma[:, :, :-1] * (phi[:, :, 1:] - phi[:, :, :-1]) / self.mesh.dz * self.mesh.A_T
        flux_T_val = F_T * phi_f_T + diff_T
        phi_fluxes['T'][:, :, :-1] = flux_T_val
        phi_fluxes['B'][:, :, 1:] = -flux_T_val

        # --- BOUNDARY TREATMENT ---
        if self.bc_handler:
            # Boundary fluxes use the BC handler
            # East
            rho_b_E = rho[-1, :, :]
            u_b_E = u[-1, :, :]
            F_b_E = rho_b_E * u_b_E * self.mesh.A_E
            phi_fluxes['E'][-1, :, :] = self.bc_handler.apply_flux(
                variable_name, 'E', phi[-1, :, :], None, Gamma[-1, :, :], self.mesh.dx, rho_b_E, u_b_E, self.mesh.A_E
            )
            # West
            rho_b_W = rho[0, :, :]
            u_b_W = -u[0, :, :]
            F_b_W = rho_b_W * u_b_W * self.mesh.A_W
            phi_fluxes['W'][0, :, :] = self.bc_handler.apply_flux(
                variable_name, 'W', phi[0, :, :], None, Gamma[0, :, :], self.mesh.dx, rho_b_W, u_b_W, self.mesh.A_W
            )
            # North
            rho_b_N = rho[:, -1, :]
            v_b_N = v[:, -1, :]
            F_b_N = rho_b_N * v_b_N * self.mesh.A_N
            phi_fluxes['N'][:, -1, :] = self.bc_handler.apply_flux(
                variable_name, 'N', phi[:, -1, :], None, Gamma[:, -1, :], self.mesh.dy, rho_b_N, v_b_N, self.mesh.A_N
            )
            # South
            rho_b_S = rho[:, 0, :]
            v_b_S = -v[:, 0, :]
            F_b_S = rho_b_S * v_b_S * self.mesh.A_S
            phi_fluxes['S'][:, 0, :] = self.bc_handler.apply_flux(
                variable_name, 'S', phi[:, 0, :], None, Gamma[:, 0, :], self.mesh.dy, rho_b_S, v_b_S, self.mesh.A_S
            )
            # Top
            rho_b_T = rho[:, :, -1]
            w_b_T = w[:, :, -1]
            F_b_T = rho_b_T * w_b_T * self.mesh.A_T
            phi_fluxes['T'][:, :, -1] = self.bc_handler.apply_flux(
                variable_name, 'T', phi[:, :, -1], None, Gamma[:, :, -1], self.mesh.dz, rho_b_T, w_b_T, self.mesh.A_T
            )
            # Bottom
            rho_b_B = rho[:, :, 0]
            w_b_B = -w[:, :, 0]
            F_b_B = rho_b_B * w_b_B * self.mesh.A_B
            phi_fluxes['B'][:, :, 0] = self.bc_handler.apply_flux(
                variable_name, 'B', phi[:, :, 0], None, Gamma[:, :, 0], self.mesh.dz, rho_b_B, w_b_B, self.mesh.A_B
            )

        # Sum all fluxes leaving the cell
        net_flux = (
            phi_fluxes['E'] + phi_fluxes['W'] +
            phi_fluxes['N'] + phi_fluxes['S'] +
            phi_fluxes['T'] + phi_fluxes['B']
        )

        # Explicit Update:
        # rho * V * (phi_new - phi_old) / dt = -net_flux + S_phi * V
        phi_new = phi - (dt / np.maximum(rho * V, 1e-8)) * (net_flux - S_phi * V)

        return phi_new, phi_fluxes
