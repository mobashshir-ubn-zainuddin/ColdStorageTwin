"""
Finite Volume Method (FVM) Transport Framework.
Implements a generic structured-Cartesian scalar transport equation:
∂(rho*phi)/∂t + div(rho*u*phi) = div(Gamma*grad(phi)) + S_phi
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union
from geometry.mesh import Mesh

class FVMTransport:
    """
    Core FVM transport logic for structured Cartesian grids.
    Prioritizes strict conservation by computing unique face fluxes.
    """
    def __init__(self, mesh: Mesh, bc_handler=None):
        self.mesh = mesh
        self.bc_handler = bc_handler

    def _compute_face_fluxes(self,
                             phi: np.ndarray,
                             rho: np.ndarray,
                             u: np.ndarray,
                             v: np.ndarray,
                             w: np.ndarray,
                             Gamma: np.ndarray,
                             variable_name: str) -> Dict[str, np.ndarray]:
        """
        Compute convective and diffusive fluxes across all six faces.
        Internal faces are computed once and shared between neighbors.

        Returns:
            A dictionary of flux arrays, each of shape (nx, ny, nz).
            flux_E[i,j,k] is the flux leaving cell (i,j,k) through its East face.
        """
        nx, ny, nz = phi.shape
        # We define fluxes as LEAVING the cell (i,j,k)
        flux_E = np.zeros_like(phi)
        flux_W = np.zeros_like(phi)
        flux_N = np.zeros_like(phi)
        flux_S = np.zeros_like(phi)
        flux_T = np.zeros_like(phi)
        flux_B = np.zeros_like(phi)

        # --- X-DIRECTION (East-West) ---
        # Face E of cell (i) is face W of cell (i+1)
        # For internal faces i = 0 ... nx-2
        # Face mass flux: F = rho_f * u_n_f * Area
        # We use arithmetic mean for rho_f and u_n_f at the face.

        # Internal East Faces
        rho_f_E = 0.5 * (rho[:-1, :, :] + rho[1:, :, :])
        u_f_E = 0.5 * (u[:-1, :, :] + u[1:, :, :])
        F_E = rho_f_E * u_f_E * self.mesh.A_E

        # Upwind phi: phi_f = phi_P if F_E > 0 else phi_N
        phi_f_E = np.where(F_E >= 0, phi[:-1, :, :], phi[1:, :, :])

        # Diffusive flux: -Gamma_f * (phi_N - phi_P) / dx * Area
        Gamma_f_E = 0.5 * (Gamma[:-1, :, :] + Gamma[1:, :, :])
        diff_E = -Gamma_f_E * (phi[1:, :, :] - phi[:-1, :, :]) / self.mesh.dx * self.mesh.A_E

        # Assign to cells
        flux_E[:-1, :, :] = F_E * phi_f_E + diff_E
        flux_W[1:, :, :] = - (F_E * phi_f_E + diff_E) # Conservation: Flux(P->N) = -Flux(N->P)

        # --- Y-DIRECTION (North-South) ---
        rho_f_N = 0.5 * (rho[:, :-1, :] + rho[:, 1:, :])
        v_f_N = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        F_N = rho_f_N * v_f_N * self.mesh.A_N

        phi_f_N = np.where(F_N >= 0, phi[:, :-1, :], phi[:, 1:, :])

        Gamma_f_N = 0.5 * (Gamma[:, :-1, :] + Gamma[:, 1:, :])
        diff_N = -Gamma_f_N * (phi[:, 1:, :] - phi[:, :-1, :]) / self.mesh.dy * self.mesh.A_N

        flux_N[:, :-1, :] = F_N * phi_f_N + diff_N
        flux_S[:, 1:, :] = - (F_N * phi_f_N + diff_N)

        # --- Z-DIRECTION (Top-Bottom) ---
        rho_f_T = 0.5 * (rho[:, :, :-1] + rho[:, :, 1:])
        w_f_T = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        F_T = rho_f_T * w_f_T * self.mesh.A_T

        phi_f_T = np.where(F_T >= 0, phi[:, :, :-1], phi[:, :, 1:])

        Gamma_f_T = 0.5 * (Gamma[:, :, :-1] + Gamma[:, :, 1:])
        diff_T = -Gamma_f_T * (phi[:, :, 1:] - phi[:, :, :-1]) / self.mesh.dz * self.mesh.A_T

        flux_T[:, :, :-1] = F_T * phi_f_T + diff_T
        flux_B[:, :, 1:] = - (F_T * phi_f_T + diff_T)

        # --- BOUNDARY TREATMENT ---
        # Now handle the outer faces (where i=nx-1 for East, i=0 for West, etc.)
        if self.bc_handler:
            # East boundary (i = nx-1)
            # rho_f and u_f at boundary are derived from boundary state or extrapolated
            # For simplicity, we use cell-center value for rho and u at the boundary face.
            rho_b_E = rho[-1, :, :]
            u_b_E = u[-1, :, :]
            F_b_E = rho_b_E * u_b_E * self.mesh.A_E
            flux_E[-1, :, :] = self.bc_handler.apply_flux(
                variable_name, 'E', phi[-1, :, :], None,
                Gamma[-1, :, :], self.mesh.dx, rho_b_E, u_b_E, self.mesh.A_E
            )

            # West boundary (i = 0)
            rho_b_W = rho[0, :, :]
            u_b_W = -u[0, :, :] # normal is -x
            F_b_W = rho_b_W * u_b_W * self.mesh.A_W
            flux_W[0, :, :] = self.bc_handler.apply_flux(
                variable_name, 'W', phi[0, :, :], None,
                Gamma[0, :, :], self.mesh.dx, rho_b_W, u_b_W, self.mesh.A_W
            )

            # North boundary (j = ny-1)
            rho_b_N = rho[:, -1, :]
            v_b_N = v[:, -1, :]
            F_b_N = rho_b_N * v_b_N * self.mesh.A_N
            flux_N[:, -1, :] = self.bc_handler.apply_flux(
                variable_name, 'N', phi[:, -1, :], None,
                Gamma[:, -1, :], self.mesh.dy, rho_b_N, v_b_N, self.mesh.A_N
            )

            # South boundary (j = 0)
            rho_b_S = rho[:, 0, :]
            v_b_S = -v[:, 0, :]
            F_b_S = rho_b_S * v_b_S * self.mesh.A_S
            flux_S[:, 0, :] = self.bc_handler.apply_flux(
                variable_name, 'S', phi[:, 0, :], None,
                Gamma[:, 0, :], self.mesh.dy, rho_b_S, v_b_S, self.mesh.A_S
            )

            # Top boundary (k = nz-1)
            rho_b_T = rho[:, :, -1]
            w_b_T = w[:, :, -1]
            F_b_T = rho_b_T * w_b_T * self.mesh.A_T
            flux_T[:, :, -1] = self.bc_handler.apply_flux(
                variable_name, 'T', phi[:, :, -1], None,
                Gamma[:, :, -1], self.mesh.dz, rho_b_T, w_b_T, self.mesh.A_T
            )

            # Bottom boundary (k = 0)
            rho_b_B = rho[:, :, 0]
            w_b_B = -w[:, :, 0]
            F_b_B = rho_b_B * w_b_B * self.mesh.A_B
            flux_B[:, :, 0] = self.bc_handler.apply_flux(
                variable_name, 'B', phi[:, :, 0], None,
                Gamma[:, :, 0], self.mesh.dz, rho_b_B, w_b_B, self.mesh.A_B
            )

        return {
            'E': flux_E, 'W': flux_W, 'N': flux_N, 'S': flux_S, 'T': flux_T, 'B': flux_B
        }

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
        rho_P * V_P * (phi_new - phi_old) / dt = - Sum(Fluxes) + S_phi * V_P
        """
        nx, ny, nz = phi.shape
        V = self.mesh.V_cell

        # 1. Compute all fluxes
        fluxes = self._compute_face_fluxes(phi, rho, u, v, w, Gamma, variable_name)

        # Net flux leaving the cell
        net_flux = (
            fluxes['E'] + fluxes['W'] +
            fluxes['N'] + fluxes['S'] +
            fluxes['T'] + fluxes['B']
        )

        # 2. Explicit Update
        # phi_new = phi_old - (dt / (rho * V)) * (net_flux - S_phi * V)
        # To be strictly conservative:
        # rho_new * V * phi_new = rho_old * V * phi_old - dt * net_flux + dt * S_phi * V
        # We assume rho is constant over the timestep for the update part.

        phi_new = phi - (dt / (rho * V)) * (net_flux - S_phi * V)

        return phi_new, fluxes
