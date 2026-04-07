"""
3D Finite Difference Method Solver for Heat and Moisture Transfer in Cold Storage

Solves coupled PDEs:
  - Energy: ρCp ∂T/∂t = k∇²T + Lv ∂w/∂t
  - Moisture: ∂w/∂t = Dm∇²w

Where:
  - T = temperature
  - w = moisture content (fraction)
  - Lv = latent heat of vaporization
  - Dm = moisture diffusivity
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class ColdStorageConfig:
    """Configuration for cold storage simulation with moisture"""
    # Domain dimensions
    Lx: float = 10.0  # Domain length in x (m)
    Ly: float = 10.0  # Domain length in y (m)
    Lz: float = 10.0  # Domain length in z (m) 
    nx: int = 10      # Grid points in x
    ny: int = 10      # Grid points in y
    nz: int = 10      # Grid points in z
    
    # Thermal properties
    k: float = 0.5           # Thermal conductivity (W/m·K)
    rho: float = 1000.0      # Density (kg/m³)
    Cp: float = 4200.0       # Specific heat capacity (J/kg·K)
    alpha: float = None      # Thermal diffusivity (m²/s), computed if None
    
    # Moisture properties
    Lv: float = 2.26e6       # Latent heat of vaporization (J/kg)
    Dm: float = 1e-6         # Moisture diffusivity (m²/s)
    
    # Initial/boundary conditions for temperature
    T_initial: float = -20.0  # Initial temperature (°C)
    T_wall: float = -20.0     # Wall boundary condition (°C)
    T_inlet: float = -25.0    # Inlet temperature (°C)
    
    # Initial/boundary conditions for moisture
    W_initial: float = 0.01   # Initial moisture fraction
    W_wall: float = 0.005     # Wall moisture boundary condition
    W_inlet: float = 0.008    # Inlet moisture fraction
    RH_ambient: float = 0.8   # Ambient relative humidity (0-1)
    
    # Time stepping
    time_steps: int = 10      # Number of simulation steps
    dt: float = 100.0         # Time step (s)
    
    def __post_init__(self):
        """Compute derived properties"""
        if self.alpha is None:
            self.alpha = self.k / (self.rho * self.Cp)


class FiniteDifference3DSolver:
    """
    Explicit Finite Difference Method (EFDM) solver for coupled 3D heat and moisture equations.
    Uses Forward Euler in time and Central Difference in space.
    
    Solves:
      - Temperature: T^{n+1} = T^n + rT*(Laplacian of T) + (Lv/ρCp)*(w^{n+1} - w^n)
      - Moisture: w^{n+1} = w^n + rW*(Laplacian of w)
    """
    
    # Magnus formula constants for dew point calculation 
    MAGNUS_A = 17.27
    MAGNUS_B = 237.7  # °C
    
    def __init__(self, config: ColdStorageConfig):
        self.config = config
        self.dx = config.Lx / (config.nx - 1)
        self.dy = config.Ly / (config.ny - 1)
        self.dz = config.Lz / (config.nz - 1)
        self.dt = config.dt
        self.alpha = config.alpha
        
        # Stability parameters
        self.rT = self.alpha * self.dt / (self.dx ** 2)  # Temperature stability number
        self.rW = config.Dm * self.dt / (self.dx ** 2)   # Moisture stability number
        
        # Initialize temperature field
        self.T = np.full((config.nx, config.ny, config.nz), 
                         config.T_initial, dtype=np.float64)
        
        # Initialize moisture field
        self.W = np.full((config.nx, config.ny, config.nz),
                         config.W_initial, dtype=np.float64)
        
        # Set inlet boundary conditions (bottom face, z=0)
        self.T[:, :, 0] = config.T_inlet
        self.W[:, :, 0] = config.W_inlet
        
        # Store history for visualization
        self.history = [self.T.copy()]
        self.moisture_history = [self.W.copy()]
        self.time_history = [0.0]
        
        # Condensation tracking
        self.condensation_rate = np.zeros_like(self.T)
        
    def is_stable(self) -> bool:
        """
        Check stability condition for explicit scheme.
        For 3D: both rT ≤ 1/6 and rW ≤ 1/6 must be satisfied
        """
        stability_threshold = 1.0 / 6.0
        return self.rT <= stability_threshold and self.rW <= stability_threshold 
    
    def get_stability_info(self) -> Dict[str, float]:
        """Return stability information for both temperature and moisture"""
        threshold = 1.0 / 6.0
        return {
            'rT': self.rT,
            'rW': self.rW,
            'threshold': threshold,
            'is_stable': self.is_stable(),
            'margin_T': threshold - self.rT,
            'margin_W': threshold - self.rW,
        }
    
    def calculate_dew_point(self, T: np.ndarray, RH: float) -> np.ndarray:
        """
        Calculate dew point temperature using Magnus formula.
        
        T_dew = b * γ(T, RH) / (a - γ(T, RH))
        where γ(T, RH) = (a * T) / (b + T) + ln(RH)
        
        Args:
            T: Temperature array (°C)
            RH: Relative humidity (0-1)
            
        Returns:
            Dew point temperature array (°C)
        """
        a = self.MAGNUS_A
        b = self.MAGNUS_B
        
        # Clamp RH to avoid log(0)
        RH_safe = np.clip(RH, 1e-10, 1.0)
        
        gamma = (a * T) / (b + T) + np.log(RH_safe)
        T_dew = b * gamma / (a - gamma)
        
        return T_dew
    
    def calculate_saturation_moisture(self, T: np.ndarray) -> np.ndarray:
        """
        Calculate saturation moisture content at given temperature.
        Uses simplified relationship for moisture fraction.
        
        Args:
            T: Temperature array (°C)
            
        Returns:
            Saturation moisture fraction
        """
        # Simplified saturation vapor pressure relationship
        # At cold temperatures, saturation moisture is lower
        T_ref = 20.0  # Reference temperature
        W_sat_ref = 0.02  # Reference saturation moisture at T_ref
        
        # Exponential relationship (simplified Clausius-Clapeyron)
        W_sat = W_sat_ref * np.exp(0.05 * (T - T_ref))
        return np.clip(W_sat, 0.001, 0.1)
    
    def apply_condensation(self, T: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply condensation when moisture exceeds saturation.
        
        When T < T_dew (or W > W_sat):
          - Excess moisture condenses
          - Latent heat is released
          
        Args:
            T: Temperature field
            W: Moisture field
            
        Returns:
            Tuple of (updated T, updated W, condensation rate)
        """
        W_sat = self.calculate_saturation_moisture(T)
        
        # Find where condensation occurs
        condensing = W > W_sat
        
        # Calculate excess moisture (amount that condenses)
        excess_moisture = np.where(condensing, W - W_sat, 0.0)
        
        # Remove condensed moisture
        W_new = np.where(condensing, W_sat, W)
        
        # Add latent heat release to temperature
        # ΔT = (Lv / (ρ * Cp)) * Δw
        latent_factor = self.config.Lv / (self.config.rho * self.config.Cp)
        T_new = T + latent_factor * excess_moisture
        
        return T_new, W_new, excess_moisture
    
    def step(self):
        """
        Perform one time step using explicit EFDM for coupled heat-moisture system.
        
        Order of operations:
        1. Update moisture field (diffusion)
        2. Update temperature field (diffusion + latent heat coupling)
        3. Apply condensation if needed
        4. Apply boundary conditions
        """
        if not self.is_stable():
            raise ValueError(
                f"Unstable configuration: rT = {self.rT}, rW = {self.rW}. "
                f"Both must be ≤ 1/6. "
                f"Increase dx, decrease dt, or adjust diffusivities."
            )
        
        T_new = self.T.copy()
        W_new = self.W.copy()
        
        # Store old moisture for latent heat calculation
        W_old = self.W.copy()
        
        # --- STEP 1: Moisture diffusion (vectorized) ---
        W_new[1:-1, 1:-1, 1:-1] = (
            self.W[1:-1, 1:-1, 1:-1] +
            self.rW * (
                self.W[2:, 1:-1, 1:-1] + self.W[:-2, 1:-1, 1:-1] +
                self.W[1:-1, 2:, 1:-1] + self.W[1:-1, :-2, 1:-1] +
                self.W[1:-1, 1:-1, 2:] + self.W[1:-1, 1:-1, :-2] -
                6 * self.W[1:-1, 1:-1, 1:-1]
            )
        )
        
        # --- STEP 2: Temperature diffusion + latent heat (vectorized) ---
        # Latent heat term: (Lv / ρCp) * (w^{n+1} - w^n)
        latent_factor = self.config.Lv / (self.config.rho * self.config.Cp)
        
        T_new[1:-1, 1:-1, 1:-1] = (
            self.T[1:-1, 1:-1, 1:-1] +
            self.rT * (
                self.T[2:, 1:-1, 1:-1] + self.T[:-2, 1:-1, 1:-1] +
                self.T[1:-1, 2:, 1:-1] + self.T[1:-1, :-2, 1:-1] +
                self.T[1:-1, 1:-1, 2:] + self.T[1:-1, 1:-1, :-2] -
                6 * self.T[1:-1, 1:-1, 1:-1]
            ) +
            latent_factor * (W_new[1:-1, 1:-1, 1:-1] - W_old[1:-1, 1:-1, 1:-1])
        )
        
        # --- STEP 3: Apply condensation ---
        T_new, W_new, self.condensation_rate = self.apply_condensation(T_new, W_new)
        
        # --- STEP 4: Boundary conditions ---
        # Temperature boundaries (Dirichlet)
        T_new[0, :, :] = self.config.T_wall   # x=0
        T_new[-1, :, :] = self.config.T_wall  # x=Lx
        T_new[:, 0, :] = self.config.T_wall   # y=0
        T_new[:, -1, :] = self.config.T_wall  # y=Ly
        T_new[:, :, -1] = self.config.T_wall  # z=Lz
        T_new[:, :, 0] = self.config.T_inlet  # z=0 (inlet)
        
        # Moisture boundaries (Dirichlet)
        W_new[0, :, :] = self.config.W_wall   # x=0
        W_new[-1, :, :] = self.config.W_wall  # x=Lx
        W_new[:, 0, :] = self.config.W_wall   # y=0
        W_new[:, -1, :] = self.config.W_wall  # y=Ly
        W_new[:, :, -1] = self.config.W_wall  # z=Lz
        W_new[:, :, 0] = self.config.W_inlet  # z=0 (inlet)
        
        # Update fields
        self.T = T_new
        self.W = W_new
        
        # Store history
        self.history.append(self.T.copy())
        self.moisture_history.append(self.W.copy())
        self.time_history.append(self.time_history[-1] + self.dt)
    
    def solve(self):
        """Run full simulation"""
        for _ in range(self.config.time_steps):
            self.step()
    
    def get_statistics(self) -> Dict[str, float]:
        """Return temperature and moisture statistics"""
        return {
            # Temperature stats
            'min_temp': float(np.min(self.T)),
            'max_temp': float(np.max(self.T)),
            'mean_temp': float(np.mean(self.T)),
            'std_temp': float(np.std(self.T)),
            # Moisture stats
            'min_moisture': float(np.min(self.W)),
            'max_moisture': float(np.max(self.W)),
            'mean_moisture': float(np.mean(self.W)),
            'std_moisture': float(np.std(self.W)),
            # Condensation stats
            'total_condensation': float(np.sum(self.condensation_rate)),
            'max_condensation': float(np.max(self.condensation_rate)),
        }
    
    def get_midplane_slice(self, field: str = 'temperature') -> np.ndarray:
        """
        Return field values at z = Lz/2 (middle plane).
        
        Args:
            field: 'temperature', 'moisture', or 'condensation'
            
        Returns:
            2D array of the specified field at midplane
        """
        k_mid = self.config.nz // 2
        if field == 'temperature':
            return self.T[:, :, k_mid]
        elif field == 'moisture':
            return self.W[:, :, k_mid]
        elif field == 'condensation':
            return self.condensation_rate[:, :, k_mid]
        else:
            raise ValueError(f"Unknown field: {field}. Use 'temperature', 'moisture', or 'condensation'")
    
    def get_temperature_field(self) -> np.ndarray:
        """Return full temperature field"""
        return self.T.copy()
    
    def get_moisture_field(self) -> np.ndarray:
        """Return full moisture field"""
        return self.W.copy()
    
    def get_dew_point_field(self) -> np.ndarray:
        """Return dew point temperature field based on current moisture"""
        # Estimate RH from moisture content
        W_sat = self.calculate_saturation_moisture(self.T)
        RH = np.clip(self.W / W_sat, 0.0, 1.0)
        return self.calculate_dew_point(self.T, RH)
    
    def get_condensation_zones(self) -> np.ndarray:
        """Return boolean array indicating where condensation is occurring"""
        W_sat = self.calculate_saturation_moisture(self.T)
        return self.W > W_sat


def create_solver_from_params(
    Lx: float, Ly: float, Lz: float,
    nx: int, ny: int, nz: int,
    alpha: float, T_initial: float,
    T_wall: float, T_inlet: float,
    time_steps: int, dt: float,
    # Moisture parameters (optional, with defaults)
    k: float = 0.5,
    rho: float = 1000.0,
    Cp: float = 4200.0,
    Lv: float = 2.26e6,
    Dm: float = 1e-6,
    W_initial: float = 0.01,
    W_wall: float = 0.005,
    W_inlet: float = 0.008,
    RH_ambient: float = 0.8,
) -> FiniteDifference3DSolver:
    """
    Factory function to create solver from parameters.
    
    Args:
        Lx, Ly, Lz: Domain dimensions (m)
        nx, ny, nz: Grid points
        alpha: Thermal diffusivity (m²/s) - if None, computed from k, rho, Cp
        T_initial: Initial temperature (°C)
        T_wall: Wall boundary temperature (°C)
        T_inlet: Inlet temperature (°C)
        time_steps: Number of simulation steps
        dt: Time step (s)
        k: Thermal conductivity (W/m·K)
        rho: Density (kg/m³)
        Cp: Specific heat capacity (J/kg·K)
        Lv: Latent heat of vaporization (J/kg)
        Dm: Moisture diffusivity (m²/s)
        W_initial: Initial moisture fraction
        W_wall: Wall moisture boundary condition
        W_inlet: Inlet moisture fraction
        RH_ambient: Ambient relative humidity (0-1)
        
    Returns:
        Configured FiniteDifference3DSolver instance
    """
    config = ColdStorageConfig(
        Lx=Lx, Ly=Ly, Lz=Lz,
        nx=nx, ny=ny, nz=nz,
        k=k, rho=rho, Cp=Cp,
        alpha=alpha,
        Lv=Lv, Dm=Dm,
        T_initial=T_initial,
        T_wall=T_wall, T_inlet=T_inlet,
        W_initial=W_initial,
        W_wall=W_wall, W_inlet=W_inlet,
        RH_ambient=RH_ambient,
        time_steps=time_steps, dt=dt
    )
    return FiniteDifference3DSolver(config)


def run_example_simulation():
    """
    Example simulation demonstrating coupled heat-moisture transfer.
    Similar to the user's original script but using the class-based solver.
    """
    import matplotlib.pyplot as plt
    
    # Physical properties (standard air)
    k = 0.024         # W/m·K
    rho = 1.2         # kg/m³
    Cp = 1005         # J/kg·K
    Lv = 2.26e6       # J/kg
    Dm = 2.2e-5       # m²/s
    
    # Domain
    L = 1.0           # m
    dx = 0.1          # m
    dt = 0.2          # s
    time_total = 20   # s
    
    nx = int(L/dx)
    nt = int(time_total/dt)
    
    # Create configuration
    config = ColdStorageConfig(
        Lx=L, Ly=L, Lz=L,
        nx=nx, ny=nx, nz=nx,
        k=k, rho=rho, Cp=Cp,
        Lv=Lv, Dm=Dm,
        T_initial=5.0,      # Start at 5°C
        T_wall=35.0,        # Outside temperature
        T_inlet=5.0,
        W_initial=0.01,
        W_wall=0.01,
        W_inlet=0.01,
        time_steps=nt,
        dt=dt
    )
    
    # Create solver and check stability
    solver = FiniteDifference3DSolver(config)
    stability = solver.get_stability_info()
    print(f"rT = {stability['rT']:.6f}, rW = {stability['rW']:.6f}")
    print(f"Stable: {stability['is_stable']}")
    
    if not stability['is_stable']:
        print("Unstable configuration!")
        return
    
    # Run simulation
    solver.solve()
    
    # Get statistics
    stats = solver.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Temperature: min={stats['min_temp']:.2f}°C, max={stats['max_temp']:.2f}°C, mean={stats['mean_temp']:.2f}°C")
    print(f"  Moisture: min={stats['min_moisture']:.4f}, max={stats['max_moisture']:.4f}, mean={stats['mean_moisture']:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Temperature midplane
    T_mid = solver.get_midplane_slice('temperature')
    im1 = axes[0].imshow(T_mid, cmap='hot', origin='lower')
    axes[0].set_title('Temperature at Mid-Plane (°C)')
    plt.colorbar(im1, ax=axes[0], label='Temperature')
    
    # Moisture midplane
    W_mid = solver.get_midplane_slice('moisture')
    im2 = axes[1].imshow(W_mid, cmap='Blues', origin='lower')
    axes[1].set_title('Moisture at Mid-Plane')
    plt.colorbar(im2, ax=axes[1], label='Moisture Fraction')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_example_simulation()
