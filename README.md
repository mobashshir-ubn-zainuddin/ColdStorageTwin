# 3D Numerical Digital Twin for Cold Storage

A sophisticated Flask web application that simulates thermal dynamics in cold storage facilities using the Finite Difference Method. This digital twin enables engineers to predict temperature distributions, optimize cooling systems, and validate facility designs.

## Features

- **3D Thermal Simulation**: Solves the heat equation using explicit Finite Difference Method (EFDM)
- **Automatic Stability Checking**: Validates Courant-Friedrichs-Lewy (CFL) condition
- **Interactive Dashboard**: Configure simulation parameters and run simulations in real-time
- **Visualization**: Matplotlib-based heatmaps showing temperature distribution
- **Educational Content**: Detailed explanations of cold storage, digital twins, and numerical methods
- **Flexible Configuration**: Customize domain dimensions, material properties, and boundary conditions

## Project Structure

```
cold-storage-digital-twin/
├── app.py                  # Flask application
├── solver_3d.py           # Numerical solver using Finite Difference Method
├── visualizer.py          # Matplotlib visualization utilities
├── templates/
│   ├── base.html          # Base template with navigation
│   ├── index.html         # Home page
│   ├── dashboard.html     # Simulation dashboard
│   ├── about_cold_storage.html
│   ├── about_digital_twin.html
│   └── numerical_model.html
├── static/                # Generated visualizations (auto-created)
└── README.md
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip



2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python app.py
```

## Usage

### Home Page
Navigate to the home page to learn about the digital twin concept and key features.

### Dashboard
1. Configure simulation parameters:
   - Domain dimensions (Lx, Ly, Lz)
   - Grid resolution (nx, ny, nz)
   - Material properties (thermal diffusivity)
   - Temperature boundary conditions
   - Time parameters (time steps, dt)

2. Click "Run Simulation"
3. View results:
   - Temperature statistics (min, max, mean, std dev)
   - Numerical stability information
   - Temperature distribution heatmap
   - Courant number visualization

### Educational Pages
- **Cold Storage**: Information about cold storage facilities and thermal challenges
- **Digital Twin**: Explanation of digital twin technology and applications
- **Numerical Model**: Mathematical details of the finite difference method with equations

## Technical Details

### Governing Equation
The 3D heat equation is solved:

∂T/∂t = α(∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)

### Numerical Method
- **Spatial Discretization**: Central differences (2nd order accurate)
- **Temporal Discretization**: Forward Euler (1st order accurate)
- **Stability**: Explicit method requires r = α·Δt/Δx² ≤ 1/6

### Boundary Conditions
- **Walls**: Dirichlet BC at constant temperature T_wall
- **Inlet (z=0)**: Cooling air inlet at temperature T_inlet
- **Initial**: Uniform temperature T_initial


### Configuration
- **GET /api/default-params**: Get default simulation parameters

## Performance Considerations

- **Grid Resolution**: Finer grids (larger nx, ny, nz) improve accuracy but require more computation
- **Time Steps**: Larger time steps are faster but must satisfy CFL condition
- **Domain Size**: Larger domains with small grid spacing require many computations
- **Typical Runtime**: 10x10x10 grid with 10 time steps: <1 second

## Stability Warnings

The application automatically checks the CFL condition. If the configuration is unstable:
- Error message: "Courant number r = X exceeds threshold 1/6"
- To fix: Increase Δx, decrease Δt, decrease α, or reduce grid resolution

## Future Enhancements

- Implicit schemes (Crank-Nicolson) for improved stability
- Multigrid methods for faster convergence
- Machine learning integration for parameter optimization
- Real-time data synchronization with physical sensors
- Advanced 3D visualization with Plotly
- GPU acceleration for large-scale simulations
- Export results to formats suitable for further analysis

## Educational Resources

The application includes:
- Detailed mathematical derivations
- Stability analysis and CFL condition explanation
- Real-world cold storage applications
- Digital twin technology overview
- Best practices for thermal simulation

## License

Educational and research use.

## Author

Created as a demonstration of:
- Finite Difference Method for solving PDEs
- Digital twin technology
- Python scientific computing (NumPy, Matplotlib)
- Flask web application development

## Support

For issues or questions:
1. Check the Numerical Model page for mathematical details
2. Review simulation parameters for physical feasibility
3. Verify stability conditions are met
4. Ensure grid resolution is appropriate for domain size
