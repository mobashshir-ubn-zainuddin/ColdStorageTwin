"""
Flask application for 3D Numerical Digital Twin for Cold Storage
"""

from flask import Flask, render_template, request, jsonify
import os
from solver_3d import create_solver_from_params
from visualizer import (
    plot_midplane_heatmap, 
    plot_temperature_profile,
    plot_3d_volume_scatter,
    plot_3d_isosurface,
    plot_3d_sliced_views,
    plot_3d_volumetric
)


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Ensure static directory exists
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard with simulation controls"""
    return render_template('dashboard.html')


@app.route('/about-cold-storage')
def about_cold_storage():
    """About cold storage facilities"""
    return render_template('about_cold_storage.html')


@app.route('/about-digital-twin')
def about_digital_twin():
    """About digital twin technology"""
    return render_template('about_digital_twin.html')


@app.route('/numerical-model')
def numerical_model():
    """Information about the numerical model"""
    return render_template('numerical_model.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    API endpoint to run coupled heat-moisture simulation with given parameters
    Expects JSON payload with simulation parameters
    """
    try:
        data = request.get_json()
        
        # Extract domain parameters
        Lx = float(data.get('Lx', 10.0))
        Ly = float(data.get('Ly', 10.0))
        Lz = float(data.get('Lz', 10.0))
        nx = int(data.get('nx', 10))
        ny = int(data.get('ny', 10))
        nz = int(data.get('nz', 10))
        
        # Thermal properties
        k = float(data.get('k', 0.024))
        rho = float(data.get('rho', 1.2))
        Cp = float(data.get('Cp', 1005.0))
        alpha = float(data.get('alpha', 0.024 / (1.2 * 1005.0)))
        
        # Moisture properties
        Lv = float(data.get('Lv', 2.26e6))
        Dm = float(data.get('Dm', 2.2e-5))
        
        # Temperature boundary/initial conditions
        T_initial = float(data.get('T_initial', 5.0))
        T_wall = float(data.get('T_wall', 7.0))
        T_inlet = float(data.get('T_inlet', 1.0))
        
        # Moisture boundary/initial conditions
        W_initial = float(data.get('W_initial', 0.01))
        W_wall = float(data.get('W_wall', 0.005))
        W_inlet = float(data.get('W_inlet', 0.008))
        RH_ambient = float(data.get('RH_ambient', 0.8))
        
        # Time stepping
        time_steps = int(data.get('time_steps', 10))
        dt = float(data.get('dt', 100.0))
        
        # Create and run solver
        solver = create_solver_from_params(
            Lx=Lx, Ly=Ly, Lz=Lz,
            nx=nx, ny=ny, nz=nz,
            alpha=alpha, T_initial=T_initial,
            T_wall=T_wall, T_inlet=T_inlet,
            time_steps=time_steps, dt=dt,
            k=k, rho=rho, Cp=Cp,
            Lv=Lv, Dm=Dm,
            W_initial=W_initial,
            W_wall=W_wall,
            W_inlet=W_inlet,
            RH_ambient=RH_ambient
        )
        
        # Check stability
        stability_info = solver.get_stability_info()
        if not stability_info['is_stable']:
            return jsonify({
                'success': False,
                'error': 'Configuration is unstable',
                'details': stability_info,
                'message': f"rT = {stability_info['rT']:.6f}, rW = {stability_info['rW']:.6f}. Both must be ≤ 1/6"
            }), 400
        
        # Run simulation
        solver.solve()
        
        # Get results
        stats = solver.get_statistics()
        config_dict = {
            'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
            'nx': nx, 'ny': ny, 'nz': nz,
            'alpha': alpha, 'dt': dt,
            'time_steps': time_steps
        }
        
        # Generate visualization
        temp_field = solver.get_temperature_field()
        
        # 2D visualizations
        plot_path = plot_midplane_heatmap(temp_field, config_dict, stats)
        profile_path = plot_temperature_profile(temp_field, config_dict)
        
        # 3D interactive visualizations removed as requested
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'stability': {
                'is_stable': stability_info['is_stable'],
                'rT': float(stability_info['rT']),
                'rW': float(stability_info['rW']),
                'threshold': float(stability_info['threshold']),
                'margin_T': float(stability_info['margin_T']),
                'margin_W': float(stability_info['margin_W'])
            },
            'heatmap_url': '/static/heatmap_current.png',
            'profile_url': '/static/profile.png',
            'simulation_time': time_steps * dt
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Simulation failed: {str(e)}'
        }), 500


@app.route('/api/default-params', methods=['GET'])
def get_default_params():
    """Return default simulation parameters for coupled heat-moisture system"""
    return jsonify({
        # Domain
        'Lx': 10.0,
        'Ly': 10.0,
        'Lz': 10.0,
        'nx': 10,
        'ny': 10,
        'nz': 10,
        # Thermal properties
        'k': 0.024,
        'rho': 1.2,
        'Cp': 1005.0,
        'alpha': 0.024 / (1.2 * 1005.0),  # k / (rho * Cp)
        # Moisture properties
        'Lv': 2.26e6,
        'Dm': 2.2e-5,
        # Temperature conditions
        'T_initial': -20.0,
        'T_wall': -20.0,
        'T_inlet': -25.0,
        # Moisture conditions
        'W_initial': 0.01,
        'W_wall': 0.005,
        'W_inlet': 0.008,
        'RH_ambient': 0.8,
        # Time stepping
        'time_steps': 10,
        'dt': 100.0
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    app.run(debug=debug, host='0.0.0.0', port=port)
