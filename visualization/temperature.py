"""
Temperature visualization for the Cold Storage Digital Twin.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .utils import create_cold_colormap, save_html_artifact, get_diverging_colorscale

def plot_midplane_heatmap(temperature_field, config_dict, solver_stats, output_path='static/heatmap_current.png'):
    """Plot temperature heatmap at midplane (z = Lz/2)"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nz = temperature_field.shape[2]
    midplane = temperature_field[:, :, nz // 2]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    im = ax.imshow(midplane.T, extent=[0, config_dict['Lx'], 0, config_dict['Ly']],
                   origin='lower', cmap=create_cold_colormap(), aspect='auto', interpolation='bicubic')
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title(f'Cold Storage Temperature Distribution\nMiddle Plane (z = {config_dict["Lz"]/2:.1f} m)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    stats_text = f"Min: {solver_stats['min_temp']:.2f}°C\nMax: {solver_stats['max_temp']:.2f}°C\nMean: {solver_stats['mean_temp']:.2f}°C"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path

def plot_temperature_profile(temperature_field, config_dict, output_path='static/profile.png'):
    """Plot 1D temperature profile along center line"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    j_mid, k_mid = temperature_field.shape[1] // 2, temperature_field.shape[2] // 2
    profile = temperature_field[:, j_mid, k_mid]
    x = np.linspace(0, config_dict['Lx'], len(profile))
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(x, profile, 'b-', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('Temperature Profile Along Center Line', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path

def plot_3d_field(field, config_dict, title, label, unit, colorscale='Blues_r', output_path='static/3d_field.html'):
    """Generalized 3D scatter plot for temperature fields"""
    nx, ny, nz = field.shape
    x, y, z = np.meshgrid(np.linspace(0, config_dict['Lx'], nx),
                          np.linspace(0, config_dict['Ly'], ny),
                          np.linspace(0, config_dict['Lz'], nz), indexing='ij')

    val_flat = field.flatten()
    c_scale = get_diverging_colorscale() if colorscale == 'diverging' else colorscale
    c_min, c_max = (val_flat.min(), val_flat.max()) if colorscale != 'diverging' else (-max(abs(val_flat.min()), abs(val_flat.max())), max(abs(val_flat.min()), abs(val_flat.max())))

    fig = go.Figure(data=[go.Scatter3d(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        mode='markers', marker=dict(size=6, color=val_flat, colorscale=c_scale, cmin=c_min, cmax=c_max, showscale=True),
        text=[f"{label}: {v:.4f}{unit}" for v in val_flat],
        hovertemplate='%{text}<extra></extra>'
    )])
    return save_html_artifact(fig, output_path, title)

def plot_3d_volumetric(field, config_dict, output_path='static/3d_volumetric.html'):
    """Continuous 3D Volumetric visualization"""
    nx, ny, nz = field.shape
    x, y, z = np.meshgrid(np.linspace(0, config_dict['Lx'], nx),
                          np.linspace(0, config_dict['Ly'], ny),
                          np.linspace(0, config_dict['Lz'], nz), indexing='ij')
    # Note: solver_stats removed as it's not used for coordinates,
    # isomin/isomax can be derived from field
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(), value=field.flatten(),
        isomin=field.min(), isomax=field.max(),
        colorscale='Blues_r', opacity=0.15, surface_count=25
    ))
    return save_html_artifact(fig, output_path, "Cold Storage 3D Volumetric Visualization")
