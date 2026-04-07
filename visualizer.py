"""
Visualization utilities for 3D temperature fields using Matplotlib and Plotly
Includes interactive 3D visualization with rotation capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime


def create_cold_colormap():
    """Create a custom colormap for cold storage temperatures"""
    colors = ['#1a237e', '#283593', '#3f51b5', '#5e6cc0', '#7986cb', 
              '#9fa8da', '#c5cae9', '#e8eaf6', '#ffffff']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('cold_storage', colors, N=n_bins)
    return cmap


def plot_midplane_heatmap(
    temperature_field: np.ndarray,
    config_dict: dict,
    solver_stats: dict,
    output_path: str = 'static/heatmap_current.png'
) -> str:
    """
    Plot temperature heatmap at midplane (z = Lz/2)
    
    Args:
        temperature_field: Full 3D temperature field (T_cold to T_hot)
        config_dict: Configuration dictionary with grid and domain parameters
        solver_stats: Statistics dictionary from solver
        output_path: Path to save the PNG image
    
    Returns:
        Path to the saved image
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract midplane
    nz = temperature_field.shape[2]
    k_mid = nz // 2
    midplane = temperature_field[:, :, k_mid]
    
    # Create figure with better aesthetics
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Plot heatmap
    cmap = create_cold_colormap()
    im = ax.imshow(
        midplane.T,  # Transpose for correct orientation
        extent=[0, config_dict['Lx'], 0, config_dict['Ly']],
        origin='lower',
        cmap=cmap,
        aspect='auto',
        interpolation='bicubic'
    )
    
    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Cold Storage Temperature Distribution\nMiddle Plane (z = {config_dict["Lz"]/2:.1f} m)',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text box
    stats_text = (
        f"Min: {solver_stats['min_temp']:.2f}°C\n"
        f"Max: {solver_stats['max_temp']:.2f}°C\n"
        f"Mean: {solver_stats['mean_temp']:.2f}°C\n"
        f"Std Dev: {solver_stats['std_temp']:.2f}°C"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_temperature_profile(
    temperature_field: np.ndarray,
    config_dict: dict,
    output_path: str = 'static/profile.png'
) -> str:
    """
    Plot 1D temperature profile along center line
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract center line (middle of y and z)
    j_mid = temperature_field.shape[1] // 2
    k_mid = temperature_field.shape[2] // 2
    profile = temperature_field[:, j_mid, k_mid]
    
    x = np.linspace(0, config_dict['Lx'], len(profile))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(x, profile, 'b-', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Temperature Profile Along Center Line', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_3d_volume_scatter(
    temperature_field: np.ndarray,
    config_dict: dict,
    solver_stats: dict,
    decimation: int = 2,
    output_path: str = 'static/3d_volume_scatter.html'
) -> str:
    """
    Create interactive 3D visualization of temperature field using scatter plot.
    Points are colored by temperature and can be rotated like in SolidWorks.
    
    Args:
        temperature_field: Full 3D temperature field (nx, ny, nz)
        config_dict: Configuration dictionary with grid and domain parameters
        solver_stats: Statistics dictionary from solver
        decimation: Factor to decimate grid (e.g., 2 = show every other point)
        output_path: Path to save the HTML file
    
    Returns:
        Path to the saved HTML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Decimate for performance if grid is large
    if decimation > 1:
        temp_decimated = temperature_field[::decimation, ::decimation, ::decimation]
    else:
        temp_decimated = temperature_field
    
    # Create coordinate grids
    nx, ny, nz = temp_decimated.shape
    x = np.linspace(0, config_dict['Lx'], nx)
    y = np.linspace(0, config_dict['Ly'], ny)
    z = np.linspace(0, config_dict['Lz'], nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten arrays
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    temp_flat = temp_decimated.flatten()
    
    # Normalize temperature for color mapping
    t_min = solver_stats['min_temp']
    t_max = solver_stats['max_temp']
    t_norm = (temp_flat - t_min) / (t_max - t_min + 1e-10)
    
    # Create scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        mode='markers',
        marker=dict(
            size=4,
            color=temp_flat,
            colorscale='Blues_r',  # Reversed Blues colorscale
            showscale=True,
            colorbar=dict(
                title="Temperature (°C)",
                thickness=15,
                len=0.7,
                x=1.02
            ),
            opacity=0.8,
            line=dict(width=0)
        ),
        text=[f"T: {t:.2f}°C<br>X: {x:.2f}m<br>Y: {y:.2f}m<br>Z: {z:.2f}m" 
              for t, x, y, z in zip(temp_flat, x_flat, y_flat, z_flat)],
        hovertemplate='%{text}<extra></extra>',
        name='Temperature'
    )])
    
    # Update layout with better camera controls
    fig.update_layout(
        title=dict(
            text=f'<b>Cold Storage 3D Temperature Distribution</b><br>' +
                 f'<sub>Rotate: Click and drag | Zoom: Scroll | Pan: Right-click and drag</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title='X Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title='Y Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            zaxis=dict(title='Z Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube'
        ),
        width=1000,
        height=800,
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=11),
        showlegend=True,
        margin=dict(l=0, r=150, b=0, t=80)
    )
    
    # Add statistics annotation
    stats_text = (
        f"<b>Statistics</b><br>"
        f"Min: {solver_stats['min_temp']:.2f}°C<br>"
        f"Max: {solver_stats['max_temp']:.2f}°C<br>"
        f"Mean: {solver_stats['mean_temp']:.2f}°C<br>"
        f"Std Dev: {solver_stats['std_temp']:.2f}°C<br>"
        f"Grid: {temperature_field.shape[0]}×{temperature_field.shape[1]}×{temperature_field.shape[2]}"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="lightgray",
        borderwidth=1,
        font=dict(size=11),
        align="left",
        xanchor="left",
        yanchor="top"
    )
    
    # Save as HTML
    fig.write_html(output_path)
    
    return output_path


def plot_3d_isosurface(
    temperature_field: np.ndarray,
    config_dict: dict,
    solver_stats: dict,
    iso_value: float = None,
    output_path: str = 'static/3d_isosurface.html'
) -> str:
    """
    Create 3D isosurface visualization of temperature field.
    Shows a surface at a specific temperature value with rotation capabilities.
    
    Args:
        temperature_field: Full 3D temperature field (nx, ny, nz)
        config_dict: Configuration dictionary with grid and domain parameters
        solver_stats: Statistics dictionary from solver
        iso_value: Temperature value for isosurface. If None, uses mean temperature
        output_path: Path to save the HTML file
    
    Returns:
        Path to the saved HTML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use mean temperature if no iso_value specified
    if iso_value is None:
        iso_value = solver_stats['mean_temp']
    
    # Create coordinate grids
    nx, ny, nz = temperature_field.shape
    x = np.linspace(0, config_dict['Lx'], nx)
    y = np.linspace(0, config_dict['Ly'], ny)
    z = np.linspace(0, config_dict['Lz'], nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create isosurface data
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=temperature_field.flatten(),
        isomin=solver_stats['min_temp'],
        isomax=solver_stats['max_temp'],
        surface_count=4,
        colorscale='Blues_r',
        colorbar=dict(
            title="Temperature (°C)",
            thickness=15,
            len=0.7
        ),
        name='Temperature Isosurface',
        caps=dict(x_show=True, y_show=True, z_show=True),
        opacity=0.9,
        text=f"Isosurface at {iso_value:.2f}°C"
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Cold Storage 3D Isosurface Visualization</b><br>' +
                 f'<sub>Multiple temperature contours | Rotate: Click and drag | Zoom: Scroll | Pan: Right-click and drag</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title='X Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title='Y Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            zaxis=dict(title='Z Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube'
        ),
        width=1000,
        height=800,
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=0, r=150, b=0, t=80)
    )
    
    # Add statistics annotation
    stats_text = (
        f"<b>Statistics</b><br>"
        f"Min: {solver_stats['min_temp']:.2f}°C<br>"
        f"Max: {solver_stats['max_temp']:.2f}°C<br>"
        f"Mean: {solver_stats['mean_temp']:.2f}°C<br>"
        f"Std Dev: {solver_stats['std_temp']:.2f}°C<br>"
        f"Isosurfaces: 4 levels"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="lightgray",
        borderwidth=1,
        font=dict(size=11),
        align="left",
        xanchor="left",
        yanchor="top"
    )
    
    # Save as HTML
    fig.write_html(output_path)
    
    return output_path


def plot_3d_sliced_views(
    temperature_field: np.ndarray,
    config_dict: dict,
    solver_stats: dict,
    output_path: str = 'static/3d_sliced_views.html'
) -> str:
    """
    Create interactive 3D visualization showing slices from different planes.
    Combines XY, XZ, and YZ plane slices for comprehensive view.
    
    Args:
        temperature_field: Full 3D temperature field (nx, ny, nz)
        config_dict: Configuration dictionary with grid and domain parameters
        solver_stats: Statistics dictionary from solver
        output_path: Path to save the HTML file
    
    Returns:
        Path to the saved HTML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    nx, ny, nz = temperature_field.shape
    
    # Extract midplane slices
    xy_slice = temperature_field[:, :, nz // 2]  # XY plane at middle Z
    xz_slice = temperature_field[:, ny // 2, :]  # XZ plane at middle Y
    yz_slice = temperature_field[nx // 2, :, :]  # YZ plane at middle X
    
    # Create coordinate grids for each plane
    x = np.linspace(0, config_dict['Lx'], nx)
    y = np.linspace(0, config_dict['Ly'], ny)
    z = np.linspace(0, config_dict['Lz'], nz)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('XY Plane (Middle Z)', 'XZ Plane (Middle Y)', 'YZ Plane (Middle X)'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # XY Plane
    fig.add_trace(
        go.Heatmap(
            z=xy_slice.T,
            x=x,
            y=y,
            colorscale='Blues_r',
            name='XY Plane',
            colorbar=dict(x=0.32, len=0.8, y=0.5),
            hovertemplate='X: %{x:.2f}m<br>Y: %{y:.2f}m<br>T: %{z:.2f}°C<extra></extra>'
        ),
        row=1, col=1
    )
    
    # XZ Plane
    fig.add_trace(
        go.Heatmap(
            z=xz_slice.T,
            x=x,
            y=z,
            colorscale='Blues_r',
            name='XZ Plane',
            colorbar=dict(x=0.65, len=0.8, y=0.5),
            hovertemplate='X: %{x:.2f}m<br>Z: %{y:.2f}m<br>T: %{z:.2f}°C<extra></extra>'
        ),
        row=1, col=2
    )
    
    # YZ Plane
    fig.add_trace(
        go.Heatmap(
            z=yz_slice.T,
            x=y,
            y=z,
            colorscale='Blues_r',
            name='YZ Plane',
            colorbar=dict(x=0.98, len=0.8, y=0.5),
            hovertemplate='Y: %{x:.2f}m<br>Z: %{y:.2f}m<br>T: %{z:.2f}°C<extra></extra>'
        ),
        row=1, col=3
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="X Position (m)", row=1, col=2)
    fig.update_yaxes(title_text="Z Position (m)", row=1, col=2)
    fig.update_xaxes(title_text="Y Position (m)", row=1, col=3)
    fig.update_yaxes(title_text="Z Position (m)", row=1, col=3)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Cold Storage Temperature - 3D Sliced Views</b><br>' +
                 f'<sub>Three orthogonal planes showing temperature distribution</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        height=500,
        width=1400,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=10),
        showlegend=False,
        margin=dict(l=60, r=150, b=60, t=100),
        hovermode='closest'
    )
    
    # Save as HTML
    fig.write_html(output_path)
    
    return output_path


def plot_3d_volumetric(
    temperature_field: np.ndarray,
    config_dict: dict,
    solver_stats: dict,
    output_path: str = 'static/3d_volumetric.html'
) -> str:
    """
    Create advanced 3D volumetric visualization with volume rendering effect.
    Provides a comprehensive 3D view of the entire temperature field.
    
    Args:
        temperature_field: Full 3D temperature field (nx, ny, nz)
        config_dict: Configuration dictionary with grid and domain parameters
        solver_stats: Statistics dictionary from solver
        output_path: Path to save the HTML file
    
    Returns:
        Path to the saved HTML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    nx, ny, nz = temperature_field.shape
    x = np.linspace(0, config_dict['Lx'], nx)
    y = np.linspace(0, config_dict['Ly'], ny)
    z = np.linspace(0, config_dict['Lz'], nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create volume data
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=temperature_field.flatten(),
        isomin=solver_stats['min_temp'],
        isomax=solver_stats['max_temp'],
        colorscale='Blues_r',
        colorbar=dict(
            title="Temperature (°C)",
            thickness=15,
            len=0.75
        ),
        name='Temperature Volume',
        opacity=0.15,
        opacityscale=[[0, 0], [0.5, 0.1], [1, 0.5]]
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Cold Storage 3D Volumetric Visualization</b><br>' +
                 f'<sub>Complete 3D temperature field | Rotate: Click and drag | Zoom: Scroll</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title='X Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title='Y Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            zaxis=dict(title='Z Position (m)', backgroundcolor="rgb(240, 240, 240)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube'
        ),
        width=1000,
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=0, r=150, b=0, t=80),
        hovermode='closest'
    )
    
    # Add statistics annotation
    stats_text = (
        f"<b>Statistics</b><br>"
        f"Min: {solver_stats['min_temp']:.2f}°C<br>"
        f"Max: {solver_stats['max_temp']:.2f}°C<br>"
        f"Mean: {solver_stats['mean_temp']:.2f}°C<br>"
        f"Std Dev: {solver_stats['std_temp']:.2f}°C<br>"
        f"Grid: {nx}×{ny}×{nz}"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="lightgray",
        borderwidth=1,
        font=dict(size=11),
        align="left",
        xanchor="left",
        yanchor="top"
    )
    
    # Save as HTML
    fig.write_html(output_path)
    
    return output_path 
