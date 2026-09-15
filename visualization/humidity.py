"""
Humidity visualization for the Cold Storage Digital Twin.
"""

import numpy as np
import plotly.graph_objects as go
from .utils import save_html_artifact

def plot_3d_field(field, config_dict, title, label, unit, colorscale='GnBu', output_path='static/moisture_3d.html'):
    """Generalized 3D scatter plot for humidity fields"""
    nx, ny, nz = field.shape
    x, y, z = np.meshgrid(np.linspace(0, config_dict['Lx'], nx),
                          np.linspace(0, config_dict['Ly'], ny),
                          np.linspace(0, config_dict['Lz'], nz), indexing='ij')

    val_flat = field.flatten()
    fig = go.Figure(data=[go.Scatter3d(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        mode='markers', marker=dict(size=6, color=val_flat, colorscale=colorscale, showscale=True),
        text=[f"{label}: {v:.4f}{unit}" for v in val_flat],
        hovertemplate='%{text}<extra></extra>'
    )])
    return save_html_artifact(fig, output_path, title)

def plot_3d_volumetric(field, config_dict, output_path='static/moisture_vol.html'):
    """Continuous 3D Volumetric visualization for humidity"""
    nx, ny, nz = field.shape
    x, y, z = np.meshgrid(np.linspace(0, config_dict['Lx'], nx),
                          np.linspace(0, config_dict['Ly'], ny),
                          np.linspace(0, config_dict['Lz'], nz), indexing='ij')
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(), value=field.flatten(),
        colorscale='GnBu', opacity=0.15, surface_count=25
    ))
    return save_html_artifact(fig, output_path, "Moisture Volumetric Distribution")
