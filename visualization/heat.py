"""
Heat and Energy visualization for the Cold Storage Digital Twin.
"""

import numpy as np
import plotly.graph_objects as go
from .utils import save_html_artifact

def plot_3d_field(field, config_dict, title, label, unit, colorscale='diverging', output_path='static/heat_3d.html'):
    """Generalized 3D scatter plot for heat fields"""
    from .utils import get_diverging_colorscale
    nx, ny, nz = field.shape
    x, y, z = np.meshgrid(np.linspace(0, config_dict['Lx'], nx),
                          np.linspace(0, config_dict['Ly'], ny),
                          np.linspace(0, config_dict['Lz'], nz), indexing='ij')

    val_flat = field.flatten()
    c_scale = get_diverging_colorscale() if colorscale == 'diverging' else colorscale
    c_min, c_max = (-max(abs(val_flat.min()), abs(val_flat.max())), max(abs(val_flat.min()), abs(val_flat.max()))) if colorscale == 'diverging' else (val_flat.min(), val_flat.max())

    fig = go.Figure(data=[go.Scatter3d(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        mode='markers', marker=dict(size=6, color=val_flat, colorscale=c_scale, cmin=c_min, cmax=c_max, showscale=True),
        text=[f"{label}: {v:.4f}{unit}" for v in val_flat],
        hovertemplate='%{text}<extra></extra>'
    )])
    return save_html_artifact(fig, output_path, title)

def plot_3d_volumetric(field, config_dict, output_path='static/heat_vol.html'):
    """Continuous 3D Volumetric visualization for heat"""
    from .utils import get_diverging_colorscale
    nx, ny, nz = field.shape
    x, y, z = np.meshgrid(np.linspace(0, config_dict['Lx'], nx),
                          np.linspace(0, config_dict['Ly'], ny),
                          np.linspace(0, config_dict['Lz'], nz), indexing='ij')
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(), value=field.flatten(),
        colorscale=get_diverging_colorscale(), opacity=0.15, surface_count=25
    ))
    return save_html_artifact(fig, output_path, "Heat Volumetric Distribution")
