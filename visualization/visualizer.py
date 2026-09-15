"""
Facade for visualization utilities to maintain backward compatibility.
"""

from .temperature import plot_midplane_heatmap as plot_temp_heatmap, plot_temperature_profile as plot_temp_profile, plot_3d_field as plot_temp_3d, plot_3d_volumetric as plot_temp_vol
from .humidity import plot_3d_field as plot_moist_3d, plot_3d_volumetric as plot_moist_vol
from .heat import plot_3d_field as plot_heat_3d, plot_3d_volumetric as plot_heat_vol

def plot_midplane_heatmap(*args, **kwargs):
    return plot_temp_heatmap(*args, **kwargs)

def plot_temperature_profile(*args, **kwargs):
    return plot_temp_profile(*args, **kwargs)

def plot_3d_field(field, config_dict, title, label, unit, colorscale='Blues_r', output_path='static/3d_field.html'):
    # Determine which module to use based on label or unit
    if 'Temperature' in title or '°C' in unit:
        return plot_temp_3d(field, config_dict, title, label, unit, colorscale, output_path)
    elif 'Moisture' in title or 'kg/kg' in unit:
        return plot_moist_3d(field, config_dict, title, label, unit, colorscale, output_path)
    else:
        return plot_heat_3d(field, config_dict, title, label, unit, colorscale, output_path)

def plot_3d_field_volumetric(field, config_dict, title, label, unit, colorscale='Blues_r', output_path='static/3d_field_volumetric.html'):
    if 'Temperature' in title or '°C' in unit:
        return plot_temp_vol(field, config_dict, output_path)
    elif 'Moisture' in title or 'kg/kg' in unit:
        return plot_moist_vol(field, config_dict, output_path)
    else:
        return plot_heat_vol(field, config_dict, output_path)

def plot_3d_volumetric(field, solver_stats, config_dict, output_path='static/3d_volumetric.html'):
    # Specifically for temperature as in original app.py
    return plot_temp_vol(field, config_dict, output_path)

def plot_3d_volume_scatter(field, config_dict, solver_stats, decimation=1, output_path='static/3d_volume_scatter.html'):
    # Legacy function from original visualizer.py
    return plot_temp_3d(field, config_dict, "3D Volume Scatter", "Temperature", "°C", output_path=output_path)

def plot_3d_isosurface(*args, **kwargs):
    # Placeholder as it was more complex, can be implemented in temperature.py
    from .temperature import plot_3d_field
    return plot_3d_field(*args, **kwargs)

def plot_3d_sliced_views(*args, **kwargs):
    # Placeholder
    from .temperature import plot_3d_field
    return plot_3d_field(*args, **kwargs)
