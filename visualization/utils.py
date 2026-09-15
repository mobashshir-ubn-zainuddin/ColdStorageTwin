"""
Common visualization utilities for the Cold Storage Digital Twin.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

def create_cold_colormap():
    """Create a custom colormap for cold storage temperatures"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#1a237e', '#283593', '#3f51b5', '#5e6cc0', '#7986cb',
              '#9fa8da', '#c5cae9', '#e8eaf6', '#ffffff']
    return LinearSegmentedColormap.from_list('cold_storage', colors, N=256)

def get_diverging_colorscale():
    """Enhanced diverging colorscale"""
    return [
        [0.0, '#4b0082'], [0.15, '#0000ff'], [0.35, '#add8e6'],
        [0.5, '#ffffff'], [0.65, '#ffff00'], [0.85, '#ffa500'], [1.0, '#ff0000']
    ]

def save_html_artifact(fig, output_path, title):
    """Save a Plotly figure as HTML"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', x=0.5, xanchor='center'),
        width=1000, height=800, margin=dict(l=0, r=150, b=0, t=80)
    )
    fig.write_html(output_path, include_plotlyjs='cdn', config={'displaylogo': False})
    return output_path
