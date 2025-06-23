"""
3D Mesh Viewer 

This Dash web application loads 3D segmented meshes in OBJ format and displays them interactively using Plotly. 
The colors of the plots represent different classes or segmentations. 
Select which mesh you want to view using the dropdown menu.

Requirements:
- OBJ files must contain vertex definitions in the format:
  'v x y z r g b' where color values are in [0,1].
- Faces must be defined with standard OBJ face syntax: 'f v1 v2 v3'.
"""

import os
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

# --- Input directory ---
MESH_DIR = '' #Path to test results
# --------Functions---------------
def get_mesh_files():
    """Gets a list of .obj files available in the directory."""
    obj_files = [f for f in os.listdir(MESH_DIR) if f.endswith('_segmented.obj')]
    mesh_data = []
    for obj_file in obj_files:
        base_name = obj_file.replace('_segmented.obj', '')
        mesh_data.append({
            'obj_path': os.path.join(MESH_DIR, obj_file),
            'name': base_name
        })
    return mesh_data


def load_mesh_with_colors(obj_path):
    """
    Loads vertices, faces, and colors (if present) from an OBJ file in the format:
    'v x y z r g b'. Returns vertices (N×3), faces (M×3), and colors (N×3).
    """
    verts, cols, faces = [], [], []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vals = list(map(float, line.strip().split()[1:]))
                x, y, z, r, g, b = vals
                verts.append((x, y, z))
                cols.append((r, g, b))  
            elif line.startswith('f '):
                idx = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:4]]
                faces.append(idx)
    return np.array(verts), np.array(faces), np.array(cols)


def make_colored_mesh(obj_path):
    verts, faces, cols = load_mesh_with_colors(obj_path)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        vertexcolor=cols,
        name='Mesh',
        opacity=1.0
    )

# --- Dash app ---
app = dash.Dash(__name__)


mesh_data = get_mesh_files()


dropdown_options = [
    {'label': mesh['name'], 'value': mesh['name']}
    for mesh in mesh_data
]


app.layout = html.Div([
    html.H1("3D Mesh Viewer"),
    dcc.Dropdown(
        id='mesh-selector',
        options=dropdown_options,
        value=dropdown_options[0]['value'] if dropdown_options else None,
        clearable=False,
        style={'width': '300px', 'margin-bottom': '20px'}
    ),
    dcc.Graph(id='mesh-graph', style={'height': '80vh'})
])


@app.callback(
    Output('mesh-graph', 'figure'),
    Input('mesh-selector', 'value')
)
def update_graph(mesh_name):
    if not mesh_name:
        return go.Figure()

    fig = go.Figure()

    mesh_info = next((m for m in mesh_data if m['name'] == mesh_name), None)
    if not mesh_info:
        return go.Figure()

    fig.add_trace(make_colored_mesh(mesh_info['obj_path']))

    fig.update_layout(
        title_text=f"3D Mesh: {mesh_name}",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# Start the Dash server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050)
