# -*- coding: utf-8 -*-
try:
    import pyvista as pv
except ImportError:
    pv = None

import numpy as np
from .visualizer_interface import IVisualizer
from ..file_parser.mesh_data import MeshData
from typing import Union
class PyVistaVisualizer(IVisualizer):
    """
    Concrete implementation of IVisualizer using PyVista.
    """

    def __init__(self, **kwargs):
        if pv is None:
            raise ImportError("PyVista is required for PyVistaVisualizer but not installed.")
        try:
            self.plotter = pv.Plotter(**kwargs)
        except Exception as e:
            # Fallback or re-raise depending on policy. Here we re-raise to let caller handle it.
            raise RuntimeError(f"Failed to initialize PyVista Plotter: {e}")

    def add_mesh(
        self,
        mesh: Union[MeshData, pv.PolyData],
        opacity: float = 0.5,
        **kwargs
    ):
        """
        Add mesh to PyVista plotter.
        """
        # PyVista expects faces array to start with the number of vertices per face (3 for triangles)
        # Format: [3, v1, v2, v3, 3, v4, v5, v6, ...]
        if isinstance(mesh, MeshData):
            triangles = mesh.triangles
            n_faces = triangles.shape[0]
            faces_vis = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), triangles]).ravel()
            pv_mesh = pv.PolyData(mesh.vertices, faces_vis)
        elif isinstance(mesh, pv.PolyData):
            triangles = mesh.faces.reshape(-1, 4)[:, 1:]
        else:
            raise ValueError("mesh must be either MeshData or pv.PolyData")
        
        
        self.plotter.add_mesh(
            pv_mesh,
            color=mesh.color_tuple,
            opacity=opacity,
            show_edges=kwargs.get('show_edges', True),
            **kwargs
        )

    def show(self, **kwargs):
        """
        Show the PyVista plot.
        """
        try:
            self.plotter.show(**kwargs)
        except Exception as e:
            print(f"Warning: Failed to display 3D model. Details: {e}")
