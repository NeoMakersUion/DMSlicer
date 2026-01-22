# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Union, Optional

@dataclass
class MeshData:
    """
    Data class representing a 3D mesh object.
    
    Attributes:
        vertices: (N, 3) float32 numpy array of vertex coordinates.
        triangles: (M, 3) int64 numpy array of triangle indices.
        color: (3,) float numpy array representing RGB color in [0, 1].
    """
    vertices: np.ndarray
    triangles: np.ndarray
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=float))

    def __post_init__(self):
        # Ensure types are correct
        if not isinstance(self.vertices, np.ndarray):
            self.vertices = np.asarray(self.vertices, dtype=np.float32)
        if not isinstance(self.triangles, np.ndarray):
            self.triangles = np.asarray(self.triangles, dtype=np.int64)
        if not isinstance(self.color, np.ndarray):
            self.color = np.asarray(self.color, dtype=float)

    @property
    def color_tuple(self) -> Tuple[float, float, float]:
        """Returns color as a tuple (r, g, b)."""
        return tuple(self.color)

    def __repr__(self):
        return (f"MeshData(vertices_shape={self.vertices.shape}, "
                f"triangles_shape={self.triangles.shape}, "
                f"color={self.color})")
