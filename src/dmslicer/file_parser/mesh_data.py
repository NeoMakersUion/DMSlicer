# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Union, Optional, List, ClassVar

@dataclass
class MeshData:
    """
    Data class representing a 3D mesh object.
    
    Attributes:
        vertices: (N, 3) float32 numpy array of vertex coordinates.
        triangles: (M, 3) int64 numpy array of triangle indices.
        color: (3,) float numpy array representing RGB color in [0, 1].
        id: Unique identifier for the mesh instance (auto-generated).
    """
    # 类属性：计数器，明确标记为 ClassVar，不参与实例初始化
    count: ClassVar[int] = 0

    # 实例属性
    vertices: np.ndarray
    triangles: np.ndarray
    # id 不需要初始化时传入，由 post_init 生成
    id: int = field(init=False)
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=float))

    def __post_init__(self):
        # Ensure types are correct
        if not isinstance(self.vertices, np.ndarray):
            self.vertices = np.asarray(self.vertices, dtype=np.float32)
        if not isinstance(self.triangles, np.ndarray):
            self.triangles = np.asarray(self.triangles, dtype=np.int64)
        if not isinstance(self.color, np.ndarray):
            self.color = np.asarray(self.color, dtype=float)
        
        # 自动分配id
        self.id = MeshData.count
        MeshData.count += 1

    @property
    def color_tuple(self) -> Tuple[float, float, float]:
        """Returns color as a tuple (r, g, b)."""
        return tuple(self.color)

    def __repr__(self):
        return (f"MeshData(id={self.id}, "
                f"vertices_shape={self.vertices.shape}, "
                f"triangles_shape={self.triangles.shape}, "
                f"color={self.color})")
