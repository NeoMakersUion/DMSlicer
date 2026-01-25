__version__ = "1.0.0"
__author__ = "QilongJiang"
__license__ = "MIT"
# -*- coding: utf-8 -*-

from .config import  DEFAULT_VISUALIZER_TYPE, GEOM_ACC, VisualizerType
from .geom_kernel import GEOM_ACC, GeometryKernel
from .intersection import clip_segment_by_aabb, intersect_triangle_triangle, intersect_triangle_with_plane
from .mesh_normalize import DEFAULT_VISUALIZER_TYPE, GEOM_ACC, Geom, IdOrder, Object, Status, TrianglesOrder, VerticesOrder, normalize_meshes
from .part import InterFaceResult, Part
from .spatial_index import SpatialIndex
from .topology3d import Topology3D

__all__ = [
    "CONFIG_VERSION",
    "DEFAULTS",
    "DEFAULT_VISUALIZER_TYPE",
    "GEOM_ACC",
    "Geom",
    "GeometryKernel",
    "IdOrder",
    "InterFaceResult",
    "Object",
    "Part",
    "SpatialIndex",
    "Status",
    "Topology3D",
    "TrianglesOrder",
    "VerticesOrder",
    "VisualizerType",
    "clip_segment_by_aabb",
    "intersect_triangle_triangle",
    "intersect_triangle_with_plane",
    "normalize_meshes",
]
