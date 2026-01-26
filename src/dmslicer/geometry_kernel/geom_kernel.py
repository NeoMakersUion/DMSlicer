# -*- coding: utf-8 -*-
"""
Geometry Kernel
===============

This module defines the core 3D geometry world of DMSlicer.

It takes parsed MeshData (Model) and builds:
- global vertex / triangle space
- adjacency topology
- spatial index
- geometric intersection queries

This is the ONLY interface slicer should talk to.
"""

from typing import List, Tuple, Dict
import numpy as np

from ..file_parser.model import Model

from .mesh_normalize import Geom
from .topology3d import Topology3D
from .spatial_index import SpatialIndex
from .intersection import (
    intersect_triangle_with_plane,
)

from .config import GEOM_ACC
class GeometryKernel:
    """
    GeometryKernel represents a complete, queryable 3D mesh world.

    It is the bridge between:
        File Parser (MeshData)
            →
        Slicer (planes, loops, paths)

    All slicing logic must go through this class.
    """

    # -------------------------
    # Construction
    # -------------------------

    def __init__(self, model: Model,acc:int=GEOM_ACC):
        """
        Build the 3D geometry kernel from a parsed Model.

        Args:
            model: Model produced by file_parser (list of MeshData)
        """
        self.acc=acc
        # Step 1: normalize all meshes into one global mesh
        self.geom = Geom(model,acc)

        self.geom.show()  
        # # Step 2: build topology (adjacency)
        # self.topology = Topology3D(self.triangles)

        # # Step 3: build spatial index (z-range, fast plane queries)
        # self.spatial = SpatialIndex(self.vertices, self.triangles)

    # ============================================================
    #                  High-level slicing API
    # ============================================================

    def query_triangles_by_plane(self, z: float) -> List[int]:
        """
        Return candidate triangle IDs that may intersect with plane Z = z.

        This uses spatial index (z_min / z_max) to prune triangles.

        This is the FIRST call slicer should make for each layer.
        """
        return self.spatial.query(z)

    def intersect_triangle_with_plane(self, tri_id: int, z: float):
        """
        Compute intersection between triangle and plane Z = z.

        Returns:
            None                → no intersection
            (p1, p2)            → one segment
            [(p1,p2), ...]      → multiple segments (degenerate cases)
        """
        tri = self.triangles[tri_id]
        v0 = self.vertices[tri[0]]
        v1 = self.vertices[tri[1]]
        v2 = self.vertices[tri[2]]

        return intersect_triangle_with_plane(v0, v1, v2, z)

    # ============================================================
    #                  Geometry & topology queries
    # ============================================================

    def get_triangle_vertices(self, tri_id: int) -> np.ndarray:
        """Return (3,3) array of triangle vertices."""
        tri = self.triangles[tri_id]
        return self.vertices[tri]

    def get_triangle_neighbors(self, tri_id: int) -> List[int]:
        """Return adjacent triangle IDs."""
        return self.topology.get_neighbors(tri_id)

    def get_triangle_meta(self, tri_id: int) -> Dict:
        """Return metadata: object_id, block_id, color, etc."""
        return self.triangle_meta.get(tri_id, {})

    # ============================================================
    #                  Debug / visualization helpers
    # ============================================================

    def get_all_triangles(self):
        return self.triangles

    def get_all_vertices(self):
        return self.vertices
