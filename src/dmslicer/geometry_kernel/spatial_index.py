# -*- coding: utf-8 -*-
"""
Spatial Index
=============

Fast Z-plane queries for slicing.

Given a plane Z = z,
returns triangle IDs whose z-range intersects that plane.
"""

import numpy as np
from bisect import bisect_left, bisect_right


class SpatialIndex:
    """
    Z-interval based spatial index for triangle meshes.

    Used to quickly find candidate triangles for slicing planes.
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        """
        Build spatial index from global vertices & triangles.

        Args:
            vertices: (N,3) array
            triangles: (M,3) index array
        """
        self.vertices = vertices
        self.triangles = triangles

        # Precompute z-ranges for each triangle
        self.tri_z_min = np.zeros(len(triangles))
        self.tri_z_max = np.zeros(len(triangles))

        for i, tri in enumerate(triangles):
            z = vertices[tri, 2]
            self.tri_z_min[i] = z.min()
            self.tri_z_max[i] = z.max()

        # Build sorted index
        self.sorted_by_zmin = sorted(range(len(triangles)), key=lambda i: self.tri_z_min[i])
        self.sorted_by_zmax = sorted(range(len(triangles)), key=lambda i: self.tri_z_max[i])

        self.zmin_values = [self.tri_z_min[i] for i in self.sorted_by_zmin]
        self.zmax_values = [self.tri_z_max[i] for i in self.sorted_by_zmax]

    # -------------------------------------------------

    def query(self, z: float):
        """
        Return triangle IDs whose Z-interval intersects z-plane.

        This is O(log N + K) where K is number of hits.
        """
        # find all triangles with z_min <= z
        idx1 = bisect_right(self.zmin_values, z)
        candidates1 = set(self.sorted_by_zmin[:idx1])

        # find all triangles with z_max >= z
        idx2 = bisect_left(self.zmax_values, z)
        candidates2 = set(self.sorted_by_zmax[idx2:])

        # intersection = triangles whose z_min <= z <= z_max
        return list(candidates1 & candidates2)
