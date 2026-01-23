# -*- coding: utf-8 -*-
"""
Topology3D
==========

Builds adjacency relationships between triangles in a 3D mesh.

This is NOT slicing topology.
This is 3D mesh connectivity (triangle neighbors via shared edges).
"""

from collections import defaultdict


class Topology3D:
    """
    3D triangle adjacency graph.

    This structure allows fast queries:
        - Which triangles share an edge?
        - What are the neighbors of a triangle?
    """

    def __init__(self, triangles):
        """
        Args:
            triangles: (M,3) array of vertex indices
        """
        self.triangles = triangles

        self.vertex_to_triangles = defaultdict(set)
        self.edge_to_triangles = defaultdict(set)
        self.triangle_neighbors = defaultdict(set)

        self._build()

    # --------------------------------------------------

    def _build(self):
        """Build vertex and edge adjacency."""
        for tri_id, tri in enumerate(self.triangles):
            v0, v1, v2 = tri

            # Vertex adjacency
            self.vertex_to_triangles[v0].add(tri_id)
            self.vertex_to_triangles[v1].add(tri_id)
            self.vertex_to_triangles[v2].add(tri_id)

            # Edge adjacency (sorted to be undirected)
            e01 = tuple(sorted((v0, v1)))
            e12 = tuple(sorted((v1, v2)))
            e20 = tuple(sorted((v2, v0)))

            self.edge_to_triangles[e01].add(tri_id)
            self.edge_to_triangles[e12].add(tri_id)
            self.edge_to_triangles[e20].add(tri_id)

        # Build triangle neighbors via shared edges
        for edge, tris in self.edge_to_triangles.items():
            if len(tris) >= 2:
                for t in tris:
                    self.triangle_neighbors[t].update(tris - {t})

    # --------------------------------------------------

    def get_neighbors(self, tri_id):
        """Return all triangles that share an edge with this triangle."""
        return list(self.triangle_neighbors.get(tri_id, []))

    def get_triangles_sharing_edge(self, v1, v2):
        """Return triangles sharing edge (v1, v2)."""
        return self.edge_to_triangles.get(tuple(sorted((v1, v2))), [])

    def get_triangles_at_vertex(self, v):
        """Return triangles touching vertex v."""
        return self.vertex_to_triangles.get(v, [])
