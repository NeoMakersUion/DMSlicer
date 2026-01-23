# -*- coding: utf-8 -*-
"""
Mesh Normalization
==================

Convert Model (list of MeshData) into a single, clean, global mesh:

- global vertices
- global triangle indices
- triangle metadata (object, block, color, etc)

This is the ONLY place where:
    - vertex deduplication
    - index remapping
    - triangle canonicalization
happens.
"""

from typing import Tuple, Dict, List
import numpy as np
from dmslicer.file_parser import Model, MeshData



def normalize_meshes(model: Model):
    """
    Normalize all MeshData into one global mesh.

    Returns:
        vertices: (N,3) float
        triangles: (M,3) int
        triangle_meta: dict tri_id -> metadata
    """
    # ------------------------  -------------------------
    # 1. Collect all vertices into one big list
    # -------------------------------------------------
    all_vertices: List[np.ndarray] = []  
    all_triangles = []
    triangle_meta = {}

    # Map from local vertex to global index
    # key = (x,y,z)  (or quantized int)
    vertex_map: Dict[Tuple[float, float, float], int] = {}

    global_vertices: List[np.ndarray] = []


    tri_id = 0
    local_tris=[]
    object_ids=[]
    colors=[]
    for _, mesh in enumerate(model.meshes):
        colors.append(mesh.color)
        object_ids.append(mesh.id)
        verts = mesh.vertices
        tris = mesh.triangles
        # local -> global vertex index map
        local_to_global = {}
        

        for i, v in enumerate(verts):
            key = (float(v[0]), float(v[1]), float(v[2]))
            if key not in vertex_map:
                vertex_map[key] = len(global_vertices)
                global_vertices.append(v.astype(float))
            local_to_global[i] = vertex_map[key]

        local_tri=[]
        # remap triangles
        for t in tris:
            gtri = (
                local_to_global[int(t[0])],
                local_to_global[int(t[1])],
                local_to_global[int(t[2])],
            )
            local_tri.append(gtri)
            all_triangles.append(gtri)
            tri_id += 1
        local_tris.append(local_tri)
    # convert to numpy
    vertices = np.asarray(global_vertices, dtype=float)
    triangles = np.asarray(all_triangles, dtype=np.int64)

    return vertices, triangles, triangle_meta
