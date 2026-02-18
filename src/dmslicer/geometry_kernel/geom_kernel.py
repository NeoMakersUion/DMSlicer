# -*- coding: utf-8 -*-
# geom_kernel.py
"""
Geometry Kernel
===============

Author//作者: QilongJiang
Date//日期: 2026-02-11

This module defines the core 3D geometry world of DMSlicer.//本模块定义了 DMSlicer 的核心 3D 几何处理核心。

It takes parsed MeshData (Model) and builds//该模块接收解析后的网格数据（Model 模型）并构建以下内容 :
- global vertex / triangle space//全局顶点 / 三角形空间
- adjacency topology//邻接拓扑结构
- spatial index//空间索引
- geometric intersection queries//几何相交查询

This is the ONLY interface slicer should talk to.//这是切片器（slicer）唯一需要对接的接口。
"""

from typing import List, Tuple, Dict
import numpy as np

from ..file_parser.model import Model

from .canonicalize import Geom
from .topology3d import Topology3D
from .spatial_index import SpatialIndex

from .config import GEOM_ACC
from ..visualizer.visualizer_interface import IVisualizer
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
        self.geom = Geom(model,acc)
        # Step 1: normalize all meshes into one global mesh
        self._build_pair_level_patch_mapping()
        pass

    def _build_pair_level_patch_mapping(self) -> None:
        """
        Build object-level adjacency graph and per-pair component mappings.
        构建对象级邻接图以及“对象对 → 组件对”映射结果。

        The graph field is a symmetric adjacency map: node -> set(neighbors).
        graph 字段是对称邻接映射：节点 → 邻接节点集合。

        The patch field is flattened into a list of {obj1_idx: comp1, obj2_idx: comp2}
        dicts per pair, using the first adjacency index as the linked component.
        patch 字段会被展平成 {obj1_idx: comp1, obj2_idx: comp2} 的列表，
        并使用第一个邻接索引作为对应组件的连接点。
        """
        # Build symmetric object-level adjacency graph from patch_info keys.
        # 基于 patch_info 的键集合构建对象级对称邻接图。
        patch_pairs = list(self.geom.patch_info.keys())
        graph = {}
        for node1, node2 in patch_pairs:
            if node1 not in graph:
                graph[node1] = set()
            graph[node1].add(node2)
            if node2 not in graph:
                graph[node2] = set()
            graph[node2].add(node1)
        self.geom.graph = graph

        # Flatten per-object patch components into per-pair component mappings.
        # 将按对象存储的组件列表展平为每个对象对的一组组件配对结果。
        tmp = {pair: elem.patch for pair, elem in self.geom.patch_info.items()}
        self.geom.patch={}
        for pair, patch_dict in tmp.items():
            obj1_idx, obj2_idx = pair
            result = []
            length = len(patch_dict[obj1_idx])
            for obj1_comp_idx in range(length):
                comp1 = patch_dict[obj1_idx][obj1_comp_idx]["component"]
                obj2_comp_idx = patch_dict[obj1_idx][obj1_comp_idx]["adj"][0]
                comp2 = patch_dict[obj2_idx][obj2_comp_idx]["component"]
                result.append({obj1_idx: comp1, obj2_idx: comp2})
            self.geom.patch[pair] = result
            
    def visualize_obj(self,lst,visualizer:IVisualizer=None,opacity:float=0.5):
        show=False
        if not isinstance(lst,list):
            lst=[lst]
        if not visualizer:
            show=True
            
            visualizer=IVisualizer.create()
        for elem in lst:
            visualizer.addObj(self.geom.objects[elem],opacity=opacity)   
        if show:
            visualizer.show()

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
