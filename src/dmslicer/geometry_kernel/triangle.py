import numpy as np
from typing import Optional, Dict, Any
from dataclasses import field
from .bvh import AABB, triangle_aabb_from_coords

class Triangle:
    user_min_edge = 0  # 将在 canonicalize 中通过静态值覆盖或依赖外部设定
    vertices: np.ndarray
    vertex_ids: np.ndarray
    normal: np.ndarray
    id: int
    min_edge: float
    edges: np.ndarray = None
    topology: Optional[Dict] = field(default_factory=dict)
    area: float = None
    aabb: AABB = None
    perimeter: float = None
    parametric_area_grade = None
    errorESB_grade = None
    AhatESB_grade = None

    def __init__(self, obj: Any, triangle_id: int):
        self.vertices = obj.vertices[obj.tri_id2vert_id[triangle_id]]
        self.aabb = triangle_aabb_from_coords(self.vertices)
        self.vertex_ids = obj.tri_id2vert_id[triangle_id]
        self.id = triangle_id
        self.process_edge_area_ESB()
        self.build_topology(obj)

    def process_edge_area_ESB(self):
        from .static_class import QualityGrade
        from .intersection import triangle_normal_vector
        self.normal = triangle_normal_vector(self.vertices)
        diff_array = self.vertices - np.array([self.vertices[-1], self.vertices[0], self.vertices[1]])
        self.edges = diff_array
        edge_lengths = np.linalg.norm(diff_array.astype(np.float64), axis=1)
        self.min_edge = float(np.min(edge_lengths))
        self.perimeter = float(np.sum(edge_lengths))
        c = np.cross(self.edges[0].astype(np.float64), self.edges[1].astype(np.float64))
        area = 0.5 * float(np.linalg.norm(c))
        self.area = area
        parametric_area_ratio = float('inf') if area <= 0 else (self.perimeter**2) / (np.pi * area)
        self.parametric_area_grade = QualityGrade.classify(parametric_area_ratio, criteria=[12, 50, 400, 1000])
        L = float(self.aabb.diag)
        errorESB = max(float(Triangle.user_min_edge), L * 1e-10)
        errorESB_ratio = float('inf') if self.min_edge <= 0 else (errorESB / self.min_edge)
        self.errorESB_grade = QualityGrade.classify(errorESB_ratio, criteria=[0.1, 1, 2, 10])
        AhatESB = 0.5 * float(np.max(edge_lengths)) * errorESB
        AhatESB_ratio = float('inf') if self.area <= 0 else (AhatESB / self.area)
        self.AhatESB_grade = QualityGrade.classify(AhatESB_ratio, criteria=[0.1, 1, 2, 10])

    def build_topology(self, obj: Any):
        tri_id = self.id
        v0, v1, v2 = self.vertex_ids
        tris_0 = set(obj.vex_id2tri_ids[v0]) - {tri_id}
        tris_1 = set(obj.vex_id2tri_ids[v1]) - {tri_id}
        tris_2 = set(obj.vex_id2tri_ids[v2]) - {tri_id}
        edge0_key = (v0, v1) if v0 < v1 else (v1, v0)
        edge0_value = list(tris_0 & tris_1)
        edge0_val = -1 if len(edge0_value) != 1 else edge0_value[0]
        if edge0_val != -1:
            tris_0.remove(edge0_val)
            tris_1.remove(edge0_val)
        edge1_key = (v1, v2) if v1 < v2 else (v2, v1)
        edge1_value = list(tris_1 & tris_2)
        edge1_val = -1 if len(edge1_value) != 1 else edge1_value[0]
        if edge1_val != -1:
            tris_1.remove(edge1_val)
            tris_2.remove(edge1_val)
        edge2_key = (v2, v0) if v2 < v0 else (v0, v2)
        edge2_value = list(tris_0 & tris_2)
        edge2_val = -1 if len(edge2_value) != 1 else edge2_value[0]
        if edge2_val != -1:
            tris_0.remove(edge2_val)
            tris_2.remove(edge2_val)
        edge_map = {edge0_key: edge0_val, edge1_key: edge1_val, edge2_key: edge2_val}
        point_map = {v0: tris_0, v1: tris_1, v2: tris_2}
        self.topology = {"vertices": point_map, "edges": edge_map}
