from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from tqdm import tqdm
from collections import deque
import logging
import sys
import pandas as pd

from .config import GEOM_ACC
from .bvh import build_bvh, BVHNode, AABB


@dataclass
class Object:
    acc: int = GEOM_ACC
    id: Optional[int] = None
    vertices: Optional[np.ndarray] = None
    tri_id2vert_id: Optional[np.ndarray] = None
    triangles: Optional[list["Triangle"]] = field(default_factory=list)
    tri_id2geom_tri_id: Optional[np.ndarray] = None
    triangle_ids_order: Optional["IdOrder"] = None
    color: Optional[np.ndarray] = None
    status: Optional["Status"] = None
    __hash_id: Optional[str] = None
    aabb: Optional[AABB] = None
    bvh: Optional[BVHNode] = None
    components: Optional[List[List[int]]] = None
    vex_id2tri_ids: Dict[int, List[int]] = field(default_factory=dict)

    def get_triangles_by_list(self, lst: List[int]):
        return [self.triangles[i] for i in lst]

    @property
    def hash_id(self) -> str:
        if self.__hash_id is None:
            self.__hash__()
        return self.__hash_id

    def show(self, include_triangles_ids=None, exclude_triangles_ids=None, opacity: float = 0.5, color=None):
        from ..visualizer import IVisualizer
        visualizer = IVisualizer.create()
        visualizer.addObj(self, include_triangles_ids, exclude_triangles_ids, opacity, color)
        visualizer.show()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__hash__()

    def __hash__(self) -> None:
        self.__hash_id = hash((
            self.acc,
            self.id,
            self.tri_id2geom_tri_id.tobytes() if self.tri_id2geom_tri_id is not None else None,
            self.color.tobytes() if self.color is not None else None,
            self.triangle_ids_order,
            self.status
        ))

    def graph_build(self, include_tri_ids: Optional[List[int]] = None, exclude_tri_ids: Optional[List[int]] = None) -> Dict[int, List[int]]:
        tri_count = len(self.triangles)
        if include_tri_ids is not None:
            tri_ids = [int(i) for i in include_tri_ids if 0 <= int(i) < tri_count]
        else:
            tri_ids = list(range(tri_count))
        if exclude_tri_ids:
            exclude_set = {int(i) for i in exclude_tri_ids if 0 <= int(i) < tri_count}
            tri_ids = [i for i in tri_ids if i not in exclude_set]

        graph: Dict[int, List[int]] = {}
        tri_ids_set = set(tri_ids)

        for tri_id in tqdm(tri_ids, desc="Building graph", total=len(tri_ids), leave=False):
            try:
                adj_vals = self.triangles[tri_id].topology['edges'].values()
                raw_adj_list = sorted({int(t) for t in adj_vals if int(t) in tri_ids_set})
                if raw_adj_list:
                    graph[tri_id] = raw_adj_list
                else:
                    graph.setdefault("isolates", []).append(tri_id)
            except Exception:
                continue
        return graph

    def connected_components(self, graph: Optional[Dict[int, List[int]]] = None) -> List[List[int]]:
        import copy
        g = copy.deepcopy(graph if graph is not None else getattr(self, "graph", {}))
        if "isolates" in g:
            g.pop("isolates", None)
        total_nodes = len(g.keys())
        logger = logging.getLogger("progress.connected_components")
        visited: set[int] = set()
        components: List[List[int]] = []
        last_len = 0
        last_msg_len = 0

        def _update_progress(cur: int, total: int) -> None:
            pct = 0.0 if total == 0 else round(cur / total * 100, 2)
            msg = "." * 11 + f"进度: {cur}/{total} ({pct}%)" + "\r"
            sys.stdout.write(msg)
            sys.stdout.flush()
            if cur == total:
                sys.stdout.write("\r" + (" " * len(msg)) + "\r")
                sys.stdout.flush()
            nonlocal last_msg_len
            last_msg_len = len(msg)

        for tri_id in sorted(g.keys()):
            if tri_id in visited:
                continue
            component: List[int] = []
            q = deque([tri_id])
            while q:
                current = q.popleft()
                if current in visited:
                    continue
                visited.add(current)
                cur_len = len(visited)
                if cur_len > last_len:
                    _update_progress(cur_len, total_nodes)
                    last_len = cur_len
                component.append(current)
                for nb in g.get(current, []):
                    if nb not in visited:
                        q.append(nb)
            components.append(component)
        if last_msg_len > 0:
            sys.stdout.write(" " * last_msg_len + "\r")
            sys.stdout.flush()
        return components

    def ensure_bvh(self):
        if not self.triangles:
            self.bvh = None
            return
        triangles = self.vertices[self.tri_id2vert_id]
        self.bvh = build_bvh(triangles)
        pass

    def repair_degenerate_triangles(self):
        triangle_data = []
        degenerate_triangles = {"safe": [], "ok": [], "warn": [], "bad": [], "critical": []}
        for tri_id, triangle in enumerate(self.triangles):
            try:
                extracted_info = {
                    "id": tri_id,
                    "AhatESB_grade": triangle.AhatESB_grade.value,
                    "errorESB_grade": triangle.errorESB_grade.value,
                    "parametric_area_grade": triangle.parametric_area_grade.value
                }
                triangle_data.append(extracted_info)
            except AttributeError as e:
                tri_id = getattr(triangle, "triangle_id", "未知ID")
                print(f"提取三角形 {tri_id} 时出错: {e}")
                continue
        df_triangles = pd.DataFrame(triangle_data)
        df_or_gt = lambda df, x: (df['AhatESB_grade'] > x) | (df['errorESB_grade'] > x) | (df['parametric_area_grade'] > x)
        df_and_lt = lambda df, x: (df['AhatESB_grade'] < x) & (df['errorESB_grade'] < x) & (df['parametric_area_grade'] < x)
        degenerate_triangles["critical"] = df_triangles[df_or_gt(df_triangles, 3)]
        degenerate_triangles["bad"] = df_triangles[df_and_lt(df_triangles, 4) & df_or_gt(df_triangles, 2)]
        degenerate_triangles["warn"] = df_triangles[df_and_lt(df_triangles, 3) & df_or_gt(df_triangles, 1)]
        degenerate_triangles["ok"] = df_triangles[df_and_lt(df_triangles, 2) & df_or_gt(df_triangles, 0)]
        degenerate_triangles["safe"] = df_triangles[df_and_lt(df_triangles, 1)]
        if len(degenerate_triangles['critical']) > 0:
            tri_id_critical = degenerate_triangles['critical']['id'].tolist()
            triangles_critical = self.get_triangles_by_list(tri_id_critical)
            g = self.graph_build(include_tri_ids=tri_id_critical)
            components = self.connected_components(g)
            isolates = g['isolates']
            g.pop('isolates')
            from collections import deque, defaultdict
            for component in components:
                comp = set(component)
                g_sub = {k: [n for n in g.get(k, []) if n in comp] for k in comp}
                ends = [k for k, nbrs in g_sub.items() if len(nbrs) == 1]
                sources = ends if len(ends) > 0 else [next(iter(comp))]
                order, parent, depth, owner, meetings, visited = self._multi_source_bfs(g_sub, sources)
                outer = [tid for tid in order if tid not in meetings]
                inner = [tid for tid in order if tid in meetings]
                repair_queue = []
                for tid in outer + inner:
                    tri = self.triangles[tid]
                    if tri.parametric_area_grade.value >= 4:
                        repair_queue.append(tid)
                if visited != comp:
                    missing = list(comp - visited)
                    print("WARNING: bfs did not cover component, missing:", len(missing))
            pass
        if len(degenerate_triangles['bad']) > 0:
            pass
        pass

    @staticmethod
    def _multi_source_bfs(g_sub: dict[int, list[int]], sources: list[int]):
        q = deque()
        parent = {}
        depth = {}
        owner = {}
        meetings = set()
        for s in sources:
            q.append(s)
            parent[s] = None
            depth[s] = 0
            owner[s] = s
        visited = set(sources)
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in g_sub.get(u, []):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    owner[v] = owner[u]
                    q.append(v)
                else:
                    if owner.get(v, None) != owner.get(u, None):
                        meetings.add(u)
                        meetings.add(v)
        return order, parent, depth, owner, meetings, visited
