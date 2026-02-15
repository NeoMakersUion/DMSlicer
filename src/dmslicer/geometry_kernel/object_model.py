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

from .transforms import rotate_z_to_vector
R=rotate_z_to_vector(np.array([1,1,100]))
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
        """Build an undirected triangle adjacency subgraph for this object.

        The method constructs an adjacency list over triangle indices using per‑triangle
        topology, restricted to a selected subset. A node is a triangle id
        i ∈ {0, …, N−1}, where N is the number of triangles in this object.
        An edge (i, j) exists if triangles i and j share an edge according to
        topology['edges'] (edge‑adjacency; no geometric checks here).

        Data definitions:
        - Vertex coordinates: local object frame, unit = model length unit (e.g., mm).
        - Triangles: indices into `self.vertices` via `self.tri_id2vert_id`.
        - Adjacency: edge‑based neighborhood derived from stored topology.
          For an adjacency set A(i), j ∈ A(i) iff j is referenced by any edge entry of i.
        - Output graph: a dict mapping tri_id → sorted, unique neighbor list.
          Isolated triangles are collected under a reserved key "isolates".

        Why this approach:
        - Subgraph selection uses include/exclude filters to support downstream
          connected‑component analysis on candidate submeshes (e.g., contact patches).
        - A set is used for membership checks, reducing filter cost from O(n) to O(1).
        - Neighbor lists are sorted to make results deterministic and easier to test.

        Args:
            include_tri_ids: Optional explicit whitelist of triangle ids to include.
                Must be convertible to int. Ids outside [0, N) are ignored.
                If None, all triangles [0, N) are considered.
            exclude_tri_ids: Optional blacklist of triangle ids to exclude from the
                final set. Applied after inclusion filtering.

        Returns:
            A dictionary where each key is a triangle id and the value is a list of
            adjacent triangle ids (sorted ascending). A special key "isolates" maps
            to a list of triangle ids with no neighbors in the filtered subgraph.

        Note:
            - The graph is conceptually undirected, but only the adjacency list for
              present nodes is emitted. If upstream topology is asymmetric, the
              neighbor relation may appear one‑sided. Downstream algorithms should
              symmetrize if necessary.
            - No geometry (angles, distances) is computed here; this is purely
              topological adjacency.

        Example:
            >>> obj = Object(id=1)
            >>> # obj.triangles[i].topology['edges'] must exist for each i
            >>> g = obj.graph_build(include_tri_ids=[0, 1, 2])
            >>> isinstance(g, dict)
            True

        AI Context:
            Input: optional include/exclude triangle id lists; relies on
            `self.triangles[i].topology['edges']` to be pre‑built.
            Output: adjacency dict with an "isolates" bucket. Pitfalls: if
            topology is missing or inconsistent, neighbor lists may be empty.
        """
        tri_count = len(self.triangles)
        # Filter the working set of triangle ids. Using ints and range checks
        # prevents invalid indices from entering adjacency construction.
        if include_tri_ids is not None:
            tri_ids = [int(i) for i in include_tri_ids if 0 <= int(i) < tri_count]
        else:
            tri_ids = list(range(tri_count))
        if exclude_tri_ids:
            exclude_set = {int(i) for i in exclude_tri_ids if 0 <= int(i) < tri_count}
            tri_ids = [i for i in tri_ids if i not in exclude_set]

        graph: Dict[int, List[int]] = {}
        # Set membership is O(1), which is critical when filtering neighbors per node.
        tri_ids_set = set(tri_ids)

        for tri_id in tqdm(tri_ids, desc="Building graph", total=len(tri_ids), leave=False):
            try:
                # Extract edge‑adjacent triangle ids from precomputed topology.
                adj_vals = self.triangles[tri_id].topology['edges'].values()
                raw_adj_list = sorted({int(t) for t in adj_vals if int(t) in tri_ids_set})
                if raw_adj_list:
                    graph[tri_id] = raw_adj_list
                else:
                    # AI: ISOLATES_KEY = 'isolates' # reserve isolated triangles for downstream handling
                    graph.setdefault("isolates", []).append(tri_id)
            except Exception:
                # TODO-20260215-assistant: Replace broad except with topology validation step.
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
        self.bvh_dot=build_bvh((R@triangles.T).T)
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
