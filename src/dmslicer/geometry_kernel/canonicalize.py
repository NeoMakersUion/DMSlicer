"""
DMSlicer · Geometry Canonicalization
构建归一化几何、缓存与邻接图

This module normalizes mesh data, builds geometry caches, and constructs
adjacency graphs used by patch-level expansion in the geometry kernel.
作者: QilongJiang
日期: 2026-02-11
"""
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Optional, TYPE_CHECKING, Any
from collections import defaultdict, OrderedDict, deque
from tqdm import tqdm
import logging
from datetime import datetime
import sys
from dataclasses import dataclass,field
import sys
import threading
import time
import pickle
from pathlib import Path
from .config import GEOM_ACC,PROCESS_ACC
import pandas as pd
if TYPE_CHECKING:
    from ..file_parser import Model
    from ..visualizer import IVisualizer
    
from ..visualizer import VisualizerType, IVisualizer
from .config import GEOM_ACC,DEFAULT_VISUALIZER_TYPE,GEOM_PARALLEL_ACC,SOFT_NORMAL_GATE_ANGLE,INITIAL_GAP_FACTOR,MAX_GAP_FACTOR,OVERLAP_RATIO_THRESHOLD
from enum import Enum
from .bvh import BVHNode,query_obj_bvh, AABB, build_bvh
from ..file_parser.workspace_utils import get_workspace_dir
EPSILON=1
from .static_class import Status, VerticesOrder, TrianglesOrder, IdOrder, QualityGrade
from .triangle import Triangle
from .patch_level import Patch


from ..cache.compute_cache import ComputeCache
from .object_model import Object

# Global cache instance
GEOM_CACHE = ComputeCache(max_size=5, ttl=7200) # Keep 5 heavy Geoms in memory



 

import numpy as np
# 确保Object类已导入，triangle_normal_vector函数可正常调用
# from xxx import Object
# from .intersection import triangle_normal_vector
from .bvh import AABB
scale = int(10 ** GEOM_ACC)                 # 坐标放大倍数
min_edge_mm = 0.5 * (10 ** (-PROCESS_ACC))  # 例如 PROCESS_ACC=2 -> 0.005mm

class Geom:
    def __init__(self,model:"Model",acc=GEOM_ACC,parallel_acc=GEOM_PARALLEL_ACC,vertices_order=VerticesOrder.ZYX,triangles_order=TrianglesOrder.P132,show=False) -> None:
        self.acc: int = acc
        self.parallel_acc:Optional[float]=parallel_acc
        self.vertices_order:VerticesOrder=vertices_order
        self.triangles_order:TrianglesOrder=triangles_order
        self.model:Optional["Model"]=model
        
        # Initialize placeholders
        self.objects: Dict[int,Object] = {}
        self.vertices: Union[np.ndarray,List] = []
        self.triangles: Union[np.ndarray,List] = []
        self.status:Optional[Status]=None
        self.__hash_id:Optional[str]=None
        
        self.triangle_index_map: Dict[Tuple[int, ...], int] = {}
        self.triangles_AABB: Optional[np.ndarray] = None
        self.AABB: Optional[np.ndarray] = None
        
        # Calculate hash early for cache lookup
        self.__hash__()
        
        if not self._load_from_cache():
            # If not cached, compute
            self.__merge_all_meshes()
            self.__sort__()
            self.__build_object_contact_graph()
            # Save to cache
            self._save_to_cache()
            if show:
                self.show()

        # Try to load from cache
        pair_level_files,pair_level_dir=self.__build_object_contact_triangles()
        patch_level_result,patch_level_dir=self.__build_object_contact_patch_level(pair_level_files,pair_level_dir,show=show)
        self.patch_info=patch_level_result

    @property
    def hash_id(self) -> str:
        # 延迟计算：仅在首次访问 hash_id 属性时才生成哈希值，避免初始化阶段不必要的计算开销
        if self.__hash_id is None:
            self.__hash__()
        return self.__hash_id


    def __hash__(self) -> None:
        # Include model hash and sorting parameters
        self.__hash_id = str(hash((
            self.acc,
            self.model.hash_id if self.model else "None",
            self.vertices_order,
            self.triangles_order,
        )))

    def _load_from_cache(self) -> bool:
        """Attempt to load the Geom state from cache."""
        cached_geom = GEOM_CACHE.get(self.model.hash_id)
        if cached_geom and isinstance(cached_geom, Geom):
            # Restore state from cached object
            # We preserve the current model reference to ensure consistency
            current_model = self.model
            self.__dict__.update(cached_geom.__dict__)
            self.model = current_model 
            if isinstance(self.objects, list):
                try:
                    self.objects = {o.id: o for o in self.objects}
                except Exception:
                    self.objects = {}
            print(f"Info: Geom loaded from cache ({self.model.hash_id})")
            return True
        print(f"Warning: Geom not found in cache ({self.model.hash_id})")
        return False

    def _save_to_cache(self) -> None:
        """Save the current Geom state to cache."""
        GEOM_CACHE.set(self.model.hash_id, self)

    def __merge_all_meshes(self):
        """
        合并模型中的所有网格对象为一个全局几何体，并执行：
        - 顶点全局去重（跨 mesh）
        - 三角形全局去重（跨 mesh）
        - Object -> 全局三角形索引映射构建
        """

        # --------------------------
        # 全局容器
        # --------------------------
        vertex_map: Dict[Tuple[int, int, int], int] = {}
        global_vertices: List[np.ndarray] = []
        all_triangles: List[Tuple[int, int, int]] = []

        global_v_count = 0
        global_t_count = 0

        # 使用类级别的 triangle_index_map（关键修正点）
        self.triangle_index_map = {}

        # --------------------------
        # 遍历所有 mesh
        # --------------------------
        # Check if model is present
        if not self.model:
            return

        for mesh in tqdm(self.model.meshes, desc="Normalizing meshes"):
            # 深拷贝，避免污染原始模型
            verts = deepcopy(mesh.vertices)
            tris  = deepcopy(mesh.triangles)

            # --------------------------
            # 顶点量化（float → int）
            # --------------------------
            verts = np.round(verts, self.acc) * (10 ** self.acc)
            verts = verts.astype(np.int64)

            # local -> global 顶点映射
            local_to_global: Dict[int, int] = {}

            for i, v in enumerate(verts):
                key = tuple(v.tolist())
                if key not in vertex_map:
                    vertex_map[key] = global_v_count
                    global_vertices.append(v)
                    global_v_count += 1
                local_to_global[i] = vertex_map[key]

            # --------------------------
            # 三角形重映射 & 全局去重
            # --------------------------
            obj_triangle_indices: List[int] = []

            for t in tris:
                gtri = (
                    local_to_global[int(t[0])],
                    local_to_global[int(t[1])],
                    local_to_global[int(t[2])]
                )

                # 忽略方向的三角形唯一键
                key = tuple(sorted(gtri))

                if key not in self.triangle_index_map:
                    self.triangle_index_map[key] = global_t_count
                    all_triangles.append(gtri)
                    global_t_count += 1

                obj_triangle_indices.append(self.triangle_index_map[key])

            # --------------------------
            # 构建 Object
            # --------------------------
            obj = Object()
            obj.update(
                tri_id2geom_tri_id=np.asarray(obj_triangle_indices, dtype=np.int64),
                id=mesh.id,
                color=deepcopy(mesh.color),
                acc=self.acc,
                status=Status.NORMAL
            )

            self.objects[obj.id] = obj


        # --------------------------
        # 转为 NumPy 数组
        # --------------------------
        self.vertices  = np.asarray(global_vertices, dtype=np.int64)
        self.triangles = np.asarray(all_triangles, dtype=np.int64)
        self.status = Status.NORMAL
        


    def __sort__(
        self,
        vert_order: VerticesOrder = VerticesOrder.ZYX,
        tri_order: TrianglesOrder = TrianglesOrder.P132,
    ):
        self.status = Status.SORTING

        vertices = np.asarray(self.vertices, dtype=np.int64)
        triangles = np.asarray(self.triangles, dtype=np.int64)

        if len(vertices) == 0:
             self.status = Status.SORTED
             return

        # ===============================
        # 1. Vertex sort
        # ===============================
        if vert_order == VerticesOrder.ZYX:
            order_vertices = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
        else:
            raise NotImplementedError(f"Unsupported VerticesOrder: {vert_order}")

        self.vertices = vertices[order_vertices]

        inv_vertices = np.empty_like(order_vertices)
        inv_vertices[order_vertices] = np.arange(len(order_vertices), dtype=np.int64)
        triangles = inv_vertices[triangles]

        # ===============================
        # 2. Triangle sort
        # ===============================
        if tri_order == TrianglesOrder.P132:
            triangles = np.sort(triangles, axis=1)
            order_triangles = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
        else:
            raise NotImplementedError(f"Unsupported TrianglesOrder: {tri_order}")

        self.triangles = triangles[order_triangles]

        inv_triangles = np.empty_like(order_triangles)
        inv_triangles[order_triangles] = np.arange(len(order_triangles), dtype=np.int64)
        objects_len=len(self.objects)
        for i, obj in enumerate(tqdm(self.objects.values(), 
                                         desc="Updating triangles indices", 
                                         total=objects_len, 
                                         leave=False)):
            obj.tri_id2geom_tri_id = np.sort(inv_triangles[obj.tri_id2geom_tri_id])
            obj.triangles_ids_order = tri_order
            obj.tri_id2vert_id=self.vertices[self.triangles[obj.tri_id2geom_tri_id]]
            
            # Reconstruct local vertices and topology
            # Using dictionary for O(1) lookups instead of O(N) list search
            local_vertices_list = []
            local_vertex_map = {} # coords_tuple -> local_index
            
            local_tri_indices = []
            obj.vex_id2tri_ids = {}
            
            # Convert triangles to list for iteration (assuming it's Nx3x3 coords)
            if obj.tri_id2vert_id is None:
                obj_triangles_iter = []
            elif isinstance(obj.tri_id2vert_id, np.ndarray):
                obj_triangles_iter = obj.tri_id2vert_id.tolist()
            else:
                obj_triangles_iter = obj.tri_id2vert_id

            for tri_id, tri_coords in tqdm(enumerate(obj_triangles_iter), 
                                         desc=f"Updating object {i+1}/{objects_len}", 
                                         total=len(obj_triangles_iter), 
                                         leave=False):
                current_tri_vertex_indices = []
                
                for p_coords in tri_coords:
                    # Convert list to tuple for dictionary key
                    p_key = tuple(p_coords)
                    
                    if p_key in local_vertex_map:
                        p_id = local_vertex_map[p_key]
                    else:
                        p_id = len(local_vertices_list)
                        local_vertices_list.append(p_coords)
                        local_vertex_map[p_key] = p_id
                    
                    current_tri_vertex_indices.append(p_id)
                    obj.vex_id2tri_ids.setdefault(p_id, []).append(tri_id)
                
                local_tri_indices.append(current_tri_vertex_indices)

            # Finalize object data
            obj.vertices = np.array(local_vertices_list)
            # Update obj.triangles to be local indices instead of coordinates
            # This aligns with standard mesh representation (vertices + faces)
            obj.tri_id2vert_id = np.array(local_tri_indices, dtype=np.int64)
            for tri_id in tqdm(range(len(obj.tri_id2vert_id)), 
                                         desc=f"Updating object {i+1}/{objects_len}", 
                                         total=len(obj.tri_id2vert_id), 
                                         leave=False):
                tri=Triangle(obj,tri_id)
                obj.triangles.append(tri)
                # pass
            # sys.stdout.write(f"\rUpdating...................................repair_degenerate_triangles {i+1}/{objects_len}"+"\r")
            # sys.stdout.flush()
            # obj.repair_degenerate_triangles()
            sys.stdout.write(f"\rUpdating...................................building BVH {i+1}/{objects_len}"+"\r")
            sys.stdout.flush()
            obj.graph=obj.graph_build()
            obj.components=obj.connected_components()
            obj.ensure_bvh()
            if obj.bvh:
                obj.aabb=obj.bvh.aabb
                obj.aabb_dot=obj.bvh_dot.aabb
        
        if self.objects:
            objs = list(self.objects.values())
            # Filter objects with valid BVH
            valid_objs = [o for o in objs if o.bvh]
            if valid_objs:
                aabb = valid_objs[0].bvh.aabb
                for obj in valid_objs[1:]:
                    aabb_current = obj.bvh.aabb
                    aabb = aabb.merge(aabb_current)
                self.AABB = aabb
            else:
                self.AABB = None
            
        self.status = Status.SORTED
        
    def __build_object_contact_graph(self,parallel_acc=None):
        obj_adj_graph={}
        obj_adj_bvh_pair={}
        objs = list(self.objects.values())
        if parallel_acc is None and self.parallel_acc is not None:
            parallel_acc=self.parallel_acc
            parellel_espsilon=np.sin(np.deg2rad(parallel_acc))
        for i, obj1 in enumerate(objs):
            if obj1.aabb is None: continue
            for j in range(i + 1, len(objs)):
                obj2 = objs[j]
                if obj2.aabb is None: continue
                overlap_flag = obj1.aabb.overlap(obj2.aabb)
                if overlap_flag: 
                    pair=(obj1.id,obj2.id)if obj1.id<obj2.id else (obj2.id,obj1.id)
                    sys.stdout.write(f"\r"+"."*35+f"obj_pair:{pair} overlap=True"+"\r")
                    sys.stdout.flush()
                    
                    parellel_espsilon=np.sin(np.deg2rad(10))
                    result=query_obj_bvh(obj1,obj2)
                    if result ==[]:
                        sys.stdout.write(f"\r"+"."*35+f"obj_pair:{pair} result_len=0"+"\r")
                        sys.stdout.flush()
                        continue
                    tris_obj_1=[elem[0] for elem in result]
                    tris_obj_2=[elem[1] for elem in result]
                    tris_obj_1_unique=list(set(tris_obj_1))
                    tris_obj_2_unique=list(set(tris_obj_2))
                    sys.stdout.write(f"\r"+"."*35+f"obj_pair:{pair} result_len={len(result)}"+"\r")
                    sys.stdout.flush()
                    visualizer = IVisualizer.create()
                    visualizer.addObj(obj1,tris_obj_1_unique)
                    visualizer.addObj(obj2,tris_obj_2_unique)
                    visualizer.show()
                    obj_adj_bvh_pair[pair]=result
                    obj_adj_graph.setdefault(obj1.id,[]).append(obj2.id)
                    obj_adj_graph.setdefault(obj2.id,[]).append(obj1.id)    
        self.obj_adj_graph=obj_adj_graph
        self.obj_adj_bvh_pair=obj_adj_bvh_pair 
        
    def __build_object_contact_triangles(self):
        """
        Build or load candidate contact triangle pairs for each object pair.
        构建（或读取缓存的）“对象对 → 三角形对”接触候选结果。

        Output directory: data/workspace/{hash_id}/pair_level/angle_{A}_gap_{G}_overlap_{O}/.
        输出目录：data/workspace/{hash_id}/pair_level/angle_{A}_gap_{G}_overlap_{O}/。

        If a JSON index already exists, load it and return without recomputation.
        若目录下已存在 JSON 索引，则直接读取并返回，避免重复计算。

        Otherwise, iterate over all object pairs and triangle pairs, compute angle/gap/overlap,
        否则遍历所有对象对与三角形对，计算 angle/gap/overlap 等几何指标，

        and persist per-pair feather files plus a global JSON index.
        并为每个对象对保存 feather 文件及全局 JSON 索引文件。
        """

        # Geometry primitives used to evaluate triangle contact metrics.
        # 几何基础函数，用于计算三角形接触相关的各类度量。
        # compute_gap:           三角形间最小距离（或某种近似 gap）
        # compute_dihedral_angle:两个法向量夹角（0~90 度的无向夹角）
        # triangle_normal_vector:计算三角形法向量
        # overlap_area_3d:       估计三角形之间的重叠覆盖率 / 交叠面积（你自定义的 3D overlap 指标）        
        from .intersection import (
            compute_gap,
            compute_dihedral_angle,
            triangle_normal_vector,
            overlap_area_3d
        )

        import pandas as pd
        import os
        import json

        # Mapping from object pair to feather file name.
        # 记录每个对象对对应的 feather 文件名。
        pair_level_result = {}

        # Compute model-scoped cache directory based on model hash.
        # 基于模型 hash 计算当前阶段的缓存目录，用于隔离不同模型或不同运行。
        hash_str = self.model.hash_id if self.model else "unknown_model"

        # Build a parameterized subdirectory to separate different gating configurations.
        # 按参数组合建立子目录，区分不同门控阈值下的结果。
        output_dir = os.path.join(
            "data", "workspace", str(hash_str), "pair_level",
            f"angle_{SOFT_NORMAL_GATE_ANGLE}_gap_{MAX_GAP_FACTOR}_overlap_{OVERLAP_RATIO_THRESHOLD}"
        )

        # Ensure the directory exists before IO.
        # 确保输出目录存在（包括嵌套路径）。
        os.makedirs(output_dir, exist_ok=True)

        # Try to reuse cached JSON index if it already exists under output_dir.
        # 若 output_dir 下已有 JSON 索引，则尝试复用缓存并直接返回。
        def find_and_parse_json(output_dir):
            """
            Recursively search output_dir for the first .json file and parse it as a dict.
            在 output_dir 中递归查找第一个 .json 文件并解析为字典。

            Return None when no JSON file is found or parsing fails.
            若未找到 JSON 文件或解析失败，则返回 None。
            """
            if not os.path.exists(output_dir):
                return None

            json_file_path = None

            # Walk the directory tree and stop at the first JSON file discovered.
            # 递归遍历目录树，在找到第一个 JSON 文件后立即停止搜索。
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.json'):
                        json_file_path = os.path.join(root, file)
                        break
                if json_file_path:
                    break

            if not json_file_path:
                return None

            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"解析 {json_file_path} 时出错: {e}")
                return None

        # Try to load existing index; skip heavy computation if successful.
        # 尝试读取已有索引；若成功则跳过后续重计算。
        res = find_and_parse_json(output_dir)

        if res is not None:
            # JSON keys are stored as strings "(i, j)"; convert them back to tuple[int, int].
            # JSON 中的 key 以 "(i, j)" 字符串形式存储；此处转换回 tuple[int, int]。
            new_keys = [tuple(map(int, key.strip('()').split(', '))) for key in res.keys()]

            # Rebuild dict with tuple keys, preserving insertion order via zip.
            # 使用 zip(res.keys(), new_keys) 保持插入顺序，将字符串 key 替换为 tuple key。
            res = {new_key: res[old_key] for old_key, new_key in zip(res.keys(), new_keys)}

            # Return cached mapping and directory; no further computation is needed.
            # 直接返回缓存结果及目录路径，不再执行后续重计算。
            return res, output_dir

        # Main loop: iterate over object pairs and compute metrics for each triangle pair.
        # 主循环：遍历“对象对 → 三角形对集合”，为每个三角形对计算几何指标。
        for obj_pair, tris_ids_pairs in tqdm(
            self.obj_adj_bvh_pair.items(),
            desc=f"Updating object contact triangles",
            total=len(self.obj_adj_bvh_pair),
            leave=False
        ):
            obj1_id, obj2_id = obj_pair
            obj1 = self.objects[obj1_id]
            obj2 = self.objects[obj2_id]

            # Collect all accepted triangle-pair rows for the current object pair.
            # 收集当前对象对下所有通过门控的三角形对结果，稍后转为 DataFrame。
            rows = []

            for t1_id, t2_id in tqdm(
                tris_ids_pairs,
                desc=f"Updating object {obj1_id} and {obj2_id}",
                total=len(tris_ids_pairs),
                leave=False
            ):
                # Prepare triangle objects and ensure normals are cached.
                # 准备 Triangle 对象并确保法向量已缓存，避免重复计算。
                tri1 = obj1.triangles[t1_id]
                tri2 = obj2.triangles[t2_id]

                # Lazily compute and cache normals if missing.
                # 若法线尚未计算，则按需计算并缓存。
                if tri1.normal is None:
                    tri1.normal = triangle_normal_vector(tri1.vertices)
                if tri2.normal is None:
                    tri2.normal = triangle_normal_vector(tri2.vertices)

                # Soft gate 1: dihedral angle between triangle normals.
                # 软门控 1：基于三角形法线夹角的筛选。
                angle_deg = compute_dihedral_angle(tri1.normal, tri2.normal)

                # Discard pairs that are too non-coplanar (angle above SOFT_NORMAL_GATE_ANGLE).
                # 若夹角大于 SOFT_NORMAL_GATE_ANGLE（极端不共面），则丢弃该三角形对。
                if angle_deg > SOFT_NORMAL_GATE_ANGLE:
                    continue

                # Soft gate 2: gap threshold scaled by local feature size.
                # 软门控 2：基于局部特征尺寸缩放的 gap 阈值。
                h = np.min([tri1.min_edge, tri2.min_edge])

                # Maximum allowed gap epsilon_max = MAX_GAP_FACTOR * h.
                # 最大允许间隙 ε_max = MAX_GAP_FACTOR * h。
                max_gap_threshold = MAX_GAP_FACTOR * h

                # compute_gap may use gap_threshold for early break or search pruning.
                # compute_gap 可能使用 gap_threshold 进行提前终止或搜索范围裁剪。
                gap = compute_gap(tri1, tri2, gap_threshold=h * INITIAL_GAP_FACTOR)

                # Discard triangle pairs whose gap exceeds the scaled threshold.
                # 若 gap 超出缩放后的阈值，则丢弃该三角形对。
                if gap > max_gap_threshold:
                    continue

                # Stage 3: overlap / cover ratio computation.
                # 第三阶段：计算 overlap / cover 比例和交叠面积。
                if tri1.parametric_area_grade==QualityGrade.CRITICAL and tri2.parametric_area_grade==QualityGrade.CRITICAL:
                    cover1=0
                    cover2=0
                    intersection_area=0
                else:
                    cover1, cover2, intersection_area = overlap_area_3d(tri1, tri2)

                # Append per-pair statistics row.
                # 追加当前三角形对的统计结果。
                rows.append({
                    "tri1": t1_id,
                    "tri2": t2_id,
                    "angle": angle_deg,
                    "gap": gap,
                    "cover1": cover1,
                    "cover2": cover2,
                    "cover_max": max(cover1, cover2),
                    "intersection_area": intersection_area,

                    # area_pass marks whether the pair passes the overlap ratio threshold.
                    # area_pass 标记该三角形对是否通过覆盖率阈值判断（是否视为接触/重叠）。
                    "area_pass": max(cover1, cover2) >= OVERLAP_RATIO_THRESHOLD
                })

            # Skip this object pair if no triangle pairs survive gating.
            # 若当前对象对没有任何候选三角形对，则直接跳过。
            if rows == []:
                continue

            # Convert accumulated rows into a DataFrame for IO and downstream processing.
            # 将累积的行转换为 DataFrame，便于持久化与后续处理。
            df = pd.DataFrame(rows)

            # Persist current object-pair results into a feather file.
            # 将当前对象对的三角形对结果保存为 feather 文件。
            def sanitize(s):
                """
                Sanitize filename component by keeping only [a-zA-Z0-9-_] characters.
                清理文件名片段，仅保留字母数字与 -/_，避免非法路径字符。
                """
                return "".join([c for c in str(s) if c.isalnum() or c in ('-', '_')])

            safe_obj1 = sanitize(obj1_id)
            safe_obj2 = sanitize(obj2_id)

            # Encode gating parameters into the feather filename for traceability.
            # 在 feather 文件名中编码门控参数，便于结果追溯。
            file_name = (
                f"tri_pairs_{safe_obj1}_{safe_obj2}"
                f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
                f"_gap_{MAX_GAP_FACTOR}"
                f"_overlap_{OVERLAP_RATIO_THRESHOLD}.feather"
            )
            full_path = os.path.join(output_dir, file_name)

            # Persist per-pair statistics in feather format for efficient IO.
            # 以 feather 格式保存每个对象对的统计结果，以获得高效的读写性能。
            df.to_feather(full_path)

            # Record mapping from object pair to feather filename for the global index JSON.
            # 记录对象对到 feather 文件名的映射，用于后续生成全局 JSON 索引。
            pair_level_result[str(obj_pair)] = file_name

        # Build and save a global JSON index mapping object pairs to feather files.
        # 构建并保存全局 JSON 索引，将对象对映射到对应的 feather 文件。
        filename = (
            f"pair_level_result"
            f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
            f"_gap_{MAX_GAP_FACTOR}"
            f"_overlap_{OVERLAP_RATIO_THRESHOLD}.json"
        )

        # Use explicit UTF-8 encoding and disable ASCII escaping for robustness.
        # 使用显式 UTF-8 编码并关闭 ASCII 转义，以更好地支持中文路径或内容。
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(pair_level_result, f, indent=4, ensure_ascii=False)

        # Return the per-pair index mapping and the output directory path.
        # 返回对象对索引映射以及输出目录路径。
        return pair_level_result, output_dir

    
    def __build_object_contact_patch_level(self, pair_level_files, pair_level_dir, show: bool = False):
        """
        Build or load patch-level artifacts for all object pairs (pair-level -> patch-level).
        构建或加载所有对象对的 patch-level 产物（从 pair-level 统计结果到 patch-level 结构）。

        Design intent / Philosophy
        设计意图 / 思想
        --------------------------
        This function is a pipeline orchestration step in the higher-level workflow (Gel pipeline).
        该函数是 Gel 大流程中的一个“编排步骤”，负责将 patch-level 作为独立阶段落地与复用。
        
        Input: per-pair statistics DataFrame files (feather) produced by the pair-level stage.
        输入：pair-level 阶段生成的对象对统计 DataFrame 文件（feather）。
        
        Output: Patch instances containing sum_df / acag / patch and a reproducible cache manifest.
        输出：Patch 实例（含 sum_df / acag / patch）以及可复现的缓存清单（manifest）。

        Caching strategy
        缓存策略
        ----------------
        Patch computation is expensive (groupby + dynamic threshold + graph build + BFS).
        Patch 计算开销较大（聚合 + 动态阈值 + 构图 + BFS 连通分量）。
        
        To avoid recomputation, we use a directory-level cache bound to model hash and parameters.
        为避免重复计算，采用目录级缓存，并绑定 model hash 与参数组合。

        Cache root: data/workspace/{model_hash}/patch_level/angle_{A}_gap_{G}_overlap_{O}/
        缓存根目录：data/workspace/{model_hash}/patch_level/angle_{A}_gap_{G}_overlap_{O}/
        
        A top-level manifest meta_patch_*.json indexes all per-pair subfolders.
        顶层 manifest（meta_patch_*.json）索引所有 pair 子目录，支持批量加载。

        Integrity / validation
        完整性 / 校验
        ----------------------
        Patch.load(..., validate_hash=True) is used to detect corrupted or partial caches.
        使用 Patch.load(..., validate_hash=True) 检测缓存是否损坏或写入不完整。
        
        If a cache is invalid, the recommended behavior is per-pair fallback recompute.
        若缓存无效，推荐行为是“按 pair 回退重算”，而不是整批失败。

        Debug visualization
        调试可视化
        ------------------
        show=True enables debug-only visualization after load/build, without altering algorithmic state.
        show=True 仅用于调试展示（load/build 后可视化），不应改变算法状态与结果。

        Parameters
        参数
        ----------
        pair_level_files : dict[tuple[int,int] | str, str]
        pair_level_files：对象对 -> feather 文件名 的映射（key 可为 tuple 或 "(i,j)" 字符串）。
        
        pair_level_dir : str
        pair_level_dir：存放 pair-level feather 文件的目录路径。

        show : bool
        show：是否进行调试可视化展示（仅调试用）。

        Returns
        返回
        -------
        (patch_level_result, output_dir)
        返回：(patch_level_result, output_dir)

        patch_level_result : dict[tuple[int,int], Patch]
        patch_level_result：对象对 -> Patch 实例（已加载或已计算）。
        
        output_dir : str
        output_dir：本阶段缓存根目录（与模型 hash 和参数绑定）。
        """
        import os
        import json
        import pandas as pd

        patch_level_result = {}

        # Workspace binding: cache directory must be tied to model identity to avoid collision.
        # 工作空间绑定：缓存目录必须与模型身份（hash）绑定，避免不同模型之间缓存冲突。
        hash_str = self.model.hash_id if self.model else "unknown_model"

        # Parameter binding: angle/gap/overlap are part of cache key to avoid mixing incompatible results.
        # 参数绑定：angle/gap/overlap 属于缓存键的一部分，避免混用不兼容的结果。
        output_dir = os.path.join(
            "data", "workspace", str(hash_str), "patch_level",
            f"angle_{SOFT_NORMAL_GATE_ANGLE}_gap_{MAX_GAP_FACTOR}_overlap_{OVERLAP_RATIO_THRESHOLD}"
        )

        # Manifest file: indexes all pair_{i}_{j} subfolders for this configuration.
        # 顶层清单文件：索引该配置下所有 pair_{i}_{j} 子目录，支持批量加载。
        filename = (
            f"meta_patch_"
            f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
            f"_gap_{MAX_GAP_FACTOR}"
            f"_overlap_{OVERLAP_RATIO_THRESHOLD}.json"
        )

        # Fast path: bulk load if output_dir and manifest exist.
        # 快速路径：若缓存目录与 manifest 存在，则直接批量加载。
        if os.path.exists(output_dir):
            manifest_path = os.path.join(output_dir, filename)
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    patch_meta = json.load(f)

                # Load each cached pair patch by pair_dir.
                # 逐个按 pair_dir 加载已缓存的 Patch。
                for obj_pair_str, sub_meta in patch_meta.get("pairs", {}).items():
                    pair_dir = sub_meta["pair_dir"]
                    patch = Patch.load(root_dir=output_dir, pair_dir=pair_dir, validate_hash=True)

                    # Optional debug visualization (no algorithmic mutation expected).
                    # 可选调试可视化（不应改变算法结果）。
                    if show:
                        # NOTE: consider using ast.literal_eval to avoid eval risks.
                        # 注意：建议用 ast.literal_eval 替代 eval，避免潜在风险。
                        obj1_idx, obj2_idx = eval(obj_pair_str)
                        obj1 = self.objects[obj1_idx]
                        obj2 = self.objects[obj2_idx]
                        patch._debug_show(obj1, obj2)

                    patch_level_result[eval(obj_pair_str)] = patch

                return patch_level_result, output_dir

        # Slow path: compute Patch for each pair from pair-level feather inputs.
        # 慢速路径：对每个对象对从 pair-level feather 输入计算 Patch，并落盘缓存。
        for obj_pair, file_name in tqdm(
            pair_level_files.items(),
            desc="build_patch_graph",
            total=len(pair_level_files),
            leave=False,
        ):
            # Legacy compatibility: allow obj_pair to be "(i,j)" string.
            # 兼容旧格式：允许 obj_pair 是 "(i,j)" 字符串。
            if isinstance(obj_pair, str):
                str_elements = obj_pair.strip("()").split(",")
                obj_pair = tuple(int(elem.strip()) for elem in str_elements)

            obj1_idx, obj2_idx = obj_pair
            obj1 = self.objects[obj1_idx]
            obj2 = self.objects[obj2_idx]

            # Pair-level DataFrame is upstream artifact; Patch decides load-or-compute internally.
            # pair-level DataFrame 是上游产物；Patch 内部决定 load-or-compute（若支持缓存）。
            file_path = os.path.join(pair_level_dir, file_name)
            df = pd.read_feather(file_path)

            # Build Patch (may compute and save inside Patch depending on cache availability).
            # 构建 Patch（可能在 Patch 内部完成计算并保存缓存）。
            patch = Patch(obj1=obj1, obj2=obj2, df=df, root_dir=output_dir, show=show)
            patch_level_result[obj_pair] = patch

        # Build top-level manifest for next-run bulk loading.
        # 构建顶层 manifest，供下次运行时批量加载使用。
        sub_metas = {}
        for obj_pair, patch in patch_level_result.items():
            obj1_idx, obj2_idx = obj_pair
            pair_dir = f"pair_{obj1_idx}_{obj2_idx}"

            # Store pair_dir into Patch.meta so manifest can locate subfolders quickly.
            # 把 pair_dir 写入 Patch.meta，使 manifest 能快速定位子目录。
            patch.meta["pair_dir"] = pair_dir
            sub_metas[obj_pair] = patch.meta

        patch_meta = {
            "pairs": {str(obj_pair): sub_meta for obj_pair, sub_meta in sub_metas.items()},
            "angle": SOFT_NORMAL_GATE_ANGLE,
            "gap": MAX_GAP_FACTOR,
            "overlap": OVERLAP_RATIO_THRESHOLD,
            "path": output_dir,
        }

        # Persist manifest atomically is recommended for crash-safety.
        # 建议使用原子写入方式落盘 manifest，以增强崩溃安全性。
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(patch_meta, f, indent=4, ensure_ascii=False)

        return patch_level_result, output_dir

    def show(self, visualizer_type: Optional[VisualizerType] = DEFAULT_VISUALIZER_TYPE, **kwargs):
        visualizer = IVisualizer.create(visualizer_type)
        visualizer.add(self, **kwargs)
        visualizer.show()

def normalize_meshes(model: "Model", acc: int = GEOM_ACC, 
                     vertices_order: VerticesOrder = VerticesOrder.ZYX,
                     triangles_order: TrianglesOrder = TrianglesOrder.P132) -> Geom:
    """
    Helper function to normalize meshes in a model.
    Creates a Geom instance which performs normalization and sorting.
    """
    return Geom(model, acc=acc, vertices_order=vertices_order, triangles_order=triangles_order)
