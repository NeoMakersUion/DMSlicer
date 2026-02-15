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
    def __init__(self,model:"Model",acc=GEOM_ACC,parallel_acc=GEOM_PARALLEL_ACC,vertices_order=VerticesOrder.ZYX,triangles_order=TrianglesOrder.P132) -> None:
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
        
        if self._load_from_cache():
            res,output_dir=self.__build_object_contact_triangles()
            res,output_dir=self.__build_object_contact_patch_level(res,output_dir)
            return
        # Try to load from cache

        # If not cached, compute
        self.__merge_all_meshes()
        self.__sort__()
        self.__build_object_contact_graph()
        self._save_to_cache()
        res,output_dir=self.__build_object_contact_triangles()
        # 
        # Save to cache
        # self.show()
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
        构建（或读取缓存的）“对象对 → 三角形对”的接触候选结果。
        输出目录: ./data/workspace/{hash_id}/pair_level/angle_x_gap_y_overlap_z/
        - 若目录下已存在 json 缓存：直接读取并返回
        - 否则：遍历 obj_adj_bvh_pair 中的对象对与三角形对，计算 angle/gap/overlap 并保存 feather + 总索引 json
        """

        # -------------------------
        # Import: 几何计算相关函数
        # -------------------------
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

        # pair_level_result: 用于记录每个 obj_pair 对应的 feather 文件名
        # 结构：{"(obj1_id, obj2_id)": "tri_pairs_obj1_obj2_....feather", ...}
        pair_level_result = {}

        # -------------------------
        # 1) 输出目录构建
        # -------------------------
        # hash_str：模型 hash_id，用于隔离不同模型/不同运行的缓存目录
        hash_str = self.model.hash_id if self.model else "unknown_model"

        # output_dir：按照参数组合建立子目录，方便区分不同门控阈值的结果
        output_dir = os.path.join(
            "data", "workspace", str(hash_str), "pair_level",
            f"angle_{SOFT_NORMAL_GATE_ANGLE}_gap_{MAX_GAP_FACTOR}_overlap_{OVERLAP_RATIO_THRESHOLD}"
        )

        # 确保目录存在（支持嵌套路径）
        os.makedirs(output_dir, exist_ok=True)

        # ⚠️ 这个 pass 没必要（不影响运行，但可删）
        pass

        # -------------------------
        # 2) 缓存读取：如果 output_dir 下已有 json，则直接读并返回
        # -------------------------
        def find_and_parse_json(output_dir):
            """
            在 output_dir 递归查找第一个 .json 文件并解析返回 dict
            找不到则返回 None
            """
            if not os.path.exists(output_dir):
                return None

            json_file_path = None

            # 递归搜索目录下任意 json（找到第一个就停）
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
                # ⚠️ BUG: 你这里写的是 'utf - 8'，中间有空格，真实编码名应为 'utf-8'
                # 建议改为 encoding='utf-8'
                with open(json_file_path, 'r', encoding='utf - 8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"解析 {json_file_path} 时出错: {e}")
                return None

        # 尝试读取缓存
        res = find_and_parse_json(output_dir)

        if res is not None:
            # 这里假设 json 的 key 是字符串形式 "(1, 2)"，你要转回 tuple(int,int)
            # 例: "(3, 5)" -> (3, 5)
            new_keys = [tuple(map(int, key.strip('()').split(', '))) for key in res.keys()]

            # 用新的 tuple key 替换旧 key
            # ⚠️ 注意：zip(res.keys(), new_keys) 默认按插入顺序对应；只要 json dump 时顺序一致一般没问题
            res = {new_key: res[old_key] for old_key, new_key in zip(res.keys(), new_keys)}

            # 直接返回缓存结果，不再重复计算
            return res, output_dir

        # -------------------------
        # 3) 主循环：遍历“对象对 → 三角形对集合”，逐 tri-pair 计算指标
        # -------------------------
        for obj_pair, tris_ids_pairs in tqdm(
            self.obj_adj_bvh_pair.items(),
            desc=f"Updating object contact triangles",
            total=len(self.obj_adj_bvh_pair),
            leave=False
        ):
            obj1_id, obj2_id = obj_pair
            obj1 = self.objects[obj1_id]
            obj2 = self.objects[obj2_id]

            # rows：用于收集当前 obj_pair 下所有通过门控的 tri-pair 结果，最后转成 DataFrame
            rows = []

            for t1_id, t2_id in tqdm(
                tris_ids_pairs,
                desc=f"Updating object {obj1_id} and {obj2_id}",
                total=len(tris_ids_pairs),
                leave=False
            ):
                # -------------------------
                # A) Triangle 对象准备与法线缓存
                # -------------------------
                # 如果 triangles dict 里没有这个 Triangle，则创建并缓存
                # 目的：避免重复构造 Triangle 与重复读取 vertices

                tri1 = obj1.triangles[t1_id]
                tri2 = obj2.triangles[t2_id]

                # 若法线未计算过，则计算并缓存
                if tri1.normal is None:
                    tri1.normal = triangle_normal_vector(tri1.vertices)
                if tri2.normal is None:
                    tri2.normal = triangle_normal_vector(tri2.vertices)

                # -------------------------
                # 1) 法线夹角门控（软门控）
                # -------------------------
                # 无向夹角：0~90°
                angle_deg = compute_dihedral_angle(tri1.normal, tri2.normal)

                # SOFT_NORMAL_GATE_ANGLE：仅剔除极端不共面（例如 >60°）
                if angle_deg > SOFT_NORMAL_GATE_ANGLE:
                    continue

                # -------------------------
                # 2) gap 门控（尺度感知阈值）
                # -------------------------
                # 局部特征长度 h：取两三角形最小边长的最小值
                # 直觉：小三角形允许的 gap 更小，大三角形允许的 gap 更大
                h = np.min([tri1.min_edge, tri2.min_edge])

                # 最大 gap 阈值 ε_max = β * h
                max_gap_threshold = MAX_GAP_FACTOR * h

                # compute_gap 内部可能是：
                # - 先用 gap_threshold 做 early break / 快速判断
                # - 或者用它限制搜索范围
                gap = compute_gap(tri1, tri2, gap_threshold=h * INITIAL_GAP_FACTOR)

                # 超出容差则丢弃
                if gap > max_gap_threshold:
                    continue

                # -------------------------
                # 3) overlap / cover 计算
                # -------------------------
                # cover1/cover2：你定义的“tri1 被 tri2 覆盖比例 / tri2 被 tri1 覆盖比例”
                # intersection_area：交叠面积（或投影交叠面积等）
                if tri1.parametric_area_grade==QualityGrade.CRITICAL and tri2.parametric_area_grade==QualityGrade.CRITICAL:
                    cover1=0
                    cover2=0
                    intersection_area=0
                else:
                    cover1, cover2, intersection_area = overlap_area_3d(tri1, tri2)

                # 保存 tri-pair 的统计结果
                rows.append({
                    "tri1": t1_id,
                    "tri2": t2_id,
                    "angle": angle_deg,
                    "gap": gap,
                    "cover1": cover1,
                    "cover2": cover2,
                    "cover_max": max(cover1, cover2),
                    "intersection_area": intersection_area,

                    # area_pass：是否达到覆盖率阈值（你最终用这个决定是否算“接触/重叠”）
                    "area_pass": max(cover1, cover2) >= OVERLAP_RATIO_THRESHOLD
                })

            # 当前 obj_pair 没有任何候选 tri-pair，则跳过
            if rows == []:
                continue

            # 结果转 DataFrame（方便保存与后处理）
            df = pd.DataFrame(rows)

            # -------------------------
            # 4) 保存当前 obj_pair 的 tri-pair 结果到 feather
            # -------------------------
            def sanitize(s):
                """
                清理文件名：只保留字母数字、-、_，避免路径字符导致保存失败
                """
                return "".join([c for c in str(s) if c.isalnum() or c in ('-', '_')])

            safe_obj1 = sanitize(obj1_id)
            safe_obj2 = sanitize(obj2_id)

            # feather 文件名携带门控参数，便于追溯
            file_name = (
                f"tri_pairs_{safe_obj1}_{safe_obj2}"
                f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
                f"_gap_{MAX_GAP_FACTOR}"
                f"_overlap_{OVERLAP_RATIO_THRESHOLD}.feather"
            )
            full_path = os.path.join(output_dir, file_name)

            # 保存（feather 读写快，适合大量数据）
            df.to_feather(full_path)

            # 记录 obj_pair -> feather 文件名（用于总索引 json）
            pair_level_result[str(obj_pair)] = file_name

            # ⚠️ 这个 pass 也没必要（可删）
            pass

        # -------------------------
        # 5) 保存总索引 pair_level_result 到 json
        # -------------------------
        filename = (
            f"pair_level_result"
            f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
            f"_gap_{MAX_GAP_FACTOR}"
            f"_overlap_{OVERLAP_RATIO_THRESHOLD}.json"
        )

        # ⚠️ 建议加 encoding='utf-8'，并 ensure_ascii=False（如果 key/路径有中文更安全）
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(pair_level_result, f, indent=4)

        # 返回：索引 dict + 输出目录
        return pair_level_result, output_dir

 
    def __build_object_contact_patch_level(self,res,output_dir):
        import os
        import pandas as pd

        patch_level_result = {}
        for obj_pair,file_name in tqdm(res.items(),desc="build_patch_graph",total=len(res),leave=False):
            if isinstance(obj_pair, str):
                obj_pair_str=obj_pair
                str_elements = obj_pair_str.strip('()').split(',')
                obj_pair = tuple(int(elem.strip()) for elem in str_elements)

            obj1,obj2=obj_pair
            obj1=self.objects[obj1]
            obj2=self.objects[obj2]
            file_path = os.path.join(output_dir, file_name)
            df=pd.read_feather(file_path)
            from .patch_level import Patch
            patch=Patch(obj1,obj2,df,show="true")
            patch_level_result[(obj1.id, obj2.id)  ] = {"patch": patch}

            
        return patch_level_result,output_dir

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
