"""
DMSlicer · Geometry Canonicalization
构建归一化几何、缓存与邻接图（Geometry Kernel: Canonical Layer）

This module defines the "canonical geometry layer" used by the geometry kernel.
It normalizes mesh data into a stable global representation and builds downstream
artifacts (object adjacency, pair-level triangle metrics, patch-level graphs).
本模块定义了几何内核所使用的“标准几何层”。
它将网格数据归一化为稳定的全局表示，并构建下游产物（对象邻接关系、三角形对级度量、面片级图结构）。

Author//作者: QilongJiang
Date//日期: 2026-02-11

========================================================================
1) Role / 本模块定位
------------------------------------------------------------------------
Geom is the canonicalization + orchestration entry for geometry processing:
Geom 是几何处理的规范化 + 调度入口：

    Model.meshes[]
        -> global vertex canonicalization (quantize + dedup)//全局顶点规范化（量化 + 去重）
        -> global triangle canonicalization (dedup as UNORIENTED facets)//全局三角形规范化（按无向面片去重）
        -> deterministic sorting (vertex + triangle)//确定性排序（顶点 + 三角形）
        -> per-object local rebuild (local vertices/faces + topology + BVH)//单对象本地重建（本地顶点 / 面片 + 拓扑 + BVH）
        -> object-pair candidate generation (AABB overlap + BVH queries)//对象对候选生成（AABB 重叠检测 + BVH 查询）
        -> pair-level metrics gating (angle/gap/overlap; cached as feather)//三角形对级度量筛选（角度 / 间隙 / 重叠；缓存为 feather）
        -> patch-level artifacts (graphs/components; cached with manifest)//面片级产物（图 / 连通分量；与清单一同缓存）

It is intentionally designed to create *stable indices* and *repeatable caches*,
which are prerequisites for patch-level expansion and contact analysis.
本模块的设计意图是创建 *稳定索引* 和 *可重复缓存*，
这是面片级扩展和接触分析的必要条件。

========================================================================
2) Pipeline overview / 全局通图（读代码的主线）
------------------------------------------------------------------------
Model (meshes[])
   |
   v
[__merge_all_meshes]
   - quantize vertices (float -> int64 grid)//全局顶点量化（浮点数 -> 整数网格）
   - global vertex dedup across meshes//跨网格全局顶点去重
   - global triangle dedup across meshes (UNORIENTED facets)//跨网格全局无向面片去重
   - build Object(id/color, tri_id2geom_tri_id)//构建对象（id/颜色，三角面片 id 到全局三角面片 id 映射）
   |
   v
[__sort__]
   - global vertex lexsort (ZYX)//全局顶点按 ZYX 字典序排序
   - remap triangles by inv_vertices//根据全局顶点索引重新映射三角形
   - per-triangle canonicalization + global triangle lexsort (P132)//每个三角面片按 P132 排序
   - remap each Object.tri_id2geom_tri_id by inv_triangles//根据全局三角形索引重新映射对象的三角面片 id 映射
   - rebuild per-object local mesh (local vertices/faces)//重建单对象本地网格（本地顶点 / 面片）
   - build per-object topology graph / components / BVH / AABB//构建单对象拓扑图 / 连通组件 / BVH / AABB
   |
   v
[__build_object_contact_graph]
   - object AABB overlap pruning//对象 AABB 重叠剪枝
   - BVH query => candidate triangle pairs per object pair//每个对象对的候选三角形对（通过 BVH 查询）
   |
   v
[__build_object_contact_triangles]  (pair-level; disk cache)
   - angle gate + gap gate + overlap gate//角度门限 + 间隙门限 + 重叠门限
   - persist feather per object-pair + JSON index//每个对象对的三角形对指标（缓存为 feather；清单索引 JSON）
   |
   v
[__build_object_contact_patch_level] (patch-level; disk cache)
   - load via manifest (fast path) OR compute Patch per pair (slow path)//通过清单加载（快速路径）或每个对象对计算 Patch（慢速路径）
   - persist meta manifest for next-run bulk load//持久化下一次运行的清单（用于批量加载）
   |
   v
Geom.patch_info : dict[(obj1,obj2)] -> Patch

========================================================================
3) Units & Quantization contract // 单位与量化协议（重要）
------------------------------------------------------------------------
- All vertices are quantized to an int64 grid for stability//所有顶点都量化到 int64 网格以确保稳定性：
    v_int = round(v_float, acc) * 10^acc  
  This reduces floating error amplification and makes hash/dedup deterministic.//所有顶点都量化到整数网格，确保哈希值和去重结果的确定性。

- The integer grid corresponds to physical units (mm) at resolution 10^-acc.//整数网格对应物理单位（mm），分辨率为 10^-acc。
  Downstream geometry computations must be consistent with this assumption.//所有下游几何计算都必须与这个假设一致。

========================================================================
4) Core invariants // 核心不变量（写注释必须一致）
------------------------------------------------------------------------
(1) vertices: np.ndarray (V,3) int64  [quantized grid]//所有顶点都量化到整数网格，确保哈希值和去重结果的确定性。
(2) triangles: np.ndarray (T,3) int64 [vertex ids; sorted per triangle in P132]//所有三角形都按 P132 排序，确保哈希值和去重结果的确定性。
(3) triangle dedup is UNORIENTED (facet occupancy)//所有三角形都去重，不考虑面片方向：
    - key = tuple(sorted(vid0, vid1, vid2))//每个三角形的顶点索引按 P132 排序，作为去重键。
    - winding / normal orientation is intentionally ignored at this stage//在这一阶段，不考虑面片的 winding / normal 方向。
    - inside/outside shells are NOT modeled here//这里不考虑内部/外部壳面。
(4) per-object local rebuild occurs in __sort__//每个对象都在 __sort__ 中重建本地网格：
    - Object.vertices: local int64 vertices//每个对象的本地顶点，索引从 0 开始。
    - Object.tri_id2vert_id: local faces (indices into Object.vertices)//每个对象的本地面片，索引指向 Object.vertices。
    - Object.triangles: list[Triangle] (per-triangle caches for later stages)//每个对象的本地面片缓存，用于后续阶段。

(5) If later stages require watertightness, inside/outside classification, or
consistent orientation, they must be implemented in a separate oriented-surface
layer. Do NOT silently change the canonical layer rules.//如果后续阶段需要 watertightness、inside/outside 分类或一致方向，必须在单独的有向表面层实现。不要静默改变canonical层规则。

========================================================================
5) Cache contract // 缓存协议（工程约束）
------------------------------------------------------------------------
Two caching layers exist//两个缓存层：

(A) In-memory heavy cache//内存heavy缓存：
    GEOM_CACHE: stores whole Geom objects to avoid repeated canonicalization.//缓存整个 Geom 对象，避免重复标准化。

(B) Workspace disk cache (reproducible artifacts)//工作空间磁盘缓存（可重复产物）：
    data/workspace/{model_hash}/pair_level/angle_{A}_gap_{G}_overlap_{O}/
      - per-pair feather files: tri-pair metrics//每个对象对的三角形对指标（缓存为 feather）
      - a JSON index mapping object-pair -> feather filename//清单索引 JSON，映射对象对到 feather 文件名

    data/workspace/{model_hash}/patch_level/angle_{A}_gap_{G}_overlap_{O}/
      - per-pair patch folders//每个对象对的 patch 文件夹
      - meta_patch_*.json manifest for bulk loading//批量加载的清单 JSON，映射对象对到 patch 文件夹

Cache correctness depends on parameter binding (angle/gap/overlap) and model hash.//缓存正确性依赖于参数绑定（角度/间隙/重叠）和模型哈希值。

========================================================================
6) Debug contract // 调试行为边界（工程约束）
------------------------------------------------------------------------
- show=True enables visualization after load/build for debugging only.//仅在调试时启用可视化。
- Debug visualization must NOT mutate algorithmic state or cached outputs.//调试可视化不能改变算法状态或缓存输出。
- Any blocking UI call should be gated by show/debug flags, otherwise the
  pipeline becomes non-batchable.//任何阻塞式 UI 调用都必须通过 show/debug 标志门控，否则管道无法批量处理。
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
from .bvh import AABB
scale = int(10 ** GEOM_ACC)                 # 坐标放大倍数
min_edge_mm = 0.5 * (10 ** (-PROCESS_ACC))  # 例如 PROCESS_ACC=2 -> 0.005mm

class Geom:
    def __init__(
        self,
        model: "Model",
        acc=GEOM_ACC,
        parallel_acc=GEOM_PARALLEL_ACC,
        vertices_order=VerticesOrder.ZYX,
        triangles_order=TrianglesOrder.P132,
        show=False
    ) -> None:
        """
        Orchestrator: Build canonical geometry + adjacency + pair/patch artifacts
        编排器：构建 canonical 全局几何 + 邻接候选 + pair/patch 产物

        High-level behavior / 总体行为
        ------------------------------
        Geom.__init__ is not a lightweight constructor; it is a pipeline entrypoint.
        It orchestrates multiple stages and attaches results onto `self`.

        Geom.__init__ 不是纯数据构造函数，而是几何内核的入口管线：
        - canonical layer: merge + sort + per-object rebuild
        - adjacency layer: object AABB overlap + BVH tri-pair candidates
        - pair-level: tri-pair metrics gating (feather + JSON cache)
        - patch-level: build/load Patch artifacts (manifest cache)

        Cache boundary / 缓存边界（非常重要）
        -----------------------------------
        There are TWO caching layers with different scopes:

        (1) In-memory heavy cache (GEOM_CACHE):
            - Stores whole Geom objects to skip canonicalization.
            - Intended for short-term reuse within a session/process.

        (2) Workspace disk cache (pair-level + patch-level):
            - Reproducible artifacts persisted under data/workspace/{model_hash}/...
            - Intended for cross-run reuse and debugging reproducibility.

        当前 __init__ 的策略是：
        - 先尝试从 GEOM_CACHE 读取（跳过 merge/sort/adjacency 等重算）
        - pair-level 和 patch-level 始终走 workspace cache 协议（命中则 load，否则 compute）

        Parameters / 参数
        -----------------
        model:
            Bound Model instance containing meshes.
        acc:
            Quantization precision (int64 grid resolution = 10^-acc mm).
        parallel_acc:
            Optional angular tolerance hint for some contact heuristics.
        vertices_order / triangles_order:
            Deterministic sorting orders for canonical indices.
        show:
            Debug visualization toggle (NOTE: some internal functions may still
            invoke visualization unconditionally; see Stage 3 notes).

        Outputs attached / 构造完成后附加的关键输出
        ----------------------------------------
        - self.vertices / self.triangles: canonical global mesh (int64)
        - self.objects: Object dict (each with local mesh, topology, BVH)
        - self.obj_adj_graph / self.obj_adj_bvh_pair: candidate adjacency artifacts
        - self.patch_info: dict[(obj1,obj2)] -> Patch
        """

        # Store configuration / 保存配置
        self.acc: int = acc
        self.parallel_acc: Optional[float] = parallel_acc
        self.vertices_order: VerticesOrder = vertices_order
        self.triangles_order: TrianglesOrder = triangles_order
        self.model: Optional["Model"] = model

        # Initialize placeholders / 初始化占位字段
        self.objects: Dict[int, Object] = {}
        self.vertices: Union[np.ndarray, List] = []
        self.triangles: Union[np.ndarray, List] = []
        self.status: Optional[Status] = None
        self.__hash_id: Optional[str] = None

        self.triangle_index_map: Dict[Tuple[int, ...], int] = {}
        self.triangles_AABB: Optional[np.ndarray] = None
        self.AABB: Optional[np.ndarray] = None

        # ------------------------------------------------------------
        # Hash id: intended to represent canonicalization configuration
        # ------------------------------------------------------------
        # hash_id includes acc + model.hash_id + order parameters.
        # This is the "ideal" cache key for canonical geometry.
        #
        # hash_id 包含 acc、model.hash_id、排序参数。
        # 设计上应当作为 canonical 结果的缓存键。
        self.gen_hash_id()

        # ============================================================
        # (A) In-memory cache branch: skip heavy canonicalization stages
        # ============================================================
        # If cached Geom exists, we restore self.__dict__ from it.
        #
        # 若命中 GEOM_CACHE，则直接恢复 Geom 状态，跳过 merge/sort/adjacency 计算。
        # 注意：workspace 缓存（pair/patch）仍可继续走命中加载流程。
        if not self._load_from_cache():
            # ========================================================
            # (B) Canonicalization pipeline (cache miss)
            # ========================================================
            # 1) merge meshes -> global vertices/triangles + Object stubs
            # 2) deterministic sort -> remap indices + rebuild per-object local mesh
            # 3) object adjacency -> candidate tri-pairs via BVH
            #
            # 该分支是 canonical 全流程：
            # - merge：量化 + 全局去重 + Object 初始映射
            # - sort：稳定排序 + inv remap + 局部重建（Triangle/graph/BVH）
            # - adjacency：AABB overlap 粗筛 + BVH tri-pair 候选
            self.__merge_all_meshes()

            # Sort & rebuild per-object local mesh structures
            # 全局排序 + Object 局部重建（生成 Triangle/graph/BVH 等）
            self.__sort__()

            # Build object-pair adjacency candidates
            # 构建对象对候选（obj_adj_graph / obj_adj_bvh_pair）
            self.__build_object_contact_graph()

            # Save to in-memory cache for reuse in current session
            # 保存到内存 cache 供本进程内复用
            self._save_to_cache()

            if show:
                # Debug show of whole geometry (visualization backend dependent)
                # 调试展示：整体几何可视化
                self.show()

        # ============================================================
        # (C) Workspace cache pipeline: pair-level -> patch-level
        # ============================================================
        # Important:
        # - pair-level and patch-level use DISK caches under data/workspace/...
        # - These stages may be expensive, but their artifacts are persisted.
        #
        # 重要：
        # - pair-level 与 patch-level 是“跨运行复用”的磁盘缓存产物
        # - 这里的返回值会绑定到 self.patch_info 供后续使用
        pair_level_files, pair_level_dir = self.__build_object_contact_triangles()

        patch_level_result, patch_level_dir = self.__build_object_contact_patch_level(
            pair_level_files,
            pair_level_dir,
            show=show
        )
        self.patch_info = patch_level_result


    @property
    def hash_id(self) -> str:
        """
        Canonicalization configuration hash (lazy)
        归一化配置 hash（延迟计算）

        This hash is intended to uniquely represent the canonical geometry configuration:
        (acc, model.hash_id, vertices_order, triangles_order).

        该 hash 用于代表 canonical 几何的配置键：
        (精度 acc, 模型 hash, 顶点排序规则, 三角排序规则)。

        NOTE / 注意
        ----------
        - This property is lazily computed to avoid unnecessary work during initialization.
        - 当前实现的 GEOM_CACHE 实际使用的是 model.hash_id 作为 key（见 _load_from_cache），
          这与本 hash_id 的“设计目标”不一致。此处用注释明确，未来建议修正缓存 key。
        """
        if self.__hash_id is None:
            self.gen_hash_id()
        return self.__hash_id


    def gen_hash_id(self) -> None:
        """
        Generate hash id for canonical configuration
        生成 canonical 配置 hash

        The hash includes:
          - self.acc
          - self.model.hash_id (identity of geometry source)
          - self.vertices_order / self.triangles_order (deterministic ordering policy)

        hash 包含：
          - 精度参数 acc
          - 模型来源标识 model.hash_id
          - 排序策略（保证索引确定性）

        Rationale / 理由
        ----------------
        If any of these change, the canonical vertices/triangles indices can change,
        so cached geometry must not be mixed across different configurations.

        若任一维度变化，canonical 的索引与结果可能变化，
        因此缓存不应混用。
        """
        self.__hash_id = str(hash((
            self.acc,
            self.model.hash_id if self.model else "None",
            self.vertices_order,
            self.triangles_order,
        )))

    def _load_from_cache(self) -> bool:
        """
        Attempt to load Geom from in-memory cache (GEOM_CACHE)
        尝试从内存缓存（GEOM_CACHE）加载 Geom

        Current behavior / 当前行为
        ---------------------------
        - Cache lookup key: self.model.hash_id
        - If hit: restore self.__dict__ from cached Geom instance.

        当前实现使用 model.hash_id 作为缓存 key：
        - 若命中：用 cached_geom.__dict__ 覆盖当前对象字段

        WARNING / 警告（重要）
        ----------------------
        This cache key ignores canonicalization configuration (acc/order).
        That means:
          - Same model.hash_id but different acc/order could incorrectly reuse cache.

        该缓存 key 忽略了 acc / 排序参数：
          - 当同一个模型以不同精度/不同排序策略构建 Geom 时，可能错误复用旧缓存。

        Recommended future fix (TODO)
        -----------------------------
        - Use self.hash_id as the cache key, not model.hash_id.
        - Or include acc/order in the GEOM_CACHE key.

        建议未来修复（TODO）：
        - GEOM_CACHE 的 key 改为 self.hash_id（或把 acc/order 纳入 key）
        """

        # NOTE: current cache key uses model.hash_id
        # 注意：当前缓存 key 使用 model.hash_id
        cached_geom = GEOM_CACHE.get(self.model.hash_id)

        if cached_geom and isinstance(cached_geom, Geom):
            # Preserve current model reference (avoid accidental model swap)
            # 保留当前 model 引用，避免被缓存对象覆盖导致引用不一致
            current_model = self.model

            # Restore full state
            # 恢复完整状态
            self.__dict__.update(cached_geom.__dict__)

            # Restore model reference explicitly
            # 显式恢复 model 引用
            self.model = current_model

            # Defensive conversion: objects may be stored as list in legacy snapshots
            # 防御性处理：某些旧缓存可能把 objects 存成 list
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
        """
        Save current Geom to in-memory cache (GEOM_CACHE)
        保存当前 Geom 到内存缓存（GEOM_CACHE）

        Current behavior / 当前行为
        ---------------------------
        - Cache key: self.model.hash_id
        - Value: the whole Geom instance (heavy object)

        WARNING / 警告
        --------------
        Using model.hash_id as key ignores acc/order configuration.
        This should be aligned with `self.hash_id` in future refactors.

        当前用 model.hash_id 作为 key，会忽略 acc/order。
        建议未来与 self.hash_id 对齐。
        """
        GEOM_CACHE.set(self.model.hash_id, self)


    def __merge_all_meshes(self):
        """
        Stage: Global canonicalization (merge + quantize + global dedup)
        阶段：全局归一化（合并 + 量化 + 全局去重）

        Role in pipeline // 本函数在全局流程中的角色
        ----------------------------------------
        This is the FIRST canonical layer stage/全局归一化第一阶段.
        It converts Model.meshes into a single global geometry representation//全局几何表示：

        - Quantize all mesh vertices onto an int64 grid (acc-controlled).//将所有 mesh 顶点量化到 int64 网格（精度受 acc 控制）。
        - Deduplicate vertices globally across all meshes./跨所有 mesh 全局去重顶点。
        - Deduplicate triangles globally across all meshes as *UNORIENTED facets*.//跨所有 mesh 全局去重三角形（无向面片，不区分绕序/法向）。
        - Build per-mesh Object records which reference global triangle IDs./为每个 mesh 构建 Object 记录，引用全局三角形 ID。

        这是整个几何内核的“入口阶段”，目标是建立稳定的全局索引：
        - 顶点：float -> int64 量化，并跨 mesh 全局唯一化
        - 三角形：跨 mesh 全局唯一化（无向面片，不区分绕序/法向）
        - Object：记录每个 mesh 的颜色/ID，并维护其三角形映射到全局 tri_id

        Inputs / 输入
        -------------
        self.model.meshes:
            each mesh provides:
              - mesh.vertices: (Nv, 3) float
              - mesh.triangles: (Nt, 3) int (local vertex indices)
              - mesh.id, mesh.color

        Outputs / 输出
        --------------
        self.vertices:  (V, 3) int64   global quantized vertices
        self.triangles: (T, 3) int64   global triangles (vertex ids; NOT sorted here)
        self.triangle_index_map: dict[facet_key -> global_tid]
            facet_key = tuple(sorted(global_vid0, global_vid1, global_vid2))
        self.objects[obj_id]: Object with
            - tri_id2geom_tri_id: (Nt,) int64  local-triangle-order -> global triangle id

        Invariants maintained / 维护的不变量
        -----------------------------------
        - Coordinates are quantized to int64 grid.
        - Triangle dedup is UNORIENTED (ignores winding/normal).
        - No topology/BVH is built here (done in __sort__ stage).

        Failure modes / 边界情况
        ------------------------
        - If self.model is None: function returns without modifying geometry.
        """

        # --------------------------
        # Global containers (canonical space)
        # 全局容器（canonical 空间）
        # --------------------------
        vertex_map: Dict[Tuple[int, int, int], int] = {}
        global_vertices: List[np.ndarray] = []
        all_triangles: List[Tuple[int, int, int]] = []

        global_v_count = 0
        global_t_count = 0

        # triangle_index_map is a *global* dedup map:
        # facet_key(sorted vertex ids) -> global triangle id
        # triangle_index_map 是全局三角形唯一化映射：
        # 无向面片键（顶点排序后的三元组） -> 全局 tri_id
        self.triangle_index_map = {}

        # --------------------------
        # Iterate meshes in the model
        # 遍历模型中的所有 mesh
        # --------------------------
        if not self.model:
            # No model bound: nothing to canonicalize
            # 未绑定模型：无需处理
            return

        for mesh in tqdm(self.model.meshes, desc="Normalizing meshes"):
            # Deepcopy to avoid mutating upstream Model data.
            # 深拷贝：避免污染上游 Model 的原始数据
            verts = deepcopy(mesh.vertices)
            tris = deepcopy(mesh.triangles)

            # --------------------------
            # 1) Vertex quantization: float -> int64 grid
            # 1) 顶点量化：float -> int64 格点
            # --------------------------
            # Contract:
            #   v_int = round(v_float, acc) * 10^acc
            # This enforces deterministic equality for dedup/hashing.
            #
            # 量化协议：
            #   v_int = round(v_float, acc) * 10^acc
            # 目的：让“顶点相等”变成可重复、可哈希的确定性判断
            verts = np.round(verts, self.acc) * (10 ** self.acc)
            verts = verts.astype(np.int64)

            # local vertex id -> global vertex id
            # 局部顶点索引 -> 全局顶点索引
            local_to_global: Dict[int, int] = {}

            for i, v in enumerate(verts):
                # Use quantized coordinate tuple as global key.
                # 以量化后的坐标 tuple 作为全局唯一键
                key = tuple(v.tolist())

                if key not in vertex_map:
                    vertex_map[key] = global_v_count
                    global_vertices.append(v)
                    global_v_count += 1

                local_to_global[i] = vertex_map[key]

            # --------------------------
            # 2) Triangle remap + global dedup (UNORIENTED facets)
            # 2) 三角形重映射 + 全局去重（无向面片）
            # --------------------------
            # Important:
            # - We ignore triangle winding/orientation here.
            # - Same vertex set => same global triangle id.
            #
            # 重要：
            # - 本阶段不区分绕序/法向，不做内外壳语义
            # - 同一组三个顶点（集合意义相同） => 视为同一面片
            obj_triangle_indices: List[int] = []

            for t in tris:
                # Remap local vertex ids to global vertex ids.
                # 把局部顶点索引映射到全局顶点索引
                gtri = (
                    local_to_global[int(t[0])],
                    local_to_global[int(t[1])],
                    local_to_global[int(t[2])]
                )

                # ------------------------------------------------------------
                # Triangle canonicalization (UNORIENTED facet)
                # 三角面片归一化（无向面片）
                #
                # We intentionally ignore triangle winding / orientation here:
                # triangles with the same 3 vertex IDs but opposite winding are
                # considered identical. This is geometric occupancy canonicalization.
                #
                # 本阶段刻意忽略三角形的绕序/法向方向：
                # 同一组三个顶点（即使绕序相反、法向相反）会被合并为同一个 tri_id。
                #
                # Rationale:
                # - Canonical layer focuses on stable indices for downstream contact
                #   candidate generation and patch expansion.
                # - Inside/outside shells are not required at this stage.
                #
                # 设计理由：
                # - 本层目标是稳定索引，为后续接触候选与 patch 扩张服务
                # - 当前不需要内外壳判定，也不要求法向一致
                # ------------------------------------------------------------
                key = tuple(sorted(gtri))

                if key not in self.triangle_index_map:
                    self.triangle_index_map[key] = global_t_count
                    all_triangles.append(gtri)
                    global_t_count += 1

                # Record object's triangle list as mapping to global tri id.
                # 记录该 object 的三角形列表（按其局部三角顺序）对应的全局 tri_id
                obj_triangle_indices.append(self.triangle_index_map[key])

            # --------------------------
            # 3) Build Object stub (will be enriched in __sort__)
            # 3) 构建 Object（在 __sort__ 阶段会进一步重建局部 mesh / topology / BVH）
            # --------------------------
            # Note:
            # - At this stage Object only stores identity/color and the mapping:
            #     local triangle order -> global tri_id
            # - Local vertices/faces/topology are NOT built here.
            #
            # 注意：
            # - 当前 Object 只保存：id/color + tri_id2geom_tri_id 映射
            # - 局部 vertices/faces、拓扑图、BVH 等均不在此阶段生成
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
        # Finalize global geometry arrays
        # 汇总为全局 NumPy 数组
        # --------------------------
        self.vertices = np.asarray(global_vertices, dtype=np.int64)
        self.triangles = np.asarray(all_triangles, dtype=np.int64)

        # Stage status update
        # 阶段状态更新
        self.status = Status.NORMAL
        

    def __sort__(
        self,
        vert_order: VerticesOrder = VerticesOrder.ZYX,
        tri_order: TrianglesOrder = TrianglesOrder.P132,
    ):
        """
        Stage: Deterministic sorting + per-object local rebuild
        阶段：确定性排序 + 每个 Object 的局部网格重建

        Role in pipeline / 本函数在全局流程中的角色
        ----------------------------------------
        This is the SECOND canonical layer stage. It makes the global geometry
        representation *deterministic* and then reconstructs per-object local mesh
        structures needed by downstream algorithms.

        本阶段的两个核心任务：
        (A) 全局几何确定性排序（稳定索引）：
            - 全局 vertices 按 ZYX lexsort 排序
            - 全局 triangles 做 canonicalization（P132）并 lexsort 排序
            - 计算 inv 映射，保证对象的引用索引同步更新
        (B) 对象级局部网格重建（为 patch/拓扑/BVH 准备）：
            - 为每个 Object 从全局 triangles 反推局部 vertices/faces
            - 建立 Triangle 实例（缓存 normal/min_edge/quality 等）
            - 构建 object 图结构、连通分量、BVH/AABB

        Inputs / 输入
        -------------
        self.vertices:  (V,3) int64  global vertices (quantized)
        self.triangles: (T,3) int64  global triangles (vertex ids)
        self.objects[*].tri_id2geom_tri_id: (Nt,) int64
            - mapping: per-object local triangle order -> global triangle id

        Outputs / 输出
        --------------
        self.vertices / self.triangles are reordered deterministically.
        Each Object is enriched with:
            - obj.vertices: (v,3) int64 local vertices (quantized)
            - obj.tri_id2vert_id: (t,3) int64 local faces (indices into obj.vertices)
            - obj.vex_id2tri_ids: dict[local_vid -> list[local_tri_id]]
            - obj.triangles: list[Triangle] (per-triangle caches)
            - obj.graph / obj.components / obj.bvh / obj.aabb ...

        Invariants maintained / 维护的不变量
        -----------------------------------
        - Global vertices order is deterministic (ZYX).
        - Global triangles order is deterministic (P132 + lexsort).
        - Object mappings are remapped consistently via inverse permutations.
        - After this stage, Object has a standard mesh representation:
            (local vertices + local faces), enabling topology and BVH.

        Failure modes / 边界情况
        ------------------------
        - If self.vertices is empty: set status SORTED and return.
        - Object rebuilding assumes obj.triangles is empty before append.
          If Object already contains Triangle instances, repeated calls will duplicate.
          （建议未来加 assert 或清理逻辑；此处用注释明确约束）
        """

        self.status = Status.SORTING

        vertices = np.asarray(self.vertices, dtype=np.int64)
        triangles = np.asarray(self.triangles, dtype=np.int64)

        if len(vertices) == 0:
            # No geometry to sort.
            # 无几何数据：直接标记排序完成
            self.status = Status.SORTED
            return

        # ============================================================
        # (1) Global vertex deterministic sort
        # (1) 全局顶点确定性排序
        # ============================================================
        # Why sort vertices?
        # - Provides deterministic global vertex IDs across runs.
        # - Stabilizes downstream caches and remapping behavior.
        #
        # 为什么要排序顶点？
        # - 让全局 vertex id 在多次运行中稳定一致（便于缓存复用、debug、对齐）
        # - 后续三角形、对象映射都依赖稳定的 vertex id
        if vert_order == VerticesOrder.ZYX:
            # Lexicographic sort by (Z, Y, X) as primary->secondary->tertiary keys.
            # 以 (Z,Y,X) 作为排序主键顺序（适配你的切片轴向习惯：先Z再Y再X）
            order_vertices = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
        else:
            raise NotImplementedError(f"Unsupported VerticesOrder: {vert_order}")

        # Apply vertex permutation
        # 应用顶点排列
        self.vertices = vertices[order_vertices]

        # Build inverse permutation:
        # inv_vertices[old_vid] = new_vid
        #
        # 构建逆置换映射：
        # inv_vertices[旧顶点索引] = 新顶点索引
        inv_vertices = np.empty_like(order_vertices)
        inv_vertices[order_vertices] = np.arange(len(order_vertices), dtype=np.int64)

        # Remap triangles to new vertex IDs
        # 把 triangles 内的顶点索引映射到“排序后的新顶点索引”
        triangles = inv_vertices[triangles]

        # ============================================================
        # (2) Global triangle canonicalization + deterministic sort
        # (2) 全局三角形归一化（P132）+ 确定性排序
        # ============================================================
        # TrianglesOrder.P132 means:
        # - within each triangle, sort vertex ids ascending (canonical form)
        # - then globally lexsort triangles by (v0, v1, v2)
        #
        # P132 的含义：
        # - 每个三角形内部先排序：保证同一面片的表示唯一（canonical form）
        # - 再对所有三角形做 lexsort：使全局 tri_id 稳定可复现
        if tri_order == TrianglesOrder.P132:
            # Per-triangle canonicalization (note: still UNORIENTED representation)
            # 三角形内部排序（注意：仍然是无向面片表达，不含绕序语义）
            triangles = np.sort(triangles, axis=1)

            # Global deterministic triangle order
            # 按 (v0,v1,v2) 排序获取全局三角形顺序
            order_triangles = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
        else:
            raise NotImplementedError(f"Unsupported TrianglesOrder: {tri_order}")

        # Apply triangle permutation
        # 应用三角形排列
        self.triangles = triangles[order_triangles]

        # Build inverse permutation:
        # inv_triangles[old_tid] = new_tid
        #
        # 构建三角形逆置换映射：
        # inv_triangles[旧 tri_id] = 新 tri_id
        inv_triangles = np.empty_like(order_triangles)
        inv_triangles[order_triangles] = np.arange(len(order_triangles), dtype=np.int64)

        # ============================================================
        # (3) Remap each Object's triangle references + rebuild local mesh
        # (3) 更新每个 Object 的三角引用 + 重建局部网格
        # ============================================================
        objects_len = len(self.objects)

        for i, obj in enumerate(
            tqdm(
                self.objects.values(),
                desc="Updating triangles indices",
                total=objects_len,
                leave=False
            )
        ):
            # --------------------------------------------------------
            # (3.1) Remap object's global triangle IDs (by inv_triangles)
            # (3.1) 用 inv_triangles 更新对象引用的全局 tri_id
            # --------------------------------------------------------
            # obj.tri_id2geom_tri_id: per-object triangle list -> global tri_id
            # After global triangle sorting, we must remap to new tri IDs.
            #
            # obj.tri_id2geom_tri_id：对象的三角列表（按对象内部顺序）-> 全局 tri_id
            # 因为全局 triangles 被重新排序了，所以必须重映射成新的 tri_id
            obj.tri_id2geom_tri_id = np.sort(inv_triangles[obj.tri_id2geom_tri_id])
            obj.triangles_ids_order = tri_order

            # --------------------------------------------------------
            # (3.2) Pull global triangles -> coordinate triples (temporary form)
            # (3.2) 从全局 triangles 拉取坐标三元组（临时态）
            # --------------------------------------------------------
            # IMPORTANT (semantic transition point):
            #
            # Here we temporarily set obj.tri_id2vert_id to *coordinates*:
            #     obj.tri_id2vert_id = global_vertices[ global_faces[tri_ids] ]
            #
            # Later in this function, we will overwrite obj.tri_id2vert_id to become
            # the FINAL standard mesh form: local faces (indices into obj.vertices).
            #
            # 重要（语义变换点）：
            # - 这里先把 obj.tri_id2vert_id 临时设置为“坐标三元组”（三角形的3个点坐标）
            # - 在本函数后段，会将 obj.tri_id2vert_id 覆盖为最终标准形式：
            #     “局部 face 索引（指向 obj.vertices）”
            #
            # 之所以这样做，是为了先基于坐标去建立局部唯一顶点表（去重 + 建映射）。
            obj.tri_id2vert_id = self.vertices[self.triangles[obj.tri_id2geom_tri_id]]

            # --------------------------------------------------------
            # (3.3) Rebuild local vertices/faces for Object
            # (3.3) 重建 Object 的局部 vertices / faces
            # --------------------------------------------------------
            # Motivation:
            # - Downstream topology (one-ring adjacency), connected components,
            #   BVH build, and per-triangle caches operate more naturally on a local mesh.
            #
            # 动机：
            # - 后续拓扑(one-ring)、连通分量、BVH、Triangle 缓存都更适合在局部 mesh 上做
            #
            # Data structures produced:
            # - local_vertices_list: list of unique vertex coords (quantized int64)
            # - local_vertex_map: coords_tuple -> local_vid
            # - local_tri_indices: list of faces using local_vid
            # - obj.vex_id2tri_ids: local_vid -> list[local_tri_id]  (vertex-to-triangle adjacency)
            local_vertices_list = []
            local_vertex_map = {}  # coords_tuple -> local_index
            local_tri_indices = []

            # Reset adjacency map for this rebuild
            # 重建前清空/重置局部邻接映射
            obj.vex_id2tri_ids = {}

            # Normalize iterator type (ndarray -> list) for safe iteration
            # 统一迭代对象类型，避免 numpy/列表混用导致的边界问题
            if obj.tri_id2vert_id is None:
                obj_triangles_iter = []
            elif isinstance(obj.tri_id2vert_id, np.ndarray):
                obj_triangles_iter = obj.tri_id2vert_id.tolist()
            else:
                obj_triangles_iter = obj.tri_id2vert_id

            # Build local vertices and faces
            # 构建局部顶点表与局部面索引
            for tri_id, tri_coords in tqdm(
                enumerate(obj_triangles_iter),
                desc=f"Updating object {i+1}/{objects_len}",
                total=len(obj_triangles_iter),
                leave=False
            ):
                current_tri_vertex_indices = []

                for p_coords in tri_coords:
                    # Convert coords to tuple for hash key
                    # 坐标转 tuple 作为哈希键
                    p_key = tuple(p_coords)

                    # Deduplicate local vertices by coordinates
                    # 按坐标去重生成局部顶点 id
                    if p_key in local_vertex_map:
                        p_id = local_vertex_map[p_key]
                    else:
                        p_id = len(local_vertices_list)
                        local_vertices_list.append(p_coords)
                        local_vertex_map[p_key] = p_id

                    current_tri_vertex_indices.append(p_id)

                    # vertex -> triangles adjacency
                    # 顶点到三角形的邻接表（用于 one-ring / 拓扑扩张）
                    obj.vex_id2tri_ids.setdefault(p_id, []).append(tri_id)

                local_tri_indices.append(current_tri_vertex_indices)

            # Finalize local mesh representation:
            # - obj.vertices: (v,3) int64
            # - obj.tri_id2vert_id: (t,3) int64  local faces (FINAL form)
            #
            # 最终落地局部 mesh 表达：
            # - obj.vertices：局部顶点坐标表
            # - obj.tri_id2vert_id：局部 faces（最终态：索引指向 obj.vertices）
            obj.vertices = np.array(local_vertices_list, dtype=np.int64)
            obj.tri_id2vert_id = np.array(local_tri_indices, dtype=np.int64)

            # --------------------------------------------------------
            # (3.4) Build Triangle objects (per-triangle caches)
            # (3.4) 构建 Triangle 实例（缓存法向/最短边/质量等级等）
            # --------------------------------------------------------
            # NOTE / 工程约束：
            # - This loop appends Triangle instances.
            # - It assumes obj.triangles is empty or has been cleared.
            #
            # 注意：
            # - 此处会 append Triangle；要求 obj.triangles 为空，否则重复构建会导致重复三角对象。
            for tri_id in tqdm(
                range(len(obj.tri_id2vert_id)),
                desc=f"Updating object {i+1}/{objects_len}",
                total=len(obj.tri_id2vert_id),
                leave=False
            ):
                tri = Triangle(obj, tri_id)
                obj.triangles.append(tri)

            # --------------------------------------------------------
            # (3.5) Build topology graph / components / BVH
            # (3.5) 构建拓扑图 / 连通分量 / BVH（为后续接触候选与 patch 做准备）
            # --------------------------------------------------------
            # graph_build / connected_components typically depend on local faces + adjacency.
            # BVH depends on Triangle geometry.
            #
            # graph_build / connected_components 依赖局部 faces 与邻接信息
            # BVH 依赖 Triangle 几何数据
            sys.stdout.write(f"\rUpdating...................................building BVH {i+1}/{objects_len}" + "\r")
            sys.stdout.flush()

            obj.graph = obj.graph_build()
            obj.components = obj.connected_components()

            # ensure_bvh should create BVH and AABB for fast collision/contact queries
            # ensure_bvh 负责建立 BVH 与包围盒，用于快速接触候选查询
            obj.ensure_bvh()

            if obj.bvh:
                obj.aabb = obj.bvh.aabb
                obj.aabb_dot = obj.bvh_dot.aabb

        # ============================================================
        # (4) Build global AABB from all valid object AABBs
        # (4) 汇总所有 Object 的 AABB 得到 Geom 全局 AABB
        # ============================================================
        if self.objects:
            objs = list(self.objects.values())

            # Filter objects with valid BVH/AABB
            # 过滤掉没有 BVH 的对象（否则无法参与全局包围盒合并）
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
      
    def __build_object_contact_graph(self, parallel_acc=None):
        """
        Stage: Object-level adjacency (coarse) + BVH candidate generation
        阶段：对象级邻接（粗筛）+ BVH 生成三角形候选对

        Role in pipeline / 本函数在全局流程中的角色
        ----------------------------------------
        This stage constructs an object adjacency graph and generates candidate
        triangle pairs for each overlapping object pair.

        本阶段负责把“全局对象集合”变成“对象对候选集”：
        1) AABB overlap: object-level coarse pruning (cheap)
           使用对象 AABB 做粗筛（成本低）
        2) BVH query: triangle-level candidate generation (expensive but localized)
           对通过 AABB 粗筛的对象对，使用 BVH 生成候选 tri-pair（更精细）

        Output artifacts / 输出产物
        --------------------------
        self.obj_adj_graph:
            dict[obj_id -> list[obj_id]]
            Object adjacency graph (only if AABB overlap and BVH returns non-empty).
            对象邻接图：只有当 AABB overlap 且 BVH 查询返回非空才记录边。

        self.obj_adj_bvh_pair:
            dict[(min_id,max_id) -> list[(tri1_id, tri2_id)]]
            Candidate triangle pair list for each object pair.
            每个对象对对应的“候选三角形对”列表（注意：是候选集，不代表最终接触）。

        Important semantic note / 重要语义说明
        -------------------------------------
        - This stage does NOT decide real contact.
          It only produces *candidates* for the downstream pair-level gating:
            angle/gap/overlap -> area_pass -> patch build.
        - 本阶段不判定“真实接触”，只给下游 pair-level 过滤提供候选集。

        Debug / 可视化副作用（必须写清）
        --------------------------------
        Current implementation invokes visualizer.show() unconditionally per hit pair,
        which is a blocking UI side-effect. This is intended for debugging.
        If batch processing is required, visualization must be gated by a debug flag.

        当前实现对每个命中的对象对都会调用 visualizer.show()，这是阻塞式调试副作用。
        如果要跑批处理，应当用 show/debug 参数控制（此处仅注释约束，暂不改逻辑）。

        Parameters / 参数
        -----------------
        parallel_acc:
            Optional angular tolerance for "parallelism" heuristics.
            NOTE: current code overwrites this internally (hard-coded 10 deg).
            平行角容差：当前代码内部会被覆盖为 10 度（调试残留/策略固定化）。
        """

        obj_adj_graph = {}
        obj_adj_bvh_pair = {}

        objs = list(self.objects.values())

        # ------------------------------------------------------------
        # Parallelism tolerance (heuristic; currently hard-coded)
        # 平行性容差（启发式；当前实现内部硬编码）
        # ------------------------------------------------------------
        # If parallel_acc is not explicitly passed, fall back to self.parallel_acc.
        # 如果不传 parallel_acc，则尝试使用 self.parallel_acc。
        #
        # Then compute epsilon in sine-space:
        #   epsilon = sin(deg2rad(parallel_acc))
        # 用 sin 空间的 epsilon 作为某种角度容差表达（具体使用在 query_obj_bvh 内部或未来扩展）
        if parallel_acc is None and self.parallel_acc is not None:
            parallel_acc = self.parallel_acc
            parellel_espsilon = np.sin(np.deg2rad(parallel_acc))
        else:
            parellel_espsilon = None  # may be overwritten below

        # ------------------------------------------------------------
        # Pairwise object scan (upper triangular)
        # 对象两两扫描（上三角遍历）
        # ------------------------------------------------------------
        # Complexity note:
        # - O(N^2) object pairs worst-case.
        # - AABB overlap is the critical cheap prune to keep it practical.
        #
        # 复杂度说明：
        # - 最坏 O(N^2) 对象对
        # - AABB overlap 粗筛是降低实际耗时的关键
        for i, obj1 in enumerate(objs):
            if obj1.aabb is None:
                continue

            for j in range(i + 1, len(objs)):
                obj2 = objs[j]
                if obj2.aabb is None:
                    continue

                # --------------------------------------------------------
                # (1) Coarse prune: AABB overlap
                # (1) 粗筛：对象 AABB 是否重叠
                # --------------------------------------------------------
                # If AABBs do not overlap, objects cannot contact.
                # 若 AABB 不重叠，则不可能接触/相交（直接跳过）
                overlap_flag = obj1.aabb.overlap(obj2.aabb)
                if not overlap_flag:
                    continue

                # Normalize pair key as (min_id, max_id) for stable dict indexing.
                # 统一对象对 key：保证字典索引稳定（min_id, max_id）
                pair = (obj1.id, obj2.id) if obj1.id < obj2.id else (obj2.id, obj1.id)

                sys.stdout.write(f"\r" + "." * 35 + f"obj_pair:{pair} overlap=True" + "\r")
                sys.stdout.flush()

                # --------------------------------------------------------
                # (2) Triangle candidate generation via BVH query
                # (2) 使用 BVH 查询生成候选三角形对
                # --------------------------------------------------------
                # WARNING:
                # - This returns CANDIDATE triangle pairs only.
                # - Downstream pair-level stage will apply angle/gap/overlap gates.
                #
                # 警告：
                # - 此处返回的是“候选 tri-pair”，不代表最终接触
                # - 真正判定依赖后续 pair-level 的 angle/gap/overlap 过滤
                #
                # NOTE:
                # Current code hard-codes parallel epsilon to 10 degrees:
                #   parellel_espsilon = sin(10°)
                # This overrides the earlier computed tolerance.
                #
                # 注意：
                # 当前代码把 parellel_espsilon 固定为 10°（sin 空间），覆盖传入参数。
                parellel_espsilon = np.sin(np.deg2rad(10))

                result = query_obj_bvh(obj1, obj2)

                # If BVH yields no triangle pairs, treat as non-adjacent and skip.
                # BVH 若无候选 tri-pair，则认为该对象对无需进入下游流程
                if result == []:
                    sys.stdout.write(f"\r" + "." * 35 + f"obj_pair:{pair} result_len=0" + "\r")
                    sys.stdout.flush()
                    continue

                # Extract triangle IDs for optional visualization and logging.
                # 提取 tri_id 列表（用于可视化与日志）
                tris_obj_1 = [elem[0] for elem in result]
                tris_obj_2 = [elem[1] for elem in result]
                tris_obj_1_unique = list(set(tris_obj_1))
                tris_obj_2_unique = list(set(tris_obj_2))

                sys.stdout.write(f"\r" + "." * 35 + f"obj_pair:{pair} result_len={len(result)}" + "\r")
                sys.stdout.flush()

                # --------------------------------------------------------
                # (3) Debug visualization (blocking side-effect)
                # (3) 调试可视化（阻塞副作用）
                # --------------------------------------------------------
                # WARNING:
                # visualizer.show() is blocking and will pause pipeline execution.
                # This is intended for debugging contact candidates.
                #
                # 警告：
                # visualizer.show() 会阻塞流程（弹窗/渲染），属于调试用途。
                # 若要跑批处理，请用 show/debug flag 将其包裹起来。
                visualizer = IVisualizer.create()
                visualizer.addObj(obj1, tris_obj_1_unique)
                visualizer.addObj(obj2, tris_obj_2_unique)
                visualizer.show()

                # --------------------------------------------------------
                # (4) Persist stage artifacts
                # (4) 落地本阶段产物：对象邻接图 + tri-pair 候选集
                # --------------------------------------------------------
                obj_adj_bvh_pair[pair] = result
                obj_adj_graph.setdefault(obj1.id, []).append(obj2.id)
                obj_adj_graph.setdefault(obj2.id, []).append(obj1.id)

        # Attach to Geom
        # 写回 Geom 实例
        self.obj_adj_graph = obj_adj_graph
        self.obj_adj_bvh_pair = obj_adj_bvh_pair
    
    def __build_object_contact_triangles(self):
        """
        Stage: Pair-level triangle-pair metrics + soft gating (disk-cached)
        阶段：对象对层面的三角形对统计 + 软门控（磁盘缓存）

        Role in pipeline / 本函数在全局流程中的角色
        ----------------------------------------
        Given object-pair candidate tri-pairs produced by Stage 3 (BVH query),
        this stage evaluates each triangle pair using a sequence of gates:

        Input:  self.obj_adj_bvh_pair[(obj1,obj2)] -> list[(tri1_id, tri2_id)]
        Output: for each object pair, a feather table recording metrics:
                  angle, gap, cover1, cover2, intersection_area, area_pass
                plus a global JSON index mapping object-pair -> feather filename.

        这是“候选集过滤层”，目标是把 BVH 候选 tri-pair（数量可能很大）通过
        角度/间隙/重叠三个层次的门控过滤，最终产出可复现的 pair-level 数据表，
        供 patch-level（连通域/扩张）使用。

        Gating philosophy / 门控思想（非常重要）
        ----------------------------------------
        Use cheap tests first, expensive tests last:
        1) angle gate:      based on triangle normals (cheap)
        2) gap gate:        distance threshold scaled by local feature size (medium)
        3) overlap/cover:   3D overlap estimation (expensive) -> area_pass

        先便宜后昂贵，避免在大规模 tri-pair 上做昂贵 overlap 计算。

        Cache contract / 缓存协议（workspace disk cache）
        ----------------------------------------
        Output directory:
          data/workspace/{model_hash}/pair_level/angle_{A}_gap_{G}_overlap_{O}/

        - Each object pair produces one feather:
            tri_pairs_{obj1}_{obj2}_angle_{A}_gap_{G}_overlap_{O}.feather
        - A global JSON index maps:
            "(obj1, obj2)" -> feather_filename

        If the JSON index already exists under the directory, this function loads it
        and returns immediately without recomputation.

        若目录下已有 JSON 索引，直接读取并返回（避免重算）。

        Outputs / 返回
        -------------
        (pair_level_result, output_dir)
        pair_level_result: dict[(obj1,obj2)-> feather_filename] (tuple keys on return)
        output_dir: the workspace directory used by this stage.

        Notes / 注意
        ------------
        - The returned mapping is used downstream by patch-level stage to load feather files.
        - This stage does NOT build patch graphs; it only computes per-tri-pair metrics.
        """

        # Geometry primitives used to evaluate triangle contact metrics.
        # 几何基础函数（角度/间隙/重叠计算）
        from .intersection import (
            compute_gap,
            compute_dihedral_angle,
            triangle_normal_vector,
            overlap_area_3d
        )

        import os
        import json
        import pandas as pd

        # Mapping: object pair -> feather filename (serialized as JSON keys)
        # 映射：对象对 -> feather 文件名（JSON 中 key 为字符串）
        pair_level_result = {}

        # Workspace binding: tie cache to model identity.
        # 工作空间绑定：缓存目录绑定 model hash，避免不同模型冲突
        hash_str = self.model.hash_id if self.model else "unknown_model"

        # Parameter binding: angle/gap/overlap are part of cache key.
        # 参数绑定：angle/gap/overlap 属于 cache key（避免阈值变化导致混用旧结果）
        output_dir = os.path.join(
            "data", "workspace", str(hash_str), "pair_level",
            f"angle_{SOFT_NORMAL_GATE_ANGLE}_gap_{MAX_GAP_FACTOR}_overlap_{OVERLAP_RATIO_THRESHOLD}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # ------------------------------------------------------------
        # Fast path: reuse cached JSON index if exists
        # 快速路径：若已有 JSON 索引则直接复用
        # ------------------------------------------------------------
        def find_and_parse_json(output_dir: str):
            """
            Recursively search output_dir for the first .json file and parse it.
            递归查找目录下第一个 .json 并解析返回 dict。
            """
            if not os.path.exists(output_dir):
                return None

            json_file_path = None
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".json"):
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

        res = find_and_parse_json(output_dir)

        if res is not None:
            # JSON keys are stored as strings "(i, j)"; convert back to tuple[int,int]
            # JSON 的 key 是 "(i, j)" 字符串形式；这里转回 tuple
            new_keys = [tuple(map(int, key.strip("()").split(", "))) for key in res.keys()]
            res = {new_key: res[old_key] for old_key, new_key in zip(res.keys(), new_keys)}
            return res, output_dir

        # ------------------------------------------------------------
        # Slow path: compute metrics per object pair and persist to feather
        # 慢速路径：逐对象对计算 tri-pair 指标，并保存 feather + JSON 索引
        # ------------------------------------------------------------
        for obj_pair, tris_ids_pairs in tqdm(
            self.obj_adj_bvh_pair.items(),
            desc="Updating object contact triangles",
            total=len(self.obj_adj_bvh_pair),
            leave=False
        ):
            obj1_id, obj2_id = obj_pair
            obj1 = self.objects[obj1_id]
            obj2 = self.objects[obj2_id]

            rows = []  # accumulate per-tri-pair metric rows

            for t1_id, t2_id in tqdm(
                tris_ids_pairs,
                desc=f"Updating object {obj1_id} and {obj2_id}",
                total=len(tris_ids_pairs),
                leave=False
            ):
                # --------------------------------------------------------
                # Prepare Triangle objects and ensure normals are cached
                # 准备 Triangle 并确保法向缓存可用（angle gate 需要）
                # --------------------------------------------------------
                tri1 = obj1.triangles[t1_id]
                tri2 = obj2.triangles[t2_id]

                # Lazy normal computation: compute once per triangle
                # 懒加载法向：只在需要时算一次，后续复用
                if tri1.normal is None:
                    tri1.normal = triangle_normal_vector(tri1.vertices)
                if tri2.normal is None:
                    tri2.normal = triangle_normal_vector(tri2.vertices)

                # --------------------------------------------------------
                # Gate-1: normal angle (cheap)
                # 门控1：法向夹角（便宜）
                # --------------------------------------------------------
                # compute_dihedral_angle returns an undirected angle in degrees
                # (typically 0~90 for absolute / folded)
                #
                # 法向夹角：无向角度（通常取绝对意义的夹角）
                angle_deg = compute_dihedral_angle(tri1.normal, tri2.normal)

                # If too non-coplanar, discard early.
                # 若夹角太大（极不共面），提前丢弃，避免后续昂贵计算
                if angle_deg > SOFT_NORMAL_GATE_ANGLE:
                    continue

                # --------------------------------------------------------
                # Gate-2: gap threshold scaled by local feature size
                # 门控2：基于局部尺度缩放的 gap 阈值（中等成本）
                # --------------------------------------------------------
                # h is the local characteristic length for this pair,
                # here chosen as min(min_edge(tri1), min_edge(tri2)).
                #
                # h 是局部特征尺度（本对三角形的最小边长下界），用于做尺度归一化，
                # 避免“同样的 gap 在大三角/小三角上意义不同”。
                h = np.min([tri1.min_edge, tri2.min_edge])

                # Maximum allowed gap for this pair:
                #   epsilon_max = MAX_GAP_FACTOR * h
                #
                # 本对三角形允许的最大间隙：
                #   gap_max = MAX_GAP_FACTOR * h
                max_gap_threshold = MAX_GAP_FACTOR * h

                # compute_gap may use a tighter threshold for early break/pruning.
                # 这里用 INITIAL_GAP_FACTOR*h 作为 compute_gap 的 early-break 阈值（更紧），
                # 目的是让 gap 计算更快结束（尤其在明显远离时）。
                gap = compute_gap(tri1, tri2, gap_threshold=h * INITIAL_GAP_FACTOR)

                # Discard if gap exceeds scaled threshold.
                # 若 gap 超过尺度化阈值，则认为不可能接触，丢弃
                if gap > max_gap_threshold:
                    continue

                # --------------------------------------------------------
                # Gate-3: overlap / cover estimation (expensive)
                # 门控3：重叠/覆盖率估计（昂贵）
                # --------------------------------------------------------
                # overlap_area_3d returns:
                #   cover1, cover2, intersection_area
                #
                # cover 通常表示“交叠面积 / 自身面积”的比例（你的实现定义为准）
                #
                # Special-case: CRITICAL triangles (degenerate / unreliable parametric area)
                # For such cases, we avoid numerical instability by forcing cover=0.
                #
                # 特殊处理：若两者 parametric_area_grade 都是 CRITICAL，
                # 说明面积/参数化质量过差，直接把 cover 置 0，避免数值病态误判。
                if (tri1.parametric_area_grade == QualityGrade.CRITICAL and
                    tri2.parametric_area_grade == QualityGrade.CRITICAL):
                    cover1 = 0
                    cover2 = 0
                    intersection_area = 0
                else:
                    cover1, cover2, intersection_area = overlap_area_3d(tri1, tri2)

                # area_pass is the final boolean gate used downstream:
                # if max(cover1, cover2) >= OVERLAP_RATIO_THRESHOLD => accept
                #
                # area_pass 是最终通过标记：下游 patch-level 会主要看这个集合
                area_pass = max(cover1, cover2) >= OVERLAP_RATIO_THRESHOLD

                # Record row for this tri-pair
                # 记录该 tri-pair 的统计数据（落盘 schema）
                rows.append({
                    "tri1": t1_id,
                    "tri2": t2_id,
                    "angle": angle_deg,
                    "gap": gap,
                    "cover1": cover1,
                    "cover2": cover2,
                    "cover_max": max(cover1, cover2),
                    "intersection_area": intersection_area,
                    "area_pass": area_pass,
                })

            # If no pair survived gating, skip this object pair.
            # 若该对象对无任何 tri-pair 通过门控，则跳过
            if rows == []:
                continue

            df = pd.DataFrame(rows)

            # ------------------------------------------------------------
            # Persist per-pair result to feather
            # 保存当前对象对结果为 feather
            # ------------------------------------------------------------
            def sanitize(s):
                """
                Sanitize filename component by keeping only [a-zA-Z0-9-_].
                清理文件名片段，仅保留字母数字与 -/_，避免非法字符。
                """
                return "".join([c for c in str(s) if c.isalnum() or c in ("-", "_")])

            safe_obj1 = sanitize(obj1_id)
            safe_obj2 = sanitize(obj2_id)

            file_name = (
                f"tri_pairs_{safe_obj1}_{safe_obj2}"
                f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
                f"_gap_{MAX_GAP_FACTOR}"
                f"_overlap_{OVERLAP_RATIO_THRESHOLD}.feather"
            )
            full_path = os.path.join(output_dir, file_name)

            # feather is chosen for fast IO and pandas compatibility
            # feather 选择理由：IO 快、pandas 友好
            df.to_feather(full_path)

            # Record mapping for global JSON index
            # 写入全局 JSON 索引（key 序列化为字符串）
            pair_level_result[str(obj_pair)] = file_name

        # ------------------------------------------------------------
        # Persist global JSON index
        # 落盘全局 JSON 索引
        # ------------------------------------------------------------
        filename = (
            f"pair_level_result"
            f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
            f"_gap_{MAX_GAP_FACTOR}"
            f"_overlap_{OVERLAP_RATIO_THRESHOLD}.json"
        )

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(pair_level_result, f, indent=4, ensure_ascii=False)

        return pair_level_result, output_dir

    
    def __build_object_contact_patch_level(self, pair_level_files, pair_level_dir, show: bool = False):
        """
        Stage: Patch-level artifacts (graph/components expansion) with disk cache
        阶段：Patch-level 产物构建（图/连通域/扩张）+ 磁盘缓存（manifest）

        Role in pipeline / 本函数在全局流程中的角色
        ----------------------------------------
        This stage converts pair-level triangle-pair statistics (feather DataFrames)
        into Patch instances. Each Patch encapsulates:
          - aggregated summary (sum_df)
          - dynamic threshold validation results
          - adjacency graph(s) and connected components
          - any patch-level metadata for reproducibility

        本阶段把 Stage 4 的 pair-level 统计（feather 表）进一步提升为 Patch 对象：
        Patch 内部通常包含汇总表、动态阈值过滤、邻接图构建、连通分量/扩张结果，
        是后续“面片级处理”的核心载体。

        Cache contract / 缓存协议（workspace disk cache）
        ----------------------------------------
        Patch computation is expensive. We use disk cache to avoid recomputation.

        Cache root:
          data/workspace/{model_hash}/patch_level/angle_{A}_gap_{G}_overlap_{O}/

        Layout:
          {root}/pair_{i}_{j}/...   (per object-pair folder; owned by Patch.save())
          {root}/meta_patch_*.json  (top-level manifest for bulk loading)

        Fast path:
          - If root exists AND meta manifest exists:
              load all patches via Patch.load(root_dir, pair_dir, validate_hash=True)

        Slow path:
          - For each object pair:
              read feather df
              Patch(obj1,obj2,df,root_dir=output_dir,show=show)  (Patch may save internally)
          - Then write meta manifest for next run.

        Integrity / 校验
        ----------------
        Patch.load(..., validate_hash=True) is used to detect corrupted/partial caches.
        If invalid, recommended behavior is per-pair fallback recompute.
        (Current implementation assumes valid cache; invalid cache will raise.)

        Debug visualization / 调试可视化
        -------------------------------
        show=True enables debug-only visualization after load/build.
        It must NOT mutate algorithmic results or cache artifacts.

        Parameters / 参数
        -----------------
        pair_level_files:
            dict[tuple[int,int] | str, str]
            object-pair -> feather filename mapping.
            (legacy: key may be "(i, j)" string)

        pair_level_dir:
            directory containing the feather files from pair-level stage.

        show:
            whether to trigger patch-level debug visualization.

        Returns / 返回
        -------------
        (patch_level_result, output_dir)
        patch_level_result:
            dict[(obj1,obj2) -> Patch]
        output_dir:
            patch-level workspace root directory.
        """

        import os
        import json
        import pandas as pd

        patch_level_result = {}

        # Workspace binding: tie cache directory to model identity.
        # 工作空间绑定：缓存目录绑定 model hash，避免不同模型混用
        hash_str = self.model.hash_id if self.model else "unknown_model"

        # Parameter binding: angle/gap/overlap belong to cache key.
        # 参数绑定：angle/gap/overlap 属于 cache key，避免阈值变化导致混用旧 patch
        output_dir = os.path.join(
            "data", "workspace", str(hash_str), "patch_level",
            f"angle_{SOFT_NORMAL_GATE_ANGLE}_gap_{MAX_GAP_FACTOR}_overlap_{OVERLAP_RATIO_THRESHOLD}"
        )

        # Top-level manifest file for bulk loading.
        # 顶层 manifest：用于下一次运行的批量加载
        filename = (
            f"meta_patch_"
            f"_angle_{SOFT_NORMAL_GATE_ANGLE}"
            f"_gap_{MAX_GAP_FACTOR}"
            f"_overlap_{OVERLAP_RATIO_THRESHOLD}.json"
        )

        # ============================================================
        # Fast path: bulk load patches if manifest exists
        # ============================================================
        # 快路径：若 manifest 存在则按清单批量加载
        if os.path.exists(output_dir):
            manifest_path = os.path.join(output_dir, filename)
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    patch_meta = json.load(f)

                # patch_meta schema (expected):
                #   {
                #     "pairs": {
                #        "(i, j)": { ... , "pair_dir": "pair_i_j", ... }
                #     },
                #     "angle": A, "gap": G, "overlap": O, "path": output_dir
                #   }
                #
                # patch_meta 结构约定如上：pairs 字典索引所有对象对的子目录
                for obj_pair_str, sub_meta in patch_meta.get("pairs", {}).items():
                    pair_dir = sub_meta["pair_dir"]

                    # validate_hash=True detects corrupted/partial cache artifacts.
                    # validate_hash=True 用于检测缓存损坏/写入不完整
                    patch = Patch.load(root_dir=output_dir, pair_dir=pair_dir, validate_hash=True)

                    # --------------------------------------------------------
                    # Debug visualization (optional; must not mutate results)
                    # 调试可视化（可选；不得影响算法结果）
                    # --------------------------------------------------------
                    if show:
                        # NOTE:
                        # Using eval on strings is unsafe in general.
                        # Here obj_pair_str is trusted because it is generated by our own manifest.
                        # If manifest may be externally modified, use ast.literal_eval instead.
                        #
                        # 注意：
                        # eval 存在安全风险；若 manifest 可能被外部修改，建议改为 ast.literal_eval。
                        obj1_idx, obj2_idx = eval(obj_pair_str)
                        obj1 = self.objects[obj1_idx]
                        obj2 = self.objects[obj2_idx]
                        patch._debug_show(obj1, obj2)

                    patch_level_result[eval(obj_pair_str)] = patch

                return patch_level_result, output_dir

        # ============================================================
        # Slow path: compute Patch per pair from pair-level feather
        # ============================================================
        # 慢路径：逐对象对从 feather 读取 df 构建 Patch
        for obj_pair, file_name in tqdm(
            pair_level_files.items(),
            desc="build_patch_graph",
            total=len(pair_level_files),
            leave=False,
        ):
            # --------------------------------------------------------
            # Legacy compatibility: obj_pair may be "(i, j)" string
            # 兼容旧格式：key 可能是 "(i, j)" 字符串
            # --------------------------------------------------------
            if isinstance(obj_pair, str):
                str_elements = obj_pair.strip("()").split(",")
                obj_pair = tuple(int(elem.strip()) for elem in str_elements)

            obj1_idx, obj2_idx = obj_pair
            obj1 = self.objects[obj1_idx]
            obj2 = self.objects[obj2_idx]

            # Pair-level DataFrame is upstream artifact (feather).
            # pair-level DataFrame 是上游产物（feather 表）
            file_path = os.path.join(pair_level_dir, file_name)
            df = pd.read_feather(file_path)

            # --------------------------------------------------------
            # Build Patch (may load-or-compute internally)
            # 构建 Patch（Patch 内部可能实现“存在则 load，否则 compute+save”）
            # --------------------------------------------------------
            # Patch is treated as the owner of per-pair cache folder:
            #   {output_dir}/pair_{obj1}_{obj2}/...
            #
            # Patch 作为每个对象对缓存目录的所有者：
            #   {output_dir}/pair_{obj1}_{obj2}/...
            patch = Patch(obj1=obj1, obj2=obj2, df=df, root_dir=output_dir, show=show)
            patch_level_result[obj_pair] = patch

        # ============================================================
        # Build and persist top-level manifest for next-run bulk loading
        # ============================================================
        # 构建并落盘顶层 manifest，供下次运行快速批量加载
        sub_metas = {}
        for obj_pair, patch in patch_level_result.items():
            obj1_idx, obj2_idx = obj_pair
            pair_dir = f"pair_{obj1_idx}_{obj2_idx}"

            # Store pair_dir into patch.meta so manifest can locate subfolders quickly.
            # 把 pair_dir 写入 patch.meta，使 manifest 能快速定位子目录
            patch.meta["pair_dir"] = pair_dir
            sub_metas[obj_pair] = patch.meta

        patch_meta = {
            "pairs": {str(obj_pair): sub_meta for obj_pair, sub_meta in sub_metas.items()},
            "angle": SOFT_NORMAL_GATE_ANGLE,
            "gap": MAX_GAP_FACTOR,
            "overlap": OVERLAP_RATIO_THRESHOLD,
            "path": output_dir,
        }

        # Persist manifest (recommended: atomic write for crash safety).
        # 落盘 manifest（建议未来使用原子写提升崩溃安全性）
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
