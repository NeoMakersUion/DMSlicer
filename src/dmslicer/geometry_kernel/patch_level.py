# patch_level.py
"""
Patch Level Module (ACAG + Components + Cache)  
Patch 层模块（ACAG + 连通组件 + 缓存）

Author//作者: QilongJiang
Date//日期: 2026-02-11

This module builds patch-level artifacts for a pair of mesh objects (obj1, obj2) from pair-level triangle interaction statistics (DataFrame).  
本模块基于 pair-level 三角面片交互统计（DataFrame），为一对网格对象（obj1, obj2）构建 patch-level 产物。

The main outputs are: summary tables (sum_df), ACAG graphs (acag), and cross-linked connected components (patch).  
主要输出包括：摘要统计表（sum_df）、ACAG 图结构（acag）、跨对象连通组件结果（patch）。

ACAG stands for AdjacencyCurvatureAreaGraph, which is a per-triangle dictionary graph containing adjacency, curvature proxy, and coverage evidence.  
ACAG 是“邻接-曲率-面积图”，以 tri_id 为键，保存邻接关系、曲率代理（角度）、以及覆盖证据（面积/对端三角）等信息。

Compared to the legacy name "graph", ACAG makes semantics explicit and avoids ambiguity about what is stored in the graph node payload.  
相比旧的 “graph” 命名，ACAG 明确表达节点载荷的语义，避免 “graph” 字段含义模糊。

--------------------------------------------------------------------------------
ACAG Schema (Per Object / Per Triangle)  
ACAG 数据结构（按对象 / 按三角）

acag: Dict[obj_id, Dict[tri_id, Node]] where Node is a dict with the following minimal contract:  
acag: Dict[obj_id, Dict[tri_id, Node]]，其中 Node 字典满足以下最小契约：

- Node["adj"]: List[int]  
  Adjacency triangle IDs within the SAME object (one-ring neighbors filtered by selected tri_ids).  
  同一对象内部的邻接三角 ID（从 one-ring 邻接中过滤得到，保证邻接点在本次选中 tri_ids 内）。

- Node["deg"]: Optional[float] (degrees)  
  Max undirected normal angle against neighbors (computed from abs(dot(n0, nj))).  
  与邻接三角的最大“无向”法线夹角（使用 abs(dot(n0, nj)) 计算）。

- Node["ddeg"]: Optional[float] (degrees)  
  Max difference of deg between this triangle and its neighbors.  
  本三角与邻接三角之间 deg 的最大差值。

- Node["cover_area"]: Dict[str, Dict[str, Any]] with keys {"true","false"}  
  Coverage evidence grouped by area_pass = True/False.  
  按 area_pass=True/False 分组的覆盖证据统计。

  Required subfields:  
  必须包含的子字段：
  - cover_area[flag]["area_ratio"]: float  
    Sum of cover ratio (e.g., cover1/cover2) for this tri under the given flag.  
    对应 flag 下 cover 比例（cover1/cover2）的累计和。
  - cover_area[flag]["area_acc"]: float  
    Sum of intersection area under the given flag.  
    对应 flag 下 intersection_area 的累计和。
  - cover_area[flag]["adj_tri_ids"]: List[int]  
    Opposite-object triangle IDs that intersect with this triangle under the given flag.  
    对端对象中与该三角相交的三角 ID 列表（用于跨对象组件链接）。

Note: component linking relies on cover_area["true"]["adj_tri_ids"] being present.  
注意：跨对象组件关联依赖 cover_area["true"]["adj_tri_ids"] 字段存在（属于硬契约）。

--------------------------------------------------------------------------------
Cache Layout and Metadata Contract  
缓存目录结构与元数据契约

When root_dir is provided, Patch instances may load/save cache artifacts under:  
当提供 root_dir 时，Patch 会在以下目录结构下加载/保存缓存：

root_dir/
  pair_{obj1_id}_{obj2_id}/
    meta.json
    sum_df/ sum_df_{obj_id}.parquet
    acag/   acag_{obj_id}.npz
    patch/  patch_{obj_id}.msgpack

meta.json provides a manifest with file paths and a stored content hash.  
meta.json 提供文件清单（manifest）以及存储的内容哈希（content hash）。

The "files" section must include keys {"sum_df","acag","patch"} and map obj_id->relative path.  
meta["files"] 必须包含 {"sum_df","acag","patch"}，并按 obj_id 映射到相对路径。

If a cache directory is incomplete or corrupted, the recommended policy is to delete the whole pair directory and recompute.  
若缓存目录不完整或损坏，推荐策略是删除整个 pair 目录并重新计算（目录级原子性）。

--------------------------------------------------------------------------------
Hashing and Validation Policy  
哈希与校验策略

hash_id is a content hash derived from (sum_df + acag + patch) after canonicalization.  
hash_id 是对（sum_df + acag + patch）进行规范化（canonicalization）后的内容哈希。

The purpose of hash_id is integrity validation (detecting partial writes, schema drift, or corrupted cache).  
hash_id 的用途是完整性校验（检测部分写入、序列化漂移、缓存损坏等）。

A recommended practice is "hash what we store": compute the hash from disk-loaded artifacts (parquet/npz/msgpack) rather than from in-memory objects.  
推荐实践是“哈希落盘内容”：从磁盘读回的产物（parquet/npz/msgpack）计算 hash，而不是直接用内存对象计算。

PATCH_CACHE_VERSION (and/or policy_version) must be bumped when canonicalization rules or saved schema change.  
当规范化规则或落盘 schema 改变时，必须提升 PATCH_CACHE_VERSION（以及/或 policy_version）以避免旧缓存误用。

--------------------------------------------------------------------------------
Design Notes  
设计说明

This module separates pure data shaping helpers (summary/thresholding) from orchestration (Patch class) and from visualization (_debug_show).  
本模块将纯函数（摘要/阈值评估）与编排逻辑（Patch 类）以及可视化（_debug_show）解耦，避免副作用传播。

The Patch class is not thread-safe because it mutates internal dicts during initialization and component linking.  
Patch 类不是线程安全的，因为初始化和组件链接过程会修改内部字典状态。

Performance hotspots typically come from repeated DataFrame slicing; consider pre-grouping by tri_id when optimizing build_patch_graph.  
性能热点通常来自 DataFrame 的重复切片；优化 build_patch_graph 时建议预先按 tri_id 分组或建立索引映射。
"""


import string
from typing import Any, Dict, List, Tuple
from collections import deque
import pandas as pd
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import os
from pathlib import Path

def build_patch_graph(obj, tri_ids, df, tri_col):
    """
    Build ACAG (AdjacencyCurvatureAreaGraph) for a single object and return a dict-based graph.  
    构建单对象的 ACAG（邻接-曲率-面积图），并以 dict 图结构返回。

    This function is the ACAG materializer: it fuses (1) mesh topology (one-ring adjacency) and (2) pair-level evidence (coverage/intersection) into per-triangle node payloads.  
    本函数是 ACAG 的“落地构造器”：把（1）网格拓扑 one-ring 邻接 与（2）pair-level 覆盖/相交证据 融合为每个 tri 的节点载荷。

    Inputs / 输入:  
    - obj: mesh object providing triangles[tri_id].normal and triangles[tri_id].topology['edges'].values().  
      obj: 网格对象，需提供 triangles[tri_id].normal 与 triangles[tri_id].topology['edges'] 邻接信息。  
    - tri_ids: selected triangle IDs for this object (usually filtered by dynamic-threshold evaluation).  
      tri_ids: 当前对象参与统计的三角形 ID 列表（通常已被动态阈值筛选）。  
    - df: pair-level DataFrame containing triangle interaction evidence for the object pair.  
      df: pair-level 统计表（对象对交互证据），用于生成 cover_area 证据。  
    - tri_col: "tri1" or "tri2", indicating which column corresponds to this object in df.  
      tri_col: "tri1" 或 "tri2"，指示 df 中哪一列对应当前对象的 tri_id。  

    Required df columns / df 必备列:  
    - {tri_col, "area_pass", "intersection_area"} plus cover column ("cover1" if tri_col=="tri1", else "cover2").  
      必须包含 {tri_col, "area_pass", "intersection_area"} 以及 cover 列（tri1->cover1，tri2->cover2）。  

    Output schema / 输出结构:  
    Returns Dict[int, Dict[str, Any]] where each tri_id maps to a node dict:  
    返回 Dict[int, Dict[str, Any]]，每个 tri_id 映射到一个节点 dict：  
      node = {  
        "adj": List[int],  
        "deg": Optional[float],  
        "ddeg": Optional[float],  
        "cover_area": {  
            "true":  {"area_ratio": float, "area_acc": float, "adj_tri_ids": List[int]},  
            "false": {"area_ratio": float, "area_acc": float, "adj_tri_ids": List[int]},  
        }  
      }  

    Meaning of fields / 字段含义:  
    - adj: one-ring neighbors within the SAME object, filtered to tri_ids for consistency.  
      adj: 同一对象内部的 one-ring 邻接三角，并与 tri_ids 取交集保证一致性。  
    - deg: max undirected normal angle (degrees) against neighbors, using abs(dot(n0, nj)); values < 1° are clamped to 0 to suppress jitter.  
      deg: 与邻接三角的最大“无向”法线夹角（°），使用 abs(dot(n0, nj))；小于 1° 的角度置 0 抑制数值抖动。  
    - ddeg: max absolute difference of deg between this triangle and its neighbors; this is a curvature-like proxy, not a strict second-order curvature.  
      ddeg: 本三角 deg 与邻接 deg 的最大差值；这是曲率代理指标，不是严格二阶曲率定义。  
    - cover_area: evidence aggregated from df for this tri_id, split by area_pass True/False; adj_tri_ids stores opposite-object triangle IDs for cross-object linking later.  
      cover_area: 基于 df 对该 tri_id 的证据聚合，并按 area_pass True/False 分组；adj_tri_ids 保存对端对象三角 ID，用于后续跨对象组件链接。  

    Robustness / 健壮性:  
    - If a triangle has no valid neighbors in tri_ids, deg/ddeg are None.  
      若 tri 无有效邻接（或邻接不在 tri_ids 中），deg/ddeg 置为 None。  
    - If normals are missing (None), angle computation for that neighbor is skipped.  
      若法线为 None，则跳过该邻接的角度计算。  

    Performance notes / 性能注意:  
    - Current implementation slices df per triangle (df[df[tri_col]==tri_id]), which can be expensive for large tri_ids; prefer pre-grouping df by tri_col for optimization.  
      当前实现会对每个 tri 做一次 df 切片（df[df[tri_col]==tri_id]），tri_ids 大时开销显著；建议预先按 tri_col 分组或构建索引映射以优化。  
    - tqdm overhead is small but not zero; keep leave=False to avoid console clutter in nested pipelines.  
      tqdm 本身也有少量开销；建议 leave=False 避免嵌套流程输出污染。  

    Raises / 异常:  
    - ValueError if tri_col is not in {"tri1","tri2"}.  
      tri_col 非 {"tri1","tri2"} 时抛 ValueError。  
    """

    g = {}
    if tri_col == "tri1":
        cover_col = "cover1"
        adj_tri_col= "tri2"
    elif tri_col == "tri2":
        cover_col = "cover2"
        adj_tri_col= "tri1"
    else:
        raise ValueError(f"tri_col must be 'tri1' or 'tri2', but got {tri_col}")
    # ---------- 一阶：adj + deg + cover ----------
    for tri_id in tqdm(tri_ids,desc="build_patch_graph_deg",total=len(tri_ids),leave=False):
        df_tri = df[df[tri_col] == tri_id]

        df_true  = df_tri[df_tri['area_pass']]
        df_false = df_tri[~df_tri['area_pass']]
        adj_tri_ids_in_df_true = list(set(df_true[adj_tri_col]))
        cover_true  = {"area_ratio": df_true[cover_col].sum(),
                    "area_acc": df_true['intersection_area'].sum(),
                    "adj_tri_ids": adj_tri_ids_in_df_true}
        adj_tri_ids_in_df_false = list(set(df_false[adj_tri_col]))
        cover_false = {"area_ratio": df_false[cover_col].sum(),
                    "area_acc": df_false['intersection_area'].sum(),
                    "adj_tri_ids": adj_tri_ids_in_df_false}

        tri = obj.triangles[tri_id]
        adj_all = list(tri.topology['edges'].values())
        adj = [t for t in adj_all if t in tri_ids]

        # --- deg ---
        n0 = tri.normal
        degs = []
        for j in adj:
            nj = obj.triangles[j].normal
            if n0 is None or nj is None:
                continue
            angle = np.degrees(
                np.arccos(np.clip(np.abs(np.dot(n0, nj)), 0.0, 1.0))
            )
            angle = angle if angle >= 1.0 else 0.0
            degs.append(angle)

        g[tri_id] = {
            "adj": adj,
            "deg": float(np.max(degs)) if degs else None,
            "cover_area": {
                "true": cover_true,
                "false": cover_false
            }
        }

    # ---------- 二阶：ddeg ----------
    for tri_id, elem in tqdm(g.items(),desc="build_patch_graph_ddeg",total=len(g),leave=False):
        dds = []
        for j in elem["adj"]:
            if j in g:
                if elem["deg"] is None or g[j]["deg"] is None:
                    continue
                dd = abs(elem["deg"] - g[j]["deg"])
                dd = dd if dd >= 1.0 else 0.0
                dds.append(dd)
        elem["ddeg"] = float(np.max(dds)) if dds else None
    return g
def build_patch_graph_fast(
    obj: Any,
    tri_ids: List[int],
    df: pd.DataFrame,
    tri_col: str,
    *,
    show_progress: bool = True,
    leave: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Build ACAG (AdjacencyCurvatureAreaGraph) with pre-grouped DataFrame to avoid per-triangle slicing.  
    使用 DataFrame 预分组来构建 ACAG，避免每个 tri 都切片 df 带来的性能损耗。

    Output schema is identical to the slow version:  
    输出结构与慢版本保持一致：
      g[tri_id] = {
        "adj": List[int],
        "deg": Optional[float],
        "cover_area": {
            "true":  {"area_ratio": float, "area_acc": float, "adj_tri_ids": List[int]},
            "false": {"area_ratio": float, "area_acc": float, "adj_tri_ids": List[int]},
        },
        "ddeg": Optional[float],
      }

    Notes:  
    说明：
    - adj is one-ring adjacency from obj.triangles[tri_id].topology['edges'], filtered by tri_ids.  
      adj 来自 topology one-ring，并与 tri_ids 取交集保证一致性。
    - deg uses abs(dot(n0,nj)) and clamps angles < 1° to 0 to suppress jitter.  
      deg 使用 abs(dot(n0,nj))，并将 <1° 的角度置 0 抑制数值抖动。
    - ddeg is max |deg_i - deg_j| over neighbors (proxy metric).  
      ddeg 为邻接 deg 的最大差值（代理指标）。
    """

    # ---------- validate tri_col / 校验 tri_col ----------
    if tri_col == "tri1":
        cover_col = "cover1"  # Coverage ratio column for obj1. | obj1 的覆盖率列
        adj_tri_col = "tri2"  # Opposite-object triangle id column. | 对端对象 tri 列
    elif tri_col == "tri2":
        cover_col = "cover2"  # Coverage ratio column for obj2. | obj2 的覆盖率列
        adj_tri_col = "tri1"  # Opposite-object triangle id column. | 对端对象 tri 列
    else:
        raise ValueError(f"tri_col must be 'tri1' or 'tri2', but got {tri_col}")

    # ---------- validate df columns / 校验 df 列 ----------
    required_cols = {tri_col, "area_pass", "intersection_area", cover_col, adj_tri_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    # ---------- prepare membership and group index / 预处理 membership 与分组索引 ----------
    tri_ids_int = [int(x) for x in tri_ids]  # Ensure int ids. | 保证 tri_id 为 int
    tri_set = set(tri_ids_int)              # O(1) membership checks. | O(1) 成员判断

    # Group once: tri_id -> sub-DataFrame view. | 一次分组：tri_id -> 子表
    # sort=False keeps original order, faster. | sort=False 更快
    groups = df.groupby(tri_col, sort=False)

    # Cache triangle normals as np arrays for speed. | 缓存法线为 numpy 向量以加速
    # NOTE: we only cache normals for tri_ids; safe and bounded. | 只缓存 tri_ids 范围内法线
    normal_cache: Dict[int, np.ndarray | None] = {}
    for tid in tri_ids_int:
        try:
            n = obj.triangles[tid].normal  # Expect length-3 vector. | 期望长度为 3 的向量
            if n is None:
                normal_cache[tid] = None
            else:
                normal_cache[tid] = np.asarray(n, dtype=np.float64)
        except Exception:
            normal_cache[tid] = None

    # ---------- build nodes (adj + deg + cover_area) / 构建节点（adj + deg + cover_area） ----------
    g: Dict[int, Dict[str, Any]] = {}

    it = tri_ids_int
    pbar = tqdm(it, desc="acag:build_deg_cover", total=len(it), leave=leave) if show_progress else it
    for tri_id in pbar:
        # --- get df group quickly / O(1) 获取 group ---
        try:
            df_tri = groups.get_group(tri_id)  # Fast group lookup. | 快速 group 查询
        except KeyError:
            # No evidence rows for this tri. | 该 tri 没有 pair-level 证据
            df_tri = None

        if df_tri is None or df_tri.empty:
            # Still build adj/deg from topology; cover_area is zeros. | 仍从拓扑构建 adj/deg；cover_area 置零
            cover_true = {"area_ratio": 0.0, "area_acc": 0.0, "adj_tri_ids": []}
            cover_false = {"area_ratio": 0.0, "area_acc": 0.0, "adj_tri_ids": []}
        else:
            # Split by area_pass once. | 一次性按 area_pass 拆分
            mask = df_tri["area_pass"].to_numpy(dtype=bool, copy=False)
            df_true = df_tri[mask]
            df_false = df_tri[~mask]

            # Unique opposite-triangle IDs. | 对端三角 ID 去重
            # Using pandas unique is faster than set(list). | pandas unique 通常更快
            adj_true = df_true[adj_tri_col].dropna().astype(int).unique().tolist()
            adj_false = df_false[adj_tri_col].dropna().astype(int).unique().tolist()

            # Aggregate coverage and intersection area. | 累计覆盖与相交面积
            cover_true = {
                "area_ratio": float(df_true[cover_col].sum()),
                "area_acc": float(df_true["intersection_area"].sum()),
                "adj_tri_ids": adj_true,
            }
            cover_false = {
                "area_ratio": float(df_false[cover_col].sum()),
                "area_acc": float(df_false["intersection_area"].sum()),
                "adj_tri_ids": adj_false,
            }

        # --- adjacency from topology / 从拓扑获取邻接 ---
        tri = obj.triangles[tri_id]
        adj_all = list(tri.topology["edges"].values())  # One-ring neighbors. | one-ring 邻接
        # Filter to tri_ids for consistency. | 与 tri_ids 取交集保证一致性
        adj = [int(t) for t in adj_all if int(t) in tri_set]

        # --- deg: max normal angle over neighbors / deg：与邻接的最大法线夹角 ---
        n0 = normal_cache.get(tri_id)
        deg_val = None
        if n0 is not None and adj:
            # Compute angles; skip neighbor if normal missing. | 计算角度；邻接法线缺失则跳过
            # Using local list and max is fast enough. | 用局部列表+max 足够快
            max_deg = None
            for j in adj:
                nj = normal_cache.get(j)
                if nj is None:
                    continue
                # cos = |dot| clipped to [0,1]. | cos = |dot| 并夹到 [0,1]
                cosv = float(np.clip(abs(float(np.dot(n0, nj))), 0.0, 1.0))
                ang = float(np.degrees(np.arccos(cosv)))
                if ang < 1.0:
                    ang = 0.0
                if max_deg is None or ang > max_deg:
                    max_deg = ang
            deg_val = float(max_deg) if max_deg is not None else None

        g[tri_id] = {
            "adj": adj,
            "deg": deg_val,
            "cover_area": {"true": cover_true, "false": cover_false},
        }

    # ---------- second pass (ddeg) / 二阶统计（ddeg） ----------
    items = list(g.items())
    pbar2 = tqdm(items, desc="acag:build_ddeg", total=len(items), leave=leave) if show_progress else items
    for tri_id, elem in pbar2:
        deg_i = elem.get("deg", None)
        if deg_i is None:
            elem["ddeg"] = None
            continue
        max_dd = None
        for j in elem.get("adj", []):
            if j not in g:
                continue
            deg_j = g[j].get("deg", None)
            if deg_j is None:
                continue
            dd = abs(float(deg_i) - float(deg_j))
            if dd < 1.0:
                dd = 0.0
            if max_dd is None or dd > max_dd:
                max_dd = dd
        elem["ddeg"] = float(max_dd) if max_dd is not None else None

    return g
        
def _stable_unique(seq: Any) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for x in seq:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out

def create_summary(df: pd.DataFrame, group_field: str, cover_field: str, tri_field: str, obj: Any) -> pd.DataFrame:
    """
    Build per-triangle coverage summary for one object from pair-level statistics.
    基于 pair-level 统计表，为单个对象构建“按三角面片聚合”的覆盖摘要表。

    Inputs / 输入:
    - df: Pair-level triangle interaction table (must contain `area_pass` and the specified fields).
          三角面片对统计表（至少包含 `area_pass` 以及下面的字段名列）。
    - group_field: Triangle id column of the *current* object to aggregate on (e.g., "tri1" or "tri2").
                   当前对象的三角面片列名，用于 groupby 聚合（例如 "tri1"/"tri2"）。
    - cover_field: Coverage ratio/score column to sum (e.g., "cover1" or "cover2").
                   覆盖比例/权重列名，用于求和（例如 "cover1"/"cover2"）。
    - tri_field: Triangle id column of the *other* object, used to collect adjacency/partner triangle ids.
                 对端对象的三角面片列名，用于收集“覆盖关联的对端 tri_id 列表”（例如 "tri2"/"tri1"）。
    - obj: Object instance that can provide triangle attributes via `get_triangles_by_list`.
           对象实例，用于通过 `get_triangles_by_list` 获取三角面片属性（如 parametric_area_grade）。

    Output / 输出:
    - DataFrame with columns:
        tri_id: current object's triangle id
        cover_sum: sum of cover_field for rows passing `area_pass`
        adj_obj_list: unique list of partner triangle ids (tri_field) that contributed to the coverage
        parametric_area_grade: per-triangle area grade from mesh metadata
      返回包含以下列的 DataFrame：
        tri_id / cover_sum / adj_obj_list / parametric_area_grade

    Notes / 注意:
    - We filter by `area_pass` here to ensure summary focuses on “valid contact/overlap evidence”.
      这里先用 `area_pass` 过滤，确保摘要只统计“有效的接触/重叠证据”。
    - Sorting by `cover_sum` ascending is kept to preserve historical behavior; downstream logic may rely on it.
      按 `cover_sum` 升序排序是为了保持历史行为；下游逻辑可能依赖该顺序。
    - `_stable_unique` should preserve deterministic order to keep hashing/cache stable.
      `_stable_unique` 应保证确定性顺序，以提升哈希/缓存稳定性。
    """
    # 1) Filter to valid rows, then aggregate per triangle in current object.
    # 1）过滤有效行，然后按当前对象的 tri_id 聚合。
    summary_df = (
        df[df["area_pass"]]
        .groupby(group_field)
        .agg(
            # Sum coverage scores contributed by all valid pair rows.
            # 汇总所有有效 pair 行贡献的覆盖分数/比例。
            cover_sum=(cover_field, "sum"),
            # Collect unique partner triangle ids on the other object side.
            # 收集对端对象的三角面片 id（去重、保持稳定顺序）。
            adj_obj_list=(tri_field, lambda x: _stable_unique(x)),
        )
        .reset_index()
    )

    # 2) Sort by cover_sum (ascending) for consistent downstream behavior.
    # 2）按 cover_sum 升序排序，保持一致的下游行为。
    summary_df_sorted = summary_df.sort_values(by="cover_sum", ascending=True).reset_index(drop=True)

    # 3) Retrieve per-triangle area grade (mesh attribute) for dynamic thresholding later.
    # 3）获取每个 tri 的面积等级（mesh 属性），用于后续动态阈值判断。
    tri_list = list(summary_df_sorted[group_field])
    parametric_area_grade_list = [
        tri.parametric_area_grade for tri in obj.get_triangles_by_list(tri_list)
    ]
    summary_df_sorted["parametric_area_grade"] = parametric_area_grade_list

    # 4) Normalize the id column name to a fixed schema: tri_id.
    # 4）统一 id 列名为固定 schema：tri_id。
    first_col_name = summary_df_sorted.columns.tolist()[0]
    summary_df_sorted_renamed = summary_df_sorted.rename(columns={first_col_name: "tri_id"})

    # 5) Keep only the contract columns to reduce payload size and stabilize caching/hashing.
    # 5）只保留契约字段，减少数据体量并提升缓存/哈希稳定性。
    summary_df_sorted_renamed = summary_df_sorted_renamed[
        ["tri_id", "cover_sum", "adj_obj_list", "parametric_area_grade"]
    ]
    return summary_df_sorted_renamed


def get_dynamic_threshold(parametric_area_grade: int) -> float:
    """
    Map a triangle's parametric area grade to a coverage threshold.
    将三角面片的 parametric_area_grade 映射为覆盖阈值（用于后续判定是否“有效接触/覆盖”）。

    Rationale / 设计意图:
    - Larger triangles (higher grade) typically require a higher coverage ratio to be considered meaningful.
      面积等级越大（通常表示面片越大），需要更高的覆盖比例才认为“有效”。
    - This function is intentionally simple and deterministic to stabilize caching and debugging.
      该函数保持简单且确定性强，便于缓存一致性与调试复现。

    Args / 参数:
    - parametric_area_grade: int, discrete grade (expected 1~4, but tolerates other values).
      三角面片面积等级（通常 1~4，但允许其他值兜底）。

    Returns / 返回:
    - float: threshold in [0,1], compared against `cover_sum`.
      覆盖阈值（0~1），用于与 `cover_sum` 比较。

    Notes / 注意:
    - Grades 1/2 share the same threshold by current policy.
      当前策略中 grade=1/2 使用相同阈值。
    """
    # High grade -> stricter threshold / 等级越高 -> 阈值越严格
    if parametric_area_grade == 4:
        return 0.10
    elif parametric_area_grade == 3:
        return 0.03
    elif parametric_area_grade == 2:
        return 0.01
    else:
        # Default fallback / 兜底默认值
        return 0.01


def _build_adj_eval(adj_list: Any, other_map: Dict[int, bool]) -> Dict[int, Any]:
    """
    Build an adjacency-evaluation lookup for one triangle.
    为“当前三角形”的邻接对端三角列表构建一个评估映射。

    Purpose / 用途:
    - In patch validation, we not only evaluate the triangle itself (main_evaluation),
      but also check whether any adjacent partner triangles are evaluated True.
      在 patch 验证阶段，不仅判断该 tri 自身是否通过（main_evaluation），
      还会检查其对端邻接 tri 是否存在通过的情况（用于“邻接提升/补偿”逻辑）。
    - This helper converts `adj_obj_list` (list/set/tuple) into:
        {adj_tri_id: other_map.get(adj_tri_id, None)}
      将 `adj_obj_list` 规范化为字典映射，便于后续 any(True) 快速判断。

    Args / 参数:
    - adj_list: Any, expected to be list/tuple/set of triangle ids; may be None/NaN/invalid.
      邻接三角列表，期望为 list/tuple/set；可能为 None/NaN/异常类型。
    - other_map: Dict[int, bool], mapping from tri_id to boolean evaluation on the *other* object.
      对端对象的 tri_id -> bool 判定结果映射。

    Returns / 返回:
    - Dict[int, Any]: tri_id -> bool/None
        - bool: if tri_id exists in other_map
        - None: if tri_id missing in other_map (unknown)
      返回 tri_id -> bool/None 的映射；缺失则为 None。

    Robustness / 鲁棒性:
    - Treat None/NaN/non-iterable as empty adjacency to avoid exceptions.
      对 None/NaN/非可迭代类型直接返回空字典，避免异常。
    - Safely cast each item to int; invalid entries are skipped.
      对每个元素尝试转 int，失败则跳过。
    """
    # Guard: missing adjacency list / 保护：邻接列表缺失
    if adj_list is None or (isinstance(adj_list, float) and np.isnan(adj_list)):
        return {}
    # Guard: unexpected type / 保护：非 list/tuple/set 直接忽略
    if not isinstance(adj_list, (list, tuple, set)):
        return {}

    out: Dict[int, Any] = {}
    # Build mapping tri_id -> evaluation result on the other side
    # 构建 tri_id -> 对端评估结果 的映射
    for t in adj_list:
        try:
            tid = int(t)
        except Exception:
            # Skip uncastable ids / 跳过无法转为 int 的条目
            continue
        # Use None if not present to explicitly represent "unknown"
        # 若对端映射中不存在，写入 None 表示“未知”
        out[tid] = other_map.get(tid, None)
    return out

def validate_patch_with_dynamic_threshold(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold_fn: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate triangles with a dynamic threshold + adjacency promotion rule.
    使用“动态阈值 + 邻接提升”规则，对两侧对象的三角面片进行筛选与最终判定。

    Core idea / 核心思想:
    1) Main evaluation / 主判定：
       - Each tri is valid if its `cover_sum` exceeds a threshold determined by its `parametric_area_grade`.
         若 cover_sum > threshold(grade)，则该 tri 主判定通过。
    2) Adjacency promotion / 邻接提升：
       - Even if a tri fails the main evaluation, it can be promoted to valid if ANY adjacent tri on the *other object*
         is valid (main_evaluation == True).
         即使主判定失败，只要该 tri 的对端邻接 tri 中存在一个通过主判定，则最终判定也通过。
       - This makes the decision more robust to fragmented / noisy intersections.
         该规则用于提升对碎片化接触、噪声统计的鲁棒性。

    Inputs / 输入:
    - df1, df2: summary tables created by `create_summary()`.
      df1/df2 是 create_summary() 产出的汇总表（分别对应 obj1 / obj2）。
    - threshold_fn: callable(grade:int)->float.
      threshold_fn 为阈值策略函数（例如 get_dynamic_threshold）。

    Required columns / 必须列:
    - tri_id: triangle id.
      tri_id: 三角面片 ID。
    - cover_sum: accumulated coverage ratio (or score) for this tri.
      cover_sum: 覆盖累计值（覆盖比例/覆盖得分）。
    - adj_obj_list: list of adjacent triangles on the other object (from pair-level statistics).
      adj_obj_list: 对端对象的“相邻/覆盖”三角列表（来自 pair-level 统计）。
    - parametric_area_grade: discrete grade controlling threshold.
      parametric_area_grade: 面片面积等级，用于阈值分级。

    Returns / 返回:
    - df1_out, df2_out: copies of inputs with extra columns:
      返回 df1_out/df2_out：在原表基础上增加若干中间与最终判定字段：
      - threshold
      - main_evaluation
      - adj_obj_tri_evaluation
      - final_evaluation

    Determinism / 确定性:
    - This function is deterministic given df1/df2 content and threshold_fn.
      在 df 与阈值函数不变时，输出确定一致，利于缓存哈希稳定性。
    """
    # -----------------------------
    # 0) Schema validation / 字段校验
    # -----------------------------
    required = {"tri_id", "cover_sum", "adj_obj_list", "parametric_area_grade"}
    missing1 = required - set(df1.columns)
    missing2 = required - set(df2.columns)
    if missing1:
        raise ValueError(f"df1 missing columns: {missing1}")
    if missing2:
        raise ValueError(f"df2 missing columns: {missing2}")

    # -----------------------------
    # 1) Work on copies / 使用副本避免污染上游数据
    # -----------------------------
    df1_out = df1.copy()
    df2_out = df2.copy()

    # ----------------------------------------
    # 2) Compute per-triangle threshold / 计算每行动态阈值
    # ----------------------------------------
    # threshold depends on parametric_area_grade
    # 阈值依赖 parametric_area_grade（面积等级）
    df1_out["threshold"] = df1_out["parametric_area_grade"].apply(threshold_fn)
    df2_out["threshold"] = df2_out["parametric_area_grade"].apply(threshold_fn)

    # ----------------------------------------
    # 3) Main evaluation / 主判定：cover_sum 是否超过阈值
    # ----------------------------------------
    df1_out["main_evaluation"] = df1_out["cover_sum"] > df1_out["threshold"]
    df2_out["main_evaluation"] = df2_out["cover_sum"] > df2_out["threshold"]

    # -------------------------------------------------------------
    # 4) Build lookup maps / 构造 tri_id -> main_evaluation 的快速映射
    # -------------------------------------------------------------
    # Used for adjacency promotion rule.
    # 用于后续“邻接提升”规则的快速查询。
    df1_eval_map: Dict[int, bool] = dict(
        zip(df1_out["tri_id"].astype(int), df1_out["main_evaluation"].astype(bool))
    )
    df2_eval_map: Dict[int, bool] = dict(
        zip(df2_out["tri_id"].astype(int), df2_out["main_evaluation"].astype(bool))
    )

    # -------------------------------------------------------------------
    # 5) Evaluate adjacency list / 计算对端邻接 tri 的评估结果（bool/None）
    # -------------------------------------------------------------------
    # For each tri in df1, look up its adjacencies on obj2 using df2_eval_map.
    # 对 df1 的每个 tri，把其 adj_obj_list 映射到 df2_eval_map 的 bool/None 结果。
    df1_out["adj_obj_tri_evaluation"] = df1_out["adj_obj_list"].apply(
        lambda adj: _build_adj_eval(adj, df2_eval_map)
    )
    # Symmetric for df2.
    # df2 同理，映射到 df1_eval_map。
    df2_out["adj_obj_tri_evaluation"] = df2_out["adj_obj_list"].apply(
        lambda adj: _build_adj_eval(adj, df1_eval_map)
    )

    # --------------------------------------------------------
    # 6) Final evaluation / 最终判定：主判定 OR 任一对端邻接通过
    # --------------------------------------------------------
    # NOTE: We only promote based on "main_evaluation" of the other side, not its final_evaluation,
    # to avoid circular dependency between df1 and df2.
    # 注意：提升逻辑只参考对端的 main_evaluation（而非 final_evaluation），避免 df1/df2 互相递归依赖。
    df1_out["final_evaluation"] = df1_out.apply(
        lambda row: bool(row["main_evaluation"])
        or any(v is True for v in dict(row["adj_obj_tri_evaluation"]).values()),
        axis=1,
    )
    df2_out["final_evaluation"] = df2_out.apply(
        lambda row: bool(row["main_evaluation"])
        or any(v is True for v in dict(row["adj_obj_tri_evaluation"]).values()),
        axis=1,
    )

    # -----------------------------
    # 7) Return / 返回增强后的结果表
    # -----------------------------
    return df1_out, df2_out

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import deque
from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class Patch:
    """
    Patch artifact builder for triangle-mesh contacts between two objects.
    面向“两个对象网格接触关系”的 Patch 产物构建器（包含统计、图结构、连通组件与缓存）。

    Purpose / 用途:
    - This class orchestrates the patch-level pipeline for a given object pair (obj1, obj2).
      本类负责对某一对对象 (obj1, obj2) 执行 patch-level 的完整流程编排与产物管理。
    - It converts pair-level triangle interaction statistics (DataFrame) into reusable artifacts.
      它把 pair-level 的三角面片对统计表（DataFrame）转换为可复用的中间产物（便于后续阶段使用）。

    Input contract / 输入契约:
    - obj1/obj2 must provide: `id`, `triangles[...]` (with `normal`, `topology`), and triangle query helpers.
      obj1/obj2 需要提供：`id`、`triangles[...]`（包含 normal/topology）以及按列表取三角形的辅助接口。
    - df must include: area_pass, tri1, tri2, cover1, cover2, intersection_area (used by downstream helpers).
      df 至少需要包含：area_pass, tri1, tri2, cover1, cover2, intersection_area（供下游函数聚合/判定）。

    Artifacts / 产物（稳定数据契约）:
    - sum_df[obj_id] (DataFrame): per-triangle summary with thresholds and evaluations.
      sum_df[obj_id]（DataFrame）：每个三角形的覆盖统计与阈值判定结果（含 main/final_evaluation）。
    - acag[obj_id] (dict): Adjacency-Curvature-Area Graph per triangle.
      acag[obj_id]（dict）：ACAG 邻接-曲率-面积图，每个 tri 对应邻接与角度/覆盖信息。
      Fields / 字段:
        * adj: List[int] adjacent triangles within the selected tri set.
          adj：邻接三角形列表（限定在 tri_ids 子集内）。
        * deg: float|None, max unsigned normal angle (degrees) against adj.
          deg：与邻接三角的最大无向法线夹角（度），无邻接则 None。
        * ddeg: float|None, max difference of deg between neighbors (degrees).
          ddeg：邻接之间 deg 差异的最大值（度），不可比则 None。
        * cover_area: dict with "true"/"false" buckets and their adjacency ids & area accumulations.
          cover_area：按 area_pass 的 true/false 分桶统计（对端 tri 列表与面积累计等）。
    - patch[obj_id] (list): connected components with cross-links.
      patch[obj_id]（list）：连通组件列表，每项为 {"component":[tri...], "adj":[other_component_ids...]}。

    Pipeline / 主流程:
    1) Filter df by area_pass to keep effective contacts.
       先按 area_pass 过滤，得到有效接触三角对。
    2) Build per-object summaries via create_summary().
       分别对 obj1/obj2 生成摘要统计 sum_df。
    3) Apply dynamic threshold + adjacency promotion via validate_patch_with_dynamic_threshold().
       用动态阈值 + 邻接提升规则得到 final_evaluation，并筛选 tri_ids。
    4) Build ACAG graphs via build_patch_graph_fast() (or build_patch_graph()).
       对筛选后的 tri_ids 构建 ACAG 图（邻接/角度/覆盖信息）。
    5) Detect connected components and cross-links via component_detect().
       对两侧图分别做连通分量，并基于覆盖关系建立跨对象组件链接。

    Caching & integrity / 缓存与一致性校验:
    - If root_dir is provided, the class tries to load artifacts from disk first.
      若传入 root_dir，则优先尝试从磁盘加载该 pair 的缓存产物。
    - On load failure (missing/corrupt/mismatch), it falls back to recomputation and overwrites cache.
      若加载失败（缺失/损坏/不匹配），则回退重算并覆盖旧缓存。
    - gen_hash_id() computes a stable content hash from (sum_df, acag, patch) for integrity checks.
      gen_hash_id() 通过 (sum_df, acag, patch) 计算稳定内容哈希，用于完整性校验与复现。

    Thread-safety / 线程安全:
    - Not thread-safe; instances mutate internal dicts during build and linking.
      非线程安全：构建与链接过程会修改内部 dict/list。

    Performance notes / 性能提示:
    - Summary groupby is typically the dominant pandas cost; ACAG build is O(M+E) in Python loops.
      摘要 groupby 通常是 pandas 的主要开销；ACAG 构建为 Python 循环主导，约 O(M+E)。
    - Hashing/caching should avoid repeatedly materializing huge JSON payloads.
      哈希/缓存应避免反复构造巨大 JSON（你已经在用 fast hash 方案，这是正确方向）。
    """
    hash_id: str
    df: pd.DataFrame
    obj_pair: Tuple[int, int]
    patch: Dict[int, List[Dict[str, Any]]]
    sum_df: Dict[int, pd.DataFrame]
    acag: Dict[int, Dict[int, Dict[str, Any]]]
    # meta: saved manifest loaded from disk / 元数据：保存/加载时的 manifest

    def __init__(
        self,
        obj1: "Object",
        obj2: "Object",
        df: pd.DataFrame,
        root_dir: str | None = None,
        show: bool = False,
    ):
        """
        Build or load Patch artifacts for an object pair.
        为一对对象构建或加载 Patch 产物（优先加载缓存，失败则重算）。

        Behavior / 行为约定:
        - If root_dir is given, try loading cache from {root_dir}/pair_{id1}_{id2}/ first.
          若提供 root_dir，优先从 {root_dir}/pair_{id1}_{id2}/ 尝试加载缓存。
        - If loading fails, delete the pair directory (best-effort) and recompute artifacts.
          若加载失败，则尽量删除该 pair 目录并回退重算。
        - show=True triggers debug visualization after build/load; it must not affect artifacts.
          show=True 仅用于构建/加载后调试显示，不应影响产物内容（避免污染 hash）。

        Edge cases / 边界情况:
        - If df has no rows with area_pass==True, outputs become empty structures.
          若 df 中不存在 area_pass==True 的记录，则输出为空结构（patch/sum_df/acag）。
        """
        self.obj_pair = (obj1.id, obj2.id)
        self.df = df

        loaded = False
        if root_dir:
            import shutil
            pair_dir_name = f"pair_{obj1.id}_{obj2.id}"
            base_dir = Patch._resolve_root_dir(root_dir)
            full_pair_path = base_dir / pair_dir_name

            # Step 1: try load cache / 步骤1：尝试从缓存加载
            try:
                if full_pair_path.exists():
                    loaded_patch = Patch.load(str(base_dir), pair_dir_name, validate_hash=True)
                    self.sum_df = loaded_patch.sum_df
                    self.acag = loaded_patch.acag
                    self.patch = loaded_patch.patch
                    self.hash_id = loaded_patch.hash_id
                    self.meta = getattr(loaded_patch, "meta", {})
                    loaded = True
            except Exception:
                # Load failed: remove corrupted cache to allow clean rebuild.
                # 加载失败：删除损坏缓存，确保下次重算是“干净目录”。
                if full_pair_path.exists():
                    shutil.rmtree(full_pair_path, ignore_errors=True)

        if not loaded:
            # Step 2: compute artifacts / 步骤2：执行计算流程
            df_true = df[df["area_pass"]]
            if df_true.empty:
                # No valid contacts / 无有效接触
                self.patch = {obj1.id: [], obj2.id: []}
                self.sum_df = {}
                self.acag = {}
                return
            else:
                # 2.1 Summaries / 摘要统计
                s1 = create_summary(df_true, "tri1", "cover1", "tri2", obj1)
                s2 = create_summary(df_true, "tri2", "cover2", "tri1", obj2)

                # 2.2 Dynamic threshold validation / 动态阈值判定
                s1e, s2e = validate_patch_with_dynamic_threshold(s1, s2, get_dynamic_threshold)
                self.sum_df = {obj1.id: s1e, obj2.id: s2e}

                # 2.3 Select triangles for graph build / 选择用于建图的三角面片
                tri1 = list(s1e[s1e["final_evaluation"]]["tri_id"])
                tri2 = list(s2e[s2e["final_evaluation"]]["tri_id"])

                # 2.4 Build ACAG graphs / 构建 ACAG 图
                g1 = build_patch_graph_fast(obj1, tri1, df, tri_col="tri1", show_progress=True, leave=False)
                g2 = build_patch_graph_fast(obj2, tri2, df, tri_col="tri2", show_progress=True, leave=False)
                self.acag = {obj1.id: g1, obj2.id: g2}

                # 2.5 Build components and cross-links / 连通组件 + 跨对象链接
                self.patch = self.component_detect(g1, g2)

            # Step 3: persist if needed / 步骤3：如需则保存到缓存
            if root_dir:
                self.meta = self.save(str(base_dir))
                # NOTE: hash_id is assigned during save() from loaded files.
                # 注意：hash_id 在 save() 内部按“落盘后再加载计算”的方式赋值。
            else:
                self.hash_id = self.gen_hash_id()

        if show:
            self._debug_show(obj1, obj2)

    def gen_hash_id(self) -> str:
        """
        Compute stable content hash for (sum_df, acag, patch).
        对 (sum_df, acag, patch) 计算稳定内容哈希（用于缓存校验/一致性验证）。

        Notes / 说明:
        - Must use canonical ordering for dict/list fields (e.g., sort adj lists) to avoid false mismatches.
          必须对 dict/list 做规范化排序（例如 adj/adj_tri_ids 排序），否则会出现“语义一致但 hash 不一致”的问题。
        """
        from ..tools.hash_utils_fast import compute_content_hash
        return compute_content_hash(self.sum_df, self.acag, self.patch, show_progress=False, leave=False)

    @staticmethod
    def _resolve_root_dir(root_dir: str | None) -> Path:
        """
        Resolve user-provided root_dir into an absolute workspace path.
        将传入的 root_dir 解析为 workspace 下的绝对路径（兼容相对/绝对/legacy 前缀）。

        Rules / 规则:
        - None -> workspace root.
          None -> workspace 根目录。
        - Absolute path is used as-is (and checked whether it lies under workspace).
          绝对路径按原样使用（并检查是否在 workspace 内）。
        - Relative path starting with "data/workspace" is remapped under workspace root.
          相对路径若以 "data/workspace" 开头，则映射为 workspace 子路径。
        - Other relative path is treated as workspace-relative.
          其他相对路径视为 workspace 的相对路径。
        """
        from ..file_parser.workspace_utils import get_workspace_dir
        workspace_dir = get_workspace_dir().resolve()
        if not root_dir:
            return workspace_dir

        raw_path = Path(root_dir)
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
            try:
                resolved.relative_to(workspace_dir)
                return resolved
            except ValueError:
                return resolved

        norm_str = os.path.normpath(str(raw_path))
        norm_path = Path(norm_str)
        try:
            rel = norm_path.relative_to(Path("data") / "workspace")
            return (workspace_dir / rel).resolve()
        except ValueError:
            return (workspace_dir / norm_path).resolve()

    def save(
        self,
        root_dir: str,
        *,
        params: dict | None = None,
        policy_version: str | None = None,
        validate_after_save: bool = True,
    ) -> dict:
        """
        Persist Patch artifacts to disk as a manifest-driven cache entry.
        将 Patch 产物以 manifest 驱动的缓存条目形式落盘。

        D-3 Key points / D-3 关键点:
        1) Each artifact writer returns the *actual* saved filename.
        每类产物写入函数返回“真实落盘文件名”（避免后缀/路径猜测）。
        2) meta.json is the single source of truth for later loading.
        meta.json 是后续 load 的唯一真相来源（single source of truth）。
        3) Optional self-check: load via manifest and recompute hash.
        可选自检：写完后用 manifest load 回读并重新计算 hash。
        """
        import json
        import time
        import pandas as pd
        import numpy as np
        from pathlib import Path

        # ---------- helpers ----------
        def _atomic_write_bytes(path: Path, data: bytes) -> None:
            """Atomic write (tmp -> replace). / 原子写（tmp -> replace）。"""
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        def _atomic_write_json(path: Path, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            _atomic_write_bytes(path, data)

        def _save_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
            """Write parquet with tmp file. / parquet 写入使用临时文件保证原子性。"""
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            os.replace(tmp, path)

        def _save_sum_df_one(sdf: pd.DataFrame, target_path: Path) -> str:
            """
            Save one summary dataframe and return actual filename.
            保存一个 sum_df 并返回真实文件名。

            Note / 注意:
            - Parquet is stable and compact; we do not guess formats.
            Parquet 稳定且紧凑；不做格式猜测。
            """
            _save_parquet_atomic(sdf, target_path)
            return target_path.name

        def _save_acag_one(g: Dict[int, Any], target_path: Path) -> str:
            """
            Save one ACAG graph and return actual filename.
            保存一个 ACAG 并返回真实文件名。

            Format / 格式:
            - CSR arrays (nodes/indptr/indices) + props (pickled object array).
            CSR（nodes/indptr/indices）+ props（对象数组 pickle）。
            """
            target_path.parent.mkdir(parents=True, exist_ok=True)

            nodes = np.array(sorted(g.keys()), dtype=np.int64)
            indptr = np.empty(len(nodes) + 1, dtype=np.int64)
            indptr[0] = 0

            indices_list: list[int] = []
            props: Dict[int, Any] = {}

            for i, n in enumerate(nodes, start=1):
                elem = g.get(int(n), {})
                neigh = elem.get("adj", [])
                indices_list.extend(int(x) for x in neigh)
                indptr[i] = len(indices_list)
                props[int(n)] = {k: v for k, v in elem.items() if k != "adj"}

            indices = np.array(indices_list, dtype=np.int64)

            # numpy requires suffix handling: ensure tmp ends with .npz
            tmp = target_path.with_name(target_path.name + ".tmp.npz")
            np.savez_compressed(
                tmp,
                nodes=nodes,
                indptr=indptr,
                indices=indices,
                props=np.array([props], dtype=object),
            )
            os.replace(tmp, target_path)
            return target_path.name

        def _save_patch_one(p: Any, target_msgpack_path: Path) -> str:
            """
            Save patch using msgpack if possible; otherwise pickle fallback.
            patch 优先 msgpack；失败则回退 pickle。

            Returns / 返回:
            - actual saved filename, e.g. "patch_1.msgpack" or "patch_1.pkl"
            返回真实文件名，例如 "patch_1.msgpack" 或 "patch_1.pkl"
            """
            target_msgpack_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import msgpack  # type: ignore
                data = msgpack.packb(p, use_bin_type=True)
                _atomic_write_bytes(target_msgpack_path, data)
                return target_msgpack_path.name
            except Exception:
                import pickle
                pkl_path = target_msgpack_path.with_suffix(".pkl")
                data = pickle.dumps(p, protocol=pickle.HIGHEST_PROTOCOL)
                _atomic_write_bytes(pkl_path, data)
                return pkl_path.name

        # ---------- dirs ----------
        obj1_id, obj2_id = self.obj_pair
        base_dir = self._resolve_root_dir(root_dir)
        pair_dir_path = base_dir / f"pair_{obj1_id}_{obj2_id}"
        pair_dir_path.mkdir(parents=True, exist_ok=True)

        sum_dir = pair_dir_path / "sum_df"
        acag_dir = pair_dir_path / "acag"
        patch_dir = pair_dir_path / "patch"
        sum_dir.mkdir(parents=True, exist_ok=True)
        acag_dir.mkdir(parents=True, exist_ok=True)
        patch_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 1) save sum_df (record actual filenames) ----------
        sum_df_saved: Dict[int, str] = {}
        for obj_id, sdf in self.sum_df.items():
            # Parquet compatibility: stringify dict keys in dict-cells.
            # parquet 兼容：dict cell 的 key 转 str，避免不同引擎/版本差异。
            sdf_save = sdf.copy()
            if "adj_obj_tri_evaluation" in sdf_save.columns:
                sdf_save["adj_obj_tri_evaluation"] = sdf_save["adj_obj_tri_evaluation"].apply(
                    lambda d: {str(k): v for k, v in d.items()} if isinstance(d, dict) else d
                )

            file_name = f"sum_df_{int(obj_id)}.parquet"
            actual = _save_sum_df_one(sdf_save, sum_dir / file_name)
            sum_df_saved[int(obj_id)] = actual

        # ---------- 2) save acag (record actual filenames) ----------
        acag_saved: Dict[int, str] = {}
        for obj_id, g in self.acag.items():
            file_name = f"acag_{int(obj_id)}.npz"
            actual = _save_acag_one(g, acag_dir / file_name)
            acag_saved[int(obj_id)] = actual

        # ---------- 3) save patch (record actual filenames; msgpack/pkl) ----------
        patch_saved: Dict[int, str] = {}
        for obj_id, p in self.patch.items():
            file_name_default = f"patch_{int(obj_id)}.msgpack"
            actual = _save_patch_one(p, patch_dir / file_name_default)
            patch_saved[int(obj_id)] = actual

        # ---------- 4) compute content hash from re-loaded artifacts ----------
        from ..tools.hash_utils_fast import compute_content_hash

        sum_df_loaded: Dict[int, pd.DataFrame] = {}
        for obj_id, fname in tqdm(sum_df_saved.items(), total=len(sum_df_saved), desc="Loading sum_df", leave=False):
            sum_df_loaded[int(obj_id)] = pd.read_parquet(sum_dir / fname)

        acag_loaded: Dict[int, Any] = {}
        for obj_id, fname in tqdm(acag_saved.items(), total=len(acag_saved), desc="Loading acag", leave=False):
            acag_loaded[int(obj_id)] = Patch._load_acag(acag_dir / fname)

        patch_loaded: Dict[int, Any] = {}
        for obj_id, fname in tqdm(patch_saved.items(), total=len(patch_saved), desc="Loading patch", leave=False):
            patch_loaded[int(obj_id)] = Patch._load_patch(patch_dir / fname)

        content_hash = compute_content_hash(sum_df_loaded, acag_loaded, patch_loaded, show_progress=False, leave=False)
        self.hash_id = content_hash

        # ---------- 5) write manifest ----------
        meta = {
            "obj_pair": [int(obj1_id), int(obj2_id)],
            "hash_id": content_hash,
            "created_at_unix": int(time.time()),
            "params": params or {},
            "policy_version": policy_version,
            "files": {
                "sum_df": {str(obj_id): f"sum_df/{sum_df_saved[int(obj_id)]}" for obj_id in sum_df_saved.keys()},
                "acag": {str(obj_id): f"acag/{acag_saved[int(obj_id)]}" for obj_id in acag_saved.keys()},
                "patch": {str(obj_id): f"patch/{patch_saved[int(obj_id)]}" for obj_id in patch_saved.keys()},
            },
        }
        _atomic_write_json(pair_dir_path / "meta.json", meta)

        # ---------- 6) optional self-check (manifest-driven load) ----------
        if validate_after_save:
            # IMPORTANT: load expects pair_dir name string, not a Path.
            # 重要：load 的 pair_dir 参数是目录名字符串，而不是 Path 对象。
            loaded_patch = Patch.load(root_dir, pair_dir_path.name, validate_hash=True)
            assert loaded_patch.hash_id == content_hash, "Hash mismatch after loading"

        return meta

    @staticmethod
    def load(root_dir: str, pair_dir: str, validate_hash: bool = True) -> "Patch":
        """
        Load Patch artifacts from disk using meta.json manifest.
        使用 meta.json（manifest）从磁盘加载 Patch 产物。

        D-2关键点 / D-2 key point:
        - patch file path must be read from meta.json, because suffix can be .msgpack or .pkl.
        patch 文件路径必须从 meta.json 读取，因为后缀可能是 .msgpack 或 .pkl。
        """
        import json
        import pandas as pd

        base_dir = Patch._resolve_root_dir(root_dir)
        pair_dir_path = base_dir / pair_dir

        meta_file = pair_dir_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found at {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        stored_hash = meta.get("hash_id")

        sum_df_files = meta["files"]["sum_df"]
        acag_files = meta["files"].get("acag")
        patch_files = meta["files"]["patch"]

        # ---------- Load sum_df ----------
        sum_df: Dict[int, pd.DataFrame] = {}
        for obj_id, sum_file in tqdm(sum_df_files.items(), total=len(sum_df_files), desc="Loading sum_df", leave=False):
            sum_df[int(obj_id)] = pd.read_parquet(pair_dir_path / sum_file)

        # ---------- Load acag ----------
        acag: Dict[int, Any] = {}
        for obj_id, acag_file in tqdm(acag_files.items(), total=len(acag_files), desc="Loading acag", leave=False):
            acag[int(obj_id)] = Patch._load_acag(pair_dir_path / acag_file)

        # ---------- Load patch (suffix can be msgpack/pkl) ----------
        patch: Dict[int, Any] = {}
        for obj_id, patch_file in tqdm(patch_files.items(), total=len(patch_files), desc="Loading patch", leave=False):
            patch[int(obj_id)] = Patch._load_patch(pair_dir_path / patch_file)

        # ---------- Reconstruct Patch shell ----------
        obj_pair_raw = meta.get("obj_pair", [])
        if isinstance(obj_pair_raw, (list, tuple)) and len(obj_pair_raw) == 2:
            obj_pair = (int(obj_pair_raw[0]), int(obj_pair_raw[1]))
        else:
            obj_pair = (0, 1)

        patch_obj = Patch.__new__(Patch)
        patch_obj.df = pd.DataFrame()
        patch_obj.obj_pair = obj_pair
        patch_obj.sum_df = sum_df
        patch_obj.acag = acag
        patch_obj.patch = patch
        patch_obj.meta = meta

        # Hash domain is serialized artifacts (same as save()).
        # hash 定义域为落盘产物（与 save() 一致）。
        patch_obj.hash_id = patch_obj.gen_hash_id()

        if validate_hash and stored_hash and stored_hash != patch_obj.hash_id:
            raise ValueError(
                f"Hash ID mismatch on load: meta={stored_hash} computed={patch_obj.hash_id} dir={pair_dir_path}"
            )

        return patch_obj


 
    @staticmethod
    def _load_acag(path: Path) -> Dict[int, Dict[str, Any]]:
        """
        Load ACAG from CSR npz + pickled props.
        从 CSR npz（nodes/indptr/indices）+ pickled props 还原 ACAG。

        Security / 安全:
        - allow_pickle=True is required for props; do not load untrusted files.
          props 需要 allow_pickle=True；不要加载不可信来源的文件。
        """
        data = np.load(path, allow_pickle=True)
        nodes = data["nodes"]
        indptr = data["indptr"]
        indices = data["indices"]

        props: Dict[int, Any] = {}
        if "props" in data:
            props_arr = data["props"]
            if props_arr.size > 0:
                props = props_arr[0]

        graph: Dict[int, Dict[str, Any]] = {}
        for i, node in enumerate(nodes):
            start, end = indptr[i], indptr[i + 1]
            adj = indices[start:end].astype(int).tolist()

            elem = props.get(int(node), {})
            # Reattach adjacency list into element dict.
            # 把 CSR 邻接列表重新挂回到 elem["adj"]。
            elem["adj"] = adj
            graph[int(node)] = elem

        return graph

    @staticmethod
    def _load_patch(path: Path) -> Any:
        """
        Load patch from msgpack, with pickle fallback.
        优先从 msgpack 读取 patch，失败则回退到 pickle。

        Rationale / 原因:
        - msgpack is compact and fast; pickle is fallback when msgpack is unavailable.
          msgpack 体积更小、读写快；pickle 作为兜底保证环境最小依赖仍可工作。

        Notes / 注意:
        - The saved fallback pickle file name is `path.with_suffix(".pkl")`.
          若保存时回退为 pickle，则文件名为 `patch_x.pkl`（由 with_suffix(".pkl") 生成）。
        """
        # Try msgpack first / 先尝试 msgpack
        try:
            import msgpack  # type: ignore
            with open(path, "rb") as f:
                return msgpack.unpackb(f.read(), raw=False)
        except Exception:
            # Fallback to pickle / 回退到 pickle（尝试同名 .pkl）
            import pickle
            pkl_path = path if path.suffix == ".pkl" else path.with_suffix(".pkl")
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

    @staticmethod
    def bfs_search_patch(graph_raw_data: Dict[int, List[int]]) -> List[List[int]]:
        """
        Find connected components from an adjacency-list graph using BFS.
        使用 BFS 在邻接表图中提取所有连通分量（connected components）。
        """
        visited = set()
        connected_components: List[List[int]] = []
        queue = deque()

        for node in graph_raw_data:
            if node in visited:
                continue
            component: List[int] = []
            visited.add(node)
            component.append(node)
            queue.append(node)

            while queue:
                current_node = queue.popleft()
                for neighbor in graph_raw_data.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)

            connected_components.append(component)

        return connected_components

    def component_detect(self, g1: Dict, g2: Dict) -> Dict[int, List[Dict[str, Any]]]:
        """
        Build connected components for each object and link them across objects.
        对两侧对象分别构建连通组件，并基于覆盖关系建立跨对象组件链接。

        Cross-link rule / 跨对象链接规则:
        - If any tri in a component of obj2 "covers" a tri belonging to a component of obj1,
          then the two components are linked (bidirectional indices in 'adj').
          若 obj2 某组件内任意 tri 的 cover_area.true.adj_tri_ids 命中 obj1 某组件的 tri，则建立双向链接。

        Output / 输出:
        - {obj1_id: [{"component":[...], "adj":[...]}, ...], obj2_id: [...]}
          返回每个对象的组件列表，每个组件带对端组件索引列表。
        """
        g1_mod = {tri: elem.get("adj", []) for tri, elem in g1.items()}
        g2_mod = {tri: elem.get("adj", []) for tri, elem in g2.items()}

        if not g1_mod or not g2_mod:
            # No graph nodes / 图为空直接返回空组件
            obj1_id, obj2_id = self.obj_pair
            return {obj1_id: [], obj2_id: []}

        connected_components_g1 = Patch.bfs_search_patch(g1_mod)
        connected_components_g2 = Patch.bfs_search_patch(g2_mod)

        g1_res = [{"component": comp, "adj": []} for comp in connected_components_g1]
        g2_res = [{"component": comp, "adj": []} for comp in connected_components_g2]

        # Link components / 建立跨对象链接
        for id_g1, component_g1 in enumerate(g1_res):
            set_g1 = set(component_g1["component"])  # accelerate membership tests / 加速成员测试
            for id_g2, component_g2 in enumerate(g2_res):
                linked = False
                for tri2 in component_g2["component"]:
                    for tri1 in g2[tri2]["cover_area"]["true"]["adj_tri_ids"]:
                        if tri1 in set_g1:
                            linked = True
                            break
                    if linked:
                        component_g1["adj"].append(id_g2)
                        component_g2["adj"].append(id_g1)
                        break

        obj1_id, obj2_id = self.obj_pair
        return {obj1_id: g1_res, obj2_id: g2_res}

    def _debug_show(self, obj1: Any, obj2: Any) -> None:
        """
        Visualize components for debugging; must not change artifact content.
        仅用于调试可视化展示组件，不应改变产物内容（避免影响 hash/caching）。
        """
        try:
            from ..visualizer.visualizer_interface import IVisualizer
        except Exception:
            return

        visualizer = IVisualizer.create()
        g1_res = self.patch[obj1.id]
        g2_res = self.patch[obj2.id]

        visualizer.addObj(obj1, opacity=0.1)
        visualizer.addObj(obj2, opacity=0.1)

        for component_g1 in g1_res:
            visualizer.addObj(obj1, component_g1["component"])
        for component_g2 in g2_res:
            visualizer.addObj(obj2, component_g2["component"])

        visualizer.show()
