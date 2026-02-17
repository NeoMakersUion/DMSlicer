"""
ACAG (AdjacencyCurvatureAreaGraph) 说明
- 核心字段:
  adj: 邻接三角面片列表 | deg: 法线夹角(一阶曲率) | cover_area: 面积权重
- 与旧graph差异:
  ACAG 显式包含曲率(角度)与面积信息，避免\"graph\"的语义模糊。
- 迁移影响:
  内部字段名已统一为`acag`，序列化元数据用`files[\"acag\"]`。
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
def build_patch_graph(obj,tri_ids,df,tri_col):
            """
            构建单对象的“面片邻接统计图”并返回字典形式的图结构。
            
            该函数基于三角形列表 tri_ids 与三角形对统计表 df（由上一阶段的 pair-level 计算生成），
            为每个三角形计算：
            - adj：与其共享边的邻接三角形且同时出现在 tri_ids 中
            - deg：与邻接三角形法线的最大无向夹角（单位°，基于 abs(dot(n0, nj)) 计算；小于 1° 的角度视为 0；若无有效邻接则为 None）
            - cover_area：按照 area_pass 筛选后的覆盖统计（true/false 两类，包含累计面积与对端三角列表）
            同时在二阶统计中计算：
            - ddeg：所有邻接三角形的 deg 差值的最大值（单位°；小于 1° 的差值视为 0；若不存在可比较的邻接 deg，则为 None）
            
            参数:
                obj: 目标 Object，包含局部三角形数据与 topology 信息
                tri_ids: 当前对象参与统计的三角形 ID 列表（来自 df_true 的 tri1/tri2 去重集合）
                df: pandas.DataFrame，面片对统计表，至少包含 [tri_col, area_pass, cover_col] 等列
                tri_col: 字段名，指示当前对象对应列，如 "tri1" 或 "tri2"
                cover_col: 字段名，对应覆盖比例列，如 "cover1" 或 "cover2"
            
            返回:
                dict[int, dict]: 
                    {
                      tri_id: {
                        "adj": List[int],                # 邻接三角形 ID
                        "deg": float,                    # 与邻接的最大法线夹角（°）
                        "cover_area": {
                            "true":  {"area": float, "list": List[List[int]]},
                            "false": {"area": float, "list": List[List[int]]}
                        },
                        "ddeg": float                    # 与邻接的最大角差（°）
                      },
                      ...
                    }
            
            说明:
            - 法线夹角计算采用 abs(dot(n0, nj)) 并通过 np.clip 保证 arccos 入参在 [0,1]
            - 角度与角差的“门槛”为 1°，用于抑制数值抖动；低于阈值记为 0
            - 邻接集合来源于三角形 topology['edges'] 的值集合，并与 tri_ids 取交集以保持一致性
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
    summary_df = (
        df[df["area_pass"]]
        .groupby(group_field)
        .agg(
            cover_sum=(cover_field, "sum"),
            adj_obj_list=(tri_field, lambda x: _stable_unique(x)),
        )
        .reset_index()
    )
    summary_df_sorted = summary_df.sort_values(by="cover_sum", ascending=True).reset_index(drop=True)
    tri_list = list(summary_df_sorted[group_field])
    parametric_area_grade_list = [
        tri.parametric_area_grade for tri in obj.get_triangles_by_list(tri_list)
    ]
    summary_df_sorted["parametric_area_grade"] = parametric_area_grade_list
    first_col_name = summary_df_sorted.columns.tolist()[0]
    summary_df_sorted_renamed = summary_df_sorted.rename(columns={first_col_name: "tri_id"})
    summary_df_sorted_renamed = summary_df_sorted_renamed[
        ["tri_id", "cover_sum", "adj_obj_list", "parametric_area_grade"]
    ]
    return summary_df_sorted_renamed


def get_dynamic_threshold(parametric_area_grade: int) -> float:
    if parametric_area_grade == 4:
        return 0.10
    elif parametric_area_grade == 3:
        return 0.03
    elif parametric_area_grade == 2:
        return 0.01
    else:
        return 0.01


def _build_adj_eval(adj_list: Any, other_map: Dict[int, bool]) -> Dict[int, Any]:
    if adj_list is None or (isinstance(adj_list, float) and np.isnan(adj_list)):
        return {}
    if not isinstance(adj_list, (list, tuple, set)):
        return {}
    out: Dict[int, Any] = {}
    for t in adj_list:
        try:
            tid = int(t)
        except Exception:
            continue
        out[tid] = other_map.get(tid, None)
    return out


def validate_patch_with_dynamic_threshold(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold_fn: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"tri_id", "cover_sum", "adj_obj_list", "parametric_area_grade"}
    missing1 = required - set(df1.columns)
    missing2 = required - set(df2.columns)
    if missing1:
        raise ValueError(f"df1 missing columns: {missing1}")
    if missing2:
        raise ValueError(f"df2 missing columns: {missing2}")

    df1_out = df1.copy()
    df2_out = df2.copy()
    df1_out["threshold"] = df1_out["parametric_area_grade"].apply(threshold_fn)
    df2_out["threshold"] = df2_out["parametric_area_grade"].apply(threshold_fn)
    df1_out["main_evaluation"] = df1_out["cover_sum"] > df1_out["threshold"]
    df2_out["main_evaluation"] = df2_out["cover_sum"] > df2_out["threshold"]
    df1_eval_map: Dict[int, bool] = dict(
        zip(df1_out["tri_id"].astype(int), df1_out["main_evaluation"].astype(bool))
    )
    df2_eval_map: Dict[int, bool] = dict(
        zip(df2_out["tri_id"].astype(int), df2_out["main_evaluation"].astype(bool))
    )
    df1_out["adj_obj_tri_evaluation"] = df1_out["adj_obj_list"].apply(lambda adj: _build_adj_eval(adj, df2_eval_map))
    df2_out["adj_obj_tri_evaluation"] = df2_out["adj_obj_list"].apply(lambda adj: _build_adj_eval(adj, df1_eval_map))
    df1_out["final_evaluation"] = df1_out.apply(
        lambda row: bool(row["main_evaluation"]) or any(v is True for v in dict(row["adj_obj_tri_evaluation"]).values()),
        axis=1,
    )
    df2_out["final_evaluation"] = df2_out.apply(
        lambda row: bool(row["main_evaluation"]) or any(v is True for v in dict(row["adj_obj_tri_evaluation"]).values()),
        axis=1,
    )
    return df1_out, df2_out

@dataclass
class Patch:
    """
    针对三角网格的层级切片与补丁生成器。

    该类负责协调和执行两个网格对象之间的面片邻接组件计算，生成用于后续处理的“补丁（Patch）”数据结构。
    它不仅处理核心的几何拓扑分析，还管理数据的持久化缓存（保存/加载）以及可选的调试可视化。

    核心流程：
    1. 输入：接收两个网格对象（obj1, obj2）和一个包含面片对统计信息的 DataFrame。
    2. 摘要生成：调用 `create_summary` 聚合覆盖率和邻接关系。
    3. 动态阈值验证：调用 `validate_patch_with_dynamic_threshold` 根据参数化面积等级过滤有效面片。
    4. 图构建：调用 `build_patch_graph` 为每个对象构建包含邻接、角度差（deg/ddeg）和覆盖信息的图结构。
    5. 组件检测：调用 `component_detect` 使用 BFS 算法识别连通组件，并建立跨对象的组件关联。
    6. 缓存管理：支持基于输入哈希的自动缓存机制，避免重复计算。

    对外暴露的主要方法：
    - `__init__`: 初始化实例，执行计算流程或加载缓存。
    - `save`: 将计算结果（摘要、图、补丁、元数据）持久化到磁盘。
    - `load`: 从磁盘加载已保存的补丁数据。
    - `component_detect`: （通常由 init 调用）执行连通组件分析和跨对象链接。

    Design Philosophy:
        The class acts as an orchestrator. It relies on pure helpers
        (create_summary, get_dynamic_threshold, validate_patch_with_dynamic_threshold)
        for data shaping and evaluation, and keeps visualization separated
        in a private method to avoid side effects during construction.
        This minimizes coupling to upstream pair-level statistics (DataFrame)
        and downstream Level-of-Detail systems, using a stable dict-based
        output contract (self.patch, self._raw_g_pair).

    Thread-Safety:
        Not thread-safe. Instances mutate internal dictionaries during
        initialization and component linking. No global shared state is used.

    Performance Notes:
        - Group-by and aggregation: O(N) with overhead proportional to the
          number of distinct triangle IDs.
        - Graph build: approximately O(M + E) where M is selected triangles
          and E is adjacency relations.
        - BFS components: O(M + E).
        Space usage is O(M + E) for graphs and component lists. Excessive
        DataFrame grouping or repeated Python/NumPy transitions may impact
        performance.
    """
    hash_id: str
    df: pd.DataFrame
    obj_pair: Tuple[int, int]
    patch: Dict[int, List[Dict[str, Any]]]
    sum_df: Dict[int, pd.DataFrame]
    acag: Dict[int, Dict[int, Dict[str, Any]]]  # ACAG: graph -> 邻接曲率面积图（含adj/deg/ddeg/cover_area）
    # meta

    def __init__(self, obj1: "Object", obj2: "Object", df: pd.DataFrame, root_dir: str | None = None, show: bool = False):
        """
        初始化 Patch 实例，执行完整的补丁生成流程或从缓存加载。

        :param obj1: 第一个网格对象，需包含 id 和 triangles 属性。
        :param obj2: 第二个网格对象，需包含 id 和 triangles 属性。
        :param df: 包含面片对交互信息的 pandas.DataFrame，需包含 'area_pass', 'tri1', 'tri2' 等列。
        :param root_dir: 可选的缓存根目录路径。若提供，将尝试从中加载或保存计算结果。
        :param show: 调试选项，"true" 则在计算完成后调用可视化展示。
        
        异常与边界:
        - 若 df 中没有满足 'area_pass' 的行，将初始化空的结果结构。
        - 若缓存加载失败（文件损坏或哈希不匹配），将自动回退到重新计算并覆盖旧缓存。
        """
        from ..tools import calculate_hash
        self.obj_pair = (obj1.id, obj2.id)
        self.df = df

        loaded = False
        if root_dir:
            import shutil
            pair_dir_name = f"pair_{obj1.id}_{obj2.id}"
            base_dir = Patch._resolve_root_dir(root_dir)
            full_pair_path = base_dir / pair_dir_name
            
            # === 步骤2: 尝试从缓存加载 ===
            try:
                if full_pair_path.exists():
                    loaded_patch = Patch.load(str(base_dir), pair_dir_name)
                    self.sum_df = loaded_patch.sum_df
                    self.acag = loaded_patch.acag
                    self.patch = loaded_patch.patch
                    self.hash_id = loaded_patch.gen_hash_id()
                    self.meta = getattr(loaded_patch, "meta", {})
                    loaded = True
            except Exception:
                # Load failed (corrupt files, missing meta, etc.)
                if full_pair_path.exists():
                    shutil.rmtree(full_pair_path, ignore_errors=True)
                

        if not loaded:
            # === 步骤3: 执行核心计算流程 ===
            df_true = df[df["area_pass"]]
            if df_true.empty:
                self.patch = {obj1.id: [], obj2.id: []}
                self.sum_df = {}  # Initialize empty
                self.acag = {}  # ACAG: graph -> ACAG 邻接曲率面积图，替换原因：保持字段一致
            else:
                # 3.1 生成摘要统计
                s1 = create_summary(df_true, "tri1", "cover1", "tri2", obj1)
                s2 = create_summary(df_true, "tri2", "cover2", "tri1", obj2)
                # 3.2 动态阈值验证
                s1e, s2e = validate_patch_with_dynamic_threshold(s1, s2, get_dynamic_threshold)
                self.sum_df = {obj1.id: s1e, obj2.id: s2e}
                tri1 = list(s1e[s1e["final_evaluation"]]["tri_id"])
                tri2 = list(s2e[s2e["final_evaluation"]]["tri_id"])
                # 3.3 构建图结构
                g1 = build_patch_graph(obj1, tri1, df, tri_col="tri1")
                g2 = build_patch_graph(obj2, tri2, df, tri_col="tri2")
                self.acag = {obj1.id: g1, obj2.id: g2}  # ACAG: graph -> ACAG 邻接曲率面积图，替换原因：突出邻接与曲率面积属性
                self.patch = self.component_detect(g1, g2)
            # === 步骤4: 保存结果到缓存（如需） ===
            if root_dir:
                self.meta=self.save(str(base_dir))
            else:
                self.hash_id=self.gen_hash_id()

        if show == True:
            self._debug_show(obj1, obj2)

    def _compute_input_hash(self, obj1: Any, obj2: Any, df: pd.DataFrame) -> str:
        """
        计算输入参数的稳定哈希值，用于缓存键生成。
        
        算法思路:
        结合两个对象的 ID 和 DataFrame 的内容哈希（基于 index 和 values）。
        由于 DataFrame 包含混合类型，直接哈希可能不稳定，因此尝试使用 pandas 内置的 `hash_pandas_object`，
        并回退到 shape 字符串作为兜底策略。
        
        :param obj1: 对象1
        :param obj2: 对象2
        :param df: 输入 DataFrame
        :return: MD5 哈希字符串
        """
        from ..tools import calculate_hash
        # Create a stable hash from inputs. 
        # Using obj IDs and dataframe content signature.
        try:
            # Hash pandas object for stability
            # Convert numpy int to python int to avoid hashing issues in tools.calculate_hash
            df_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
        except Exception:
            df_hash = str(df.shape)
        return calculate_hash([obj1.id, obj2.id, df_hash], use_md5=True)

    def _normalize_for_hash(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._normalize_for_hash(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._normalize_for_hash(v) for v in obj]
        if isinstance(obj, set):
            return {self._normalize_for_hash(v) for v in obj}
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return self._normalize_for_hash(obj.tolist())
        return obj

    def gen_hash_id(self):
        from ..tools.hash_utils import compute_content_hash
        return compute_content_hash(self.sum_df, self.acag, self.patch)

    @staticmethod
    def _resolve_root_dir(root_dir: str | None) -> Path:
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
        Save Patch artifacts to disk.

        功能概述:
        将当前 Patch 对象的关键数据（sum_df, graph, patch）序列化并保存到指定目录。
        采用原子写操作（写入临时文件后重命名）以防止写入中断导致的数据损坏。

        :param root_dir: 基础目录路径。
                         - 如果是绝对路径且位于 workspace 根目录内，则按原样使用。
                         - 如果是绝对路径但不在 workspace 内，则按原样使用。
                         - 如果是相对路径且以前缀 `data/workspace` 开头，将自动转换为
                           相对于 workspace 根目录的子路径。
                         - 其他相对路径将被视为 workspace 根目录下的子目录。
        :param params: 可选参数字典（如 angle/gap/overlap/hash_str 等），将记录在 meta.json 中
        :param policy_version: 可选的策略版本标识字符串
        :return: 包含元数据信息的字典（同时也写入了 meta.json）

        WARNING:
        - 涉及文件 IO 操作，请确保磁盘空间充足且有写入权限。
        - Graph 数据保存为 .npz 格式，包含 pickle 序列化的属性对象，加载时需注意安全性（allow_pickle=True）。
        """
        import json
        import time
        import pandas as pd
        import numpy as np
        from pathlib import Path

        # ---------- helpers ----------
        def _atomic_write_bytes(path: Path, data: bytes) -> None:
            # TIP: 原子写入模式：先写 .tmp 文件，再执行 rename
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
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            os.replace(tmp, path)

        def _acag_to_csr_npz(g: Dict[int, Any], path: Path) -> None:
            # Store adjacency list as CSR arrays: nodes, indptr, indices
            # Store other properties as a pickled object in 'props'
            path.parent.mkdir(parents=True, exist_ok=True)
            nodes = np.array(sorted(g.keys()), dtype=np.int64)
            indptr = np.empty(len(nodes) + 1, dtype=np.int64)
            indptr[0] = 0

            # flatten
            indices_list: list[int] = []
            props = {}
            for i, n in enumerate(nodes, start=1):
                elem = g.get(int(n), {})
                # Handle adjacency
                neigh = elem.get("adj", [])
                indices_list.extend(int(x) for x in neigh)
                indptr[i] = len(indices_list)
                # Handle properties
                props[int(n)] = {k: v for k, v in elem.items() if k != "adj"}

            indices = np.array(indices_list, dtype=np.int64)
            # Ensure tmp file ends with .npz so numpy doesn't append it again
            tmp = path.with_name(path.name + ".tmp.npz")
            # Use object array for props to allow pickling
            np.savez_compressed(tmp, nodes=nodes, indptr=indptr, indices=indices, props=np.array([props], dtype=object))
            os.replace(tmp, path)

        # msgpack is optional; fall back to pickle if you don't want extra dep
        def _save_msgpack_atomic(obj, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import msgpack  # type: ignore
                data = msgpack.packb(obj, use_bin_type=True)
                _atomic_write_bytes(path, data)
            except Exception:
                import pickle
                data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                _atomic_write_bytes(path.with_suffix(".pkl"), data)

        obj1_id, obj2_id = self.obj_pair
        base_dir = self._resolve_root_dir(root_dir)
        pair_dir = base_dir / f"pair_{obj1_id}_{obj2_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # ---------- save summaries ----------
        sum_dir = pair_dir / "sum_df"
        sum_df_saved: Dict[int, pd.DataFrame] = {}
        for obj_id, sdf in self.sum_df.items():
            # Convert dict keys in 'adj_obj_tri_evaluation' to str for Parquet compatibility
            # Use a copy to avoid mutating the in-memory dataframe
            sdf_save = sdf.copy()
            if "adj_obj_tri_evaluation" in sdf_save.columns:
                sdf_save["adj_obj_tri_evaluation"] = sdf_save["adj_obj_tri_evaluation"].apply(
                    lambda d: {str(k): v for k, v in d.items()} if isinstance(d, dict) else d
                )
            file_name=f"sum_df_{obj_id}.parquet"
            sum_df_saved[int(obj_id)] = file_name
            _save_parquet_atomic(sdf_save, sum_dir / file_name)

        # ---------- save ACAG (AdjacencyCurvatureAreaGraph) ----------
        acag_dir = pair_dir / "acag"
        acag_saved: Dict[int, Dict[int, Any]] = {}
        for obj_id, g in self.acag.items():
            file_name=f"acag_{obj_id}.npz"
            acag_saved[int(obj_id)] = file_name
            _acag_to_csr_npz(g, acag_dir / file_name)

        # ---------- save patch (raw structure) ----------
        # NOTE: raw nested dict is not stable for long-term; but msgpack/pickle at least keeps int keys.
        patch_dir = pair_dir / "patch"
        patch_saved: Dict[int, Dict[int, Any]] = {}
        for obj_id, p in self.patch.items():
            file_name=f"patch_{obj_id}.msgpack"
            patch_saved[int(obj_id)] = file_name
            _save_msgpack_atomic(p, patch_dir / file_name)

        # Load Saved Files
        from ..tools.hash_utils import compute_content_hash
        sum_df = {}
        # sum_file already contains "sum_df/" prefix
        for obj_id, sum_file in tqdm(sum_df_saved.items(), total=len(sum_df_saved), desc="Loading sum_df",leave=False):
            sum_df[int(obj_id)] = pd.read_parquet(sum_dir / sum_file)

        # ---------- Load ACAG (AdjacencyCurvatureAreaGraph) ----------
        acag = {}
        # acag_file already contains "acag/" or legacy \"graph/\" prefix
        for obj_id, acag_file in tqdm(acag_saved.items(), total=len(acag_saved), desc="Loading acag",leave=False):
            acag[int(obj_id)] = Patch._load_acag(acag_dir / acag_file)

        # ---------- Load patches ----------
        patch = {}
        # patch_file already contains "patch/" prefix
        for obj_id, patch_file in tqdm(patch_saved.items(), total=len(patch_saved), desc="Loading patch",leave=False):
            patch[int(obj_id)] = Patch._load_patch(patch_dir / patch_file)
        content_hash = compute_content_hash(sum_df, acag, patch)

        # ---------- meta / manifest ----------

        self.hash_id = content_hash
        meta = {
            "obj_pair": [int(obj1_id), int(obj2_id)],
            "hash_id": content_hash,
            "created_at_unix": int(time.time()),
            "params": params or {},
            "policy_version": policy_version,
            "files": {
                "sum_df": {str(obj_id): f"sum_df/sum_df_{obj_id}.parquet" for obj_id in self.sum_df.keys()},
                "acag": {str(obj_id): f"acag/acag_{obj_id}.npz" for obj_id in self.acag.keys()},
                "patch": {str(obj_id): f"patch/patch_{obj_id}.msgpack" for obj_id in self.patch.keys()},
            },
        }
        _atomic_write_json(pair_dir / "meta.json", meta)
        return meta

            
    @staticmethod
    def load(root_dir: str, pair_dir: str,validate_hash: bool = False) -> "Patch":
        """
        Load Patch artifacts from disk.

        功能概述:
        从指定目录加载之前保存的 Patch 数据（sum_df, graph, patch）。
        
        :param root_dir: 基础目录路径，解析规则与 save 方法一致。
        :param pair_dir: 具体的配对目录名，例如 `pair_10_20`
        :return: 完整初始化后的 Patch 实例
        
        :raises FileNotFoundError: 如果 meta.json 不存在
        :raises ValueError: 如果加载的数据结构不符合预期（由底层 pandas/numpy 抛出）
        """
        import json
        import pandas as pd
        from tqdm import tqdm
        import numpy as np
        base_dir = Patch._resolve_root_dir(root_dir)
        pair_dir_path = base_dir / pair_dir

        meta_file=pair_dir_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found at {meta_file}")

        # Load metadata to get file structure and parameters
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Retrieve hash_id from meta to verify data integrity
        stored_hash = meta.get("hash_id")

        # Retrieve files from meta
        sum_df_files = meta["files"]["sum_df"]
        # 优先使用 ACAG 字段，兼容旧版本的 graph 字段
        acag_files = meta["files"].get("acag")
        patch_files = meta["files"]["patch"]

        # ---------- Load summaries (sum_df) ----------
        sum_df = {}
        # sum_file already contains "sum_df/" prefix
        for obj_id, sum_file in tqdm(sum_df_files.items(), total=len(sum_df_files), desc="Loading sum_df",leave=False):
            sum_df[int(obj_id)] = pd.read_parquet(pair_dir_path / sum_file)

        # ---------- Load ACAG (AdjacencyCurvatureAreaGraph) ----------
        acag = {}
        # acag_file already contains "acag/" or legacy \"graph/\" prefix
        for obj_id, acag_file in tqdm(acag_files.items(), total=len(acag_files), desc="Loading acag",leave=False):
            acag[int(obj_id)] = Patch._load_acag(pair_dir_path / acag_file)

        # ---------- Load patches ----------
        patch = {}
        # patch_file already contains "patch/" prefix
        for obj_id, patch_file in tqdm(patch_files.items(), total=len(patch_files), desc="Loading patch",leave=False):
            patch[int(obj_id)] = Patch._load_patch(pair_dir_path / patch_file)

        # Removed _raw_g_pair loading logic as it is redundant with graph
        # ---------- Check for metadata ----------
        meta_file = pair_dir_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found at {meta_file}")
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
        patch_obj.hash_id = patch_obj.gen_hash_id()
        if validate_hash and stored_hash and stored_hash != patch_obj.hash_id:
            debug_dir = pair_dir_path / "hash_mismatch_debug"
            Patch._dump_hash_mismatch_debug(
                debug_dir=debug_dir,
                meta=meta,
                stored_hash=str(stored_hash),
                computed_hash=str(patch_obj.hash_id),
                sum_df=patch_obj.sum_df,
                acag=patch_obj.acag,
                patch=patch_obj.patch,
            )
            raise ValueError(f"Hash ID mismatch on load: meta={stored_hash} computed={patch_obj.hash_id} debug_dir={debug_dir}")
        return patch_obj

    @staticmethod
    def _dump_hash_mismatch_debug(
        *,
        debug_dir: Path,
        meta: dict,
        stored_hash: str,
        computed_hash: str,
        sum_df: Dict[int, Any],
        acag: Dict[int, Any],
        patch: Dict[int, Any],
    ) -> None:
        import json
        import platform
        import sys
        from ..tools.hash_utils import build_content_hash_payload, stable_json_dumps_bytes

        debug_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "stored_hash": stored_hash,
            "computed_hash": computed_hash,
        }

        (debug_dir / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        payload = build_content_hash_payload(sum_df, acag, patch)
        payload_bytes = stable_json_dumps_bytes(payload)
        (debug_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (debug_dir / "payload.bin").write_bytes(payload_bytes)
        (debug_dir / "payload.hex.txt").write_text(payload_bytes.hex(), encoding="utf-8")

    @staticmethod
    def _load_acag(path: Path) -> Dict[int, Dict[str, Any]]:
        """
        Load the graph from a CSR npz format with pickled properties.
        
        算法思路:
        NPZ 文件包含 CSR 格式的邻接矩阵（indices, indptr）以及一个 pickled 对象数组（props）。
        函数将这些分散的数据重组为 `Dict[int, Dict]` 格式的图结构。
        
        :param path: .npz 文件路径
        :return: 恢复后的图字典
        """
        data = np.load(path, allow_pickle=True)
        nodes = data["nodes"]
        indptr = data["indptr"]
        indices = data["indices"]
        
        # Load properties if available
        props = {}
        if "props" in data:
            props_arr = data["props"]
            if props_arr.size > 0:
                props = props_arr[0]

        graph: Dict[int, Dict[str, Any]] = {}
        for i, node in enumerate(nodes):
            start, end = indptr[i], indptr[i+1]
            adj = indices[start:end].astype(int).tolist()
            # Combine adjacency with properties
            elem = props.get(int(node), {})
            elem["adj"] = adj
            graph[int(node)] = elem
        return graph

    @staticmethod
    def _load_patch(path: Path) -> Any:
        """
        Load the patch from a msgpack or pickle file.
        
        优先尝试 msgpack 加载（更高效、跨语言），失败则回退到 pickle。
        """
        try:
            import msgpack
            with open(path, "rb") as f:
                return msgpack.unpackb(f.read(), raw=False)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)


    @staticmethod
    def bfs_search_patch(graph_raw_data: Dict[int, List[int]]) -> List[List[int]]:
        """Return connected components using BFS on an adjacency-list graph.

        功能概述:
        使用广度优先搜索（BFS）算法，在给定的邻接表图结构中查找所有连通分量。

        Args:
            graph_raw_data: Mapping from tri_id to a list of adjacent tri_ids.
                Must be a well-formed adjacency list; None is not allowed.

        Returns:
            A list of components, each component represented by a list of tri_ids.

        Raises:
            KeyError: If adjacency entries are malformed during traversal.

        Side Effects:
            None.
        """
        visited = set()
        connected_components: List[List[int]] = []
        queue = deque()
        for node in graph_raw_data:
            if node not in visited:
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

    def component_detect(self, g1: Dict, g2: Dict):
        """Populate cross-linked components between two graphs.

        功能概述:
        识别两个对象图（g1, g2）各自的内部连通组件，并根据覆盖关系建立组件间的跨对象链接（Cross-Link）。
        
        算法思路:
        1. 提取简化的邻接表（仅包含 adj 字段）。
        2. 分别对 g1 和 g2 执行 BFS 查找连通组件。
        3. 遍历 g2 中每个组件的每个三角形，检查其覆盖的 g1 三角形是否属于 g1 的某个组件。
        4. 若存在覆盖关系，则在两个组件之间建立双向索引链接。

        Args:
            g1: Graph dict for obj1; entries expected to include "adj" and
                "cover_area" with subkey "true" -> "adj_tri_ids".
            g2: Graph dict for obj2; same structure as `g1`.

        Returns:
            None. Updates `self.patch` in-place with:
            {obj_id: [{"component": List[int], "adj": List[int]}], ...}

        Raises:
            KeyError: If required keys are missing in graph entries.

        Side Effects:
            Mutates `self.patch` based on detected adjacency relationships.
        """
        g1_mod = {tri: elem.get('adj', []) for tri, elem in g1.items()}
        g2_mod = {tri: elem.get('adj', []) for tri, elem in g2.items()}
        g1_res =[]
        g2_res= []
        if not g1_mod or not g2_mod:
            return g1_res, g2_res
        # === 步骤1: 内部连通组件检测 ===
        connected_components_g1 = Patch.bfs_search_patch(g1_mod)
        connected_components_g2 = Patch.bfs_search_patch(g2_mod)
        for component in connected_components_g1:
            g1_res.append({"component": component, "adj": []})
        for component in connected_components_g2:
            g2_res.append({"component": component, "adj": []})
        
        # === 步骤2: 跨组件关联检测 ===
        for id_g1,component_g1 in enumerate(g1_res):
            for id_g2,component_g2 in enumerate(g2_res):
                # 检查 component_g2 中的元素是否覆盖了 component_g1 中的元素
                for elem in component_g2['component']:
                    status=False
                    # TIP: 这里假设 g2[elem] 数据结构完整，包含 'cover_area'['true']['adj_tri_ids']
                    for adj_g1 in g2[elem]['cover_area']['true']['adj_tri_ids']:
                        if adj_g1 in component_g1['component']:
                            status=True
                            break
                    if status:
                        component_g1['adj'].append(id_g2)
                        component_g2['adj'].append(id_g1)
                        break
        obj1_id, obj2_id = self.obj_pair
        return {obj1_id: g1_res, obj2_id: g2_res}

    def _debug_show(self,obj1,obj2) -> None:
        """Render components using the active visualizer; for debug only.

        Pre-conditions:
            `self.patch` must already be populated by `component_detect`.
        Post-conditions:
            No mutation of algorithmic state; only visualization side effects.

        Args:
            obj1: First object to render; used to add meshes.
            obj2: Second object to render.

        Returns:
            None.

        Raises:
            ImportError: If visualization interface cannot be imported.

        Side Effects:
            Performs I/O via the visualizer; does not change global state.
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
            tris1 = component_g1["component"]
            visualizer.addObj(obj1, tris1)
        for component_g2 in g2_res:
            tris2 = component_g2["component"]
            visualizer.addObj(obj2, tris2)
        visualizer.show()
        
        
