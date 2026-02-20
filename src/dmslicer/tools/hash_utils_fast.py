# hash_utils_fast.py
# fast & stable content hash for Patch artifacts (sum_df + acag + patch)
# Patch 产物（sum_df + acag + patch）的快速稳定内容哈希（支持进度条）
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------
# versioning / 版本号
# --------------------------
PATCH_CACHE_VERSION = 1
"""
Hash Contract (fast streaming implementation) / 哈希契约（流式快速实现）

English:
- This module computes a deterministic MD5 for Patch cache artifacts:
  sum_df (DataFrame), acag (AdjacencyCurvatureAreaGraph), patch (components).
- It MUST match the legacy implementation (hash_utils.py) for semantically
  equivalent content.
- Determinism is guaranteed by:
  (1) sorting keys (object_id, tri_id),
  (2) sorting list fields that are semantically sets (adj, adj_tri_ids),
  (3) normalizing numpy/pandas objects into JSON-stable primitives,
  (4) feeding a stable token stream into md5.update() (no giant payload JSON).

中文：
- 本模块用于为 Patch 缓存产物计算“确定性 MD5”：
  sum_df（DataFrame）、acag（邻接-曲率-面积图）、patch（连通组件）。
- 对语义等价的数据，本模块的输出必须与旧实现（hash_utils.py）一致。
- 确定性由以下规则保证：
  (1) key 排序（object_id、tri_id），
  (2) 对“语义是集合”的列表字段排序（adj、adj_tri_ids），
  (3) 归一化 numpy/pandas 对象为 JSON 稳定基础类型，
  (4) 用稳定 token 流 md5.update()（不构造巨大 payload JSON）。
"""


# --------------------------
# helpers: md5 / 哈希工具
# --------------------------
def _md5() -> "hashlib._Hash":
    return hashlib.md5()


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _json_bytes(obj: Any) -> bytes:
    """
    Stable json bytes for hashing (sorted keys, compact separators, no NaN).
    用于哈希的稳定 JSON bytes（key 排序 + 紧凑分隔符 + 禁止 NaN）。
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,  # critical / 关键：禁止 NaN，避免不稳定
    ).encode("utf-8")


# --------------------------
# normalize primitives / 基础类型归一化
# --------------------------
def _normalize_primitive(x: Any) -> Any:
    """
    Normalize to JSON-stable primitives, fail-fast on unknown types.
    归一化为 JSON 稳定的基础类型；遇到未知类型直接抛错（开发期更安全）。
    """
    # None / 空值
    if x is None:
        return None

    # python scalars / Python 标量
    if isinstance(x, (str, int, bool)):
        return x

    # float special values / float 特殊值
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return None
        # normalize -0.0 -> 0.0 / 归一化 -0.0
        if x == 0.0:
            return 0.0
        return x

    # numpy scalar / numpy 标量
    if isinstance(x, np.generic):
        v = x.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, float) and v == 0.0:
            return 0.0
        return v

    # numpy ndarray / numpy 数组：用 dtype/shape + md5(bytes) 表示，避免巨大展开
    if isinstance(x, np.ndarray):
        try:
            b = x.tobytes(order="C")
            return ["ndarray", x.dtype.name, list(x.shape), _md5_hex(b)]
        except Exception:
            return ["ndarray_list", x.dtype.name, list(x.shape), x.tolist()]

    # dict / 字典：key 转 string，value 递归归一化
    if isinstance(x, dict):
        return {str(k): _normalize_primitive(v) for k, v in x.items()}

    # list/tuple / 列表元组：递归归一化
    if isinstance(x, (list, tuple)):
        return [_normalize_primitive(v) for v in x]

    # set / 集合：排序确保确定性
    if isinstance(x, set):
        return sorted(_normalize_primitive(v) for v in x)

    # pandas Timestamp / Timedelta
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return x.isoformat()

    # fail-fast / 开发期：直接报错，避免 repr 不稳定
    raise TypeError(f"Unsupported type for hashing: {type(x)}")


def _sort_int_list(x: Any) -> Any:
    """
    Sort list-like with int casting when possible.
    对“集合语义”的 list 做 int 化排序（尽量稳定）。
    """
    if not isinstance(x, list):
        return x
    try:
        return sorted(int(v) for v in x)
    except Exception:
        # fallback: stable sort on raw values
        return sorted(x)


# --------------------------
# DataFrame fingerprint (fast-ish, stable) / DataFrame 指纹（相对快且稳定）
# --------------------------
def _canonical_cell(x: Any) -> Any:
    """
    Canonicalize one DataFrame cell (object dtype).
    规范化 DataFrame 的单元格（主要处理 object 列）。
    """
    return _normalize_primitive(x)


def stable_df_fingerprint(
    df: pd.DataFrame,
    *,
    show_progress: bool = False,
    leave: bool = False,
    desc: str = "hash:sum_df",
) -> int:
    """
    Stable DataFrame fingerprint tolerant to object cells (ndarray/dict/list).
    对 DataFrame 生成稳定指纹（兼容 object 列里的 ndarray/dict/list）。

    Notes / 注意:
    - Only object columns are canonicalized to JSON strings.
      仅对 object 列进行规范化，避免 pandas factorize(unhashable)。
    - NaN/Inf are converted to None and JSON forbids NaN.
      NaN/Inf 会被归一化成 None，且 JSON 禁止 NaN，避免不稳定。
    """
    if df is None or df.empty:
        return 0

    # shallow copy is enough for column operations / 浅拷贝足够
    df2 = df.copy()

    # stable column order / 列顺序稳定
    df2 = df2.reindex(sorted(df2.columns), axis=1)

    # stable row order / 行顺序稳定（优先 tri_id）
    if "tri_id" in df2.columns:
        df2 = df2.sort_values(["tri_id"], kind="mergesort")
    else:
        # fallback: stable sort by index only (avoid expensive all-columns sort)
        # 兜底：只按 index 排序，避免对混合列做全列排序导致慢和不稳定
        df2 = df2.sort_index(kind="mergesort")

    cols = list(df2.columns)
    it_cols = cols
    if show_progress:
        it_cols = tqdm(cols, total=len(cols), desc=desc, leave=leave)

    # canonicalize object columns / 规范化 object 列
    for c in it_cols:
        if df2[c].dtype == "object":
            df2[c] = df2[c].map(
                lambda v: json.dumps(
                    _canonical_cell(v),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,  # critical / 关键：禁止 NaN
                )
            )

    # stable hash via pandas / 使用 pandas 稳定 hash
    return int(pd.util.hash_pandas_object(df2, index=True).sum())


# --------------------------
# ACAG token stream / ACAG 流式 token
# --------------------------
def _iter_acag_tokens(acag: Dict[int, Dict[int, Dict[str, Any]]]):
    """
    Yield stable tokens for ACAG for streaming hash.
    以稳定顺序产出 ACAG token，用于流式 hash。

    Canon rules / 规范规则:
    - obj_id sorted
      obj_id 升序
    - tri_id sorted
      tri_id 升序
    - elem['adj'] sorted (int)
      adj 作为集合语义，int 化排序
    - cover_area.true/false.adj_tri_ids sorted (int)
      cover_area.*.adj_tri_ids 作为集合语义，int 化排序
    - all other fields normalized to JSON-stable primitives
      其余字段统一归一化为 JSON 稳定基础类型
    """
    for obj_id in sorted(acag.keys()):
        g = acag[obj_id]
        yield ("obj", int(obj_id))

        for tri in sorted(g.keys()):
            elem = dict(g[tri])

            # 1) sort adjacency / 邻接排序
            if "adj" in elem:
                elem["adj"] = _sort_int_list(elem.get("adj"))

            # 2) sort cover_area adj_tri_ids / cover_area 中 adj_tri_ids 排序
            ca = elem.get("cover_area")
            if isinstance(ca, dict):
                ca2 = dict(ca)
                for key in ("true", "false"):
                    sub = ca2.get(key)
                    if isinstance(sub, dict):
                        sub2 = dict(sub)
                        if "adj_tri_ids" in sub2:
                            sub2["adj_tri_ids"] = _sort_int_list(sub2.get("adj_tri_ids"))
                        ca2[key] = sub2
                elem["cover_area"] = ca2

            # normalize everything / 统一归一化
            yield ("tri", int(tri), _normalize_primitive(elem))


# --------------------------
# Patch token stream / Patch 流式 token
# --------------------------
def _iter_patch_tokens(patch: Dict[int, List[Dict[str, Any]]]):
    """
    Yield stable tokens for patch components for streaming hash.
    以稳定顺序产出 patch token，用于流式 hash。

    Canon rules / 规范规则:
    - obj_id sorted
      obj_id 升序
    - each component:
        - component list sorted(int)
        - adj list sorted(int)
      component/adj 作为集合语义，int 化排序
    - component list sorted by min(component)
      组件列表按 component 的最小 tri_id 排序（稳定）
    """
    for obj_id in sorted(patch.keys()):
        comps = patch[obj_id] or []

        norm_comps: List[Dict[str, Any]] = []
        for comp in comps:
            comp2 = dict(comp)
            if "component" in comp2 and isinstance(comp2["component"], list):
                comp2["component"] = sorted(int(x) for x in comp2["component"])
            if "adj" in comp2 and isinstance(comp2["adj"], list):
                comp2["adj"] = sorted(int(x) for x in comp2["adj"])
            norm_comps.append(_normalize_primitive(comp2))

        def key_fn(c: Dict[str, Any]) -> int:
            xs = c.get("component", [])
            return xs[0] if xs else 10**18

        norm_comps.sort(key=key_fn)

        yield ("obj", int(obj_id))
        for c in norm_comps:
            yield ("comp", c)


# --------------------------
# Public API / 对外接口
# --------------------------
def compute_content_hash(
    sum_df: Dict[int, pd.DataFrame],
    acag: Dict[int, Any],
    patch: Dict[int, Any],
    *,
    show_progress: bool = False,
    leave: bool = False,
) -> str:
    """
    Compute stable content hash for Patch artifacts.
    计算 Patch 产物的稳定内容哈希。

    Design / 设计:
    - streaming md5.update(), avoid building huge payload JSON
      流式 md5.update()，避免构造巨大 payload JSON
    - deterministic order for dict/list fields
      dict/list 字段采用确定性顺序
    - optional tqdm progress bars (leave configurable)
      可选 tqdm 进度条（leave 可控）
    """
    h = _md5()

    # version / 版本号
    h.update(b"PATCH_CACHE_VERSION=" + str(PATCH_CACHE_VERSION).encode("utf-8") + b"\n")

    # --------------------------
    # sum_df / 汇总表
    # --------------------------
    items = list(sum_df.items())
    items.sort(key=lambda kv: int(kv[0]))

    it_items = items
    if show_progress:
        it_items = tqdm(items, total=len(items), desc="hash:sum_df", leave=leave)

    for obj_id, df in it_items:
        fp = stable_df_fingerprint(df, show_progress=False)  # avoid nested tqdm / 避免嵌套 tqdm
        h.update(b"SUM_DF_OBJ=" + str(int(obj_id)).encode("utf-8") + b":")
        h.update(str(int(fp)).encode("utf-8"))
        h.update(b"\n")

    # --------------------------
    # acag / 邻接曲率面积图
    # --------------------------
    acag_tokens = _iter_acag_tokens(acag)
    if show_progress:
        # rough total = obj_count + tri_count / 粗略 total
        total = 0
        for obj_id in acag:
            total += 1
            total += len(acag[obj_id])
        acag_tokens = tqdm(acag_tokens, total=total, desc="hash:acag", leave=leave)

    for tok in acag_tokens:
        h.update(_json_bytes(tok))
        h.update(b"\n")

    # --------------------------
    # patch / 组件补丁
    # --------------------------
    patch_tokens = _iter_patch_tokens(patch)
    if show_progress:
        # rough total = obj_count + comp_count / 粗略 total
        total = 0
        for obj_id in patch:
            total += 1
            total += len(patch[obj_id] or [])
        patch_tokens = tqdm(patch_tokens, total=total, desc="hash:patch", leave=leave)

    for tok in patch_tokens:
        h.update(_json_bytes(tok))
        h.update(b"\n")

    return h.hexdigest()


