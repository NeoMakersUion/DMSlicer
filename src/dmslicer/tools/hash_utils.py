# hash_utils.py
"""# 通用哈希工具：将任意嵌套数据结构规范化为可哈希形式并计算稳定指纹 # General hash utilities: canonicalize nested data and compute stable fingerprints

本模块提供两类核心能力：                                                        # This module provides two core capabilities:
1) recursive_hashable：将 dict / list / set / pandas.DataFrame 等嵌套结构        # 1) recursive_hashable: convert nested structures like dict/list/set/
   规范化为可哈希的不可变元组表示，保证相同语义的数据获得一致表示；             #    pandas.DataFrame into hashable, immutable tuples so equal data share
2) calculate_hash：在规范化表示上计算 MD5（跨环境稳定）或内置 hash（进程内快）。   #    identical canonical forms; 2) calculate_hash: compute MD5 (stable across
                                                                                 #    environments) or built-in hash (fast, process-local) on the canonical form.

复杂度与实现说明：                                                               # Complexity and implementation notes:
- 对 dict 按键排序再递归处理，时间复杂度近似 O(N log N)（N 为键数）；             # - dict keys are sorted then processed recursively, ~O(N log N) for N keys;
- 对 list/set 线性遍历，set 先排序以保证确定性，复杂度 O(N log N)；               # - lists are linear, sets are sorted first for determinism, O(N log N);
- 对 DataFrame 先拷贝列，再把字典单元格转为有序元组，最后按索引排序展平。        # - DataFrame is copied, dict cells are canonicalized, then rows are sorted by
                                                                                 #   index and flattened into a tuple.
"""
from __future__ import annotations

from typing import Any, Union
import hashlib
# 尝试延迟依赖 pandas（仅在传入 DataFrame 时才需要）                       # Defer pandas import; only required when DataFrame is actually provided
try:
    import pandas as pd  # type: ignore  # 中文：仅在环境安装了 pandas 时启用  # English: enable only if pandas is available
    _PANDAS_AVAILABLE = True  # 中文：标记 pandas 可用                         # English: flag indicating pandas availability
except Exception:
    pd = None  # type: ignore  # 中文：占位，避免类型检查报错                 # English: placeholder to satisfy type checkers
    _PANDAS_AVAILABLE = False  # 中文：标记 pandas 不可用                     # English: flag indicating pandas not available

from tqdm import tqdm
def _process_dict(d: dict) -> tuple:
    """# 将字典按键排序并递归归一化为不可变元组 # Canonicalize dict by sorted keys into an immutable tuple

    功能概述：                                                                   # Overview:
    - 中文：对输入字典按键排序，确保不同插入顺序得到一致表示；对每个值递归规范化。   # - Sort keys to ensure deterministic order; recursively canonicalize values.
    - English: Sort keys deterministically and recursively canonicalize values.

    参数/Parameters:
    - d (dict)：输入字典 # Input dictionary

    返回/Returns:
    - tuple：形如 ((k1,v1'),(k2,v2'),...) 的不可变表示 # Immutable tuple of (key, canonical_value)
    """
    # 中文：对每个键值对，递归处理其值，最后按键排序并转为元组                    # English: recursively process values, sort by key, and build tuple
    return tuple(sorted((k, recursive_hashable(v)) for k, v in d.items()))


def _process_list_or_set(seq: Union[list, set, Any]) -> tuple:
    """# 规范化列表/集合（集合先排序保证确定性）为不可变元组 # Canonicalize list/set to immutable tuple (sets sorted first)

    功能/Overview:
    - 中文：集合先排序；随后递归处理每个元素并打包为元组。                       # - Sort sets for determinism; recursively process each element; pack into tuple.
    - English: Sort set (if any), recurse over elements, return a tuple.

    参数/Parameters:
    - seq (list|set|Any)：输入序列 # Input sequence

    返回/Returns:
    - tuple：不可变表示 # Immutable tuple representation
    """
    # 中文：若为 set，先排序以确保哈希稳定                                      # English: sort set to ensure deterministic hashing
    if isinstance(seq, set):
        seq = sorted(seq)
    # 中文：逐元素递归规范化，收集为元组                                         # English: recursively canonicalize elements and collect into tuple
    return tuple(recursive_hashable(item) for item in seq)


def _process_dataframe(df: Any) -> tuple:
    """# 规范化 pandas.DataFrame：处理字典单元格、按索引排序并展平 # Canonicalize pandas.DataFrame: normalize dict cells, sort by index, flatten

    功能/Overview:
    - 中文：对包含字典的列逐元规范化；随后按索引排序；最后将二维表展平成元组。      # - Normalize dict-valued cells; sort by index; flatten into a tuple.
    - English: Normalize dict cells column-wise, sort by index, then flatten rows.

    参数/Parameters:
    - df (Any)：DataFrame 实例（运行时需 pandas 支持） # DataFrame instance (requires pandas at runtime)

    返回/Returns:
    - tuple：不可变元组表示 # Immutable tuple canonical form

    复杂度/Complexity:
    - 中文：拷贝 O(n)，列遍历 O(n)，字典规范化引入 O(k log k)，整体与数据量线性相关。 # O(n) copy, O(n) column pass; dict cells add O(k log k); overall ~linear.
    - English: Overall near-linear in data size; dict-heavy columns add sorting cost.
    """
    # 中文：复制 DataFrame，避免修改调用方数据                                   # English: copy to avoid mutating caller data
    df_copy = df.copy()
    # 中文：逐列检查字典单元格，进行递归规范化                                    # English: scan columns, canonicalize dictionary cells
    for col in df_copy.columns:
        if df_copy[col].apply(lambda x: isinstance(x, dict)).any():
            df_copy[col] = df_copy[col].apply(
                lambda x: _process_dict(x) if isinstance(x, dict) else recursive_hashable(x)
            )
    # 中文：按索引排序后，将二维表展平为一维元组                                   # English: sort by index and flatten the 2D table into a 1D tuple
    return tuple(df_copy.sort_index().values.flatten())


def recursive_hashable(data: Any) -> Any:
    """# 递归将任意嵌套数据转为可哈希表示 # Recursively convert nested data into a hashable representation

    功能/Overview:
    - 中文：按类型分派：DataFrame→有序元组；dict→有序键值对元组；               # - Type-dispatch: DataFrame→ordered tuple; dict→sorted key-value tuple;
      list/set→元组；基础可哈希类型直接返回；其他类型抛出异常。                  #   list/set→tuple; hashable scalars returned as-is; others raise.
    - English: Dispatch by type to produce a deterministic, hashable canonical form.

    参数/Parameters:
    - data (Any)：任意嵌套输入 # Arbitrary nested input

    返回/Returns:
    - Any：可哈希的不可变表示（通常为元组/标量） # A hashable immutable form (tuples/scalars)

    异常/Exceptions:
    - TypeError：遇到未支持的不可哈希类型时抛出 # Raised when encountering unsupported unhashable types
    """
    # 中文：如可用且为 DataFrame，走 DataFrame 规范化路径                         # English: if available and DataFrame, use the DataFrame path
    if _PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):  # type: ignore
        return _process_dataframe(data)
    # 中文：dict → 有序键值对元组                                                 # English: dict → sorted (key, value) tuple
    elif isinstance(data, dict):
        return _process_dict(data)
    # 中文：list/set → 元组（set 已排序）                                         # English: list/set → tuple (sets pre-sorted)
    elif isinstance(data, (list, set)):
        return _process_list_or_set(data)
    # 中文：基础可哈希类型直接返回                                                # English: hashable scalar types returned as-is
    elif isinstance(data, (int, str, tuple, bool, float, frozenset)):
        return data
    elif data is None:
        return None
    else:
        raise TypeError(f"Unsupported unhashable type: {type(data)}")


def calculate_hash(data: Any, use_md5: bool = True) -> Union[int, str]:
    """# 计算数据指纹：MD5（稳定）或内置 hash（进程内快速） # Compute fingerprint: MD5 (stable) or built-in hash (process-local fast)

    功能/Overview:
    - 中文：先递归规范化数据，再根据 use_md5 选择 MD5 十六进制摘要或内置 hash。     # - Canonicalize input, then choose MD5 hex digest or built-in hash.
    - English: Canonicalize first; MD5 yields 32-hex stable string; built-in is int.

    参数/Parameters:
    - data (Any)：任意数据结构（需支持 recursive_hashable） # Any data supported by recursive_hashable
    - use_md5 (bool)：True 返回 32 位十六进制字符串；False 返回内置整数哈希         # True for 32-hex string; False for Python's int hash

    返回/Returns:
    - str|int：MD5 十六进制字符串或内置整数哈希                                   # str (MD5 hex) or int (built-in hash)

    异常/Exceptions:
    - TypeError：当数据包含未支持类型时由 recursive_hashable 抛出                  # Propagates TypeError from recursive_hashable on unsupported types
    """
    # 中文：将任意嵌套结构转为稳定、可哈希的规范化表示                              # English: canonicalize to stable, hashable representation
    hashable_data = recursive_hashable(data)
    if use_md5:
        # 中文：转字符串再编码，计算 MD5 摘要（跨环境稳定）                         # English: stringify then encode; compute MD5 digest (stable across envs)
        data_str = str(hashable_data).encode("utf-8")
        return hashlib.md5(data_str).hexdigest()
    else:
        # 中文：使用内置 hash，速度快但受进程种子影响（非跨进程稳定）                # English: built-in hash is fast but not stable across processes/seeds
        return hash(hashable_data)
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List

PATCH_CACHE_VERSION = 1


def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _stable_json_dumps(obj: Any) -> bytes:
    # bytes version for hashing
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_normalize(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _normalize(obj.tolist())
    return obj

def _sort_int_list(x: Any) -> Any:
    if isinstance(x, list):
        try:
            return sorted(int(v) for v in x)
        except Exception:
            return sorted(x)
    return x


def _canonical_cell(x: Any) -> Any:
    """把 DataFrame object 单元格变成稳定可序列化结构。"""
    if x is None:
        return None
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        # 用 dtype/shape + md5(bytes) 避免巨大展开
        try:
            b = x.tobytes(order="C")
            return ["ndarray", x.dtype.name, list(x.shape), _md5_bytes(b)]
        except Exception:
            return ["ndarray_list", x.dtype.name, list(x.shape), x.tolist()]
    if isinstance(x, dict):
        return {str(k): _canonical_cell(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canonical_cell(v) for v in x]
    if isinstance(x, set):
        return sorted(_canonical_cell(v) for v in x)
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return x.isoformat()
    return x


def stable_df_fingerprint(df: pd.DataFrame) -> int:
    """对 DataFrame 生成稳定指纹（支持 object 列里含 ndarray/dict/list）。"""
    if df is None or df.empty:
        return 0

    df2 = df.copy()
    df2 = df2.reindex(sorted(df2.columns), axis=1)

    if "tri_id" in df2.columns:
        df2 = df2.sort_values(["tri_id"], kind="mergesort")
    else:
        df2 = df2.sort_values(list(df2.columns), kind="mergesort")

    for c in tqdm(df2.columns, total=len(df2.columns), desc="Normalizing object columns", leave=False):
        if df2[c].dtype == "object": 
            # 转成稳定 JSON 字符串，避免 pandas factorize unhashable
            df2[c] = df2[c].map(lambda v: json.dumps(_canonical_cell(v), ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    return int(pd.util.hash_pandas_object(df2, index=True).sum())


def stable_acag_fingerprint(acag: Dict[int, Dict[int, Dict[str, Any]]]) -> Any:
    """
    ACAG stable canonical form:
    - tri_id sorted
    - adj sorted
    - cover_area.true/false.adj_tri_ids sorted
    - everything normalized into json-stable primitives
    """
    out: Dict[str, Any] = {}

    for obj_id in sorted(acag.keys()):
        g = acag[obj_id]
        g_out: Dict[str, Any] = {}

        for tri in sorted(g.keys()):
            elem = g[tri]
            elem2 = dict(elem)

            # 1) sort adjacency
            if "adj" in elem2:
                elem2["adj"] = _sort_int_list(elem2.get("adj"))

            # 2) sort cover_area adj_tri_ids
            ca = elem2.get("cover_area")
            if isinstance(ca, dict):
                ca2 = dict(ca)
                for key in ("true", "false"):
                    sub = ca2.get(key)
                    if isinstance(sub, dict):
                        sub2 = dict(sub)
                        if "adj_tri_ids" in sub2:
                            sub2["adj_tri_ids"] = _sort_int_list(sub2.get("adj_tri_ids"))
                        ca2[key] = sub2
                elem2["cover_area"] = ca2

            g_out[str(int(tri))] = _normalize(elem2)

        out[str(int(obj_id))] = g_out

    return out


def stable_patch_fingerprint(patch: Dict[int, List[Dict[str, Any]]]) -> Any:
    out: Dict[str, Any] = {}
    for obj_id in tqdm(sorted(patch.keys()), total=len(patch.keys()), desc="Normalizing patch", leave=False):
        comps = patch[obj_id] or []
        norm_comps: List[Dict[str, Any]] = []

        for comp in tqdm(comps, total=len(comps), desc="Normalizing patch", leave=False):
            comp2 = dict(comp)
            if "component" in comp2 and isinstance(comp2["component"], list):
                comp2["component"] = sorted(int(x) for x in comp2["component"])
            if "adj" in comp2 and isinstance(comp2["adj"], list):
                comp2["adj"] = sorted(int(x) for x in comp2["adj"])
            norm_comps.append(_normalize(comp2))

        def key_fn(c: Dict[str, Any]) -> int:
            xs = c.get("component", [])
            return xs[0] if xs else 10**18

        norm_comps.sort(key=key_fn)
        out[str(int(obj_id))] = norm_comps
    return out


def compute_input_hash(obj1_id: int, obj2_id: int, df: pd.DataFrame, params: dict | None, policy_version: str | None) -> str:
    df_sig = int(pd.util.hash_pandas_object(df, index=True).sum()) if df is not None and not df.empty else 0
    payload = {
        "v": PATCH_CACHE_VERSION,
        "obj_pair": [int(obj1_id), int(obj2_id)],
        "df_sig": int(df_sig),
        "params": _normalize(params or {}),
        "policy_version": policy_version or "",
    }
    return _md5_bytes(_stable_json_dumps(payload))


def compute_content_hash(sum_df: Dict[int, pd.DataFrame], acag: Dict[int, Any], patch: Dict[int, Any]) -> str:
    payload = build_content_hash_payload(sum_df, acag, patch)
    return _md5_bytes(stable_json_dumps_bytes(payload))


def build_content_hash_payload(sum_df: Dict[int, pd.DataFrame], acag: Dict[int, Any], patch: Dict[int, Any]) -> Dict[str, Any]:
    sum_dict={}
    for k, v in tqdm(sum_df.items(), total=len(sum_df.items()), desc="Normalizing sum_df", leave=False):
        sum_dict[str(int(k))] = stable_df_fingerprint(v)
    return {
        "v": PATCH_CACHE_VERSION,
        "sum_df": sum_dict,
        "acag": stable_acag_fingerprint(acag),
        "patch": stable_patch_fingerprint(patch),
    }


def stable_json_dumps_bytes(obj: Any) -> bytes:
    return _stable_json_dumps(obj)
