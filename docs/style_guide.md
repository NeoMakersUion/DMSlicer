# DMSlicer Style Guide
## DMSlicer 编码与注释规范

This document defines code, documentation, and caching conventions for DMSlicer.
本文档定义 DMSlicer 的代码、文档、缓存落盘等统一规范。

---

### 1. Goals
### 1. 目标

Write code that is readable, testable, cacheable, and reproducible.
编写可读、可测、可缓存、可复现的工程代码。

Prefer deterministic outputs over "clever" shortcuts.
优先保证结果确定性，而不是追求“技巧性写法”。

Make expensive geometry stages resumable by disk artifacts.
让昂贵的几何计算阶段可以通过磁盘产物断点续跑。

---

### 2. Language & Comment Policy
### 2. 语言与注释策略

All public docstrings and key logic comments must be bilingual.
所有公共接口 docstring 与关键逻辑注释必须双语。

Rule: one English sentence followed by one Chinese sentence.
规则：英文一句后紧跟中文一句。

Keep pairs aligned (1:1) whenever possible.
尽量保持一一对应的语义配对。

Avoid overly long comment lines; wrap at ~88 characters.
避免超长注释行；建议在 ~88 字符处换行。

Use concise engineering terms; avoid rhetorical or emotional wording.
使用简洁工程术语，避免抒情/口语化表达。

---

### 3. Naming Conventions
### 3. 命名规范

Use `snake_case` for functions/variables and `PascalCase` for classes.
函数/变量用 `snake_case`，类用 `PascalCase`。

Prefer semantic names that reveal structure and role.
优先使用能体现结构与角色的语义化命名。

Examples: `pair_level_files`, `pair_dir`, `acag`, `content_hash`.
示例：`pair_level_files`、`pair_dir`、`acag`、`content_hash`。

Avoid `res`, `tmp`, `data1` outside tiny local scopes.
避免在非极小局部范围使用 `res/tmp/data1` 等弱命名。

Use explicit units in names when relevant (e.g., `_mm`, `_deg`).
涉及单位时在命名中体现（如 `_mm`、`_deg`）。

---

### 4. Type Hints & Data Shapes
### 4. 类型标注与数据形状

Use Python type hints for public APIs and non-trivial internals.
公共 API 与重要内部函数必须有类型标注。

Use `dict[int, list[int]]` style for adjacency lists.
邻接表使用 `dict[int, list[int]]` 这类明确结构。

Document DataFrame schemas in docstrings.
DataFrame 的列名/含义需要在 docstring 中说明。

For numpy arrays, document dtype and shape (e.g., `int64[N,3]`).
numpy 数组注明 dtype 与 shape（如 `int64[N,3]`）。

---

### 5. Docstring Template
### 5. Docstring 模板

Use the following structure for public functions/classes.
公共函数/类 docstring 建议使用下述结构。

- One-line summary.
- 一行摘要。

- Design intent / Philosophy.
- 设计意图 / 思想。

- Parameters / Returns / Raises.
- 参数 / 返回 / 异常。

- Side effects (IO, visualization, mutation).
- 副作用（IO/可视化/状态修改）。

- Performance notes (Big-O, memory).
- 性能说明（复杂度与内存）。

Example skeleton:
示例骨架：

```text
Summary.
摘要。

Design intent.
设计意图。

Parameters.
参数。

Returns.
返回。

Raises.
异常。

Side effects.
副作用。

Performance notes.
性能说明。


## Hash Contract (Patch Cache) / 哈希契约（Patch 缓存）

### 1. Purpose / 目的

**English:**  
The Patch cache uses a deterministic content hash to guarantee that cached artifacts
(sum_df / acag / patch) exactly match the in-memory computation result.  
The hash is used as an integrity check during `Patch.load(..., validate_hash=True)`.

**中文：**  
Patch 缓存通过“确定性内容哈希”保证落盘产物（sum_df / acag / patch）与内存计算结果严格一致。  
在 `Patch.load(..., validate_hash=True)` 时使用该哈希做完整性校验。

---

### 2. What must be stable / 必须稳定的内容

The content hash MUST be invariant under:  
- dict insertion order differences  
- list order differences where the list is semantically a set (e.g., adjacency lists)  
- numpy scalar types (np.int64 vs int)  
- pandas column order / row order differences (if not semantically meaningful)

内容哈希必须对以下变化保持不变：  
- dict 插入顺序不同  
- 列表顺序不同但语义上是集合的字段（例如邻接列表）  
- numpy 标量类型差异（np.int64 vs int）  
- pandas 列顺序 / 行顺序差异（若这些顺序不表达语义）

---

### 3. Hash scope / 哈希覆盖范围

**The hash covers ONLY these artifacts:**
1) `sum_df`: per-object summary DataFrame  
2) `acag`: adjacency-curvature-area graph per object  
3) `patch`: connected components + cross-links per object

哈希只覆盖这三类产物：  
1) `sum_df`：每个 object 的摘要 DataFrame  
2) `acag`：每个 object 的邻接-曲率-面积图  
3) `patch`：每个 object 的连通组件及跨对象链接

**The hash MUST NOT depend on:**  
- filesystem paths  
- timestamps (created_at)  
- visualization flags (show)  
- any debug-only fields

哈希不得依赖：  
- 文件路径  
- 时间戳（created_at）  
- 可视化开关（show）  
- 任何调试字段

---

### 4. Canonicalization rules / 规范化规则

To ensure determinism, we canonicalize data before hashing:

#### 4.1 DataFrame (`sum_df`) canonicalization
**English:**  
- Reindex columns by sorted column names  
- Sort rows by `tri_id` if present; otherwise sort by all columns  
- For object dtype cells (dict/list/ndarray), convert each cell to a stable JSON string  
- Finally compute `hash_pandas_object(...).sum()` as the fingerprint

**中文：**  
- 列按列名排序重排（sorted columns）  
- 行优先按 `tri_id` 排序；否则按全部列排序  
- 对 object 列（dict/list/ndarray）逐单元转换为“稳定 JSON 字符串”  
- 最终用 `hash_pandas_object(...).sum()` 得到指纹

#### 4.2 ACAG canonicalization
**English:**  
- sort object ids ascending  
- within each object, sort tri ids ascending  
- normalize `adj` into sorted integer list  
- normalize `cover_area.true/false.adj_tri_ids` into sorted integer list  
- keep other scalar fields as JSON-stable primitives

**中文：**  
- object_id 升序  
- 每个 object 内 tri_id 升序  
- `adj` 统一为“排序后的 int 列表”  
- `cover_area.true/false.adj_tri_ids` 统一为“排序后的 int 列表”  
- 其他标量字段保持为 JSON 稳定基础类型

#### 4.3 Patch canonicalization
**English:**  
- sort object ids ascending  
- normalize each component:
  - `component` sorted int list
  - `adj` sorted int list
- sort components by `min(component)` as a stable ordering

**中文：**  
- object_id 升序  
- 每个 component 规范化：
  - `component` 排序后的 int 列表
  - `adj` 排序后的 int 列表
- component 以 `min(component)` 作为排序键保证稳定顺序

---

### 5. Streaming token protocol / 流式 token 协议

**English:**  
`compute_content_hash()` MUST use a deterministic token stream:
- header token includes version
- then sum_df fingerprints in sorted object_id order
- then acag tokens (obj, tri) in stable order
- then patch tokens (obj, comp) in stable order
Each token is serialized with stable JSON and fed into MD5 incrementally.

**中文：**  
`compute_content_hash()` 必须采用确定性的 token 流：
- 头部 token 包含版本号
- sum_df 指纹按 object_id 排序写入
- acag token（obj, tri）按稳定顺序写入
- patch token（obj, comp）按稳定顺序写入
每个 token 以稳定 JSON 序列化后 `md5.update()` 流式累积。

---

### 6. Compatibility (classic vs fast) / 兼容性（classic 与 fast）

**Rule / 规则：**  
`hash_utils.py` (classic) and `hash_utils_fast.py` (fast streaming) MUST compute
the SAME hash for semantically equivalent content.

**Important / 重要：**  
If `acag.cover_area.*.adj_tri_ids` is collected from `set(...)`, its order is undefined.
Therefore, canonicalization MUST sort those lists before hashing.

---

### 7. Debug workflow for hash mismatch / 哈希不一致调试流程

When `validate_hash=True` fails:

1) Dump debug bundle: env.json / meta.json / payload.json / payload.bin  
2) Compare canonical forms:
   - compare canonicalized ACAG (sorted lists) first
   - then compare patch canonical form
   - finally inspect DataFrame object columns

当 `validate_hash=True` 失败时：

1) 落盘 debug 包：env.json / meta.json / payload.json / payload.bin  
2) 对比 canonical form：
   - 先对比 ACAG（关注排序后的列表字段）
   - 再对比 patch canonical
   - 最后检查 DataFrame object 列的规范化结果

**Common root causes / 常见原因：**
- any unsorted list that is semantically a set (adj lists, adj_tri_ids)
- msgpack/pickle load returning different numeric types (np.int64 vs int)
- DataFrame object cells not canonicalized into stable strings

常见原因：
- 语义是集合但未排序的 list（adj、adj_tri_ids）
- msgpack/pickle 读回类型差异（np.int64 vs int）
- DataFrame object 单元格未规范化成稳定字符串
