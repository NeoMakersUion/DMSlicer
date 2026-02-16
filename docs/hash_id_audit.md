# Hash ID 审计报告

- 扫描范围：src/dmslicer
- 生成时间：自动脚本（scripts/hash_id_audit.py）支持增量扫描

## 摘要结论
- 内容地址化（Content-addressed）：已在 AMF 解析阶段对上游文件采用 SHA-256（工作空间与缓存目录基于 `model.hash_id`）。这是稳定且可追溯的方案。
- 内部对象与几何归一化层：`Object.hash_id` 与 `Geom.hash_id` 使用 Python 内置 `hash()`，非确定性（受 PYTHONHASHSEED 影响），不满足跨进程稳定与固定长度要求，不适合作为持久化/路径键。
- Patch 层：`Patch.hash_id` 字段存在但未正确实现（赋值为元组，占位语义明显）。

## 发现明细

### 1) 文件哈希（内容哈希）
- 路径：`src/dmslicer/file_parser/workspace_utils.py`
- 接口：`sha256_of_file(filepath) -> str`
- 算法：SHA-256（`hashlib.sha256`），hexdigest，64 个十六进制字符
- 盐值：无（内容地址化不需要盐）
- 冲突处理：未见显式逻辑（SHA-256 冲突概率可忽略）
- 下游：`amf_parser.read_amf_objects` → `Model.hash_id`、workspace 查找与缓存
- 业务要求评估：
  - 唯一性：强（基于文件内容）
  - 不可预测性：强（对于未知内容）
  - 固定长度：是（64 hex）
- 单测：`tests/test_amf_parser.py` 通过猴补替换稳定值覆盖调用路径（间接覆盖）
- 风险与建议：无明显风险；建议保留现状，或在大文件场景采用分块读取（已实现）。

### 2) 几何对象哈希（Object）
- 路径：`src/dmslicer/geometry_kernel/object_model.py`
- 接口：`Object.hash_id`（属性）
- 实现：`__hash__` 使用内置 `hash((acc, id, tri_id2geom_tri_id.tobytes(), color.tobytes(), triangle_ids_order, status))`
- 算法：Python 内置 `hash()`（SipHash，存在随机种子）
- 盐值：解释器级（PYTHONHASHSEED），非稳定
- 长度/字符集：整数转字符串，非固定宽度
- 冲突处理：未见
- 下游：未发现跨进程持久化用法；主要为内存内标识/调试
- 业务要求评估：
  - 唯一性：弱（冲突可能且依赖运行时状态）
  - 不可预测性：高但非确定；对缓存/路径不稳定
  - 固定长度：否
- 单测：未见
+- 风险与建议：若需持久化/跨进程稳定，建议改为 `SHA-256` 基于稳定序列化（例如将关键 ndarray 以 C-order bytes 拼接并附上 shape/dtype，再做哈希），并缓存计算结果。

### 3) 归一化几何哈希（Geom）
- 路径：`src/dmslicer/geometry_kernel/canonicalize.py`
- 接口：`Geom.hash_id`（属性）
- 实现：`__hash__` 使用内置 `hash((acc, model.hash_id, vertices_order, triangles_order))` → 转为字符串
- 算法：Python 内置 `hash()`（非稳定）
- 盐值：解释器级
- 长度/字符集：整数转字符串，非固定宽度
- 冲突处理：未见
- 下游：GEOM_CACHE 键使用 `model.hash_id`（稳定）；但 `Geom.hash_id` 自身若用于其他缓存/日志可能不稳定
- 业务要求评估：
  - 唯一性：弱（取决于内置 hash 特性）
  - 不可预测性：有，但非确定
  - 固定长度：否
- 单测：未见
- 风险与建议：若需要对不同排序参数组合进行稳定区分，建议使用 `SHA-256( model.hash_id + json.dumps(params, sort_keys=True) )` 并截断为短哈希（如前 12～16 hex）。

### 4) Patch 层哈希（Patch）
- 路径：`src/dmslicer/geometry_kernel/patch_level.py`
- 字段：`hash_id: str`
- 现状：构造中出现 `hash_id = obj1, obj2, df` 的占位式赋值；未作为稳定 ID 实现
- 建议：若需要标识 patch 产物，建议定义：`SHA-256( obj1.id | obj2.id | params(angle/gap/overlap) | df fingerprint )`，其中 `df fingerprint` 可使用 `pandas.util.hash_pandas_object(...).sum()` 或持久化输入 parquet 的文件哈希。

## 差异对比与评估

| 组件 | 算法 | 稳定性 | 安全性 | 性能 | 是否满足唯一/不可预测/固定长 |
| --- | --- | --- | --- | --- | --- |
| 文件哈希 | SHA-256 | 稳定 | 强 | O(file size) | 是/是/是 |
| Object.hash_id | 内置 hash | 不稳定 | 弱 | O(data size) | 否/否/否 |
| Geom.hash_id | 内置 hash | 不稳定 | 弱 | 低 | 否/否/否 |
| Patch.hash_id | 未实现 | 无 | 无 | 无 | 否 |

## 改进建议
1. 将 `Object.hash_id` 与 `Geom.hash_id` 迁移为基于 `SHA-256` 的稳定计算，对大数组采用一次性缓存（lazy 计算后存入私有字段）。
2. 为 Patch 定义稳定 `hash_id`（包含对象 id、参数组、输入 df 指纹），并用于落盘与复现。
3. 在 `Model.load` 增加可选一致性校验（例如携带源文件 SHA-256 或元数据哈希），避免外部篡改。
4. 在 CI 中增加脚本校验：运行 `python scripts/hash_id_audit.py --root src --out docs/hash_id_latest.md`，对新增内置 `hash()` 用途进行提示或阻断。

## 测试与覆盖
- 现有：`tests/test_amf_parser.py` 通过猴补 `sha256_of_file` 覆盖上游路径。
- 建议新增：
  - `tests/test_hash_object_geom.py`：验证新方案下 `Object.hash_id` 与 `Geom.hash_id` 的稳定性（同一输入多进程一致）。
  - `tests/test_patch_hash_id.py`：验证 Patch 构成的参数变化是否反映到 `hash_id`。

## 附录：快速扫描输出
- 使用脚本：`python scripts/hash_id_audit.py --root src --out docs/hash_id_latest.md`
- 表头说明：File/Line/Pattern/Snippet，其中 Pattern ∈ {hashlib, hexdigest, uuid, secrets, builtins_hash, name}

