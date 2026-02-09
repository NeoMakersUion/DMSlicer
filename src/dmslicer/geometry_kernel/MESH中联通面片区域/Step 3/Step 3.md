
# 总体视角：我们现在到底在做什么？

一句话版目标（先钉死）：

> **从两个三角网格 object 中，识别“几何意义上真实存在的公共接触区域”，  
> 在允许离散误差与不同网格划分的前提下，输出稳定、可聚合的区域级结果。**

这句话决定了一切设计。

---

# 顶层流程（完全重来版）

```
Input:
    Mesh A, Mesh B

Output:
    A set of contact regions:
        { Region_1, Region_2, ... }
    where each Region is a group of paired surface patches
```

---

# 模块 0：BVH 查询（唯一继承旧世界的部分）

### 模块名

**Candidate Pair Generation (BVH-based)**

### 职责（只做一件事）

> **快速找出“空间上可能接近”的三角形对**

### 输出（非常关键）

不是“相交 / 不相交”，而是：

```
CandidateTrianglePairs = {
    (triA_i, triB_j)
}
```

⚠️ 注意：

- **不在这里做任何几何判定**
    
- 不管平行、不管投影、不管相交
    
- BVH 只是**空间候选生成器**
    

---

# Step 1：Triangle-level 几何关系抽象（不做判决）

### 模块名

**Local Geometric Relation Extraction**

### 输入

```
(triA, triB) from CandidateTrianglePairs
```

### 输出

一个**结构化几何关系描述**，而不是布尔值：

```
LocalRelation {
    triA_id
    triB_id

    distance_stats:
        min_gap
        avg_gap

    normal_relation:
        abs_dot = |nA · nB|

    projection_hint:
        estimated local tangent plane (optional)
}
```

### 关键思想

> **Step 1 不回答“是不是接触”，  
> 只回答“它们之间的几何关系是什么样的”。**

这里没有 ε 判死刑，  
只有“信息抽取”。

---

# Step 2：Patch Pair 构建（从点对到“区域假设”）

这是**你整个方法的核心升级点**。

---

## Step 2.1：Triangle Pair → Patch Pair Seed

### 模块名

**Patch Pair Seeding**

### 逻辑

```
Initialize empty PatchPair list

For each LocalRelation:
    if normal_relation.abs_dot < normal_seed_threshold:
        discard   # 连切平面都对不上，直接丢
    else:
        assign (triA, triB) to a PatchPair seed
```

### PatchPair（再次强调）

```
PatchPair P = (P^A, P^B)

where:
    P^A = set of triangles on Object A
    P^B = set of triangles on Object B
```

⚠️ 注意：

- Patch 是**跨两个 object 的**
    
- 不是单侧 patch
    

---

## Step 2.2：Patch Pair 内部支持度计算（区域级）

### 模块名

**Support Evaluation**

对每一个 PatchPair P：

```
Compute projected region R_A from P^A
Compute projected region R_B from P^B

Compute overlap area:
    A_overlap = Area(R_A ∩ R_B)

Compute support ratio:
    SR(P) = A_overlap / min(Area(R_A), Area(R_B))
```

这里你已经选择了 **B（min-area 归一化）**，这是对的。

---

## Step 2.3：两阶段门控（防“一巴掌打死”）

### 模块名

**Two-stage Gating**

```
if SR(P) < seed_threshold:
    discard P
else:
    keep P as valid patch candidate
```

你后面可以有：

- seed_threshold（宽松）
    
- final_threshold（严格）
    

但逻辑层面**先不锁死参数**。

---

# Step 3：Patch 聚合与拓扑扩展（区域恢复）

现在我们终于来到 **“真实公共区域”的恢复**。

---

## Step 3.1：构建 Patch 邻接图

### 模块名

**Patch Adjacency Graph Construction**

定义图：

```
Node = PatchPair P
Edge(P_i, P_j) exists if:
    P_i^A and P_j^A are adjacent on mesh A
    OR
    P_i^B and P_j^B are adjacent on mesh B
```

⚠️ 这一步**直接保证了**：

- 凳子四脚不会合并
    
- 空间分离的接触不会被“面化”
    

---

## Step 3.2：可合并性判定（Mergeability Test）

对相邻 PatchPair：

```
CanMerge(P_i, P_j) iff:

1) Undirected normal consistency:
   |n_i · n_j| > θ_n

2) Gap compatibility:
   gap(P_i ∪ P_j) is consistent

3) Support monotonicity:
   SR_agg(P_i ∪ P_j) ≥ min(SR_agg(P_i), SR_agg(P_j)) − δ

4) Intra-object connectivity (strict):
   Conn_A(P_i, P_j) AND Conn_B(P_i, P_j)

```

其中：

- `Conn_S` = 在 mesh S 上属于同一连通分量
    
- 不需要投影
    
- 不区分 A/B 角色
    
- 不受网格密度影响。

---

## Step 3.3：区域级聚合（Region Formation）

```
Initialize empty Region list

For each PatchPair not yet assigned:
    start new Region
    BFS / DFS expand:
        add neighbor PatchPairs
        if CanMerge holds
```

最终输出：

```
Region_k = { P_1, P_2, ... }
```

---

# 最终输出是什么？

不是：

- triangle 对
    
- 也不是 patch 对
    

而是：

> **一组“区域级公共接触结构”**

每个 Region：

- 可统计总接触面积
    
- 可用于制造 / 物理 / 语义分析
    
- 对离散误差鲁棒
    

---

# 非常重要的一点（你刚才的决定是对的）

你现在做的事情，本质上是：

> **从“几何判交算法”  
> 升级为  
> “区域级接触建模算法”**

这两个东西**在研究范式上完全不同**。

你现在这套逻辑：

- 不依赖“理想数学相交”
    
- 明确承认离散误差
    
- 通过 patch + 支持度 + 拓扑恢复真实结构
    

👉 这是**研究级方法**，不是工程 trick。

---

## 下一步（按你原计划）

你说的是：

> ✔ 用真实问题走一遍  
> ✔ 反思有没有漏洞  
> ✔ 没问题再画流程图 / 伪代码图

所以我现在只问你一句：

> **你想先用哪一个真实场景，来“手推”这套新流程？**  
> 1️⃣ 平面–平面（不同网格密度）  
> 2️⃣ 曲面壳–核（球壳 / 曲面）  
> 3️⃣ 凳子/椅子多接触点结构  
> 4️⃣ 你真实数据里“最烦”的那个案例（你描述）

你选一个，我们下一步就**只干这一件事**。