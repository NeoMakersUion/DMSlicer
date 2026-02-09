
# Step 3｜Patch Aggregation & Topological Expansion（规则列表·定稿版）

## 3.1 目标

将 Step 2 输出的多个局部候选 patch，聚合为若干个**拓扑连通且几何一致**的公共区域；同时避免将空间上分离的多接触点结构错误合并。

---

## 3.2 输入与符号

对每个候选 patch 对 $P_k=(P_k^A,P_k^B)$，已知：

- 投影区域 $R_{A,k}, R_{B,k}$
    
- 支持度：局部 $\mathrm{SR}_k$，以及（可计算的）聚合支持度 $\mathrm{SR}_{\text{agg}}$
    
- gap 统计：$\bar g_k, \mathrm{Var}(g_k)$
    
- 法线统计：局部法线集合或其代表方向 $n_k$
    

---

## 3.3 正式定义：可合并性（Mergeability）

定义两个 patch $P_i, P_j$ 可合并，当且仅当同时满足以下四类条件。

---

### 条件 1：拓扑邻接门控（Topological Adjacency Gate）

**硬条件**：非邻接永不合并。

- 在 Object A 上：若 $P_i^A$ 与 $P_j^A$ 存在网格邻接（共享边的 1-ring 邻接，或允许 k-ring），则记为相邻；
    
- 或在 Object B 上：若 $P_i^B$ 与 $P_j^B$ 相邻。
    

形式化：  
$$ \mathrm{Adj}(P_i,P_j)=\mathrm{Adj}_A(P_i^A,P_j^A)\ \lor\ \mathrm{Adj}_B(P_i^B,P_j^B) $$  
要求：  
$$ \mathrm{Adj}(P_i,P_j)=\text{True} $$

**意义**：  
保证空间分离的多接触点（如凳子四脚）不会因同处一个平面而被错误合并。

---

### 条件 2：无向法线一致性（Undirected Normal Consistency）

你刚才确认的关键点：**法线只比较切平面方向，不比较朝向**。

对于代表法线 $n_i, n_j$，定义无向一致性：  
$$ |n_i \cdot n_j| > \cos(\theta_{\text{merge}}) $$

- 使用绝对值，使 $n$ 与 $-n$ 等价；
    
- $\theta_{\text{merge}}$ 控制允许的曲率变化幅度；
    
- 对于锐边/立面跳变（~90°），该条件会自然拒绝合并。
    

---

### 条件 3：gap 连续性约束（Gap Continuity Constraint）

合并不得引入显著更差的 gap 统计。

可用一个简单、可辩护的形式：  
$$ \bar g_{i\cup j} \le \max(\bar g_i,\bar g_j) + \delta_g $$  
并可选加入方差约束：  
$$ \mathrm{Var}(g_{i\cup j}) \le \max(\mathrm{Var}(g_i),\mathrm{Var}(g_j)) + \delta_v $$

**意义**：  
防止用“gap 大的误配片段”去桥接“gap 小的主接触区域”。

---

### 条件 4：支持度单调性（Support Monotonicity / Non-Degradation）

合并后的区域级支持度不能显著下降。

定义合并后的聚合支持度：  
$$ \mathrm{SR}_{\text{agg}}(P_i\cup P_j)=  
\frac{\mathrm{Area}(\tilde{R}_A \cap \tilde{R}_B)}  
{\min(\mathrm{Area}(\tilde{R}_A),\mathrm{Area}(\tilde{R}_B))} $$

要求（强版本）：  
$$ \mathrm{SR}_{\text{agg}}(P_i\cup P_j)  
\ge  
\min(\mathrm{SR}_{\text{agg}}(P_i),\mathrm{SR}_{\text{agg}}(P_j)) $$

或（弱版本，允许轻微波动）：  
$$ \mathrm{SR}_{\text{agg}}(P_i\cup P_j)  
\ge  
\min(\mathrm{SR}_{\text{agg}}(P_i),\mathrm{SR}_{\text{agg}}(P_j))-\delta_{sr} $$

**意义**：

- 允许“碎片化覆盖”在聚合后变得更强（这是你要的公平性）
    
- 同时拒绝“错误桥接”导致的覆盖度崩塌
    

---

## 3.4 聚合策略（Expansion Strategy，概念层）

在满足上述 mergeability 的前提下，采用拓扑扩展形成聚合区域：

- 从通过 seed 阶段门控的 patch 作为种子；
    
- 仅向其拓扑邻接 patch 扩展；
    
- 每次扩展必须满足“法线一致 + gap 连续 + 支持度不降”；
    
- 直到再无可合并邻居为止。
    

输出为若干个聚合组件：  
$$ {\mathcal{G}_1,\mathcal{G}_2,\dots} $$  
每个 $\mathcal{G}$ 对应一个公共接触区域候选。

---

## 3.5 Step 3 的边界声明（必写）

- Step 3 旨在恢复公共区域的拓扑完整性，而非强制形成单一最大区域；
    
- 对于几何突变或支持度不足的区域，公共面应自然分裂为多个 patch；
    
- 多接触点结构（如椅子脚/凳子脚）在拓扑上分离，应输出多个独立 patch。
    

---

# ✅ 到这里，你的 Step 3 已经是“可以写进论文”的规则版本

接下来，如果你愿意，我们可以做两件非常实用的事（你选一个）：

1. **把 Step 1–3 串成一张“方法流程图/伪代码级描述”**（方便放论文）
    
2. **用你真实数据里的一个典型问题（误判相交、平面延伸、曲面壳核）把三步映射到你当前代码结构**（方便落地到 DMSlicer）
    

你回一个数字就行。