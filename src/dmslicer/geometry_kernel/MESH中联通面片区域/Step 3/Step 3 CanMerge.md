# 修正版：对称的“同侧连通 + 另一侧不分裂”合并条件

我们先定义两个概念（都对 A/B 对称）：

## 1) 同侧连通性：Connected on side $S$

对任意一侧 $S \in {A,B}$，令  
$$ U_S = P_i^S \cup P_j^S $$  
定义  
$$ \mathrm{Conn}_S(P_i,P_j)=1 \quad\Longleftrightarrow\quad U_S \text{ 在 mesh } S \text{ 的三角邻接图上是单连通分量} $$  
（也就是：把 $U_S$ 的三角形拿出来，看它们在该物体表面是否连成一片）
同理判断$\mathrm{Conn}_S({\bar S}(P_i,P_j)=1$，其中$\bar S$（若 $S=A$ 则 $\bar S=B$，反之亦然）

---

## 2) 另一侧不分裂：Not-split on the opposite side $\bar S$

对另一侧 $\bar S$（若 $S=A$ 则 $\bar S=B$，反之亦然），令投影并集区域：  
$$ \tilde R_{\bar S}(U_{\bar S}) = \mathrm{proj}_\Pi\big(P_i^{\bar S}\big)\ \cup\ \mathrm{proj}_\Pi\big(P_j^{\bar S}\big) $$  
定义“**不分裂**”为：  
$$ \mathrm{NotSplit}_{\bar S}(P_i,P_j)=1 $$  
只要满足以下任一条（给你两个等价选项，你选更好实现/更符合直觉的那条）：

**选项 (a) 连通性版本：**  
$$ \tilde R_{\bar S} \text{ 是单个连通区域} $$

**选项 (b) 距离版本（更工程更稳定）：**  
$$ \mathrm{dist}\big(\tilde R_{\bar S}(P_i^{\bar S}),\ \tilde R_{\bar S}(P_j^{\bar S})\big) < d_{\text{bridge}} $$  
（意思：在另一侧它俩不能隔得太远，否则就是“两个孤岛硬拼”）

---

# ✅ 最终你要的那条合并条件（对称、无角色假设）

把第 4 条写成：

$$ \boxed{ \mathrm{IntraOK}(P_i,P_j) ;:=; \Big(\mathrm{Conn}_A(P_i,P_j)\ \wedge\ \mathrm{NotSplit}_B(P_i,P_j)\Big) ;\vee; \Big(\mathrm{Conn}_B(P_i,P_j)\ \wedge\ \mathrm{NotSplit}_A(P_i,P_j)\Big) } $$

用人话解释就是：

> **只要存在一侧（A 或 B）能把两块 patch 连成一整片，  
> 同时另一侧不会表现为两个离得很远的孤岛，  
> 就允许合并。**  
> （完全不管哪一侧是“地面”、哪一侧是“凳子”）

---

# 把它塞回 Step 3.2（完整版本一句话）

```
CanMerge(P_i, P_j) iff:
    normal consistency
AND gap compatibility
AND support monotonicity
AND IntraOK(P_i, P_j)   # 上面的对称条件
```