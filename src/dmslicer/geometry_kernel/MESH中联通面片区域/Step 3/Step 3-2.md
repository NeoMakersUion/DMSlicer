好的！根据你的要求——**段内公式用 `$...$`，段间公式用 `$$...$$`**——我对全文的 LaTeX 公式格式进行了规范化处理，并保留了所有技术内容、逻辑结构和强调语气。以下是修改后的版本：

---

# Step 2｜Patch Support：把“面积”变成一等公民

你要的不是“有没有相交”，而是：

> **一片公共区域是否有足够的“面积支持”，从而可信。**

这里的关键是：  
在独立网格 + 离散误差下，**真正的 3D 交集面积未必能直接算**。  
所以我们引入一个可计算、可解释、可控的替代量：

## ✅ 2.1 “区域支持度”定义（Support, not exact intersection）

对一个候选 patch 对 $(P^A, P^B)$，定义一个局部参考平面 $\Pi$（patch 的主平面）：

- $\Pi$ 可以由 patch 的平均法线 $n_{\text{ref}}$ 和 patch 质心定义
    
- 或者用 PCA 拟合得到（你后面实现时选一种即可）
    

然后把 patch 内的三角形投影到 $\Pi$ 的 2D 参数域，得到：

- $\text{proj}(P^A)$
    
- $\text{proj}(P^B)$
    

### 支持度（核心量）：

$$ S(P) = \mathrm{Area}\big(\text{proj}(P^A)\ \cap\ \text{proj}(P^B)\big) $$

> 这不是“精确 3D 交集面积”，  
> 但它是一个**区域级的、对网格错位鲁棒、与“公共接触面”高度相关**的量。  
> 而且它天然过滤点/边接触（面积≈0）。

---

# Step 2 的核心：用 Support 做“验证与抑制误吸”

你刚才最担心的误差类型，我们用 Support + 其他指标一并解决。

---

## ✅ 2.2 Patch Validity（有效 patch）必须满足的最小条件

一个 patch 被接受，当且仅当同时满足：

### (A) 面积支持度足够

$$ S(P) > \varepsilon_S $$  
其中 $\varepsilon_S$ 不是拍脑袋，而是用局部尺度 $h$ 定义（继承第一步思想）：  
$$ \varepsilon_S = \eta \cdot h^2 $$  
（因为面积量纲是长度²）

> 这条直接把“零测度接触”排除出去了。

---

### (B) gap 一致（你第一步的 $\varepsilon$ 框架在这里派上用场）

$$ \bar g(P) < \varepsilon_k,\quad \mathrm{Var}(g(P)) < \sigma_g^2 $$

---

### (C) 法线统计一致（软门控+patch-level 统计）

$$ \mathrm{Var}(\angle(n_i, n_{\text{ref}})) < \sigma_\theta^2 $$

---

## ✅ 2.3 解决你提到的“平面延伸误吸”的关键：紧致度/边界复杂度

你指出“局部平滑也可能是错的、延伸出来一大片也平滑”，非常对。  
所以 Step 2 必须加一条**形状约束**：

### 紧致度（compactness）：

$$ C(P) = \frac{\mathrm{Perimeter}(\text{proj}(P))^2}{4\pi\ \mathrm{Area}(\text{proj}(P))} $$

直觉：

- 圆形最紧致，$C$ 最小
    
- 细长、蛇形、延伸的区域，$C$ 会变大
    

你可以设阈值：  
$$ C(P) < \kappa $$  
这样就能强力抑制“沿平面无限延伸吸进去”的错误 patch。

---

# Step 2 完成后，你会得到一个非常清晰的输出定义

> 输出不是 tri 对，而是 patch 对：  
> $(P^A_k, P^B_k)$，每个 patch 有一个支持度 $S_k$、平均 gap、法线方差、紧致度。  
> 你可以排序、筛选、可视化、统计。

这对后续可行性非常关键。

---

# 现在只问你一个问题（锁定 Step 2 的最核心选择）

在计算 support $S(P)$ 时，你更倾向于：

A) **严格交集面积**：投影后算 $A \cap B$ 的面积（更严格，计算稍重）  
B) **重叠覆盖度**：用 $\min(\mathrm{Area}(A), \mathrm{Area}(B))$ 与交叠比例（更鲁棒）

你只选 A 或 B。  
选完我们就把 Step 2 的“Support 指标”正式定稿（可写进论文），然后再进入 Step 3（怎么从 tri-pair 聚合成 patch）。