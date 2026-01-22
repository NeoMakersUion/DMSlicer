# 🧩 05 — Segments → Loops

> 把 STL 切片得到的零散线段，拼成闭合轮廓

在真实切片中，一个 Z 平面切到 STL 网格后，得到的不是多边形，而是：

```
[(p1,p2), (p3,p4), (p5,p6), ...]
```

也就是一堆**无序的线段**。
切片器必须把它们拼成闭合 loop，才能进入 Clipper 的世界。

---

## 1️⃣ 为什么不能直接用 pyclipper？

Clipper 需要的是**闭合多边形**。
而 STL slice 给你的是**开口线段**。

所以你必须先做这一步：

> **Segment stitching（线段拓扑缝合）**

---

## 2️⃣ 关键工具：端点量化（容错）

```python
def snap_point(pt, eps):
    return (round(x/eps)*eps, round(y/eps)*eps)
```

STL 交线存在浮点误差：

```
(40.0000001, 30.0)
(39.9999999, 30.0)
```

它们应该是同一点。
`snap_point` 把它们吸附到同一网格上，让端点可以被正确匹配。

> 这是 slicer 能否闭合轮廓的生死线。

---

## 3️⃣ 建立“端点 → 线段”的拓扑表

```python
adj = {}
for i, (a,b) in enumerate(segs):
    adj[a].append((i,0))
    adj[b].append((i,1))
```

这一步构造的是：

> **“这个点连接了哪几条线段？”**

这是一个典型的**图结构**：

* 节点 = 端点
* 边 = 线段

---

## 4️⃣ 用图遍历把线段串成环

核心逻辑：

```python
used = [False] * len(segs)

for each unused segment:
    从它开始
    一直沿着端点连接走
    直到回到起点
```

关键代码：

```python
loop = [a, b]
curr = b
```

表示：

> 我从一条线段的 a→b 开始沿着轮廓走

---

## 5️⃣ 沿着端点一直“走轮廓”

```python
candidates = adj[curr]
找一条还没用过的线段
让它从 curr 继续走
```

这一步就是 slicer 里的：

> **Follow contour**

你不是在搜索几何最近点，而是在走**拓扑连接**。

---

## 6️⃣ 闭合检测

```python
if curr == loop[0]:
    这是一个闭合 loop
```

这一步就是 slicer 的：

> **“我围了一圈，又回到起点了”**

此时这个轮廓完成。

---

## 7️⃣ 输出的 loops 是什么？

你最终得到的是：

```
[
  [p0,p1,p2,p3,...],   # 一个闭合轮廓
  [q0,q1,q2,...],     # 另一个岛
]
```

这正是 Clipper 所需要的输入格式。

---

## 一句话总结

> **`segments_to_loops` 是 slicer 中“几何→拓扑”的桥梁。**

STL 给你碎线，
你把它变成可打印的轮廓。

之后才轮到：

* Union / Difference
* Offset
* Infill

你现在已经站在 slicer 内核的入口了。
