


# 🧩 03 — OFFSET (Shell / Wall / Nozzle Compensation)
![alt text](03_offset_shell.png)
在切片软件中，`OFFSET` 表示：
> **把实体轮廓向内或向外平移一定距离，生成打印路径**

它对应的是：

| 切片术语   | Clipper   |
| ------ | --------- |
| 壁厚     | Offset    |
| 外壳     | Offset    |
| 喷嘴半径补偿 | Offset    |
| 多道壳    | 多次 Offset |

---

## 1️⃣ 本节用的场景

我们使用一个最简单的实体轮廓（来自上一节的结果）：

```python
solid = [(0,0),(20,0),(20,40),(0,40)]   # 左半实体
```

这是某一层中一个可打印区域。

---

## 2️⃣ 用 ClipperOffset 生成外壳

```python
co = pyclipper.PyclipperOffset()

co.AddPath(scale(solid),
           pyclipper.JT_MITER,           # 直角保持
           pyclipper.ET_CLOSEDPOLYGON)   # 闭合多边形

shells = co.Execute(int(2.0 * SCALE))   # 向外扩 2mm
```

含义是：

> 把这个实体轮廓向外平移 2mm
> 得到打印外壳路径

---

## 4️⃣ 在切片器中的意义

如果：

* 喷嘴直径 = 0.4mm
* 你要 3 层外壳

那么你做的是：

```text
原轮廓
→ offset 0.4
→ offset 0.8
→ offset 1.2
```

这就是：

> slicer 的外壳生成算法

---

## 5️⃣ 一句话总结

> **Offset = 把“实体边界”变成“打印路径”**

Union / Difference 决定

> **哪里有材料**

Offset 决定

> **打印机在哪里走**


