# 布尔运算：UNION

> **把两个重叠的矩形，变成一个“可打印的外轮廓”。**
![01_basic_boolean.png](01_basic_boolean.png)

也就是 slicer 的：

> **Layer region = rect1 ∪ rect2**

---

## 代码四个关键点

### ① 必须用整数

```python
SCALE = 1000000
(x, y) → (int(x*SCALE), int(y*SCALE))
```

Clipper 只在 **int64** 中保证拓扑稳定。
这是切片软件必须做的事。

---

### ② 一个是 SUBJECT，一个是 CLIP

>SUBJECT 是“主体物体”
>CLIP 是“用来作用在主体上的东西”

```python
pc.AddPath(scale(rect1), pyclipper.PT_SUBJECT, True)
pc.AddPath(scale(rect2), pyclipper.PT_CLIP, True)
```

含义是：

> 用 rect2 去参与 rect1 的布尔运算

这是 Clipper 的拓扑语义，不是装饰。

---

### ③ UNION = “这一层哪里是实体”

```python
result = pc.Execute(pyclipper.CT_UNION)
```

你得到的不是两个矩形，
而是：

> **rect1 ∪ rect2 的外边界**

也就是 slicer 里真正要打印的区域。

---

### ④ 返回的是“闭合轮廓列表”

```python
result = [ [ (x1,y1), (x2,y2), ... ], ... ]
```

每一个 path = 一个**闭合实体边界**
这是你后续做：

* topo rings
* infill
* shell
  的基础。

---

## 一句话总结

这 20 行代码做的事情是：

> **把多个物体在这一层的投影，转换成一个“真实材料区域”的拓扑边界**

这一步就是 slicer 的几何心脏。
