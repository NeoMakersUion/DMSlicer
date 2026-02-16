# ACAG迁移关键点

## 1. 概述
ACAG（AdjacencyCurvatureAreaGraph，邻接曲率面积图）用于描述三角网格面片之间的邻接关系、法线角度变化以及覆盖面积等几何属性。

## 2. 关键映射

| 旧名              | 新名              | 说明               |
|-------------------|-------------------|--------------------|
| graph             | acag              | 内部字段名         |
| files["graph"]    | files["acag"]     | 磁盘元数据字段     |
| _graph_to_csr_npz | _acag_to_csr_npz  | 序列化方法名       |

> ✅ 旧数据通过 `files["graph"]` 兼容读取，新数据强制用 `acag`。

