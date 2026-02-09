

```python
import math
from typing import List, Tuple

# 假设 Triangle 类结构（实际使用时需与项目定义一致）
# class Triangle:
#     def __init__(self, vertices: List[Tuple[float, float]]):
#         self.vertices = vertices  # [(x1,y1), (x2,y2), (x3,y3)]

def compute_overlap_area(tri1, tri2) -> float:
    """
    计算两个三角形的重叠率（交并比 IoU: Intersection over Union）。
    
    说明：
    - 函数名含 "area" 但问题描述要求"重叠率"，在计算机视觉/几何计算中，
      "重叠率" 通常指 IoU（范围 [0, 1]），故返回 IoU 值。
    - 若需重叠面积（intersection area），可修改返回值为 intersection_area。
    - 假设 Triangle 对象有 .vertices 属性：List[Tuple[float, float]]（三个顶点）
    
    参数:
        tri1, tri2: Triangle 对象
    返回:
        float: 重叠率 (IoU)，范围 [0.0, 1.0]
               0.0 表示无重叠或退化三角形；1.0 表示完全重合
    """
    EPS = 1e-10
    
    # ========== 辅助函数 ==========
    def polygon_area(points: List[Tuple[float, float]]) -> float:
        """鞋带公式计算多边形有向面积"""
        area = 0.0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def ensure_ccw(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """确保顶点为逆时针顺序（面积为正）"""
        if polygon_area(points) < 0:
            return points[::-1]
        return points[:]

    def is_point_left_of_edge(p, a, b) -> bool:
        """判断点 p 是否在有向边 a->b 的左侧（含边界）"""
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        return cross >= -EPS

    def line_intersection(p1, p2, q1, q2) -> Tuple[float, float]:
        """计算直线 p1p2 与 q1q2 的交点（已知相交）"""
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = q1; x4, y4 = q2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # 理论上 denom 不应为0（因一点内一点外），但防浮点误差
        if abs(denom) < EPS:
            # 退化情况：返回线段中点（保守处理）
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def sutherland_hodgman(subject, clip) -> List[Tuple[float, float]]:
        """凸多边形裁剪：返回 subject ∩ clip 的顶点列表（clip 需为凸且 CCW）"""
        output = subject
        n = len(clip)
        
        for i in range(n):
            if not output:
                break
            cp1, cp2 = clip[i], clip[(i + 1) % n]
            input_pts = output
            output = []
            s = input_pts[-1]
            
            for p in input_pts:
                if is_point_left_of_edge(p, cp1, cp2):
                    if not is_point_left_of_edge(s, cp1, cp2):
                        output.append(line_intersection(s, p, cp1, cp2))
                    output.append(p)
                elif is_point_left_of_edge(s, cp1, cp2):
                    output.append(line_intersection(s, p, cp1, cp2))
                s = p
        return output

    # ========== 主逻辑 ==========
    # 获取并规范化顶点（逆时针）
    try:
        verts1 = ensure_ccw(tri1.vertices)
        verts2 = ensure_ccw(tri2.vertices)
    except AttributeError:
        raise ValueError("Triangle 对象需包含 .vertices 属性（三个 (x,y) 元组的列表）")
    
    # 计算原始面积（取绝对值）
    area1 = abs(polygon_area(verts1))
    area2 = abs(polygon_area(verts2))
    
    # 处理退化三角形（面积≈0）
    if area1 < EPS or area2 < EPS:
        return 0.0
    
    # 计算交集多边形（tri1 为裁剪窗口，tri2 为被裁剪多边形）
    intersection_pts = sutherland_hodgman(verts2, verts1)
    
    # 计算交集面积
    intersection_area = abs(polygon_area(intersection_pts)) if len(intersection_pts) >= 3 else 0.0
    
    # 避免浮点误差导致负面积
    intersection_area = max(0.0, min(intersection_area, area1, area2))
    
    # 计算 IoU（重叠率）
    union_area = area1 + area2 - intersection_area
    if union_area < EPS:
        return 0.0
    iou = intersection_area / union_area
    return max(0.0, min(1.0, iou))  # 严格限制在 [0,1]
```

### 关键说明：
1. **返回值含义**：  
   返回 **交并比 (IoU)**，即 `重叠面积 / (area1 + area2 - 重叠面积)`，符合“重叠率”在CV/几何计算中的通用定义（如目标检测中的 bbox 重叠率）。  
   *若实际需求为重叠面积（非比率），将最后一行改为 `return intersection_area` 即可。*

2. **鲁棒性设计**：
   - 自动处理顶点顺序（转为逆时针）
   - 浮点误差容错（EPS 阈值）
   - 退化三角形（面积≈0）安全返回 0.0
   - 交集多边形顶点数 < 3 时面积视为 0
   - 最终结果 clamp 到 [0, 1]

3. **算法选择**：  
   使用 **Sutherland-Hodgman 算法**（适用于凸多边形裁剪），高效可靠。两个三角形的交集必为凸多边形（0~6 边形），该算法能精确求解。

4. **使用前提**：  
   假设 `Triangle` 对象有 `.vertices` 属性（含三个 `(x, y)` 坐标元组）。若项目中 `Triangle` 结构不同，需调整顶点提取逻辑。

> 此实现平衡了准确性、健壮性与可读性，适用于大多数几何重叠计算场景。如需极致性能（如高频调用），可增加包围盒快速排斥检测优化。