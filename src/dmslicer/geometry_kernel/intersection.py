# -*- coding: utf-8 -*-
"""
Intersection
============

Pure geometry intersection utilities.

This module must be stateless:
- no dependency on GeometryKernel / topology / index
- only takes raw points/triangles/planes and returns results

Primary use in slicing:
    plane Z = z  ∩  triangle(v0,v1,v2)
"""
from __future__ import annotations
from .config import GEOM_ACC
from copy import deepcopy
from typing import Optional, Tuple, List, Union
import numpy as np
from enum import Enum
# 几何精度阈值（复用你前序的精度控制，适配浮点误差）
LEN_EPSILON = 10 ** int(GEOM_ACC/2)

class PointLocation(Enum):
    """点与三角形的精确位置关系（互斥状态）"""
    INSIDE = 1
    ON_EDGE = 0
    OUTSIDE = -1



class InterSect(Enum):
    NO_INTERSECT = 0
    INTERSECT = 1
    UNKNOWN = -1

def triangle_normal_vector(verts_3d):
    """
    由三角形的3个3D顶点坐标计算单位法向量
    :param verts_3d: 三角形顶点坐标NumPy数组，形状(3,3)，每行是(x,y,z)
    :return: 单位法向量NumPy数组，形状(3,)；若三点共线，返回None（无效三角形）
    """
    # 步骤1：提取3个顶点并计算两个边向量
    v0, v1, v2 = verts_3d[0], verts_3d[1], verts_3d[2]
    vec1 = v1 - v0  # 边向量1: P1-P0
    vec2 = v2 - v0  # 边向量2: P2-P0
    
    # 步骤2：叉乘计算原始法向量（NumPy内置cross函数，高效且准确）
    raw_normal = np.cross(vec1, vec2)
    
    # 步骤3：计算法向量模长，判断是否为有效三角形（三点不共线）
    norm_len = np.linalg.norm(raw_normal)
    if norm_len < 0.1:
        # 模长接近0，说明三点共线，无有效法向量
        return None
    
    # 步骤4：单位化法向量（模长归一化为1）
    unit_normal = raw_normal / norm_len
    return unit_normal

import numpy as np

# ---------- 法向 ----------
def triangle_normal(tri):
    n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        return None
    return n / ln


def triangles_3Dto2D(tri1,tri2):
    tri1=deepcopy(tri1)
    tri2=deepcopy(tri2)
    center=np.mean(tri1,axis=0)
    tri1=tri1-center
    tri2=tri2-center
    n=triangle_normal(tri1)
    target=np.array([0., 0., 1.])
    axis = np.cross(n, target)
    kx, ky, kz = axis
    R = np.array([
        [kx**2,      kx*ky,     ky],
        [kx*ky,      ky**2,    -kx],
        [-ky,         kx,       0.]
    ])
    tri_1_rotated = (R @ tri1.T).T
    tri_2_rotated = (R @ tri2.T).T
    return tri_1_rotated[:,:2], tri_2_rotated[:,:2]

def point_in_triangle_cross(p, tri, eps=1):
    """
    叉乘符号法（同向法）判断点在三角形内/边/外
    :param p: 待判断点(x,y)，a/b/c：三角形顶点（顺/逆时针顺序均可，需统一）
    :param eps: 浮点精度阈值，处理数值误差
    :return: 状态
    """
    from .intersection import PointLocation
    # 转换为numpy数组，方便向量计算
    p0=tri[0]
    p1=tri[1]
    p2=tri[2]
    v0=p1-p0
    v1=p2-p1
    v2=p0-p2
    vp0=p-p0
    vp1=p-p1
    vp2=p-p2
    c1=np.cross(v0,vp0)
    c2=np.cross(v1,vp1)
    c3=np.cross(v2,vp2)

    # 处理浮点误差：接近0的数视为0
    c1 = 0 if abs(c1) < eps else c1
    c2 = 0 if abs(c2) < eps else c2
    c3 = 0 if abs(c3) < eps else c3
    
    # 判断符号：全正/全负=内部；有0=边上；有正有负=外部
    if (c1 > 0) and (c2 > 0) and (c3 > 0):
        return PointLocation.INSIDE
    elif (c1 < 0) and (c2 < 0) and (c3 < 0):
        return PointLocation.INSIDE
    elif (c1 == 0) or (c2 == 0) or (c3 == 0):
        return PointLocation.ON_EDGE
    else:
        return PointLocation.OUTSIDE

def orient(a, b, c, eps=1e-9):
    """
    返回:
    +1: 左侧
    -1: 右侧
     0: 共线（在 eps 范围内）
    """
    val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    if val > eps:
        return 1
    elif val < -eps:
        return -1
    else:
        return 0
        
def segments_strictly_intersect(seg1, seg2, eps=1e-9):
    A, B = seg1
    C, D = seg2
    o1 = orient(A, B, C, eps)
    if o1 == 0:
        return False
    o2 = orient(A, B, D, eps)
    if o2==0:
        return False
    if o2 == o1:
        return False
    o3 = orient(C, D, A, eps)
    if o3==0:
        return False
    o4 = orient(C, D, B, eps)
    if o4==0:
        return False
    if o3 == o4:
        return False
    return True
    # 严格相交：四个方向值必须全非 0，且成对异号


def seg_intersect(seg1,seg2,eps:float=1e-8):
    return segments_strictly_intersect(seg1, seg2, eps)

def tri_edge_intersect(tri1,tri2):
    """
    通过穷举所有边对边（Edge-to-Edge）的相交性，判断两个三角形是否发生边界穿插。
    
    算法意图 (Algorithm Intent):
    检测两个三角形的边界线段是否存在严格相交（Strict Intersection）。
    如果存在任意一对边相交，则认为两个三角形在几何上发生了重叠或穿插。
    
    注意：
    - 此函数通常用于共面三角形或 2D 投影后的三角形检测。
    - 仅检测边界相交，不处理包含关系（包含关系由 check_cross_or_not_by_p 处理）。
    
    :param tri1: 三角形1的3个顶点坐标 NumPy 数组 (3, 3)
    :param tri2: 三角形2的3个顶点坐标 NumPy 数组 (3, 3)
    :return: 
        - InterSect.INTERSECT: 发现边与边相交
        - InterSect.UNKNOWN: 未发现边相交（可能分离，也可能包含）
    """
    # 1. 构建两个三角形的边段列表 (Edges Construction)
    segs_tri1=[[tri1[i],tri1[(i+1)%3]] for i in range(3)]
    segs_tri2=[[tri2[i],tri2[(i+1)%3]] for i in range(3)]
    
    # 2. 穷举所有边对 (3x3=9 pairs) 进行相交检测
    for seg1 in segs_tri1:
        for seg2 in segs_tri2:
            # 使用严格线段相交判定
            if seg_intersect(seg1,seg2):
                return InterSect.INTERSECT
                
    # 3. 若无边相交，返回未知状态（需进一步检测包含关系）
    return InterSect.UNKNOWN


import numpy as np
from .canonicalize import Triangle
import numpy as np

import numpy as np

def nearest_point_on_triangle(
    point: np.ndarray, 
    triangle: "Triangle",
    eps: float = 1e-8
) -> tuple[np.ndarray, float]:
    """
    Nearest Point Projection onto Triangle (Ericson §5.1.5)
    
    专为 DMSlicer 整数化坐标系优化（×1000），仅依赖 Triangle.vertices，
    不依赖 normal/edges/topology 等衍生属性，避免实现耦合。
    
    参数:
        point: (3,) 空间查询点（float64，单位与 Triangle.vertices 一致）
        triangle: Triangle 实例（vertices 形状必须为 (3, 3)）
        eps: 数值容差（整数坐标推荐 1e-8；浮点坐标可设 1e-12）
    
    返回:
        (closest_point, distance)
        - closest_point: (3,) 三角形上最近点（float64）
        - distance: float 点到三角形的最短欧氏距离（单位同输入）
    
    设计原则:
        ✅ 零依赖：仅使用 triangle.vertices，规避 edges/topology 计算顺序歧义
        ✅ 整数友好：避免法向量归一化，全程点积运算（整数坐标下精确）
        ✅ 退化兜底：area=0 时自动回退至顶点最近点
        ✅ 无魔法数字：eps 显式传参，适配不同坐标尺度
    """
    # === 安全校验（开发期调试用，生产环境可注释）===
    if triangle.vertices.shape != (3, 3):
        raise ValueError(f"Triangle.vertices shape {triangle.vertices.shape} != (3, 3)")
    if point.shape != (3,):
        raise ValueError(f"point shape {point.shape} != (3,)")
    
    # === 提取顶点（避免重复索引）===
    v0, v1, v2 = triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]
    
    # 强制转换为 float64 以避免大整数点积运算溢出 (e.g. 10^18 * 10^18)
    v0 = v0.astype(np.float64)
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    point = point.astype(np.float64)
    
    # === 区域分类核心逻辑（Ericson §5.1.5）===
    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0
    
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    
    # 区域 V0
    if d1 <= eps and d2 <= eps:
        return v0, np.linalg.norm(point - v0)
    
    bp = point - v1
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    # 区域 V1
    if d3 >= -eps and d4 <= d3 + eps:
        return v1, np.linalg.norm(point - v1)
    
    vc = d1 * d4 - d3 * d2
    # 区域 E01
    if vc <= eps and d1 >= -eps and d3 <= eps:
        denom = d1 - d3
        t = np.clip(d1 / denom if abs(denom) > eps else 0.0, 0.0, 1.0)
        q = v0 + t * ab
        return q, np.linalg.norm(point - q)
    
    cp = point - v2
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    # 区域 V2
    if d6 >= -eps and d5 <= d6 + eps:
        return v2, np.linalg.norm(point - v2)
    
    vb = d5 * d2 - d1 * d6
    # 区域 E02
    if vb <= eps and d2 >= -eps and d6 <= eps:
        denom = d2 - d6
        t = np.clip(d2 / denom if abs(denom) > eps else 0.0, 0.0, 1.0)
        q = v0 + t * ac
        return q, np.linalg.norm(point - q)
    
    va = d3 * d6 - d5 * d4
    # 区域 E12
    if va <= eps and (d4 - d3) >= -eps and (d5 - d6) >= -eps:
        denom = (d4 - d3) + (d5 - d6)
        t = np.clip((d4 - d3) / denom if abs(denom) > eps else 0.0, 0.0, 1.0)
        q = v1 + t * (v2 - v1)
        return q, np.linalg.norm(point - q)
    
    # === 区域 F（三角形内部）===
    denom = va + vb + vc
    if abs(denom) < eps:  # 退化三角形兜底（area ≈ 0）
        # 返回到三个顶点的最小距离（比随机选点更稳健）
        dists = [np.linalg.norm(point - v) for v in (v0, v1, v2)]
        return [v0, v1, v2][int(np.argmin(dists))], float(np.min(dists))
    
    v = vb / denom
    w = vc / denom
    q = v0 + v * ab + w * ac
    return q, np.linalg.norm(point - q)

import numpy as np


def point2tri_distance_approx(t1:Triangle, t2:Triangle, gap_threshold=LEN_EPSILON):
    # Step 1: 顶点采样快速筛选（您的实现）
    min_dis=10e20
    for p1 in t1.vertices:
        
        _,dis=nearest_point_on_triangle(p1,t2)
        if min_dis>dis:
            min_dis=dis
    for p2 in t2.vertices:
        _,dis=nearest_point_on_triangle(p2,t1)
        if min_dis>dis:
            min_dis=dis
    if min_dis<gap_threshold:
        return 0
    else:
        return min_dis  
      

def segment_segment_min_distance(A, B, C, D):
    """计算两条线段间的最短距离及最近点对
    
    参数:
    A, B -- 第一条线段的起点和终点 (numpy数组)
    C, D -- 第二条线段的起点和终点 (numpy数组)
    
    返回:
    P, Q -- 最近点对 (numpy数组)
    distance -- 最短距离 (标量)
    """
    # 确保输入为NumPy数组并转为float64防止溢出
    A, B, C, D = map(lambda x: np.array(x, dtype=np.float64), [A, B, C, D])
    
    # 计算方向向量和偏移向量
    u = B - A
    v = D - C
    w = A - C
    
    # 计算关键点积
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    
    # 处理线段关系
    denom = a * c - b * b
    if np.abs(denom) < 1e-10:  # 近平行或平行
        s = 0.0
        t = e / c if c > 1e-10 else 0.0
    else:  # 一般情况
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    
    # 限制参数到线段范围
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    # 边界修正逻辑
    if s < 1e-10 or s > 1.0 - 1e-10:
        if c > 1e-10:
            t = np.clip(np.dot(v, (A + s * u - C)) / c, 0.0, 1.0)
        else:
            t = 0.0

    if t < 1e-10 or t > 1.0 - 1e-10:
        if a > 1e-10:
            s = np.clip(np.dot(u, (C + t * v - A)) / a, 0.0, 1.0)
        else:
            s = 0.0
    
    # 计算最近点和距离
    P = A + s * u
    Q = C + t * v
    distance = np.linalg.norm(P - Q)
    
    return P, Q, distance

def seg2tri_distance_approx(t1:Triangle, t2:Triangle, gap_threshold=LEN_EPSILON):
    # Step 1: 顶点采样快速筛选（您的实现）
    min_dis=10e20
    [v0_t1,v1_t1,v2_t1]=t1.vertices
    [v0_t2,v1_t2,v2_t2]=t2.vertices
    seg0_t1=[v0_t1,v1_t1]
    seg1_t1=[v1_t1,v2_t1]
    seg2_t1=[v2_t1,v0_t1]
    seg0_t2=[v0_t2,v1_t2]
    seg1_t2=[v1_t2,v2_t2]
    seg2_t2=[v2_t2,v0_t2]
    segs_t1=[seg0_t1,seg1_t1,seg2_t1]
    segs_t2=[seg0_t2,seg1_t2,seg2_t2]
    for seg_t1 in segs_t1:
        for seg_t2 in segs_t2:
            _,_,dis=segment_segment_min_distance(seg_t1[0],seg_t1[1],seg_t2[0],seg_t2[1])
            if dis<min_dis:
                min_dis=dis
    if min_dis<gap_threshold:
        return 0
    else:
        return min_dis  
      



def  compute_gap(tri1:Triangle, tri2:Triangle, gap_threshold=LEN_EPSILON,type='mean'):
    """
    计算两个近似平行三角形之间的间隙（平面距离）。
    策略：计算 Tri2 的所有顶点到 Tri1 所在平面的平均距离。
    
    Args:
        tri1_coords: shape (3, 3) -> obj1 的三角形顶点
        tri2_coords: shape (3, 3) -> obj2 的三角形顶点
        normal1: (可选) tri1 的法向量，如果之前算过可以传进来加速
    Returns:
        float: 两个三角形平面之间的平均距离
    """
    # 1. 如果没有传入法线，现场计算 Tri1 的法线
    # 向量 AB 和 AC
    min_p=point2tri_distance_approx(tri1,tri2,gap_threshold=gap_threshold)
    min_s=seg2tri_distance_approx(tri1,tri2,gap_threshold=gap_threshold)
    if min_p<min_s:
        return min_p
    else:
        return min_s

def compute_dihedral_angle(normal_1: np.ndarray, normal_2: np.ndarray) -> float:
    """
    计算两个三角形所在平面的夹角（二面角）。
    
    参数:
        normal_1: (3,) 数组，三角形1的法向量
        normal_2: (3,) 数组，三角形2的法向量
        
    返回:
        float: 两个平面的夹角，单位为度（°）。
               范围 [0.0, 90.0]。始终返回非负值。
               如果夹角 > 90°，则返回其补角（180° - 夹角）。
               
    异常:
        ValueError: 如果任一法向量无效（None 或 长度为0）。
        
    精度:
        至少保留 6 位小数精度。
    """
    if normal_1 is None:
        raise ValueError(f"Failed to compute normal for triangle 1: {normal_1}")
    if normal_2 is None:
        raise ValueError(f"Failed to compute normal for triangle 2: {normal_2}")
        
    # 计算 cos(theta) = |n1 · n2| / (|n1| * |n2|)
    # triangle_normal_vector 应该已经归一化了，但为了稳健再次归一化或直接点乘
    # 假设 triangle_normal_vector 返回已归一化的向量
    
    dot_product = np.abs(np.dot(normal_1, normal_2))
    
    # 钳位以防止数值误差导致超出 [-1, 1] (虽然 abs 后是 [0, 1])
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)

import numpy as np

def overlap_area_3d(tri1, tri2):
    """
    Symmetric overlap under near-coplanar assumption:
    1) choose common normal n_proj ~ normalize(n1 + n2)
    2) symmetrically translate both triangles along n_proj onto the same plane
    3) compute 2D polygon intersection area on that plane
    Returns: cover1, cover2, intersection_area
    """
    EPS = 1e-9

    # ----------------------------
    # 0) data
    # ----------------------------
    V1 = np.asarray(tri1.vertices, dtype=np.float64)  # (3,3)
    V2 = np.asarray(tri2.vertices, dtype=np.float64)

    n1 = np.asarray(tri1.normal, dtype=np.float64)
    n2 = np.asarray(tri2.normal, dtype=np.float64)

    # ----------------------------
    # 1) common normal (bisector)
    # ----------------------------
    n_sum = n1 + n2
    n_len = np.linalg.norm(n_sum)
    if n_len < EPS:
        return 0.0, 0.0, 0.0
    n_proj = n_sum / n_len

    # ----------------------------
    # 2) symmetric "snap" to common plane along n_proj
    #    plane is defined by n_proj · x = d_mid
    # ----------------------------
    # Use centroid-projected distances as robust offsets
    c1 = V1.mean(axis=0)
    c2 = V2.mean(axis=0)
    d1 = np.dot(n_proj, c1)
    d2 = np.dot(n_proj, c2)
    d_mid = 0.5 * (d1 + d2)

    # Translate along n_proj only (no rotation), to make both coplanar
    V1p = V1 + (d_mid - d1) * n_proj
    V2p = V2 + (d_mid - d2) * n_proj

    # Choose origin on the common plane (midpoint of translated centroids)
    origin = 0.5 * (V1p.mean(axis=0) + V2p.mean(axis=0))

    # ----------------------------
    # 3) build stable tangent basis (t1, t2) on the plane
    # ----------------------------
    # pick a helper axis not parallel to n_proj
    if abs(n_proj[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    t1 = np.cross(u, n_proj)
    t1_len = np.linalg.norm(t1)
    if t1_len < EPS:
        u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t1 = np.cross(u, n_proj)
        t1_len = np.linalg.norm(t1)
        if t1_len < EPS:
            return 0.0, 0.0, 0.0
    t1 /= t1_len
    t2 = np.cross(n_proj, t1)  # unit

    def project_to_2d(verts3):
        vecs = verts3 - origin
        return np.column_stack((vecs @ t1, vecs @ t2))

    poly1 = project_to_2d(V1p)
    poly2 = project_to_2d(V2p)

    # ----------------------------
    # 4) signed area & ensure CCW (IMPORTANT for clipping)
    # ----------------------------
    def signed_area(coords):
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def polygon_area(coords):
        return abs(signed_area(coords))

    def ensure_ccw(poly):
        return poly[::-1].copy() if signed_area(poly) < 0 else poly

    poly1 = ensure_ccw(poly1)
    poly2 = ensure_ccw(poly2)

    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    if area1 < EPS or area2 < EPS:
        return 0.0, 0.0, 0.0

    # ----------------------------
    # 5) convex polygon clipping (Sutherland–Hodgman), clip must be CCW
    # ----------------------------
    def clip_polygon(subject, clip):
        def inside(p, a, b):
            # left side of directed edge a->b for CCW clip polygon
            return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= -EPS

        def intersect(a, b, s, e):
            # line intersection between segment s-e and infinite line a-b
            da = a - b
            db = s - e
            dp = a - s
            denom = da[0]*db[1] - da[1]*db[0]
            if abs(denom) < EPS:
                return 0.5 * (s + e)  # fallback
            t = (dp[0]*db[1] - dp[1]*db[0]) / denom
            return a + t * (b - a)

        output = subject
        cp1 = clip[-1]
        for cp2 in clip:
            input_list = output
            output = []
            if len(input_list) == 0:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        output.append(intersect(cp1, cp2, s, e))
                    output.append(e)
                elif inside(s, cp1, cp2):
                    output.append(intersect(cp1, cp2, s, e))
                s = e
            cp1 = cp2
            output = np.array(output, dtype=np.float64)
        return np.array(output, dtype=np.float64)

    inter_poly = clip_polygon(poly1, poly2)
    if inter_poly.shape[0] < 3:
        return 0.0, 0.0, 0.0

    intersection_area = polygon_area(inter_poly)
    if intersection_area < EPS:
        return 0.0, 0.0, 0.0

    # ----------------------------
    # 6) sanity check: intersection must not exceed either area
    # ----------------------------
    # allow tiny numerical tolerance
    if intersection_area > area1 * (1.0 + 1e-6) or intersection_area > area2 * (1.0 + 1e-6):
        # If this triggers, you can print poly1/poly2/inter_poly to debug
        # It usually indicates numerical degeneracy or unexpected non-convex input.
        intersection_area = min(intersection_area, area1, area2)

    cover1 = intersection_area / area1
    cover2 = intersection_area / area2
    return cover1, cover2, intersection_area


import numpy as np

def overlap_area_3d_normal(tri1, tri2):
    """
    面向近共面三角形的稳健重叠面积估计（平移 → 旋转 → 投影 → 2D 剪裁）
    
    法向方向约束（重点）:
        tri1.normal 与 tri2.normal 需在方向上对齐，采用“无向法向”规则确保稳定：
            - 若 dot(n1, n2) >= 0，则使用角平分向量 n_proj = normalize(n1 + n2)
            - 否则使用 n_proj = normalize(n1 - n2)
        等价于“dot 为正值，否则反向计算”的约束，避免相反方向导致的错误投影。
        任一法向退化（模长 < EPS）直接返回 (0.0, 0.0, 0.0) 以防止后续数值错误。
    
    参数:
        tri1: Triangle 实例，vertices 形状为 (3,3)，normal 为 (3,)
        tri2: Triangle 实例，同上
    
    返回:
        (cover1, cover2, intersection_area)
            - cover1: 交叠面积相对于 tri1 投影面积的比例，范围 [0, 1]
            - cover2: 交叠面积相对于 tri2 投影面积的比例，范围 [0, 1]
            - intersection_area: 在公共投影平面上的交叠面积（单位与输入一致）
    
    数值稳健性:
        - EPS=1e-12 作为法向/面积门槛；法向或三角形面积退化直接返回 0
        - 罗德里格斯旋转将 n_proj 旋到 Z 轴，避免奇异方向
        - 投影后统一顶点方向为 CCW，再用 Sutherland–Hodgman 做三角形裁剪
        - 最终交叠面积做 clamp，不超过任一侧投影面积；cover 也被 clip 到 [0, 1]
    """

    EPS = 1e-12

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def normalize(v):
        n = np.linalg.norm(v)
        if n < EPS:
            return None
        return v / n

    def triangle_area_3d(V):
        return 0.5 * np.linalg.norm(np.cross(V[1]-V[0], V[2]-V[0]))

    def signed_area_2d(P):
        x, y = P[:, 0], P[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def area_2d(P):
        return abs(signed_area_2d(P))

    def ensure_ccw(P):
        return P[::-1].copy() if signed_area_2d(P) < 0 else P

    def clip_triangle(subject, clip):
        """
        Sutherland–Hodgman clipping for convex polygons (triangle vs triangle).
        Assumes clip is CCW.
        """
        def inside(p, a, b):
            return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= -EPS

        def intersect(a, b, s, e):
            se = e - s
            ab = b - a
            denom = se[0]*ab[1] - se[1]*ab[0]
            if abs(denom) < EPS:
                return 0.5*(s+e)
            t = ((a-s)[0]*ab[1] - (a-s)[1]*ab[0]) / denom
            return s + t*se

        output = subject
        cp1 = clip[-1]
        for cp2 in clip:
            input_list = output
            output = []
            if len(input_list) == 0:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        output.append(intersect(cp1, cp2, s, e))
                    output.append(e)
                elif inside(s, cp1, cp2):
                    output.append(intersect(cp1, cp2, s, e))
                s = e
            cp1 = cp2
        return np.array(output, dtype=np.float64)

    # -------------------------------------------------
    # 0) input data
    # -------------------------------------------------
    V1 = np.asarray(tri1.vertices, dtype=np.float64)
    V2 = np.asarray(tri2.vertices, dtype=np.float64)

    n1 = normalize(np.asarray(tri1.normal, dtype=np.float64))
    n2 = normalize(np.asarray(tri2.normal, dtype=np.float64))
    if n1 is None or n2 is None:
        return 0.0, 0.0, 0.0

    A1_3d = triangle_area_3d(V1)
    A2_3d = triangle_area_3d(V2)
    if A1_3d < EPS or A2_3d < EPS:
        return 0.0, 0.0, 0.0

    # -------------------------------------------------
    # 1) translate to common center
    # -------------------------------------------------
    c1 = V1.mean(axis=0)
    c2 = V2.mean(axis=0)
    center = 0.5 * (c1 + c2)

    V1 -= center
    V2 -= center

    # -------------------------------------------------
    # 2) rotate: bisector normal → Z axis
    # -------------------------------------------------
    
    if np.dot(n1, n2) >= 0:
        n_proj = normalize(n1 + n2)
    else:
        n_proj = normalize(n1 - n2)
    if n_proj is None:
        return 0.0, 0.0, 0.0

    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n_proj, z)
    sin_theta = np.linalg.norm(axis)
    cos_theta = np.dot(n_proj, z)

    if sin_theta < EPS:
        R = np.eye(3)
    else:
        axis = axis / sin_theta
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = (
            np.eye(3)
            + K * sin_theta
            + K @ K * (1 - cos_theta)
        )

    V1r = (R @ V1.T).T
    V2r = (R @ V2.T).T

    # -------------------------------------------------
    # 3) project to XY plane
    # -------------------------------------------------
    P1 = V1r[:, :2]
    P2 = V2r[:, :2]

    P1 = ensure_ccw(P1)
    P2 = ensure_ccw(P2)

    area1 = area_2d(P1)
    area2 = area_2d(P2)

    # projection degeneracy guard
    if area1 < EPS or area2 < EPS:
        return 0.0, 0.0, 0.0

    # -------------------------------------------------
    # 4) 2D triangle intersection
    # -------------------------------------------------
    inter = clip_triangle(P1, P2)
    if len(inter) < 3:
        return 0.0, 0.0, 0.0

    inter_area = area_2d(inter)
    if inter_area < EPS:
        return 0.0, 0.0, 0.0

    # -------------------------------------------------
    # 5) sanity clamp
    # -------------------------------------------------
    inter_area = min(inter_area, area1, area2)

    cover1 = inter_area / area1
    cover2 = inter_area / area2

    cover1 = float(np.clip(cover1, 0.0, 1.0))
    cover2 = float(np.clip(cover2, 0.0, 1.0))

    return cover1, cover2, inter_area

import numpy as np

def overlap_area_3d_rotated(tri1, tri2):
    """
    面向近共面三角形的稳健重叠面积估计（平移 → 旋转 → 投影 → 2D 剪裁）
    
    强调的法向方向约束:
        为确保投影方向一致性与数值稳定，要求 tri1.normal 与 tri2.normal 在方向上“对齐”。
        内部实现为“无向法向”处理：
            - 若 dot(n1, n2) >= 0，则使用角平分向量 n_proj = normalize(n1 + n2)
            - 否则使用 n_proj = normalize(n1 - n2)
        即“dot 为正值，否则反向计算”原则的程序化落地。
        若任一法向退化（模长过小），返回 (0.0, 0.0, 0.0)。
    
    参数:
        tri1: Triangle 实例，要求 vertices 形状为 (3,3)，normal 为 (3,)
        tri2: Triangle 实例，同上
    
    返回:
        (cover1, cover2, intersection_area)
            - cover1: 交叠面积相对于 tri1 投影面积的比例，范围 [0, 1]
            - cover2: 交叠面积相对于 tri2 投影面积的比例，范围 [0, 1]
            - intersection_area: 在公共投影平面上的交叠面积（单位与输入一致）
    
    数值与鲁棒性:
        - EPS = 1e-12 作为长度与面积门槛；法向/三角形面积/投影面积退化时直接返回 0
        - 旋转采用罗德里格斯公式将 n_proj 旋到 Z 轴，避免奇异方向
        - 投影后统一为 CCW，使用 Sutherland–Hodgman 对三角形进行凸多边形裁剪
        - 最终对交叠面积做 sanity clamp：不超过任一单侧投影面积
    
    适用范围:
        - 近共面、近平行三角形的重叠估计；非近共面场景下该方法仍给出有界、保守的结果
    
    边界行为:
        - 法向或面积退化、投影退化：返回 (0.0, 0.0, 0.0)
        - 返回的 cover1/cover2 经 clip 保证在 [0, 1]
    """

    EPS = 1e-12

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def normalize(v):
        n = np.linalg.norm(v)
        if n < EPS:
            return None
        return v / n

    def triangle_area_3d(V):
        return 0.5 * np.linalg.norm(np.cross(V[1]-V[0], V[2]-V[0]))

    def signed_area_2d(P):
        x, y = P[:, 0], P[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def area_2d(P):
        return abs(signed_area_2d(P))

    def ensure_ccw(P):
        return P[::-1].copy() if signed_area_2d(P) < 0 else P

    def clip_triangle(subject, clip):
        """
        Sutherland–Hodgman clipping for convex polygons (triangle vs triangle).
        Assumes clip is CCW.
        """
        def inside(p, a, b):
            return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= -EPS

        def intersect(a, b, s, e):
            se = e - s
            ab = b - a
            denom = se[0]*ab[1] - se[1]*ab[0]
            if abs(denom) < EPS:
                return 0.5*(s+e)
            t = ((a-s)[0]*ab[1] - (a-s)[1]*ab[0]) / denom
            return s + t*se

        output = subject
        cp1 = clip[-1]
        for cp2 in clip:
            input_list = output
            output = []
            if len(input_list) == 0:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        output.append(intersect(cp1, cp2, s, e))
                    output.append(e)
                elif inside(s, cp1, cp2):
                    output.append(intersect(cp1, cp2, s, e))
                s = e
            cp1 = cp2
        return np.array(output, dtype=np.float64)

    # -------------------------------------------------
    # 0) input data
    # -------------------------------------------------
    V1 = np.asarray(tri1.vertices, dtype=np.float64)
    V2 = np.asarray(tri2.vertices, dtype=np.float64)

    n1 = normalize(np.asarray(tri1.normal, dtype=np.float64))
    n2 = normalize(np.asarray(tri2.normal, dtype=np.float64))
    if n1 is None or n2 is None:
        return 0.0, 0.0, 0.0

    A1_3d = triangle_area_3d(V1)
    A2_3d = triangle_area_3d(V2)
    if A1_3d < EPS or A2_3d < EPS:
        return 0.0, 0.0, 0.0

    # -------------------------------------------------
    # 1) translate to common center
    # -------------------------------------------------
    c1 = V1.mean(axis=0)
    c2 = V2.mean(axis=0)
    center = 0.5 * (c1 + c2)

    V1 -= center
    V2 -= center

    # -------------------------------------------------
    # 2) rotate: bisector normal → Z axis
    # -------------------------------------------------

    if np.dot(n1, n2) >= 0:
        n_proj = normalize(n1 + n2)
    else:
        n_proj = normalize(n1 - n2)
    if n_proj is None:
        return 0.0, 0.0, 0.0

    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n_proj, z)
    sin_theta = np.linalg.norm(axis)
    cos_theta = np.dot(n_proj, z)

    if sin_theta < EPS:
        R = np.eye(3)
    else:
        axis = axis / sin_theta
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = (
            np.eye(3)
            + K * sin_theta
            + K @ K * (1 - cos_theta)
        )

    V1r = (R @ V1.T).T
    V2r = (R @ V2.T).T

    # -------------------------------------------------
    # 3) project to XY plane
    # -------------------------------------------------
    P1 = V1r[:, :2]
    P2 = V2r[:, :2]

    P1 = ensure_ccw(P1)
    P2 = ensure_ccw(P2)

    area1 = area_2d(P1)
    area2 = area_2d(P2)

    # projection degeneracy guard
    if area1 < EPS or area2 < EPS:
        return 0.0, 0.0, 0.0

    # -------------------------------------------------
    # 4) 2D triangle intersection
    # -------------------------------------------------
    inter = clip_triangle(P1, P2)
    if len(inter) < 3:
        return 0.0, 0.0, 0.0

    inter_area = area_2d(inter)
    if inter_area < EPS:
        return 0.0, 0.0, 0.0

    # -------------------------------------------------
    # 5) sanity clamp
    # -------------------------------------------------
    inter_area = min(inter_area, area1, area2)

    cover1 = inter_area / area1
    cover2 = inter_area / area2

    cover1 = float(np.clip(cover1, 0.0, 1.0))
    cover2 = float(np.clip(cover2, 0.0, 1.0))

    return cover1, cover2, inter_area

def overlap_area_3d(tri1, tri2):
    """
    重叠面积估计的统一入口（封装）
    
    采用“无向法向”约束以防止误用：
        - tri1.normal 与 tri2.normal 的方向必须对齐；内部按 dot(n1, n2) 的符号自动选择
          角平分向量：dot>=0 用 n1+n2；dot<0 用 n1−n2（已在被调函数中实现）
        - 该约束避免法向相反导致的错误投影与异常面积
    
    错误防护（已在实现链路内完成）:
        - 法向/三角形面积/投影退化时返回 (0.0, 0.0, 0.0)
        - 交叠面积不超过任一侧投影面积；cover1/cover2 钳制到 [0, 1]
    
    注意:
        - 本函数仅作为规范化入口，实际计算由 overlap_area_3d_rotated 完成
        - Triangle.normal 应为可用单位向量；若为 None/退化，将触发稳健回退
    """
    return overlap_area_3d_rotated(tri1, tri2)
   