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
from typing import Optional, Tuple, List, Union
import numpy as np


Point3 = Union[np.ndarray, Tuple[float, float, float]]
Segment3 = Tuple[np.ndarray, np.ndarray]


def _as_np(p: Point3) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"Point must be shape (3,), got {p.shape}")
    return p


def _lerp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    return p0 + t * (p1 - p0)


def _edge_plane_intersection_z(
    p0: np.ndarray, p1: np.ndarray, z: float, eps: float
) -> Optional[np.ndarray]:
    """
    Intersect segment p0-p1 with plane Z=z.
    Returns point if intersects within segment, else None.
    Handles near-parallel / endpoint on plane with eps.
    """
    z0 = p0[2] - z
    z1 = p1[2] - z

    # both on plane -> ambiguous (handled at triangle level)
    if abs(z0) <= eps and abs(z1) <= eps:
        return None

    # one endpoint on plane -> return that endpoint
    if abs(z0) <= eps:
        return p0
    if abs(z1) <= eps:
        return p1

    # same side -> no intersection
    if (z0 > 0 and z1 > 0) or (z0 < 0 and z1 < 0):
        return None

    # proper crossing
    # z0 + t*(z1-z0) = 0 => t = -z0/(z1-z0)
    denom = (z1 - z0)
    if abs(denom) <= eps:
        return None

    t = -z0 / denom
    if t < -eps or t > 1 + eps:
        return None
    t = min(1.0, max(0.0, t))
    return _lerp(p0, p1, t)


def intersect_triangle_with_plane(
    v0: Point3, v1: Point3, v2: Point3, z: float, eps: float = 1e-9
) -> Optional[Union[Segment3, List[Segment3]]]:
    """
    Compute intersection between triangle (v0,v1,v2) and plane Z=z.

    Returns:
        None                   -> no intersection
        (pA, pB)               -> one segment (most common case)
        [(pA,pB), ...]         -> multiple segments (rare / degenerate)
    
    Notes:
    - For slicing, we typically want a single segment or none.
    - Degenerate cases (triangle lying on plane) can be treated specially later.
    """
    p0 = _as_np(v0)
    p1 = _as_np(v1)
    p2 = _as_np(v2)

    d0 = p0[2] - z
    d1 = p1[2] - z
    d2 = p2[2] - z

    on0 = abs(d0) <= eps
    on1 = abs(d1) <= eps
    on2 = abs(d2) <= eps

    # Case A: triangle fully on plane (coplanar)
    # In slicing, this is a degeneracy: it produces a 2D triangle area.
    # We return None here and let higher-level slicer decide how to handle it.
    if on0 and on1 and on2:
        return None

    # Collect intersection points from edges
    pts: List[np.ndarray] = []

    for a, b in ((p0, p1), (p1, p2), (p2, p0)):
        ip = _edge_plane_intersection_z(a, b, z, eps)
        if ip is not None:
            pts.append(ip)

    if len(pts) == 0:
        return None

    # Remove duplicates (caused by hitting vertices)
    uniq: List[np.ndarray] = []
    for p in pts:
        if not any(np.linalg.norm(p - q) <= 1e-7 for q in uniq):
            uniq.append(p)
    pts = uniq

    # Typical: two intersection points -> one segment
    if len(pts) == 2:
        return (pts[0], pts[1])

    # Sometimes: one point (touching at a vertex) -> treat as no segment for slicing
    if len(pts) == 1:
        return None

    # Rare: more than 2 points (can happen due to numerical issues or weird degeneracy)
    # Strategy: choose two farthest points to form a segment
    # (slicing generally expects a segment)
    if len(pts) > 2:
        max_d = -1.0
        pa, pb = None, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d > max_d:
                    max_d = d
                    pa, pb = pts[i], pts[j]
        if pa is not None and pb is not None and max_d > 1e-12:
            return (pa, pb)
        return None

    return None


# ============================================================
# (Optional future expansion)
# ============================================================

def intersect_triangle_triangle(*args, **kwargs):
    """
    Placeholder for triangle-triangle intersection (future use):
    - collision detection
    - self-intersection checks
    """
    raise NotImplementedError


def clip_segment_by_aabb(*args, **kwargs):
    """
    Placeholder for segment clipping by AABB (future use):
    - speed up
    - restrict to local region
    """
    raise NotImplementedError
