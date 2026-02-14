# -*- coding: utf-8 -*-
"""
Contour Operations Module
=========================
Handles 2D contour/loop operations including quantization, normalization, and deduplication.
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union

# Numeric Policy Constants
SCALE_FACTOR = 10000.0  # 1 unit = 0.0001 mm
INV_SCALE_FACTOR = 1.0 / SCALE_FACTOR
EPS_COINCIDE = 10       # 0.001 mm
EPS_COLLINEAR = 1       # 0.0001 mm
MIN_SEGMENT_LEN = 50    # 0.005 mm
MIN_LOOP_AREA = 10000   # 0.0001 mm^2

class Orientation(Enum):
    CCW = 1
    CW = -1
    UNKNOWN = 0

def to_int64(points: np.ndarray) -> np.ndarray:
    """Quantize float points to int64."""
    return np.round(points * SCALE_FACTOR).astype(np.int64)

def to_float64(points: np.ndarray) -> np.ndarray:
    """Dequantize int64 points to float."""
    return points.astype(np.float64) * INV_SCALE_FACTOR

def snap_points(points: np.ndarray, threshold: int = EPS_COINCIDE) -> np.ndarray:
    """
    Merge points that are closer than threshold.
    Simple greedy approach (for robust, use KDTree/Grid).
    For contour points which are ordered, we usually only check neighbors,
    but for "soup of segments", we need global snapping.
    """
    # TODO: Implement spatial hashing for global snapping if needed.
    # For now, assume points are pre-snapped or use simple pairwise (slow).
    # MVP: No-op or neighbor snapping for loops.
    return points

def calculate_signed_area(loop: np.ndarray) -> float:
    """Calculate signed area of a 2D loop (Shoelace formula)."""
    # Input: (N, 2) int64
    # Shoelace: 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
    # Using np.roll(-1) shifts indices: i -> i-1 (effectively next element in array terms for +1?)
    # Wait, roll(-1): [0,1,2] -> [1,2,0]. So x[i] pairs with y[i+1].
    # Formula: x[i]*y[i+1] - x[i+1]*y[i].
    # Implementation below:
    # x dot roll(y, 1)? roll(y, 1) shifts [0,1,2] -> [2,0,1]. So y[i-1].
    # x[i]*y[i-1]. This is effectively x_{i+1}*y_i if we re-index.
    # Standard formula: 0.5 * sum(x_i*y_{i+1} - x_{i+1}*y_i)
    
    x = loop[:, 0].astype(np.float64)
    y = loop[:, 1].astype(np.float64)
    
    # Correct implementation for x_i * y_{i+1}
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    
    return 0.5 * (np.dot(x, y_next) - np.dot(x_next, y))

def normalize_loop_orientation(loop: np.ndarray, target_ccw: bool = True) -> np.ndarray:
    """Ensure loop has desired orientation."""
    area = calculate_signed_area(loop)
    current_ccw = area > 0
    if current_ccw != target_ccw:
        return loop[::-1]
    return loop

def align_start_point(loop: np.ndarray) -> np.ndarray:
    """Cyclic shift loop so the lexicographically smallest point is first."""
    # Find min index (lexicographical: min x, then min y)
    # np.lexsort sorts by keys in reverse order (y, x) -> sort by x then y
    min_idx = np.lexsort((loop[:, 1], loop[:, 0]))[0]
    return np.roll(loop, -min_idx, axis=0)

def clean_loop(loop: np.ndarray) -> np.ndarray:
    """
    Remove degenerate segments and collinear points.
    Input: (N, 2) int64
    """
    if len(loop) < 3:
        return loop # Too short to clean properly
    
    # 1. Remove short segments (neighbor duplicates)
    # Calculate dists
    # Roll to get p[i] and p[i+1]
    diff = loop - np.roll(loop, -1, axis=0)
    dist_sq = np.sum(diff**2, axis=1)
    
    # Keep points where distance to NEXT is > threshold
    # Note: This is a simple filter.
    mask = dist_sq >= MIN_SEGMENT_LEN**2
    
    cleaned = loop[mask]
    
    # 2. Remove collinear points
    if len(cleaned) < 3:
        return cleaned
        
    # Check cross product of (p[i-1]->p[i]) and (p[i]->p[i+1])
    # Use roll
    p_prev = np.roll(cleaned, 1, axis=0)
    p_curr = cleaned
    p_next = np.roll(cleaned, -1, axis=0)
    
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    
    # Cross product 2D: x1*y2 - x2*y1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Keep points where cross product is significant (not collinear)
    # Using float64 for cross calculation check to avoid overflow?
    # v1, v2 are int64. Max diff ~ 2*10^9. Product ~ 4*10^18.
    # int64 max is 9*10^18. Borderline safe, but float64 is safer.
    cross_f = cross.astype(np.float64)
    mask_collinear = np.abs(cross_f) > EPS_COLLINEAR
    
    return cleaned[mask_collinear]

class LoopNormalizer:
    """
    Pipeline to normalize 2D contours.
    """
    @staticmethod
    def process(segments: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """
        Convert soup of segments to normalized loops.
        
        Args:
            segments: List of (p1, p2) pairs, float64.
            
        Returns:
            List of (N, 2) float64 loops.
        """
        if not segments:
            return []
            
        # 1. Quantize
        # Flatten to (M, 2, 2/3)
        # Assume 2D for now (XY). If 3D, project or ignore Z.
        seg_int = []
        for p1, p2 in segments:
            p1_i = to_int64(p1[:2])
            p2_i = to_int64(p2[:2])
            seg_int.append((p1_i, p2_i))
            
        # 2. Chain Segments (Naive O(N^2) for MVP, improve later)
        # Build graph: point_tuple -> list of connected point_tuples
        adj = {}
        for p1, p2 in seg_int:
            t1, t2 = tuple(p1), tuple(p2)
            # Skip zero length
            if t1 == t2: continue
            
            adj.setdefault(t1, []).append(t2)
            # Directed or Undirected? Slicer segments usually directed (CCW).
            # But safe to assume undirected for stitching and recover direction later.
            # adj.setdefault(t2, []).append(t1) # If undirected
        
        # Simple walker (Directed assumption for closed loops)
        loops = []
        visited = set()
        
        # Start from any node
        # Optimize: this is just a sketch.
        # Ideally use a robust stitching lib or Union-Find.
        
        # Placeholder for simple closed loop extraction
        # Assumes perfect matching (1-in, 1-out)
        
        # Let's implement a simple "follow next"
        nodes = list(adj.keys())
        processed_edges = set()
        
        for start_node in nodes:
            if start_node in visited:
                continue
                
            # Start path
            path = [start_node]
            visited.add(start_node)
            curr = start_node
            
            while True:
                if curr not in adj or not adj[curr]:
                    break
                
                # Get next
                next_node = adj[curr][0] # Take first
                # Remove edge (directed)
                adj[curr].pop(0)
                
                if next_node == start_node:
                    # Closed loop found
                    loops.append(np.array(path, dtype=np.int64))
                    break
                
                if next_node in visited:
                    # Hit visited node (merge or complex topology)
                    # For MVP, stop.
                    break
                    
                path.append(next_node)
                visited.add(next_node)
                curr = next_node
                
        # 3. Clean & Normalize
        normalized_loops = []
        for loop in loops:
            # Clean
            loop = clean_loop(loop)
            
            # Filter Area
            area = abs(calculate_signed_area(loop))
            if area < MIN_LOOP_AREA:
                continue
                
            # Normalize Orientation (CCW)
            loop = normalize_loop_orientation(loop, target_ccw=True)
            
            # Align Start
            loop = align_start_point(loop)
            
            # Dequantize
            normalized_loops.append(to_float64(loop))
            
        return normalized_loops

