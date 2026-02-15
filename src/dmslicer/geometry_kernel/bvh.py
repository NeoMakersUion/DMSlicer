import numpy as np
from typing import List, Tuple, Optional
import sys
from ..visualizer.visualizer_interface import IVisualizer
class AABB:
    """Axis-aligned bounding box with float64 precision.

    Stores per-axis minima and maxima as 3D float vectors. All operations
    are pure and return new AABB instances unless otherwise stated.

    Args:
        min_point: Minimum corner (x_min, y_min, z_min).
        max_point: Maximum corner (x_max, y_max, z_max).
    """
    def __init__(self, min_point, max_point):
        self.min = np.array(min_point, dtype=np.float64)
        self.max = np.array(max_point, dtype=np.float64)
    @property
    def diag(self) -> np.ndarray:
        """返回 AABB 对角线向量"""
        return np.linalg.norm(self.max - self.min)

    def merge(self, other: 'AABB') -> 'AABB':
        """合并两个 AABB"""
        new_min = np.minimum(self.min, other.min)
        new_max = np.maximum(self.max, other.max)
        return AABB(new_min, new_max)
    
    def overlap(self, other: 'AABB') -> bool:
        """检查两个 AABB 是否重叠"""
        # AABB overlap condition:
        # max(a.min, b.min) <= min(a.max, b.max)
        # equivalent to: a.max >= b.min AND a.min <= b.max
        # Note: Must hold for all dimensions
        
        # Check x-axis
        if self.max[0] < other.min[0] or self.min[0] > other.max[0]:
            return False
        # Check y-axis
        if self.max[1] < other.min[1] or self.min[1] > other.max[1]:
            return False
        # Check z-axis
        if self.max[2] < other.min[2] or self.min[2] > other.max[2]:
            return False
        return True
    
    def extent(self) -> np.ndarray:
        """Return per-axis extent vector e = max - min."""
        return (self.max - self.min).astype(np.float64, copy=False)

    def center(self) -> np.ndarray:
        """Return box center c = (min + max) / 2."""
        return (self.min + self.max) * 0.5

    def surface_area(self) -> float:
        """Return surface area S = 2*(xy + yz + zx)."""
        ex, ey, ez = self.extent()
        return float(2.0 * (ex * ey + ey * ez + ez * ex))

    def volume(self) -> float:
        """Return box volume V = ex * ey * ez."""
        ex, ey, ez = self.extent()
        return float(ex * ey * ez)

    def expand(self, margin: float) -> 'AABB':
        """Return a new AABB expanded by `margin` in all directions.

        Args:
            margin: Non-negative scalar expansion.
        """
        if margin < 0:
            raise ValueError("margin must be non-negative")
        delta = np.array([margin, margin, margin], dtype=np.float64)
        return AABB(self.min - delta, self.max + delta)

    def corners(self) -> np.ndarray:
        """Return the 8 corner points of the box as an array of shape (8, 3)."""
        xmin, ymin, zmin = self.min
        xmax, ymax, zmax = self.max
        return np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ], dtype=np.float64)
    def visualize(self, visualizer: IVisualizer):
        """可视化 AABB"""
        visualizer.addAABB(self)

class BVHNode:
    def __init__(self):
        self.aabb: Optional[AABB] = None
        self.tri_index: Optional[int] = None
        self.left: Optional['BVHNode'] = None
        self.right: Optional['BVHNode'] = None
        self.is_leaf = False

def triangle_aabb_from_coords(tri: np.ndarray) -> AABB:
    """Create an AABB from a triangle vertex array.

    Args:
        tri: Array of shape (3, 3) with triangle vertices.

    Returns:
        AABB tightly enclosing the triangle.
    """
    tri_min = np.min(tri, axis=0).astype(np.float64, copy=False)
    tri_max = np.max(tri, axis=0).astype(np.float64, copy=False)
    return AABB(tri_min, tri_max)


def build_bvh(triangles: np.ndarray) -> BVHNode:
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must be shape (N, 3, 3)")

    N = len(triangles)
    if N == 0:
        raise ValueError("Empty triangle list")
    total_Node=2*N-1
    tri_indices = list(range(N))
    finished = 0
    processed_node_list=[]
    def process_print(finished, total=total_Node):
        percentage=int(finished/total*100)
        sys.stdout.write(
            f"\rProgress BVH:{percentage}%|{finished}/{total}"+" "*10+"\r"
        )
        if finished >= total:
            sys.stdout.write("\r")
        sys.stdout.flush()

    def _build(indices: List[int]) -> BVHNode:
        nonlocal finished

        node = BVHNode()

        total_aabb = triangle_aabb_from_coords(triangles[indices[0]])
        for i in indices[1:]:
            total_aabb = AABB.merge(
                total_aabb,
                triangle_aabb_from_coords(triangles[i])
            )
        node.aabb = total_aabb

        if len(indices) == 1:
            node.tri_index = indices[0]
            node.is_leaf = True
            if node not in processed_node_list:
                finished += 1
                processed_node_list.append(node)
                process_print(finished)
            return node

        extent = total_aabb.max - total_aabb.min
        axis = int(np.argmax(extent))

        centroids = [np.mean(triangles[i], axis=0) for i in indices]
        sorted_indices = [
            idx for _, idx in sorted(
                zip(centroids, indices),
                key=lambda x: x[0][axis]
            )
        ]

        mid = len(sorted_indices) // 2
        node.left = _build(sorted_indices[:mid])
        node.right = _build(sorted_indices[mid:])
        if node not in processed_node_list:
            processed_node_list.append(node)
            finished += 1
            process_print(finished)
        return node

    return _build(tri_indices)
from ..tools.progress_bar import RecursionSpinner 
def query_obj_bvh(obj1,obj2,**kwargs):
    spinner = RecursionSpinner(interval=0.1, max_dots=12)
    def query_bvh(node1: BVHNode, node2: BVHNode) -> List[Tuple[int, int]]:
        nonlocal spinner
        spinner.tick()
        results = []
        
        # 1. Base case: Check for None nodes
        if node1 is None or node2 is None:
            return results

        # 2. Pruning: Check AABB overlap
        if not node1.aabb.overlap(node2.aabb):
            return results

        # 3. Leaf check
        if node1.is_leaf and node2.is_leaf:
            results.append((node1.tri_index, node2.tri_index))
            return results

        # 4. Recursion
        if node1.is_leaf:
            results.extend(query_bvh(node1, node2.left))
            results.extend(query_bvh(node1, node2.right))
        elif node2.is_leaf:
            results.extend(query_bvh(node1.left, node2))
            results.extend(query_bvh(node1.right, node2))
        else:
            results.extend(query_bvh(node1.left, node2.left))
            results.extend(query_bvh(node1.left, node2.right))
            results.extend(query_bvh(node1.right, node2.left))
            results.extend(query_bvh(node1.right, node2.right))
        return results
    result=query_bvh(obj1.bvh,obj2.bvh)
    result_dot=query_bvh(obj1.bvh_dot,obj2.bvh_dot)
    result=list(set(result+result_dot))
    spinner.done()
    return result  


