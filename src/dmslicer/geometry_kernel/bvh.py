import numpy as np
from typing import List, Tuple, Optional
import sys
class AABB:
    def __init__(self, min_point, max_point):
        self.min = np.array(min_point, dtype=np.float64)
        self.max = np.array(max_point, dtype=np.float64)
    
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

class BVHNode:
    def __init__(self):
        self.aabb: Optional[AABB] = None
        self.tri_index: Optional[int] = None
        self.left: Optional['BVHNode'] = None
        self.right: Optional['BVHNode'] = None
        self.is_leaf = False
def triangle_aabb_from_coords(tri: np.ndarray) -> AABB:
    """tri: (3, 3) array of vertex coordinates"""
    return AABB(np.min(tri, axis=0), np.max(tri, axis=0))


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
            f"\rProgress:{percentage}%|{finished}/{total}"+" "*10+"\r"
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
    spinner.done()
    return result


