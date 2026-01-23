from typing import List
from dataclass import dataclass
from file_parser import read_amf_objects

@dataclass
class InterFaceResult:
    interface_edges: Dict[Tuple[int,int], Set[Tuple[int,int]]]
    interface_tris: Dict[int, Set[int]]
    interface_pairs: List[Tuple[int,int,int,int,Tuple[int,int]]]
    pairs_by_objpair: Optional[Dict[Tuple[int,int], int]] = None
    # ★ 新增：每个对象对 -> (N,2) int32 数组，每一行是 (triA, triB)
    pairs_map: Optional[Dict[Tuple[int,int], np.ndarray]] = None
    # 新增：每个对象对 -> [trisA, trisB]，其中 trisA/trisB 是可用三角形索引数组
    pairs_tri_map: Optional[Dict[Tuple[int, int], List[np.ndarray]]] = None
    tri_intervals: Optional[Dict[int, List[Tuple[float, float]]]] = None

class Part:
    def __init__(self):
        self.vertices: List = []
        self.triangles: List = []