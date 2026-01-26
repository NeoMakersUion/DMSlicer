import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List,Union,Optional,TYPE_CHECKING
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..file_parser import Model
    from ..visualizer import IVisualizer
    
from ..visualizer import VisualizerType
from .config import GEOM_ACC,DEFAULT_VISUALIZER_TYPE
from enum import Enum
class Status(Enum):
    NORMAL=0
    UPDATING=1
class VerticesOrder(Enum):
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZXY = 4
    ZYX = 5

class TrianglesOrder(Enum):
    P123 = 0
    P132 = 1
    P213 = 2
    P231 = 3
    P312 = 4
    P321 = 5
class IdOrder(Enum):
    ASC = 0
    DESC = 1


@dataclass
class Object:
    acc:int=GEOM_ACC
    id:Optional[int]=None
    triangles_ids: Optional[np.ndarray] = None       
    color: Optional[np.ndarray] = None
    triangle_ids_order:Optional[IdOrder]=None
    status:Optional[Status]=None
    __hash_id:Optional[str]=None

    @property
    def hash_id(self) -> str:
        if self.__hash_id is None:
            self.__hash__()
        return self.__hash_id

    def visualize(self,vertices,triangles):
        visualize_vertices_and_triangles(self.vertices[self.triangles_ids],self.triangles[self.triangles_ids])

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__hash__()

    def __hash__(self) -> None:
        self.__hash_id = hash((
            self.acc,
            self.id,
            self.triangles_ids.tobytes() if self.triangles_ids is not None else None,
            self.color.tobytes() if self.color is not None else None,
            self.triangle_ids_order,
            self.status
        ))


class Geom:
    def __init__(self,model:"Model",acc=GEOM_ACC) -> None:
        self.acc=acc
        self.model=model
        self.objects = []

        self.acc: int = acc
        self.vertices: Union[np.ndarray,List] = []
        self.triangles: Union[np.ndarray,List] = []
        self.objects: List[Object] = []
        self.vertices_order:Optional[VerticesOrder]=None
        self.triangles_order:Optional[TrianglesOrder]=None
        self.model:Optional["Model"]=model
        self.cache={}
        self.__hash_id:Optional[str]=None
        self.status:Optional[Status]=None
        self.triangle_index_map: Dict[Tuple[int, ...], int] = {}
        self.__merge_all_meshes()

    @property
    def hash_id(self) -> str:
        if self.__hash_id is None:
            self.__hash__()
        return self.__hash_id


    def __hash__(self) -> None:
        self.__hash_id = hash((
            self.acc,
            self.vertices.tobytes() if isinstance(self.vertices, np.ndarray) else tuple(self.vertices),
            self.triangles.tobytes() if isinstance(self.triangles, np.ndarray) else tuple(self.triangles),
            tuple(hash(o) for o in self.objects),
            self.vertices_order,
            self.triangles_order,
            self.status
        ))


    def __merge_all_meshes(self):
        """
        合并模型中的所有网格对象为一个全局几何体，并执行顶点去重与统一化。

        功能逻辑：
        1. 遍历模型中的所有网格（Mesh）。
        2. 对每个网格的顶点进行整型量化（基于 self.acc），以消除浮点误差并支持精确匹配。
        3. 使用全局顶点映射表（vertex_map）合并重复顶点：
           - 如果顶点坐标已存在，复用现有全局索引。
           - 如果是新顶点，分配新的全局索引并存入全局顶点列表。
        4. 重映射与三角形去重：
           - 将本地三角形顶点索引转换为全局索引。
           - 使用 `self.triangle_index_map` (key=tuple(sorted(gtri))) 对三角形进行去重（忽略顶点顺序/法向）。
           - 如果三角形已存在，复用全局索引；否则创建新条目。
        5. 生成对象（Object）记录：为每个原始网格创建一个对应的 Object 实例，记录其在全局列表中的三角形索引列表（非连续范围）及属性。

        关键步骤：
        - 顶点量化：`round(v, acc) * 10^acc` -> int64
        - 顶点去重：通过 `vertex_map` (Dict[Tuple, int]) 实现跨网格的顶点共享。
        - 三角形去重：通过 `triangle_index_map` 实现基于顶点集合（frozenset语义）的唯一化。
        - 索引重映射：`local_index` -> `global_index`。

        副作用：
        - 更新 `self.vertices`: (N, 3) int64 全局唯一顶点数组。
        - 更新 `self.triangles`: (M, 3) int64 全局唯一三角形数组。
        - 更新 `self.triangle_index_map`: 记录三角形顶点集合到全局索引的映射。
        - 更新 `self.objects`: 包含每个子网格元数据的 Object 列表。
        - 更新 `self.status`: 设置为 Status.NORMAL。

        注意：
        - 此过程会修改内部状态，是几何处理的初始化步骤。
        - 依赖 `self.model` 和 `self.acc`。
        """
        vertex_map: Dict[Tuple[int, int, int], int] = {}
        global_vertices: List[np.ndarray] = []
        all_triangles = []
        global_v_count = 0
        global_t_count = 0

        # --------------------------
        # 1. Merge all meshes (合并所有网格)
        # --------------------------
        for mesh in tqdm(self.model.meshes, desc="Normalizing meshes"):
            # 深拷贝以避免修改原始模型数据
            verts = deepcopy(mesh.vertices)
            tris  = deepcopy(mesh.triangles)

            # quantize to integer grid (量化至整数网格)
            # 乘以 10^acc 并转为 int64，确保坐标精度一致性
            verts = np.round(verts, self.acc) * (10 ** self.acc)
            verts = verts.astype(np.int64)

            local_to_global = {}

            # unify vertices (统一顶点)
            # 遍历当前网格的所有顶点，建立本地索引到全局索引的映射
            for i, v in enumerate(verts):
                key = tuple(v.tolist())
                # 检查顶点是否已在全局映射中存在
                if key not in vertex_map:
                    vertex_map[key] = global_v_count
                    global_vertices.append(v)
                    global_v_count += 1
                local_to_global[i] = vertex_map[key]

            # remap triangles (重映射三角形)
            # 将三角形引用的本地顶点索引转换为全局顶点索引
            obj_triangle_indices = []
            for t in tris:
                gtri = (
                    local_to_global[int(t[0])],
                    local_to_global[int(t[1])],
                    local_to_global[int(t[2])]
                )
                
                # Deduplication logic (Triangles unique with frozenset)
                # 使用排序后的元组作为键，代表顶点集合的唯一性
                key = tuple(sorted(gtri))
                
                if key not in self.triangle_index_map:
                    self.triangle_index_map[key] = global_t_count
                    all_triangles.append(gtri)
                    global_t_count += 1
                
                # 获取全局索引（可能是新建的，也可能是复用的）
                idx = self.triangle_index_map[key]
                obj_triangle_indices.append(idx)

            # 创建对应的几何对象实例
            obj = Object()
            # 初始化对象属性，记录其在全局数据中的三角形索引列表
            obj.update(
                triangles_ids=np.array(obj_triangle_indices, dtype=np.int64),
                id=mesh.id,
                color=deepcopy(mesh.color),
                acc=self.acc,
                status=Status.NORMAL
            )

            self.objects.append(obj)
        
        # 转换为 NumPy 数组以优化后续计算性能
        self.vertices = np.asarray(global_vertices, dtype=np.int64)
        self.triangles = np.asarray(all_triangles, dtype=np.int64)
        self.status = Status.NORMAL


    def __sort__(self,vert_order:VerticesOrder=VerticesOrder.ZYX,tri_order:TrianglesOrder=TrianglesOrder.P132):
        self.vertices_order=vert_order
        self.vertices=self.vertices[:,::-1]
        self.triangles=self.triangles[:,::-1]
        self.triangles_order=tri_order

    def visualize_union(self):
        visualize_vertices_and_triangles(self.vertices,self.triangles)

    def show(self,visualizer:Optional["IVisualizer"]=None,visualizer_type:Optional[VisualizerType]=None,**kwargs):
        """
        Show the Geom object using a specified visualizer.

        Args:
            visualizer: Optional visualizer instance. If None, a new one will be created.
            visualizer_type: Type of visualizer to use if creating a new one.
            **kwargs: Additional keyword arguments passed to the visualizer.
                Specifically accepts the following key-value pairs:
                - object_list: List of objects to include in the visualization.
                - opacity_list: List of opacity values corresponding to the objects.
        """
        if visualizer_type is None:
            visualizer_type=DEFAULT_VISUALIZER_TYPE
        if visualizer is None:
            from ..visualizer import IVisualizer
            visualizer=IVisualizer.create(visualizer_type,**kwargs)
            is_shown=True
        else:
            is_shown=False

        # Pass the entire Geom object to the visualizer, not individual objects
        visualizer.add(self,**kwargs)
        if is_shown:
            visualizer.show()






def normalize_meshes(model, acc: int):
    """
    Normalize all MeshData into one global canonicalized mesh.

    Steps:
      1) Quantize vertices to int grid
      2) Merge all meshes into one global vertex pool
      3) Build global triangles
      4) Sort vertices (Z,Y,X)
      5) Remap triangles using inverse permutation
      6) Verify topology consistency

    Returns:
        vertices_sorted : (N,3) int64
        triangles_sorted: (M,3) int64   (correctly remapped, topology preserved)
    """

    vertex_map: Dict[Tuple[int, int, int], int] = {}
    global_vertices: List[np.ndarray] = []
    all_triangles = []
    colors=[]
    mesh_ids=[]
    local_tris=[]
    global_v_count = 0
    global_t_count = 0

    # --------------------------
    # 1. Merge all meshes
    # --------------------------
    for mesh in tqdm(model.meshes, desc="Normalizing meshes"):
        obj=Object()
        obj.id = mesh.id
        obj.color=deepcopy(mesh.color)
        obj.acc=acc

        verts = deepcopy(mesh.vertices)
        tris  = deepcopy(mesh.triangles)
        color = deepcopy(mesh.color)

        # quantize to integer grid
        verts = np.round(verts, acc) * (10 ** acc)
        verts = verts.astype(np.int64)

        local_to_global = {}

        # unify vertices
        for i, v in enumerate(verts):
            key = tuple(v.tolist())
            if key not in vertex_map:
                vertex_map[key] = global_v_count
                global_vertices.append(v)
                global_v_count += 1
            local_to_global[i] = vertex_map[key]

        # remap triangles
        local_triangles=[]
        start=global_t_count
        for t in tris:
            gtri = (
                local_to_global[int(t[0])],
                local_to_global[int(t[1])],
                local_to_global[int(t[2])]
            )
            all_triangles.append(gtri)
            global_t_count+=1
        end=global_t_count
        obj.triangles_ids=np.arange(start,end,dtype=np.int64)
        obj.color=(color)
        self.objects.append(obj)

    self.vertices= np.asarray(global_vertices, dtype=np.int64)
    self.triangles = np.asarray(all_triangles, dtype=np.int64)
    self.visualize_union()

    # --------------------------
    # 2. Sort vertices (Z,Y,X)
    # --------------------------
    vertices = self.vertices
    triangles = self.triangles
    order_vzyx = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
    vertices_sorted_by_zyx = vertices[order_vzyx]

    # build inverse permutation: old_index -> new_index
    inv = np.empty_like(order_vzyx)
    inv[order_vzyx] = np.arange(len(order_vzyx), dtype=order_vzyx.dtype)
    # remap triangles correctly
    triangles_sorted_by_vzyx = inv[triangles]
    triangles_sorted_by_vzyx=np.sort(triangles_sorted_by_vzyx,axis=1)
    order_t123_vzyz=np.lexsort((triangles_sorted_by_vzyx[:, 1], triangles_sorted_by_vzyx[:, 2], triangles_sorted_by_vzyx[:, 0]))

    triangles_sorted_by_t123_vzyz=triangles_sorted_by_vzyx[order_t123_vzyz]
    inv =np.empty_like(order_t123_vzyz)
    inv[order_t123_vzyz] = np.arange(len(order_t123_vzyz), dtype=order_t123_vzyz.dtype)
    for i in range(len(local_tris)):
        local_tris[i]=inv[local_tris[i]]
        local_tris[i]=np.sort(local_tris[i])
        
    
    # --------------------------
    # 4. Verify topology preserved
    # --------------------------
    edges_after = _edge_set(triangles_sorted_by_vzyx)
    assert edges_before == edges_after, "❌ Topology changed after sorting! Mapping is broken."

    # --------------------------
    # 5. (Optional) canonical triangle key for hashing / dedup
    #    DO NOT use for rendering
    # --------------------------
    triangle_key = np.sort(triangles_sorted_by_vzyx, axis=1)

    # --------------------------
    # 6. Visualize (debug)
    # --------------------------
    try:
        from ..visualizer.pyvista_visualizer import visualize_vertices_and_triangles
        visualize_vertices_and_triangles(vertices_sorted_by_zyx, triangles_sorted_by_vzyx)
    except Exception as e:
        print("Visualizer skipped:", e)

    return vertices_sorted_by_zyx, triangles_sorted_by_vzyx, triangle_key
