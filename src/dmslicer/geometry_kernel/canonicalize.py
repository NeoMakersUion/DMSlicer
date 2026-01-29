import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List,Union,Optional,TYPE_CHECKING
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
import sys
if TYPE_CHECKING:
    from ..file_parser import Model
    from ..visualizer import IVisualizer
    
from ..visualizer import VisualizerType
from .config import GEOM_ACC,DEFAULT_VISUALIZER_TYPE
from enum import Enum
from .bvh import BVHNode,query_obj_bvh, AABB, build_bvh
class Status(Enum):
    NORMAL=0
    SORTING=1
    SORTED=2
    UPDATING=3
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
    vertices: Optional[np.ndarray] = None
    triangles: Optional[np.ndarray] = None
    tri_id2geom_tri_id: Optional[np.ndarray] = None       
    triangle_ids_order:Optional[IdOrder]=None
    color: Optional[np.ndarray] = None
    status:Optional[Status]=None
    __hash_id:Optional[str]=None
    aabb: Optional[AABB] = None
    bvh: Optional[BVHNode] = None

    @property
    def hash_id(self) -> str:
        if self.__hash_id is None:
            self.__hash__()
        return self.__hash_id

    def visualize(self,vertices,triangles):
        visualize_vertices_and_triangles(self.vertices[self.tri_id2geom_tri_id],self.triangles[self.tri_id2geom_tri_id])

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__hash__()

    def __hash__(self) -> None:
        self.__hash_id = hash((
            self.acc,
            self.id,
            self.tri_id2geom_tri_id.tobytes() if self.tri_id2geom_tri_id is not None else None,
            self.color.tobytes() if self.color is not None else None,
            self.triangle_ids_order,
            self.status
        ))
    def ensure_bvh(self):
        """
        Ensure the BVH is built and object is sorted.
        If already sorted and BVH exists, this method returns immediately.
        
        Args:
            geom_triangles_AABB: Global triangle AABBs from Geom. 
                               If None, BVH cannot be built.
        """
        triangles=self.triangles
        self.bvh = build_bvh(triangles)
        pass
    
        


class Geom:
    def __init__(self,model:"Model",acc=GEOM_ACC,vertices_order=VerticesOrder.ZYX,triangles_order=TrianglesOrder.P132) -> None:
        self.acc=acc
        self.model=model
        self.objects = []
        self.acc: int = acc
        self.vertices: Union[np.ndarray,List] = []
        self.triangles: Union[np.ndarray,List] = []
        self.objects: List[Object] = []
        self.obj_by_id: Dict[int, Object] = {} # Map obj.id to Object instance
        self.vertices_order:VerticesOrder=vertices_order
        self.triangles_order:TrianglesOrder=triangles_order
        self.model:Optional["Model"]=model
        self.cache={}
        self.__hash_id:Optional[str]=None
        self.status:Optional[Status]=None
        self.triangle_index_map: Dict[Tuple[int, ...], int] = {}
        self.triangles_AABB: Optional[np.ndarray] = None
        self.AABB: Optional[np.ndarray] = None
        self.__merge_all_meshes()
        self.__sort__()
        self.__build_object_contact_graph()
        # self.__detect_interface_pairs()

    @property
    def hash_id(self) -> str:
        if self.__hash_id is None:
            self.__hash__()
        return self.__hash_id


    def __hash__(self) -> None:
        self.__hash_id = hash((
            self.acc,
            self.model.hash_id,
            self.vertices_order,
            self.triangles_order,
        ))


    def __merge_all_meshes(self):
        """
        合并模型中的所有网格对象为一个全局几何体，并执行：
        - 顶点全局去重（跨 mesh）
        - 三角形全局去重（跨 mesh）
        - Object -> 全局三角形索引映射构建
        """

        # --------------------------
        # 全局容器
        # --------------------------
        vertex_map: Dict[Tuple[int, int, int], int] = {}
        global_vertices: List[np.ndarray] = []
        all_triangles: List[Tuple[int, int, int]] = []

        global_v_count = 0
        global_t_count = 0

        # 使用类级别的 triangle_index_map（关键修正点）
        self.triangle_index_map = {}

        # --------------------------
        # 遍历所有 mesh
        # --------------------------
        for mesh in tqdm(self.model.meshes, desc="Normalizing meshes"):
            # 深拷贝，避免污染原始模型
            verts = deepcopy(mesh.vertices)
            tris  = deepcopy(mesh.triangles)

            # --------------------------
            # 顶点量化（float → int）
            # --------------------------
            verts = np.round(verts, self.acc) * (10 ** self.acc)
            verts = verts.astype(np.int64)

            # local -> global 顶点映射
            local_to_global: Dict[int, int] = {}

            for i, v in enumerate(verts):
                key = tuple(v.tolist())
                if key not in vertex_map:
                    vertex_map[key] = global_v_count
                    global_vertices.append(v)
                    global_v_count += 1
                local_to_global[i] = vertex_map[key]

            # --------------------------
            # 三角形重映射 & 全局去重
            # --------------------------
            obj_triangle_indices: List[int] = []

            for t in tris:
                gtri = (
                    local_to_global[int(t[0])],
                    local_to_global[int(t[1])],
                    local_to_global[int(t[2])]
                )

                # 忽略方向的三角形唯一键
                key = tuple(sorted(gtri))

                if key not in self.triangle_index_map:
                    self.triangle_index_map[key] = global_t_count
                    all_triangles.append(gtri)
                    global_t_count += 1

                obj_triangle_indices.append(self.triangle_index_map[key])

            # --------------------------
            # 构建 Object
            # --------------------------
            obj = Object()
            obj.update(
                triangles_ids=np.asarray(obj_triangle_indices, dtype=np.int64),
                id=mesh.id,
                color=deepcopy(mesh.color),
                acc=self.acc,
                status=Status.NORMAL
            )

            self.objects.append(obj)
            self.obj_by_id[obj.id] = obj

        # --------------------------
        # 转为 NumPy 数组
        # --------------------------
        self.vertices  = np.asarray(global_vertices, dtype=np.int64)
        self.triangles = np.asarray(all_triangles, dtype=np.int64)
        self.status = Status.NORMAL
        


    def __sort__(
        self,
        vert_order: VerticesOrder = VerticesOrder.ZYX,
        tri_order: TrianglesOrder = TrianglesOrder.P132,
    ):
        self.status = Status.SORTING

        vertices = np.asarray(self.vertices, dtype=np.int64)
        triangles = np.asarray(self.triangles, dtype=np.int64)

        # ===============================
        # 1. Vertex sort
        # ===============================
        if vert_order == VerticesOrder.ZYX:
            order_vertices = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
        else:
            raise NotImplementedError(f"Unsupported VerticesOrder: {vert_order}")

        self.vertices = vertices[order_vertices]

        inv_vertices = np.empty_like(order_vertices)
        inv_vertices[order_vertices] = np.arange(len(order_vertices), dtype=np.int64)
        triangles = inv_vertices[triangles]

        # ===============================
        # 2. Triangle sort
        # ===============================
        if tri_order == TrianglesOrder.P132:
            triangles = np.sort(triangles, axis=1)
            order_triangles = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
        else:
            raise NotImplementedError(f"Unsupported TrianglesOrder: {tri_order}")

        self.triangles = triangles[order_triangles]

        inv_triangles = np.empty_like(order_triangles)
        inv_triangles[order_triangles] = np.arange(len(order_triangles), dtype=np.int64)
        objects_len=len(self.objects)
        for i,obj in enumerate(self.objects):
            obj.tri_id2geom_tri_id = np.sort(inv_triangles[obj.tri_id2geom_tri_id])
            obj.triangles_ids_order = tri_order
            obj.triangles=self.vertices[self.triangles[obj.tri_id2geom_tri_id]]
            obj.vertices=[]
            p_id=0
            obj.tri_id2vex_ids=[]
            obj.vex_id2tri_ids={}
            for tri_id,tri in enumerate(obj.triangles.tolist()):
                t_p_ids=[]
                for p in tri:
                    if p in obj.vertices:
                        p_id=obj.vertices.index(p)
                    else:
                        obj.vertices.append(p)
                        p_id+=1
                    t_p_ids.append(p_id)
                    obj.vex_id2tri_ids.setdefault(p_id,[]).append(tri_id)
                obj.tri_id2vex_ids.append(t_p_ids)
            sys.stdout.write(f"\rUpdating triangles_ids of object {i+1}/{objects_len}"+" "*30+"\r")
            sys.stdout.flush()
            obj.ensure_bvh()
            obj.aabb=obj.bvh.aabb
        aabb=obj.bvh.aabb
        for i,obj in enumerate(self.objects):
            if i==0:
                continue
            aabb_current=obj.bvh.aabb
            aabb=aabb.merge(aabb_current)
        self.AABB=aabb
        self.status = Status.SORTED
        pass
        #To do:保存self 



            
    def __build_object_contact_graph(self):
        # build AABB
        if self.status!=Status.SORTED:
            self.__sort__()
        self.status=Status.UPDATING
        # build tmp object_adj_graph_dict
        object_adj_graph_dict={}
        len_objects=len(self.objects)
        for i in range(len_objects):
            obj1=self.objects[i]
            for j in range(i+1,len_objects):
                obj2=self.objects[j]
                if not obj1.aabb.overlap(obj2.aabb):
                    continue
                object_adj_graph_dict.setdefault(obj1.id,set()).add(obj2.id)
                object_adj_graph_dict.setdefault(obj2.id,set()).add(obj1.id)
                pass
        visited=set()
        candidate_pairs_dict={}
        for target_obj,adj_objs in object_adj_graph_dict.items():
            if target_obj in visited:
                continue
            visited.add(target_obj)
            for adj_obj in list(adj_objs):
                if adj_obj in visited:
                    continue
                sys.stdout.write(f"\rUpdating contact graph of object {target_obj} and {adj_obj}"+" "*30+"\r")
                sys.stdout.flush()
                candidate_pairs=self.check_contact(target_obj,adj_obj)
                if candidate_pairs == []:
                    object_adj_graph_dict[target_obj].remove(adj_obj)
                    object_adj_graph_dict[adj_obj].remove(target_obj)
                    continue
                if target_obj>adj_obj:
                    key=(adj_obj,target_obj)
                    candidate_pairs=[(b, a) for a, b in candidate_pairs]
                else:
                    key=(target_obj,adj_obj)
                candidate_pairs_dict[key]=candidate_pairs
        self.object_adj_graph_dict=object_adj_graph_dict
        self.candidate_pairs_dict=candidate_pairs_dict
        
                
    
    def check_contact(self,obj1_id:int,obj2_id:int):
        obj1=self.obj_by_id[obj1_id]
        obj2=self.obj_by_id[obj2_id]
        # BVH Query
        if obj1.bvh and obj2.bvh:
            candidate_pairs = query_obj_bvh(obj1,obj2)
            return candidate_pairs
        
        return []



                
        
        

            

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
        obj.tri_id2geom_tri_id=np.arange(start,end,dtype=np.int64)
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
