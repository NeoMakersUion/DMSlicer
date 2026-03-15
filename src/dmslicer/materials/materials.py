from abc import ABCMeta,abstractmethod
from typing import List,Optional
import json
import os
import uuid
import warnings
import logging
from ..geometry_kernel.geom_kernel import GeometryKernel
try:
    from .composition import ConstantComposition,GradientComposition
except ImportError:
    from composition import ConstantComposition,GradientComposition

from dmslicer.visualizer.visualizer_interface import IVisualizer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger=logging.getLogger(__name__)


class Adj_checker():
    def __init__(self,center_obj,include_obj_list,exclude_obj_list):
        self.center_obj=center_obj
        self.include_obj_list=include_obj_list
        self.exclude_obj_list=exclude_obj_list

def input_obj_index_list_input(mark="include",
                                protential_objs_list=None,
                                input_obj_index_list=None):
    if input_obj_index_list is None:
        input_obj_index_list=[]
        input_include_obj_id=None
        while protential_objs_list!=[] and input_include_obj_id!="q":
            print(f"\n\n please input the {mark} object index in {protential_objs_list},"\
                    "\n press 'q' to stop," \
                  f"\n press 'd' to add default list {protential_objs_list} to {mark} object list",\
                    "\n default value is 'd'")
            input_include_obj_id=input("Please input,you can input a single index ,e.g idx ,or a list of indices,e.g. [idx1,idx2],:")or"d"
            try:
                input_include_obj_id=eval(input_include_obj_id)
                protential_objs_list.remove(input_include_obj_id)
                input_obj_index_list.append(input_include_obj_id)
            except Exception:
                if isinstance(input_include_obj_id,list):
                    for item in input_include_obj_id:
                        protential_objs_list.remove(item)
                        input_obj_index_list.append(item)
                elif isinstance(input_include_obj_id,str):
                    input_include_obj_id=input_include_obj_id.lower()
                    if input_include_obj_id=='q':
                        break
                    if input_include_obj_id=='d':
                        input_obj_index_list+=protential_objs_list
                        for elem in input_obj_index_list:
                            protential_objs_list.remove(elem)
                        break
        return input_obj_index_list
    else:
        return input_obj_index_list


# 补全 AbstractMaterial 抽象基类（核心：初始化 material_name）
class Material_Property():
    _materials_db = []
    _db_path = os.path.join(os.getcwd(), "data", "materials_db.json")

    def __init__(self, name, color=None,melting_temperature=None, soft_temperature=None,
                 composition=None, density=None, elastic_modulus=None, poisson_ratio=None, id=None, **kwargs):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.color = color
        self.melting_temperature = melting_temperature
        self.melting_temperature_max = melting_temperature
        self.melting_temperature_min = melting_temperature
        self.soft_temperature = soft_temperature
        self.soft_temperature_max = soft_temperature
        self.soft_temperature_min = soft_temperature
        self.composition = composition
        
        # New fields
        self.density = density
        self.elastic_modulus = elastic_modulus
        self.poisson_ratio = poisson_ratio

    def to_dict(self):
        """Convert material property to dictionary, excluding empty values and private attributes."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and v is not None and not (isinstance(v, (list, dict, set, tuple)) and len(v) == 0)
        }

    @classmethod
    def load_from_json(cls):
        """Load materials from JSON file."""
        if not os.path.exists(cls._db_path):
            cls._materials_db = []
            return
        try:
            with open(cls._db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cls._materials_db = [cls(**item) for item in data]
        except Exception as e:
            logging.error(f"Failed to load materials: {e}")
            cls._materials_db = []

    @classmethod
    def save_to_json(cls):
        """Save materials to JSON file."""
        data = [m.to_dict() for m in cls._materials_db]
        os.makedirs(os.path.dirname(cls._db_path), exist_ok=True)
        with open(cls._db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def validate_material(cls, name: str):
        """
        Validate material data.
        
        Args:
            data: Dictionary containing material properties.
            
        Raises:
            ValueError: If validation fails.
        """
        if not name:
            warnings.warn("Material name is required.")
        else:
            # Ensure db is loaded for duplicate check
            if not cls._materials_db and os.path.exists(cls._db_path):
                cls.load_from_json()
            for m in cls._materials_db:
                if m.name == name:
                    warnings.warn(f"Material '{name}' already exists.")
                    return False
        return True

    @classmethod
    def add_material(cls, data: dict):
        """
        Add a new material.
        
        Args:
            data: Dictionary containing material properties.
            
        Returns:
            Material_Property: The created material object.
        """
        # Ensure db is loaded
        if not cls._materials_db and os.path.exists(cls._db_path):
            cls.load_from_json()
            
        if not cls.validate_material(data.get("name")):
             raise ValueError(f"Material '{data.get('name')}' invalid or already exists.")
             
        obj = cls(**data)
        cls._materials_db.append(obj)
        cls.save_to_json()
        return obj

    @classmethod
    def remove_material(cls, uuid_str):
        """
        Remove a material by ID.
        
        Args:
            uuid_str: The UUID of the material to remove.
        """
        if not cls._materials_db and os.path.exists(cls._db_path):
            cls.load_from_json()
        cls._materials_db = [m for m in cls._materials_db if m.id != uuid_str]
        cls.save_to_json()

    @classmethod
    def get_all_materials(cls):
        """Get all materials."""
        if not cls._materials_db:
            cls.load_from_json()
        return cls._materials_db

class Material(metaclass=ABCMeta):
    name_list=set()
    center_list=set()
    @abstractmethod
    def terminal_input():
        pass
    def __init__(self, material_name: str,material_property:Material_Property=None):
        # 关键：给实例赋值 material_name，子类才能继承到
        self.material_name = material_name
        self.material_property=material_property
    def __str__(self):
        return self.material_name
    def __repr__(self):
        return self.material_name
    def property(self):
        return self.material_property
    def set_property(self,material_property:Material_Property):
        self.material_property=material_property

#指定待处理的材料
class PendingMaterial(Material):
    def __init__(self,material_name="Pending"):
        super().__init__(material_name)
        
    def terminal_input(protential_objs_list,center_obj_index,input_include_obj_index_list:Optional[List[int]]=None,input_exclude_obj_index_list:Optional[List[int]]=None):
        pass
# 修复后的 SourceMaterial 类
class SourceMaterial(Material):
    # 修复：可变默认参数改为 None，避免共享列表
    def terminal_input(protential_objs_list,center_obj_index,input_include_obj_index_list:Optional[List[int]]=None,input_exclude_obj_index_list:Optional[List[int]]=None):
        pass
    def __init__(
        self, 
        material_name: str, 
        material_composition=None, 
        restricted_materials: Optional[List] = None  # 改为 None
    ):
        # 第一步：正确初始化 AbstractMaterial（仅传 material_name）
        super().__init__(material_name)
        # 第二步：单独初始化 ConstantComposition（而非传给 AbstractMaterial）
        self.composition = ConstantComposition(material_name, material_composition)
        # 修复：默认空列表的正确写法（每次实例化新建列表）
        self._restricted_materials = restricted_materials or []

    @property
    def restricted_materials(self):
        return self._restricted_materials

    def contacts(self):
        """返回接触规则：restrict 为禁止接触的材料名称列表"""
        return {
            "material": self.material_name,  # 现在能正确获取到 material_name
            "restrict": self.restricted_materials
        }

# 修复后的 IsolationMaterial 类
class IsolationMaterial(SourceMaterial):
    def terminal_input(protential_objs_list,center_obj_index,input_include_obj_index_list:Optional[List[int]]=None,input_exclude_obj_index_list:Optional[List[int]]=None):
        pass
    def __init__(
        self, 
        material_name: str, 
        material_composition=None
    ):
        # 第一步：调用父类 SourceMaterial 的构造方法（参数正确传递）
        super().__init__(material_name, material_composition)
        # 第二步：设置隔离材料的规则（None 表示禁止接触所有材料）
        self._restricted_materials = None


class GradientMaterial(Material):
    def terminal_input(protential_objs_list,center_obj_index,input_include_obj_index_list:Optional[List[int]]=None,input_exclude_obj_index_list:Optional[List[int]]=None):
        material_name=input("please input the material name:")
        while material_name in Material.name_list:
            material_name=input(f"the material name:{material_name} already exists,please input another name:")
        Material.name_list.add(material_name)

        center_include_exclude_dict={
            "center":None,
            "include":None,
            "exclude":None
        }

        if center_obj_index in Material.center_list:
            print(f"the center object index:{center_obj_index} already exists,please input another index:")
            return None,None
        Material.center_list.add(center_obj_index)

        center_include_exclude_dict["center"]=center_obj_index        
        input_include_list=input_obj_index_list_input(mark="include",
                                       protential_objs_list=protential_objs_list,
                                       input_obj_index_list=input_include_obj_index_list)
        input_exclude_list=input_obj_index_list_input(mark="exclude",
                                       protential_objs_list=protential_objs_list,
                                       input_obj_index_list=input_exclude_obj_index_list)
        center_include_exclude_dict["include"]=input_include_list
        center_include_exclude_dict["exclude"]=input_exclude_list
        return material_name,center_include_exclude_dict
    def __init__(self,material_name,target_obj,center_include_exclude_dict):
        super().__init__(material_name)
        self.composition=GradientComposition(material_name)
        source_obj_list=center_include_exclude_dict["include"]
        exclude_obj_list=center_include_exclude_dict["exclude"] or []
        nbr_objs=getattr(target_obj,"nbr_objects",[]) or []
        nbr_set=set(nbr_objs)
        include_set={i for i in (nbr_set if source_obj_list is None else source_obj_list) if i in nbr_set}
        exclude_set={i for i in exclude_obj_list if i in nbr_set}
        self.include=include_set
        self.exclude=exclude_set


class TestObject:
    def __init__(self,obj_name,material_type,composition=None,nbr_objects=None):
        self.name=obj_name
        self.material_type=material_type
        self.composition=composition
        self.nbr_objects=nbr_objects or []

class Abs_Materializer(metaclass=ABCMeta):
    def __init__(self,geom_kernel:GeometryKernel):
        objects=geom_kernel.geom.objects
        self.pending_obj=[]
        for obj_index,obj in objects.items():
            obj.material=PendingMaterial()
            obj.nbr_objects=geom_kernel.geom.graph[obj_index]
            self.pending_obj.append(obj_index)
        self.geom_kernel=geom_kernel
    
    @abstractmethod
    def resolve_material(self,object_id):
        pass
    def terminal_input(self,adj_obj_list,object_id):
        pass
    def load_materials(self):
        from dmslicer.materials import load_materials
        load_materials(self)

class Materializer(Abs_Materializer):
    def resolve_material(self,object_id):
        pass
    def terminal_input(self,adj_obj_list,object_id):
        pass
class DefaultMaterializer(Abs_Materializer):
    def resolve_material(self,object_id):  
        adj_obj_set=self.geom_kernel.graph[object_id]  
        adj_obj_list=list(adj_obj_set)
        objects=self.geom_kernel.geom.objects
        gradient,isolate,source=True,True,True
        if len(adj_obj_list)==1:
            print(f"Object id {object_id} cannot be gradient material,because it has only one neighbor")
            gradient=False
        else:
            length=len(adj_obj_list)  
            count=0
            for adj_obj_id in adj_obj_list:
                if isinstance(objects[adj_obj_id].material,IsolationMaterial):
                    count+=1
            if length-count<2:
                print(f"Object id {object_id} cannot be gradient material,because it has only {length-count} neighbor(s) that are not isolation materials")
                gradient=False
        print("\n\nPlease choice the material type, as follow:")
        if source:
            print("Source Material, Choose:",1)
        if isolate:
            print("Isolation Material, Choose:",2)
        if gradient:
            print("Gradient Material, Choose:",3)
        input_material_type=input("Please input the material type:")
        if input_material_type.isdigit():
            input_material_type=int(input_material_type)
            obj=objects[object_id]
            if input_material_type==1 and isolate:
                obj.material=SourceMaterial()
            elif input_material_type==2 and isolate:
                obj.material=IsolationMaterial()
            elif input_material_type==3 and gradient:
                material_name,center_include_exclude_dict=GradientMaterial.terminal_input(adj_obj_list,object_id)
                obj.material=GradientMaterial(material_name,obj,center_include_exclude_dict)
            else:
                print("invalid input")
                return False
        else:
            print("invalid input")
            return False
        return True
        
    def resolver(self):
        while True:
            flag=True
            for obj_index,obj in self.geom_kernel.geom.objects.items():
                if isinstance(obj.material,PendingMaterial):
                    flag=False
                    break
            if flag:
                break
            else:
                print(f"input object index {self.pending_obj} to resolve:")
                self.show()
                input_obj_index=input("Please input the object index:")
                self.show(eval(input_obj_index))
                if input_obj_index.isdigit():
                    input_obj_index=int(input_obj_index)
                    if input_obj_index in self.pending_obj:
                        if self.resolve_material(input_obj_index):
                            self.pending_obj.remove(input_obj_index)
                else:
                    print("invalid input")
    def visualizer_create(self,include_triangles_ids=None,opacity=0.1,labels=False):
        visualizer=IVisualizer.create()
        object_labels=[]
        if labels:
            for obj_index,obj in self.geom_kernel.geom.objects.items():
                object_labels.append(str(obj_index))
        if include_triangles_ids is None:
            for obj_index,obj in self.geom_kernel.geom.objects.items():
                visualizer.addObj(obj,opacity=opacity,label=str(obj_index) if labels else None)
        else:
            for obj_index,obj in self.geom_kernel.geom.objects.items():
                visualizer.addObj(obj,include_triangles_ids=include_triangles_ids,opacity=opacity,label=str(obj_index) if labels else None)
        if labels:
            visualizer.add_legend(object_labels,(0.3,0.2))
        return visualizer

    def visualizer_addObj(self,visualizer,obj_index,include_triangles_ids=None,opacity=0.1,label=None):
        if isinstance(obj_index,int):
            obj=self.geom_kernel.geom.objects[obj_index]
            if include_triangles_ids is None:
                visualizer.addObj(obj,opacity=opacity,label=label)
            else:
                visualizer.addObj(obj,include_triangles_ids=include_triangles_ids,opacity=opacity,label=label)
        elif isinstance(obj_index,list):
            obj_list=obj_index
            if include_triangles_ids is not None:
                if len(obj_list)!=len(include_triangles_ids):
                    raise ValueError("obj_list and include_triangles_ids must have the same length")
            if label is not None:
                if len(obj_list)!=len(label):
                    raise ValueError("obj_list and label must have the same length")
                labels=label
            else:
                labels=[None for _ in obj_list]
            for idx,obj_idx in enumerate(obj_list):
                obj=self.geom_kernel.geom.objects[obj_idx]
                if include_triangles_ids is not None:
                    visualizer.addObj(obj,include_triangles_ids=include_triangles_ids[idx],opacity=opacity,label=labels[idx])
                else:
                    visualizer.addObj(obj,opacity=opacity,label=labels[idx])
        return visualizer
    
    def show(self,*args):
        if args==():
            visualizer=self.visualizer_create(opacity=0.5,labels=True)
            visualizer.show()
            return
        elif len(args)==1:
            object_id=args[0]
            visualizer=self.visualizer_create(opacity=0.05)
            visualizer=self.visualizer_addObj(visualizer,object_id,opacity=0.8,label=str(object_id))
            visualizer.show()
            return
        elif len(args)==2:
            object_id=args[0]
            adj_obj_list=args[1]
            adj_obj_list_labels=[str(obj_index) for obj_index in adj_obj_list]
            visualizer=self.visualizer_create(opacity=0.05)
            visualizer=self.visualizer_addObj(visualizer,object_id,opacity=0.8,label=str(object_id))
            visualizer=self.visualizer_addObj(visualizer,adj_obj_list,opacity=0.2,label=adj_obj_list_labels)
            visualizer.add_legend(labels=[str(object_id)]+adj_obj_list_labels,size=(0.1,0.1))
            visualizer.show()
            return 


               
        
    
if __name__=="__main__":
    target_source_exclude_dict={
    "target":TestObject("target","GradientMaterial"),
    "source":[TestObject("source1","SourceMaterial"),TestObject("source2","SourceMaterial")],
    "exclude":[TestObject("exclude1","SourceMaterial")]
}
    gradient_composition=GradientComposition("GradientComposition")


    const_composition=ConstantComposition(["PLA","TPU"],[0.9,0.1])
def load_materials(materializer):
    import os,json
    objects=materializer.geom_kernel.geom.objects
    for _obj in objects.values():
        setattr(_obj,"_geom_objects",objects)
    hash_id=materializer.geom_kernel.geom.model.hash_id
    ws_dir=os.path.join("d:\\DMSlicer","data","workspace",hash_id,"material")
    assign_path=os.path.join(ws_dir,"material_assignment.json")
    props_path=os.path.join(ws_dir,"material_properties.json")
    with open(assign_path,"r",encoding="utf-8") as f:
        assign_cfg=json.load(f)
    with open(props_path,"r",encoding="utf-8") as f:
        props_cfg=json.load(f)
    props_map={m["name"]:m for m in props_cfg.get("materials",[])}
    def build_property(name:str):
        data=props_map.get(name)
        if not data:
            return None
        return Material_Property(
            id=data.get("id"),
            name=data.get("name"),
            melting_temperature=data.get("melting_temperature"),
            soft_temperature=data.get("soft_temperature"),
            composition=data.get("composition"),
        )
    order_keys=list(assign_cfg.get("objects",{}).keys())
    for k in order_keys:
        v=assign_cfg["objects"][k]
        idx=int(k)
        t=v.get("material_type")
        n=v.get("material_name")
        composition=props_map.get(n,{}).get("composition",{})
        if t in ("SourceMaterial","IsolationMaterial"):
            if t=="SourceMaterial":
                m=SourceMaterial(n,composition,v.get("restricted_materials"))
            else:
                m=IsolationMaterial(n,composition)
            prop=build_property(n)
            if prop:
                m.set_property(prop)
            objects[idx].material=m
    for k in order_keys:
        v=assign_cfg["objects"][k]
        idx=int(k)
        t=v.get("material_type")
        if t=="GradientMaterial":
            include=v.get("include") or []
            exclude=v.get("exclude") or []
            target_obj=objects[idx]
            cfg={"center":idx,"include":include,"exclude":exclude}
            g=GradientMaterial(v.get("material_name"),target_obj,cfg)
            objects[idx].material=g
    return props_map
