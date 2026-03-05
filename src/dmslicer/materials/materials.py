from abc import ABCMeta,abstractmethod
from typing import List,Dict,Optional
import logging
try:
    from .composition import ConstantComposition,GradientComposition
except ImportError:
    from composition import ConstantComposition,GradientComposition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger=logging.getLogger(__name__)


# 补全 AbstractMaterial 抽象基类（核心：初始化 material_name）
class Material(metaclass=ABCMeta):
    def __init__(self, material_name: str):
        # 关键：给实例赋值 material_name，子类才能继承到
        self.material_name = material_name
#指定待处理的材料
class ImpendingMaterial(Material):
    def __init__(self,material_name):
        super().__init__(material_name)
        
# 修复后的 SourceMaterial 类
class SourceMaterial(Material):
    # 修复：可变默认参数改为 None，避免共享列表
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
    def __init__(
        self, 
        material_name: str, 
        material_composition=None
    ):
        # 第一步：调用父类 SourceMaterial 的构造方法（参数正确传递）
        super().__init__(material_name, material_composition)
        # 第二步：设置隔离材料的规则（None 表示禁止接触所有材料）
        self._restricted_materials = None

class Adj_checker():
    def __init__(self,center_obj,include_obj_list,exclude_obj_list):
        self.center_obj=center_obj
        self.include_obj_list=include_obj_list
        self.exclude_obj_list=exclude_obj_list

class GradientMaterial(Material):
    def __init__(self,material_name,center_include_exclude_dict):
        super().__init__(material_name)
        target_obj=center_include_exclude_dict["center"]
        source_obj_list=center_include_exclude_dict["include"]
        exclude_obj_list=center_include_exclude_dict["exclude"]
        composition={}
        if source_obj_list==None:
            source_obj_list=target_obj.nbr_objects
        for source_obj in source_obj_list:
            if source_obj in exclude_obj_list:
                continue
            if source_obj not in target_obj.nbr_objects:
                continue
            composition[source_obj]=source_obj.composition
        
        self.composition=GradientComposition(material_name)
        self.composition.set_composition(composition)

class TestObject:
    def __init__(self,obj_name,material_type,composition=None,nbr_objects=None):
        self.name=obj_name
        self.material_type=material_type
        self.composition=composition
        self.nbr_objects=nbr_objects or []

target_source_exclude_dict={
    "target":TestObject("target","GradientMaterial"),
    "source":[TestObject("source1","SourceMaterial"),TestObject("source2","SourceMaterial")],
    "exclude":[TestObject("exclude1","SourceMaterial")]
}
gradient_composition=GradientComposition("GradientComposition")


# const_composition=ConstantComposition(["PLA","TPU"],[0.9,0.1])