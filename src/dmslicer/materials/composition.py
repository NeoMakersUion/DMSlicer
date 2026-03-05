from abc import ABCMeta,abstractmethod
from typing import List,Dict,Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger=logging.getLogger(__name__)

# 校验并规范化材料组成字典：
# - 允许传入字符串（自动转为 {name: 1.0}），或 Dict[str, number]
# - 每个值必须在 [0, 1]（带 esp 容差）；所有值之和必须等于 1（带 esp 容差）
# - 返回规范化后的 Dict[str, float]，不修改入参
def _ensure_valid_comp_dict(dict_material_composition: Dict,esp:float=1e-6) -> Dict[str, float]:
    if isinstance(dict_material_composition,str):
        dict_material_composition={dict_material_composition:1.0}
    if not isinstance(dict_material_composition, Dict):
        raise TypeError("material_composition must be Dict")
    result: Dict[str, float] = {}
    s = 0.0
    for k, v in dict_material_composition.items():
        if not isinstance(v, (float, int)):
            raise ValueError("composition values must be numeric")
        vv = float(v)
        if vv < -esp or vv > 1+esp:
            raise ValueError("each composition value must be in [0, 1]")
        result[k] = vv
        s += vv
    if abs(s - 1.0) > esp:
        raise ValueError("sum of composition values must be 1")
    return result

class CompositionBehavior(metaclass=ABCMeta):
    # 抽象“组成行为”基类：
    # - 维护名称 _name 与当前组成 _composition
    # - 要求派生类实现 composition 属性与 set_composition 方法
    def __init__(self,name):
        self._name=name
        self._composition=None
    @property
    @abstractmethod
    def composition(self):
        pass
    @abstractmethod
    def set_composition(self,composition):
        pass
    def __str__(self):
        return self._name
        
    @property
    def name(self):
        return self._name
    


class ConstantComposition(CompositionBehavior):
    # 常量组成：
    # - 初始化时立即调用 set_composition 完成校验与命名
    # - 名称规则：按键排序，格式 "key:ratio*1000" 连接，用下划线分隔；可追加 suffix
    # 示例：
    # - {"ABS":0.8,"PLA":0.2} -> ABS:800_PLA:200
    # - {"PLA":1.0} -> PLA:1000
    # - 加 suffix='x' -> ABS:800_PLA:200_x
    def __init__(self,material_composition:Dict=None,suffix:str=None):
        self._composition=None
        self.set_composition(material_composition,suffix)
    @property
    def composition(self):
        return self._composition
    def set_composition(self,material_composition:Dict,suffix:str=None):
        # 统一使用校验函数保证 Dict 且满足数值约束（范围与和为 1）
        composition=_ensure_valid_comp_dict(material_composition)
        if composition!=None:
            # 生成易读的名称标签：键升序、值按千分制整数显示
            name=""
            name_seq=[k for k in composition.keys()]
            name_seq.sort()
            for i,n in enumerate(name_seq):
                value=composition[n]
                if i!=0:
                    name+="_"
                name+=f"{n}:{round(value,3)*1000:.0f}"
        if suffix!=None:
            name+=f"_{suffix}"
        self._name=name
        self._composition=composition

       




class GradientComposition(CompositionBehavior):
    def __init__(self,name):
        super().__init__(name)
        self._composition=None
    def set_composition(self,composition):
        self._composition=composition
    @property
    def composition(self):
        return self._composition

if __name__=="__main__":
    constant_composition=ConstantComposition("PLA")
    print(constant_composition.composition)
    constant_composition.set_composition({"PLA":0.2,"ABS":0.8})
    print(constant_composition.composition)
    gradient_composition=GradientComposition("gradient")
    print(gradient_composition.composition)
