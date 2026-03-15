from abc import ABCMeta,abstractmethod
from typing import List,Dict,Optional,Any,Sequence,Mapping,Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger=logging.getLogger(__name__)

def _normalize_position_key(key: Union[int, Sequence[float], Mapping[str, Any]]) -> Union[int, tuple]:
    """
    规范化位置键 / Normalize position key
    ----------------------------------
    支持以下入参形式（键名大小写不敏感）/ Supported forms (case-insensitive keys):
      1) 对象 id：int → 返回相同整数
      2) 空间坐标序列：list/tuple 长度为 3（元素为 int/float）→ 返回三元组 (x,y,z)
      3) 字典：
         - {'x': v, 'y': v, 'z': v} 或大小写/混合写法（例如 'X','y','Z'）
         - {'xyz': [vx, vy, vz]} 或大小写写法（例如 'XYZ'）
    
    返回 / Returns:
      - int（对象 id）或 (x,y,z) 三元组（全 int 或全 float）
    
    异常 / Raises:
      - TypeError: 入参形态不受支持，或字典缺少必要键，或坐标维度非数值
    """
    if isinstance(key, int):
        return int(key)
    if isinstance(key, (list, tuple)) and len(key) == 3 and all(isinstance(i, (int, float)) for i in key):
        k0, k1, k2 = key
        if all(isinstance(i, int) for i in (k0, k1, k2)):
            return (int(k0), int(k1), int(k2))
        return (float(k0), float(k1), float(k2))
    if isinstance(key, Mapping):
        lower_map = {str(k).lower(): v for k, v in key.items()}
        if "xyz" in lower_map:
            seq = lower_map["xyz"]
            if isinstance(seq, (list, tuple)) and len(seq) == 3 and all(isinstance(i, (int, float)) for i in seq):
                a, b, c = seq
                if all(isinstance(i, int) for i in (a, b, c)):
                    return (int(a), int(b), int(c))
                return (float(a), float(b), float(c))
            raise TypeError("xyz must map to a 3-length list/tuple of numbers")
        required = ("x", "y", "z")
        if all(r in lower_map for r in required):
            a, b, c = lower_map["x"], lower_map["y"], lower_map["z"]
            if all(isinstance(i, (int, float)) for i in (a, b, c)):
                if all(isinstance(i, int) for i in (a, b, c)):
                    return (int(a), int(b), int(c))
                return (float(a), float(b), float(c))
            raise TypeError("x,y,z values must be numbers")
    raise TypeError("position key must be int (object id), a 3-number list/tuple, or a dict with x/y/z or xyz")

def _ensure_valid_comp_dict(dict_material_composition: Dict,esp:float=1e-6) -> Dict[str, float]:
    """
    校验并规范化材料组成字典 / Validate and normalize material composition mapping.

    约定 / Contract:
    - 接受字符串或 Dict[str, number]；字符串将被视为单一组分 {name: 1.0}
    - 各组分值需位于 [0, 1]（含容差 esp），总和需等于 1（含容差）

    参数 / Args:
        dict_material_composition: 组分映射或单一材料名字符串
        esp: 容差（用于单项范围与总和判断）

    返回 / Returns:
        规范化后的 Dict[str, float]（不修改入参对象）

    异常 / Raises:
        TypeError: 输入类型不合法（既不是字符串也不是字典）
        ValueError: 组分值非数值、单项不在 [0,1]、或总和不等于 1（容差外）

    说明 / Notes:
        本函数仅进行数值与约束校验，不重命名或排序键。
        EN: Validates numeric constraints only; does not rename or sort keys.
    """
    if isinstance(dict_material_composition,str):
        dict_material_composition={dict_material_composition:1.0}
    # 类型检查：必须为字典 / EN: Type check – must be dict
    if not isinstance(dict_material_composition, Dict):
        raise TypeError("material_composition must be Dict")
    # 结果映射与累计和 / EN: Result map and running sum
    result: Dict[str, float] = {}
    s = 0.0
    # 遍历条目并校验/规范化 / EN: Iterate entries; validate and normalize
    for k, v in dict_material_composition.items():
        # 数值类型校验 / EN: Numeric type check
        if not isinstance(v, (float, int)):
            raise ValueError("composition values must be numeric")
        # 转为浮点数 / EN: Cast to float
        vv = float(v)
        # 范围校验：[0,1]（含容差）/ EN: Range check [0,1] with epsilon
        if vv < -esp or vv > 1+esp:
            raise ValueError("each composition value must be in [0, 1]")
        # 记录规范值并累计 / EN: Store normalized value and accumulate
        result[k] = vv
        s += vv
    # 总和校验：需等于1（含容差）/ EN: Sum must be 1 within epsilon
    if abs(s - 1.0) > esp:
        # 不满足则抛出异常 / EN: Raise error if sum mismatch
        raise ValueError("sum of composition values must be 1")
    return result

class CompositionBehavior(metaclass=ABCMeta):
    """
    组成行为抽象基类 / Abstract base class for composition behavior.

    职责 / Responsibility:
    - 维护名称（_name）与当前组成（_composition）
    - 约束派生类提供“读取当前组成”的接口与“设置/更新组成”的接口

    适配 / Adaptation:
    - ConstantComposition：返回/设置全局常量型组成（Dict[str, float]）
    - GradientComposition：返回/设置空间位置到组成的映射（(x,y,z) → Dict[str, float]）

    说明 / Notes:
    - 本类仅定义行为契约，不涉及具体的校验细节；数值与结构校验应由派生类或工具函数完成。
      EN: Defines the behavior contract only; numeric/structure validation is implemented by subclasses or helpers.
    """
    # 抽象“组成行为”基类：
    # - 维护名称 _name 与当前组成 _composition
    # - 要求派生类实现 composition 属性与 set_composition 方法
    def __init__(self,name):
        self._name=name
        self._composition=None

    @abstractmethod
    def composition(self):
        """
        获取当前材料组成 / Return current material composition.

        返回 / Returns:
            任意可表示组成的结构，例如：
            - 常量组成：Dict[str, float]
            - 梯度组成：Dict[Tuple[int,int,int], Dict[str, float]]
        """
        pass
    @abstractmethod
    def set_composition(self,composition):
        """
        设置或更新材料组成 / Set or update material composition.

        参数 / Args:
            composition: 目标组成结构；形态由派生类自行约定
                - 常量：Dict[str, float] 或 str
                - 梯度：Dict[(x,y,z) -> Dict[str, float]] 或 (xyz, dict) 形式
        """
        pass
    def __str__(self):
        return self._name
        
    @property
    def name(self):
        return self._name
    


class ConstantComposition(CompositionBehavior):
    """
    常量组成 / Constant material composition
    --------------------------------------
    为材料提供“常量型”的组成定义（不随位置变化）。在构造或设置时会进行数值与约束校验，
    并生成可读名称标签（基于键升序与千分制数值）。

    行为 / Behavior:
    - 校验与规范化：调用 _ensure_valid_comp_dict 保证各组分在 [0,1] 且总和为 1；
      接受 str 或 Dict[str, number]（字符串将被视为单一组分 {name:1.0}）
    - 命名规则：按键名升序，名称片段为 "key:ratio*1000"；多个片段用下划线连接
      例如 {"ABS":0.8,"PLA":0.2} → "ABS:800_PLA:200"
    - 可选后缀：若提供 suffix，则在名称尾部追加 "_{suffix}"

    示例 / Examples:
    - {"ABS":0.8, "PLA":0.2} → ABS:800_PLA:200
    - {"PLA":1.0} → PLA:1000
    - suffix='x' → ABS:800_PLA:200_x
    """
    def __init__(self,material_composition:Dict=None,suffix:str=None):
        """
        初始化常量组成 / Initialize constant composition

        参数 / Args:
            material_composition: Dict[str, float] 或 str；将通过 _ensure_valid_comp_dict 校验与规范化
            suffix: 可选名称后缀，用于区分同配比的不同语义标签
        """
        self._composition=None
        self.set_composition(material_composition,suffix)
    @property
    def composition(self):
        """
        获取当前常量组成 / Return current constant composition

        返回 / Returns:
            Dict[str, float]：键为材料名称，值为占比（满足范围与和约束）
        """
        return self._composition
    def set_composition(self,material_composition:Dict,suffix:str=None):
        """
        设置常量组成并更新名称 / Set constant composition and update display name

        参数 / Args:
            material_composition: Dict[str, float] 或 str；各值位于 [0,1] 且总和为 1（含容差）
            suffix: 可选名称后缀，若提供则在生成名称后追加

        说明 / Notes:
            名称生成遵循“键名升序 + 千分制数值”的规则，便于人类可读与差异化比较。
        """
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
    """
    梯度组成 / Gradient material composition
    -------------------------------------
    在不同“位置键”上绑定不同的材料组成映射，位置键通过 _normalize_position_key 统一：
      - 对象 id（int）
      - 空间坐标：list/tuple 三元数（int/float）
      - 字典：{'x','y','z'} 或 {'xyz':[x,y,z]}，键名大小写不敏感

    组成值通过 _ensure_valid_comp_dict 校验并规范化（支持 Dict[str,float] 或 str→{name:1.0}）。
    若值为 None，则表示该位置未定义具体组成。

    EN:
    Binds material compositions to “position keys” normalized by _normalize_position_key:
      - object id (int)
      - spatial coordinate: 3-number list/tuple (int/float)
      - dict forms: {'x','y','z'} or {'xyz':[x,y,z]} (case-insensitive keys)
    The composition is validated by _ensure_valid_comp_dict (Dict[str,float] or str→{name:1.0}).
    None denotes undefined composition at that position.
    """
    def __init__(self,name):
        super().__init__(name)
        self._composition={}
    def set_composition(self,pos=None,composition=None):
        """
        设置梯度组成 / Set gradient composition.

        参数 / Args:
            pos:
                - int: 对象 id
                - list/tuple: 长度为 3 的坐标（int/float）
                - dict: {'x','y','z'} 或 {'xyz':[x,y,z]}，键名大小写不敏感
            composition:
                - Dict[str, float]：将通过 _ensure_valid_comp_dict 校验
                - str：表示单一材料名，将规范化为 {name:1.0}
                - None：表示该位置没有定义具体组成

        调用形式 / Call forms:
            1) set_composition(pos, composition)  → 设置单个位置
            2) set_composition({pos: composition, ...})  → 批量设置位置

        行为 / Behavior:
            - 使用 _normalize_position_key 统一位置键
            - 使用 _ensure_valid_comp_dict 统一并校验组成（若非 None）
        """
        if pos==None or composition==None:
            raise TypeError("set_composition expects (pos, composition) or (dict)")
        # 批量形式：set_composition({pos: composition, ...})
        if composition is None and isinstance(pos, dict):
            for k, v in pos.items():
                key=_normalize_position_key(k)
                if v is None:
                    self._composition[key]=None
                else:
                    self._composition[key]=_ensure_valid_comp_dict(v)
            return
        # 单点形式：set_composition(pos, composition)
        key=_normalize_position_key(pos)
        v=composition
        if v is None:
            self._composition[key]=None
        else:
            self._composition[key]=_ensure_valid_comp_dict(v)
    def composition(self,xyz):
        """
        获取指定位置的组成 / Get composition at a given position.

        参数 / Args:
            xyz: 与 set_composition 中 pos 相同的“位置键”形态
        返回 / Returns:
            Dict[str,float]｜None：该位置的组成映射；若未定义则返回 None
        """
        return self._composition.get(_normalize_position_key(xyz),None) 

if __name__=="__main__":
    constant_composition=ConstantComposition("PLA")
    print(constant_composition.composition)
    constant_composition.set_composition({"PLA":0.2,"ABS":0.8})
    print(constant_composition.composition)
    gradient_composition=GradientComposition("gradient")
    print(gradient_composition.composition)
