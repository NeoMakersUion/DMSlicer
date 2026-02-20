from abc import ABC,abstractmethod
from enum import Enum
class Material_Constant(Enum):
    GRADIENT = 1
    CONSTANT = 2
class Material_Action(Enum):
    DEPENDENT = 1
    INDEPENDENT = 2

class Material(ABC):
    def __init__(self,material_type:Material_Constant,material_action:Material_Action):
        self.material_type=material_type
        self.material_action=material_action
        self.include=set()
        self.exclude=set()


    @abstractmethod
    def action(self):
        raise NotImplementedError

class Gradient_Material(Material):
    def __init__(self):
        super().__init__(Material_Constant.GRADIENT,Material_Action.DEPENDENT)
    def action(self):
        pass
        

class Constant_Material(Material):
    def __init__(self,material_value):
        super().__init__(Material_Constant.CONSTANT,Material_Action.DEPENDENT)
        self.composition=material_value
    def action(self):
        pass
class Independent_Material(Material):
    def __init__(self,material_value):
        super().__init__(Material_Constant.CONSTANT,Material_Action.INDEPENDENT)
        self.composition=material_value
    def action(self):
        pass 