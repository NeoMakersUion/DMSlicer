from dataclasses import dataclass, field
from typing import List, Optional, Dict

from dmslicer.file_parser import MeshData
from .build_role import BuildRole
from .process_profile import ProcessProfile,Material


# ===========================
# Build-time object
# ===========================

@dataclass
class BuildObject:
    """
    One geometry + manufacturing intent.
    """
    mesh: MeshData
    role: BuildRole
    material: Material
    #通过adjacent_objects来表示与其他对象的联系,从而可将Gradient材料与其他对象区分开来
    adjacent_objects: List[BuildObject] = field(default_factory=list)



# ===========================
# Build model
# ===========================

@dataclass
class BuildModel:
    """
    Geometry + manufacturing semantics.
    """
    objects: List[BuildObject] = field(default_factory=list)

    def get_printable_objects(self) -> List[BuildObject]:
        return [o for o in self.objects if o.is_manufacturable()]

    def get_all_processes(self) -> List[ProcessProfile]:
        return list({o.process for o in self.objects if o.process is not None})
