from dataclasses import dataclass
from .materials import Material

@dataclass
class Material:
    name:str
    # 如果不用bed可以的话设置成-300 设置都
    nozzle_temp_normal: float
    nozzle_temp_min: float
    nozzle_temp_max: float
    bed_temp_normal: float
    bed_temp_min: float
    bed_temp_max: float
   

@dataclass
class ProcessProfile:
    name: str
    printer: str
    material_name: str   # "PLA", "TPU", "PETG"
    
    # Safe processing window

    # Flow & speed limits
    flow_rate: float      # mm^3/s
    speed: float         # mm/s

    # Retraction / cooling
    retract_length: float
    cooling_fan_percent: float

@dataclass
class Filament:
    """
    A specific spool sold by a manufacturer.
    """
    material: Material
    brand: str
    color: Tuple[int, int, int]        # RGB 0–255
    batch_id: Optional[str] = None
    diameter: float = 1.75             # mm

