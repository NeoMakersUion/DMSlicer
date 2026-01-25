try:
    from ..default_config import DEFAULTS
    from ..visualizer.visualizer_type import VisualizerType
except ImportError:
    CONFIG_VERSION = "sub_1.0.0"
    from typing import Any, Dict
    # Fallback mock if import fails
    class VisualizerType:
        PyVista = 0
        
    DEFAULTS: Dict[str, Any] = {
        "DEFAULT_VISUALIZER":VisualizerType.PyVista,
        "GEOM_ACC": 6                   # 几何精度 (保留小数位数)
    }
GEOM_ACC = DEFAULTS["GEOM_ACC"]
DEFAULT_VISUALIZER_TYPE=DEFAULTS["DEFAULT_VISUALIZER"]
