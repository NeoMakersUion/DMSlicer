from typing import Any, Dict
CONFIG_VERSION = "1.0.0"
from .visualizer.visualizer_interface import VisualizerType
DEFAULTS: Dict[str, Any] = {
    # AMF Parser Defaults
    "AMF_TARGET_REDUCTION": 0.0,      # 网格简化比例 (0~1)
    "AMF_DECIMATE_MIN_TRIS": 100,     # 仅当三角数 >= 该阈值时才执行简化
    "DEFAULT_VISUALIZER": VisualizerType.PyVistaVisualizer,  # 默认可视化器
}