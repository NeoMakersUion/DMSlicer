from typing import Any, Dict
CONFIG_VERSION = "1.0.0"
from .visualizer.visualizer_interface import VisualizerType
DEFAULTS: Dict[str, Any] = {
    # AMF Parser Defaults
    "AMF_TARGET_REDUCTION": 0.0,      # 网格简化比例 (0~1)
    "AMF_DECIMATE_MIN_TRIS": 100,     # 仅当三角数 >= 该阈值时才执行简化
    "DEFAULT_VISUALIZER": VisualizerType.PyVista,  # 默认可视化器

    # Geometry Kernel Defaults
    "GEOM_ACC": 4,                   # 几何精度 (保留小数位数)
    "GEOM_PARALLEL_ACC":1e-1,       # 计算三角形平时的精度 (角度阈值)
    
    # Processing Defaults
    "PROCESS_ACC":2, #加工精度

    ## ===== Step 3-1: Soft Normal Gating (无向法线软门控) =====
    "SOFT_NORMAL_GATE_ANGLE": 25,  # 单位: 度 | 文档依据: "仅当 θ > 45° 时剔除"
                                   # 命名逻辑: "软门控" + "法线" + "角度阈值"

    # ===== Step 3-1: Scale-Aware Gap Tolerance (尺度感知间隙容差) =====
    "INITIAL_GAP_FACTOR": 0.01,   # α | ε₀ = α · h (初始容差比例)
    "MAX_GAP_FACTOR": 0.5,        # β | ε_max = β · h (最大容差比例)
                                # 命名逻辑: "初始/最大" + "间隙" + "比例因子"
                                # ✅ 避免使用 magic number 10 (β/α)
    "OVERLAP_RATIO_THRESHOLD": 0.01  # γ | 重叠区域占比阈值 (默认 0.5)
}