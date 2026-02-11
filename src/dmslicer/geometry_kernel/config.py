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
        "GEOM_ACC": 4,                  # 几何精度 (保留小数位数)
        "GEOM_PARALLEL_ACC":1e-1,       # 并行精度 (角度阈值)
        "PROCESS_ACC":2, #加工精度
            
        ## ===== Step 3-1: Soft Normal Gating (无向法线软门控) =====
        "SOFT_NORMAL_GATE_ANGLE": 60.0,  # 单位: 度 | 文档依据: "仅当 θ > 60° 时剔除"
                                    # 命名逻辑: "软门控" + "法线" + "角度阈值"

        # ===== Step 3-1: Scale-Aware Gap Tolerance (尺度感知间隙容差) =====
        "INITIAL_GAP_FACTOR": 0.01,   # α | ε₀ = α · h (初始容差比例)
        "MAX_GAP_FACTOR": 0.5,        # β | ε_max = β · h (最大容差比例)
                                    # 命名逻辑: "初始/最大" + "间隙" + "比例因子"
                                    # ✅ 避免使用 magic number 10 (β/α)
        "OVERLAP_RATIO_THRESHOLD": 0.1  # γ | 重叠区域占比阈值 (默认 0.5)
    }

PROCESS_ACC=DEFAULTS["PROCESS_ACC"]    
GEOM_PARALLEL_ACC=DEFAULTS["GEOM_PARALLEL_ACC"]    
GEOM_ACC = DEFAULTS["GEOM_ACC"]
DEFAULT_VISUALIZER_TYPE=DEFAULTS["DEFAULT_VISUALIZER"]

# ===== Step 3-1: Soft Normal Gating (无向法线软门控) =====
SOFT_NORMAL_GATE_ANGLE = DEFAULTS["SOFT_NORMAL_GATE_ANGLE"]  # 单位: 度 | 文档依据: "仅当 θ > 60° 时剔除"
                                   # 命名逻辑: "软门控" + "法线" + "角度阈值"

# ===== Step 3-1: Scale-Aware Gap Tolerance (尺度感知间隙容差) =====
INITIAL_GAP_FACTOR = DEFAULTS["INITIAL_GAP_FACTOR"]   # α | ε₀ = α · h (初始容差比例)
MAX_GAP_FACTOR = DEFAULTS["MAX_GAP_FACTOR"]        # β | ε_max = β · h (最大容差比例)   
                            # 命名逻辑: "初始/最大" + "间隙" + "比例因子"
                            # ✅ 避免使用 magic number 10 (β/α)
OVERLAP_RATIO_THRESHOLD=DEFAULTS["OVERLAP_RATIO_THRESHOLD"]
