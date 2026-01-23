from enum import Enum
class BuildRole(str, Enum):
    ASSEMBLY = "assembly"     # 装配体（不切，但用于层级与定位）
    PART = "part"            # 正常可打印零件
    SUPPORT = "support"      # 支撑
    GRADIENT = "gradient"    # 梯度材料
    GLUE = "glue"            # 胶水结构
    INSERT = "insert"        # 嵌件 / 占位 / 预留
    IGNORE = "ignore"        # 完全不参与