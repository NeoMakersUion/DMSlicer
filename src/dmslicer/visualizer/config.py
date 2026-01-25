# -*- coding: utf-8 -*-
"""
DMSlicer File Parser Configuration
==================================

This module stores default configuration values for file parsers.
It supports versioning and provides a fallback mechanism through direct attribute access.

Attributes:
    CONFIG_VERSION (str): Version of the configuration schema.
    DEFAULTS (dict): Dictionary containing default values.
"""


try:
    from ..default_config import DEFAULTS
except:
    # Default configuration values
    # These values can be adjusted without modifying the core parser logic
    CONFIG_VERSION = "sub_1.0.0"
    from typing import Any, Dict
    from .visualizer_type import VisualizerType
    DEFAULTS: Dict[str, Any] = {
        # AMF Parser Defaults
        "AMF_TARGET_REDUCTION": 0.0,      # 网格简化比例 (0~1)
        "AMF_DECIMATE_MIN_TRIS": 100,     # 仅当三角数 >= 该阈值时才执行简化
        "DEFAULT_VISUALIZER": VisualizerType.PyVista, # 默认可视化器类型
    }



# Explicitly expose configuration variables for direct import
AMF_TARGET_REDUCTION = DEFAULTS["AMF_TARGET_REDUCTION"]
AMF_DECIMATE_MIN_TRIS = DEFAULTS["AMF_DECIMATE_MIN_TRIS"]
DEFAULT_VISUALIZER = DEFAULTS["DEFAULT_VISUALIZER"]
