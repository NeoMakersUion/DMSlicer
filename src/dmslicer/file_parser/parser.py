# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import numpy as np

from .amf_parser import read_amf_objects

def file_parser(
    file_path: Union[str, Path],
    **kwargs
) -> List[Dict[str, Union[np.ndarray, Tuple[float, float, float]]]]:
    """
    通用文件解析入口函数。根据文件后缀分发到具体的解析器。

    Args:
        file_path: 文件路径
        **kwargs: 传递给具体解析器的参数 (如 target_reduction, show, decimate_min_tris, progress 等)

    Returns:
        解析后的对象列表，格式同具体 parser (如 read_amf_objects) 的返回。
        List[dict]，每个元素包含 'vertices', 'triangles', 'color' 等信息。

    Raises:
        ValueError: 如果文件格式不支持或文件不存在。
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == '.amf':
        return read_amf_objects(file_path, **kwargs)
    else:
        # 可以在这里扩展其他格式，如 .stl, .obj 等
        raise ValueError(f"Unsupported file format: {suffix}. Currently only .amf is supported.")
