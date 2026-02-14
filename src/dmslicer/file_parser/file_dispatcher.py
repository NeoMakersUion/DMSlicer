# -*- coding: utf-8 -*-
"""
File Dispatcher
统一的文件解析分发入口，根据文件后缀调用具体解析器。

主要接口:
- parse_file: 通用入口，返回 Model

依赖:
- 标准库: pathlib, typing
- 第三方: numpy (类型引用)
- 本地模块: amf_parser.read_amf_objects

作者: DMSlicer Team
创建日期: 2026-02-14
"""

from pathlib import Path
from typing import Any, Union

from .model import Model
from .amf_parser import read_amf_objects


def parse_file(file_path: Union[str, Path], **kwargs: Any) -> Model:
    """Dispatch file parsing based on file suffix and return Model.

    Args:
        file_path: 文件路径，支持 str 或 Path。
        **kwargs: 传递给具体解析器的参数，例如 progress、show、visualizer_type 等。

    Returns:
        Model: 解析得到的模型对象。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        ValueError: 当文件类型暂不支持时抛出。

    Examples:
        >>> from pathlib import Path
        >>> from dmslicer.file_parser.file_dispatcher import parse_file
        >>> model = parse_file(Path("sample.amf"), progress=False, show=False)
        >>> assert model.count >= 0
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".amf":
        return read_amf_objects(file_path, **kwargs)
    raise ValueError(f"Unsupported file format: {suffix}. Only .amf is supported.")
