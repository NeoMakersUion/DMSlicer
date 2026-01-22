# -*- coding: utf-8 -*-
"""
DMSlicer File Parser Package
============================

This package provides functionalities to parse various 3D file formats
for the DMSlicer application.

Currently supported formats:
- AMF (Additive Manufacturing File)

Usage Example:
    from file_parser import read_amf_objects
    objects = read_amf_objects("model.amf")
"""

__version__ = "1.0.0"
__author__ = "QilongJiang"
__license__ = "MIT"

from .amf_parser import read_amf_objects
from .mesh_data import MeshData
from .parser import file_parser

__all__ = [
    "read_amf_objects",
    "MeshData",
    "file_parser"
]
