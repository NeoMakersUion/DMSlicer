"""
DMSlicer Visualization Package
==============================

该包提供了 DMSlicer 项目的几何可视化功能，旨在为用户提供统一、易用的接口来显示和交互 3D 模型数据。
核心设计基于工厂模式和抽象基类，支持多种后端可视化引擎（目前主要支持 PyVista）。

主要功能 (Key Features)
-----------------------
- **统一接口**: 通过 `IVisualizer` 提供标准化的可视化操作接口。
- **多后端支持**: 架构设计支持扩展不同的可视化库（如 PyVista, OpenGL 等）。
- **工厂模式**: 支持通过 `IVisualizer.create()` 动态创建可视化实例。
- **交互式显示**: 提供窗口化的 3D 模型查看、旋转、缩放等交互功能。

模块说明 (Modules)
------------------
- `visualizer_interface`: 定义抽象基类 `IVisualizer`，规范所有可视化器的行为。
- `pyvista_visualizer`: 基于 PyVista 库的具体实现 `PyVistaVisualizer`。
- `visualizer_type`: 定义 `VisualizerType` 枚举，用于指定可视化后端类型。

使用示例 (Usage Example)
------------------------
.. code-block:: python

    from dmslicer.visualizer import IVisualizer, VisualizerType

    # 1. 使用工厂方法创建可视化器实例 (推荐)
    visualizer = IVisualizer.create(VisualizerType.PYVISTA)

    # 2. 设置窗口标题
    visualizer.set_title("3D Mesh Viewer")

    # 3. 添加网格数据 (vertices, faces)
    # 假设 vertices 是 (N, 3) 的 numpy 数组，faces 是 (M, 3) 或 (M, 4) 的 numpy 数组
    visualizer.add_mesh(vertices, faces, color="lightblue")

    # 4. 启动可视化窗口
    visualizer.show()

注意事项 (Notes)
----------------
- **依赖库**: 默认实现依赖 `pyvista` 和 `vtk`。请确保环境中已安装这些库 (`pip install pyvista`)。
- **线程安全**: 可视化窗口通常需要在主线程中运行，多线程环境下请注意 GUI 事件循环的限制。
"""

__version__ = "1.0.0"
__author__ = "QilongJiang"
__license__ = "MIT"
# -*- coding: utf-8 -*-

from .visualizer_type import VisualizerType
from .visualizer_interface import IVisualizer
from .pyvista_visualizer import PyVistaVisualizer 

__all__ = ["IVisualizer", "PyVistaVisualizer", "VisualizerType"]
