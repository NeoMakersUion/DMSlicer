# -*- coding: utf-8 -*-
try:
    import pyvista as pv
except ImportError:
    pv = None

import numpy as np
from .visualizer_interface import IVisualizer
from ..file_parser.mesh_data import MeshData
from typing import Union,Optional
from ..geometry_kernel import Geom,Object


class PyVistaVisualizer(IVisualizer):
    """
    Concrete implementation of IVisualizer using PyVista.
    """

    def __init__(self, **kwargs):
        if pv is None:
            raise ImportError("PyVista is required for PyVistaVisualizer but not installed.")
        try:
            try:
                self.plotter = pv.Plotter(**kwargs)
            except TypeError as e:
                import logging
                logging.warning(f"PyVistaVisualizer init failed with kwargs: {e}. Fallback to default.")
                self.plotter = pv.Plotter()
        except Exception as e:
            # Fallback or re-raise depending on policy. Here we re-raise to let caller handle it.
            raise RuntimeError(f"Failed to initialize PyVista Plotter: {e}")

    def add(
        self,
        obj: Union[MeshData,Geom, pv.PolyData],
        opacity: float = 0.5,
        **kwargs
    ):
        """
        Add mesh to PyVista plotter.
        """
        # PyVista expects faces array to start with the number of vertices per face (3 for triangles)
        # Format: [3, v1, v2, v3, 3, v4, v5, v6, ...]
        if isinstance(obj, MeshData):
            mesh=obj
            triangles=mesh.triangles
            vertices=mesh.vertices
            opacity=kwargs.get("opacity",opacity)
            self.plotter=visualize_vertices_and_triangles(vertices,triangles,plotter=self.plotter,color=mesh.color_tuple,opacity=opacity)
        elif isinstance(obj, Geom):
            geom=obj
            object_list= kwargs.get("object_list",[])
            opacity_list= kwargs.get("opacity_list",[])
            vertices=geom.vertices.astype(np.float32)  # Convert int64 to float32 for PyVista
            
            # If object_list is empty, show all objects
            if not object_list:
                object_list = [o.id for o in geom.objects]
                
            if not opacity_list:
                opacity_list=[opacity]*len(object_list)
                
            for object in geom.objects:
                if object.id not in object_list:
                    continue
                else:
                    index=object_list.index(object.id)
                color=object.color
                triangles_ids=object.triangle_ids
                triangles=geom.triangles[triangles_ids]
                self.plotter=visualize_vertices_and_triangles(vertices,triangles,plotter=self.plotter,color=color,opacity=opacity_list[index])
                

        elif isinstance(obj, pv.PolyData):
            triangles = obj.faces.reshape(-1, 4)[:, 1:]
        else:
            raise ValueError("mesh must be either MeshData,Object or pv.PolyData")
        
        

    def show(self, **kwargs):
        """
        Show the PyVista plot.
        """
        try:
            self.plotter.show(**kwargs)
        except Exception as e:
            print(f"Warning: Failed to display 3D model. Details: {e}")



def visualize_vertices_and_triangles(
    vertices: np.ndarray, triangles: np.ndarray,
    plotter:Optional[pv.Plotter]=None,
    color="#66b3ff",
    show_edges=True,  # 显示三角面的边（方便看网格结构）
    edge_color="black",  # 边的颜色
    opacity=0.8,  # 透明度（0-1，1不透明）
    smooth_shading=True  # 平滑着色，更美观):
    ):
    """
    Visualize vertices and triangles using PyVista.
    """
    faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).flatten()

    # ===================== 3. 创建PyVista网格对象 =====================
    # PolyData是PyVista表示面网格的核心对象
    mesh = pv.PolyData(vertices, faces)

    # ===================== 4. 可视化配置（美化+交互）=====================
    # 创建绘图器
    no_return = False
    if plotter is None:
        no_return=True
        plotter = pv.Plotter(window_size=(1200, 800))
    # 添加网格：设置颜色、显示边、透明度
    plotter.add_mesh(
        mesh,
        color=color,  # 浅蓝色（也可以用RGB：color=[0.4, 0.7, 1.0]）
        show_edges=show_edges,  # 显示三角面的边（方便看网格结构）
        edge_color=edge_color,  # 边的颜色
        opacity=opacity,  # 透明度（0-1，1不透明）
        smooth_shading=smooth_shading  # 平滑着色，更美观
    )
    # 添加坐标轴、网格背景
    if no_return is True:
        plotter.add_axes()  # 显示xyz坐标轴
        plotter.show_grid()  # 显示网格线
        # 设置背景色
        plotter.set_background("white")
        # 显示交互窗口（支持鼠标旋转、缩放、平移）
        plotter.show()
    else:
        return plotter

def visualize_local_tris(
    vertices_sorted_by_zyx: np.ndarray, triangles_sorted_by_t123_vzyz: np.ndarray,local_tris:list,colors:list,
    plotter:Optional[pv.Plotter]=None,
    show_edges=True,  # 显示三角面的边（方便看网格结构）
    edge_color="black",  # 边的颜色
    opacity=0.8,  # 透明度（0-1，1不透明）
    smooth_shading=True  # 平滑着色，更美观):
    ):
    """
    Visualize vertices and triangles using PyVista.
    """
    if colors is None:
        colors=[color for color in pv.plotting.BUILTIN_COLORS]
    for vertices,triangles,color in zip(vertices_list,triangles_list,colors):
        visualize_vertices_and_triangles(vertices,triangles,plotter=plotter,color=color,show_edges=show_edges,edge_color=edge_color,opacity=opacity,smooth_shading=smooth_shading)
    visualize_vertices_and_triangles(vertices_sorted_by_zyx,triangles_sorted_by_t123_vzyz[local_tris[0]],color=colors[0])