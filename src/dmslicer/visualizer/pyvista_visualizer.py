# -*- coding: utf-8 -*-
try:
    import pyvista as pv
except ImportError:
    pv = None

import numpy as np
import os
from datetime import datetime
from .visualizer_interface import IVisualizer
from ..file_parser.mesh_data import MeshData
from typing import Union,Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..geometry_kernel import Geom, Object

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
    def addObject(self,geom:"Geom",object:"Object",triangles_ids=None,opacity:float=0.5,color=None):
        """
        Add object to PyVista plotter.
        """
        vertices=geom.vertices.astype(np.float32)  # Convert int64 to float32 for PyVista
        if color is None:
            color=object.color
        if triangles_ids is None:
            triangles_ids=object.tri_id2geom_tri_id
        
        if triangles_ids is None:
             import logging
             logging.warning(f"Object {object.id if object else 'Unknown'} has no triangles_ids. Skipping visualization.")
             return

        triangles=geom.triangles[triangles_ids]
        self.plotter=visualize_vertices_and_triangles(vertices,triangles,plotter=self.plotter,color=color,opacity=opacity)
    def addTriangle(self,triangle:"Triangle",opacity:float=0.5,color=None):
        """
        Add triangle to PyVista plotter.
        """
        if color is None:
            color="red"
        vertices=triangle.vertices.astype(np.float32)
        triangles=np.array([[0,1,2]])
        self.plotter=visualize_vertices_and_triangles(vertices,triangles,plotter=self.plotter,color=color,opacity=opacity)
    
    def addObj(self,object:"Object",include_triangles_ids=None,exclude_triangles_ids=None,opacity:float=0.5,color=None):
        """
        Add object to PyVista plotter.
        """
        if color is None:
            color=object.color
        vertices=object.vertices.astype(np.float32)
        tri_len=len(object.tri_id2vert_id)
        triangles_ids=list(range(tri_len))
        if include_triangles_ids is not None:
            if not isinstance(include_triangles_ids,list):
                raise ValueError(f"include_triangles_ids must be a list of triangle ids.")
            for tri_id in include_triangles_ids:
                if not tri_id in triangles_ids:
                    include_triangles_ids.remove(tri_id)
            triangles_ids=include_triangles_ids
  
        if exclude_triangles_ids is not None:
            if not isinstance(exclude_triangles_ids,list):
                raise ValueError(f"exclude_triangles_ids must be a list of triangle ids.")
            exclude_triangles_ids=sorted(exclude_triangles_ids,reverse=True)
            for tri_id in exclude_triangles_ids:
                if not tri_id in triangles_ids:
                    continue
                triangles_ids.remove(tri_id)
        triangles=object.tri_id2vert_id[np.array(triangles_ids)]
        self.plotter=visualize_vertices_and_triangles(vertices,triangles,plotter=self.plotter,color=color,opacity=opacity)

    def add(
        self,
        obj: Union[MeshData,"Geom", pv.PolyData],
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
        elif hasattr(obj, 'objects') and hasattr(obj, 'vertices'):  # Duck typing for Geom
            geom=obj
            object_list= kwargs.get("object_list",[])
            opacity_list= kwargs.get("opacity_list",[])
            vertices=geom.vertices.astype(np.float32)  # Convert int64 to float32 for PyVista
            
            # If object_list is empty, show all objects
            if not object_list:
                object_list = list(geom.objects.keys())
                
            if not opacity_list:
                opacity_list=[opacity]*len(object_list)
                
            for object in geom.objects.values():
                if object.id not in object_list:
                    continue
                else:
                    index=object_list.index(object.id)
                opacity=opacity_list[index]
                # self.addObject(geom,object,opacity=opacity)
                self.addObj(object,opacity=opacity)

                

        elif isinstance(obj, pv.PolyData):
            triangles = obj.faces.reshape(-1, 4)[:, 1:]
        else:
            raise ValueError("mesh must be either MeshData,Object or pv.PolyData")
    def addContactTriangles(self,objects,obj_pair:tuple,tris_ids_pairs:list,select=None,opacity:float=0.5):
        """
        Add contact triangles to PyVista plotter.
        """
        obj1_id,obj2_id=obj_pair
        obj1=objects[obj1_id]
        obj2=objects[obj2_id]
        if select is None or select==obj1_id or obj1_id in select:
            tri_ids_obj1=[elem[0] for elem in tris_ids_pairs]
            self.addObj(obj1,tri_ids_obj1,opacity=opacity)
        if select is None or select==obj2_id or obj2_id in select:
            tri_ids_obj2=[elem[0] for elem in tris_ids_pairs]
            self.addObj(obj2,tri_ids_obj2,opacity=opacity)
    

    def show(self, **kwargs):
        """
        Show the PyVista plot.
        """
        try:
            self.plotter.show(**kwargs)
        except Exception as e:
            print(f"Warning: Failed to display 3D model. Details: {e}")

    def save(self, file_path: str = None, name: str = None, **kwargs):
        """
        Save the PyVista plot to a file.
        
        Supports:
        - Image: .png, .jpg, .jpeg, .bmp, .tiff (via screenshot)
        - Interactive 3D: .gltf, .glb (Recommended), .vtkjs (Scientific)
        - Geometry: .stl, .ply, .vtk
        
        Args:
            file_path: Target file path. If None, auto-generates in 'screenshots/'.
            name: Optional tag to include in the auto-generated filename (e.g. 'layer_1').
            **kwargs: 
                - format (str): Force specific format if file_path is None (default: 'png').
                - off_screen (bool): For image/screenshot (default: True).
                - inline (bool): For vtkjs export (default: False).
        """
        try:
            # 1. Determine Format & Path
            fmt = kwargs.pop('format', None)
            
            if file_path is None:
                output_dir = "screenshots"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                ext = f".{fmt}" if fmt else ".png"
                
                if name:
                    # Sanitize name to be safe for filenames
                    safe_name = "".join([c for c in name if c.isalnum() or c in ('-', '_')]).strip()
                    filename = f"snapshot_{timestamp}_{safe_name}{ext}"
                else:
                    filename = f"snapshot_{timestamp}{ext}"
                
                file_path = os.path.join(output_dir, filename)
            
            # Normalize extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # 2. Dispatch based on extension
            if ext in ['.gltf', '.glb']:
                # Interactive 3D (GLTF/GLB) - Recommended
                # Note: GLTF export usually requires the plotter to be rendered once or just scene export.
                self.plotter.export_gltf(file_path, **kwargs)
                print(f"Saved Interactive 3D (GLTF) to: {file_path}")
                
            elif ext == '.vtkjs':
                # Interactive 3D (VTKJS) - Not supported in current PyVista, fallback to GLTF
                print(f"Warning: .vtkjs export is not supported by this PyVista version. Falling back to .gltf.")
                new_path = os.path.splitext(file_path)[0] + ".gltf"
                self.plotter.export_gltf(new_path, **kwargs)
                print(f"Saved Interactive 3D (GLTF) to: {new_path}")
                
            elif ext == '.html':
                # HTML Export via Embedded GLTF (Robust for Offline/Local View)
                # 1. Export scene to temporary GLTF
                import tempfile
                import base64
                
                with tempfile.NamedTemporaryFile(suffix='.gltf', delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    # Export to GLTF with inline data (single file JSON)
                    self.plotter.export_gltf(tmp_path, inline_data=True)
                    
                    # 2. Read GLTF content
                    with open(tmp_path, 'rb') as f:
                        gltf_content = f.read()
                        
                    # 3. Encode to Base64
                    gltf_b64 = base64.b64encode(gltf_content).decode('utf-8')
                    
                    # 4. Generate HTML with embedded model
                    # Using jsDelivr for model-viewer which is generally accessible
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DMSlicer 3D View</title>
    <script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; font-family: sans-serif; }}
        model-viewer {{ width: 100vw; height: 100vh; background-color: #f0f0f0; }}
        .controls {{ position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="controls">
        <b>Controls:</b> Left Click: Rotate | Right Click: Pan | Scroll: Zoom
    </div>
    <model-viewer 
        src="data:model/gltf+json;base64,{gltf_b64}" 
        camera-controls 
        auto-rotate
        shadow-intensity="1"
        camera-orbit="45deg 55deg 2.5m"
        field-of-view="30deg">
    </model-viewer>
</body>
</html>"""
                    
                    # 5. Write HTML file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                        
                    print(f"Saved Self-Contained HTML to: {file_path}")
                    
                finally:
                    # Cleanup temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)


            elif ext in ['.stl', '.ply', '.vtk', '.vtp']:
                # Geometry Export (Scene-level export is tricky, usually exports the whole scene as one or supported by some methods)
                # PyVista plotter.save_mesh is for single mesh. plotter.export_obj exists.
                if ext == '.obj':
                    self.plotter.export_obj(file_path, **kwargs)
                else:
                    # For STL/PLY, we might need to export individual meshes or merge them.
                    # Plotter doesn't have a direct 'save_scene_to_stl'.
                    # We'll try generic export if available, or warn.
                    # Actually, let's stick to what Plotter supports directly.
                    # Fallback to saving the 'active' mesh if possible, or warn.
                    print(f"Warning: Scene export to {ext} might not support all objects. Using GLTF is recommended.")
                    # Attempting generic save if implemented in newer PyVista, otherwise warn.
                    # self.plotter.save(file_path) # This exists in recent PyVista for some formats
                    pass 
                    
            else:
                # Default: Image Screenshot (.png, .jpg, etc.)
                
                # Note: 'off_screen' handling for screenshot
                off_screen = kwargs.pop('off_screen', True)
                if self.plotter is not None:
                    self.plotter.off_screen = off_screen

                self.plotter.show(screenshot=file_path, **kwargs)
                # print(f"Saved Snapshot to: {file_path}")

        except Exception as e:
            print(f"Error: Failed to save to {file_path}. Details: {e}")



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
