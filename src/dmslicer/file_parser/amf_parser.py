from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from tqdm import tqdm
from ..visualizer.visualizer_interface import IVisualizer
from .mesh_data import MeshData
from .model import Model
from ..visualizer.visualizer_type import VisualizerType

def read_amf_objects(
    uploaded_file: Union[str, Path],
    progress: bool = True,
    show: bool = False,
    visualizer_type: Optional[VisualizerType] = None,
    **kwargs
) -> Model:
    """
    解析 AMF 文件（Additive Manufacturing File）并构建 3D 模型对象。

    该函数读取 AMF 文件中的 XML 数据，提取顶点、三角形和颜色信息，
    将其转换为 MeshData 对象，并聚合到 Model 实例中。
    同时支持通过工厂模式初始化的可视化器进行实时预览。

    Args:
        uploaded_file (Union[str, Path]): AMF 文件的路径。
        progress (bool, optional): 是否显示进度条 (tqdm)。默认为 True。
        show (bool, optional): 是否在解析完成后显示 3D 预览。默认为 False。
        visualizer_type (Optional[VisualizerType], optional): 指定可视化器的类型。
            如果为 None 且 show=True，将使用默认的可视化器（通常是 PyVista）。
        **kwargs: 传递给其他潜在扩展功能的额外参数。

    Returns:
        Model: 包含所有解析出的 MeshData 对象的模型容器。
            如果解析失败（如文件不存在或格式错误），返回一个空的 Model 实例。
    """
    try:
        with open(uploaded_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        # 预处理 XML：移除命名空间前缀以简化 XPath 查询
        xml_content = xml_content.replace('amf:', '').replace('ns:', '').replace(':', '')
        root = ET.fromstring(xml_content)
    except FileNotFoundError:
        print(f"Error: File '{uploaded_file}' not found.")
        return Model()
    except ET.ParseError as e:
        print(f"Error: Failed to parse XML content. Details: {e}")
        return Model()
    except Exception as e:
        print(f"Error: Unexpected error occurred while reading file. Details: {e}")
        return Model()

    model = Model()
    object_xml = root.findall(".//object")
    
    # 主进度条：遍历所有 object 节点
    for index, obj in enumerate(tqdm(object_xml, desc="AMF objects", unit="obj", disable=not progress)):
        try:
            # 查找顶点和三角形数据的 XML 节点
            verts_xml = obj.findall(".//vertex")
            tris_xml  = obj.findall(".//triangle")

            # ---- 解析顶点 (Vertices) ----
            vertices = []
            # 子进度条：当数据量较大时显示顶点解析进度
            it_v = tqdm(verts_xml, desc=f"[obj {index}] vertices", unit="v",
                        leave=False, disable=not progress or len(verts_xml) < 2000)
            for vertex in it_v:
                coords = vertex.find("coordinates")
                if coords is not None:
                    x = float(coords.find("x").text.strip())
                    y = float(coords.find("y").text.strip())
                    z = float(coords.find("z").text.strip())
                    vertices.append([x, y, z])
            vertices = np.array(vertices, dtype=np.float32)

            # ---- 解析三角形 (Triangles) ----
            triangles = []
            # 子进度条：当数据量较大时显示三角形解析进度
            it_t = tqdm(tris_xml, desc=f"[obj {index}] triangles", unit="tri",
                        leave=False, disable=not progress or len(tris_xml) < 2000)
            for triangle in it_t:
                v1 = int(triangle.find("v1").text.strip())
                v2 = int(triangle.find("v2").text.strip())
                v3 = int(triangle.find("v3").text.strip())
                triangles.append([v1, v2, v3])
            triangles = np.array(triangles, dtype=np.int64)

            # ---- 解析颜色 (Color) ----
            # 尝试从 XML 中提取颜色信息，默认为灰色
            color_elem = obj.find(".//color")
            r, g, b = 0.5, 0.5, 0.5 
            if color_elem is not None:
                r = float(color_elem.find("r").text.strip())
                g = float(color_elem.find("g").text.strip())
                b = float(color_elem.find("b").text.strip())

            # 构建 MeshData 对象并添加到 Model 中
            mesh_data_tmp = MeshData(
                vertices=np.asarray(vertices),
                triangles=np.asarray(triangles, dtype=np.int64),
                color=np.array([r, g, b], dtype=float)
            )
            model.add_mesh(mesh_data_tmp)

        except Exception as e:
            # 捕获单个 object 解析过程中的错误，避免中断整个文件的解析
            print(f"Error: processing object {index}. Details: {e}")
            continue

    # ---- 可视化处理 ----
    # 如果请求显示 (show=True)，则初始化并运行可视化器
    if show:
        try:
            # 使用工厂方法根据指定的类型创建可视化器实例
            visualizer = IVisualizer.create(visualizer_type)
        except Exception as e:
            print(f"Warning: Failed to initialize visualizer. Details: {e}")
            show = False
            visualizer = None
            
    if show and visualizer:
        # 将模型数据添加到可视化器并展示
        model.show(visualizer)
        visualizer.show()

    return model
