from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any, TYPE_CHECKING
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from tqdm import tqdm

if TYPE_CHECKING:
    from ..visualizer.visualizer_interface import IVisualizer
    
from .mesh_data import MeshData
from .model import Model
from ..visualizer.visualizer_type import VisualizerType
from .workspace_utils import sha256_of_file, check_workspace_folder_exists


def read_amf_objects(
    uploaded_file: Union[str, Path],
    progress: bool = True,
    show: bool = False,
    visualizer_type: Optional[VisualizerType] = None,
    **kwargs
) -> Model:
    """
    解析 AMF 文件（Additive Manufacturing File）并构建 3D 模型对象。
    ... (保持原有文档字符串不变) ...
    """
    try:
        from ..visualizer import IVisualizer
    except ImportError:
        pass

    try:
        # 计算文件哈希并检查缓存是否存在
        sha256_hash = sha256_of_file(uploaded_file)
        _,cached_flag= check_workspace_folder_exists(sha256_hash)

        
        if cached_flag:
            print(f"Info: Cache found in workspace for hash {sha256_hash}")
            cached_model = Model.load(sha256_hash)
            if cached_model:
                print(f"Info: Successfully loaded model from cache.")
                # 如果请求显示 (show=True)，则初始化并运行可视化器
                if show:
                    try:
                        visualizer = IVisualizer.create(visualizer_type)
                        if visualizer:
                            cached_model.show(visualizer)
                            visualizer.show()
                    except Exception as e:
                        print(f"Warning: Failed to initialize visualizer for cached model. Details: {e}")
                return cached_model
            else:
                print(f"Warning: Failed to load model from cache despite folder existence. Reparsing...")
        
        model = Model()
        model.hash_id = sha256_hash
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

    # Save to cache
    model.save()

    # ---- 可视化处理 ----
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
