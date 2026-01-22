from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from tqdm import tqdm

try:
    from .config import DEFAULTS
except ImportError:
    # Default configuration values
    DEFAULTS: Dict[str, Any] = {
        "AMF_TARGET_REDUCTION": 0.0,
        "AMF_DECIMATE_MIN_TRIS": 100,
    }
AMF_TARGET_REDUCTION = DEFAULTS["AMF_TARGET_REDUCTION"]
AMF_DECIMATE_MIN_TRIS = DEFAULTS["AMF_DECIMATE_MIN_TRIS"]

from visualizer.visualizer_interface import IVisualizer
from visualizer.pyvista_visualizer import PyVistaVisualizer
from .mesh_data import MeshData

def read_amf_objects(
    uploaded_file: Union[str, Path],
    target_reduction: float = AMF_TARGET_REDUCTION,
    show: bool = False,
    decimate_min_tris: int = AMF_DECIMATE_MIN_TRIS,
    progress: bool = True,
    visualizer: Optional[IVisualizer] = None,
) -> List[MeshData]:
    """
    解析 AMF 文件（Additive Manufacturing File），按 object 输出网格数据。

    Args:
        uploaded_file: AMF 文件路径
        target_reduction: 网格简化比例（0~1）。大网格且 >= decimate_min_tris 时才会执行 decimate
        show: 是否用 PyVista 实时预览每个 object（半透明+边框）
        decimate_min_tris: 仅当三角数 ≥ 该阈值时才对该 object 执行网格简化
        progress: 是否显示 tqdm 进度条
        visualizer: 可选的可视化器实例。如果为 None 且 show=True，默认使用 PyVistaVisualizer。

    Returns:
        List[MeshData]: 包含解析后的网格数据对象的列表。
    """
    try:
        with open(uploaded_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        # 去掉命名空间，简化 XPath 查找
        xml_content = xml_content.replace('amf:', '').replace('ns:', '').replace(':', '')
        root = ET.fromstring(xml_content)
    except FileNotFoundError:
        print(f"Error: File '{uploaded_file}' not found.")
        return []
    except ET.ParseError as e:
        print(f"Error: Failed to parse XML content. Details: {e}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error occurred while reading file. Details: {e}")
        return []

    object_xml = root.findall(".//object")
    result = []
    
    # Initialize visualizer if showing is requested
    if show and visualizer is None:
        try:
            visualizer = PyVistaVisualizer()
        except Exception as e:
            print(f"Warning: Failed to initialize visualizer. Details: {e}")
            show = False

    # 顶层进度：对象个数
    for index, obj in enumerate(tqdm(object_xml, desc="AMF objects", unit="obj", disable=not progress)):
        try:
            # 子进度：顶点/三角（大规模时才开，避免刷新频繁）
            verts_xml = obj.findall(".//vertex")
            tris_xml  = obj.findall(".//triangle")

            # ---- 顶点 ----
            vertices = []
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

            # ---- 三角 ----
            triangles = []
            it_t = tqdm(tris_xml, desc=f"[obj {index}] triangles", unit="tri",
                        leave=False, disable=not progress or len(tris_xml) < 2000)
            for triangle in it_t:
                v1 = int(triangle.find("v1").text.strip())
                v2 = int(triangle.find("v2").text.strip())
                v3 = int(triangle.find("v3").text.strip())
                triangles.append([v1, v2, v3])
            triangles = np.array(triangles, dtype=np.int64)

            # 构造临时 PolyData 做简化/一致化处理
            # 注意：这里仍然使用 pyvista 做网格处理，因为这是 core logic 的一部分 (mesh simplification)
            # 如果要完全解耦 pyvista，需要引入其他几何处理库(如 trimesh)，或者只在需要 simplify 时才 import pyvista
            faces = np.hstack([np.full((len(triangles), 1), 3, dtype=np.int64), triangles]).ravel()
            mesh = pv.PolyData(vertices, faces)

            # 仅在“大网格 & 明确要求”时简化（避免小网格精度损失）
            if target_reduction > 0.0 and triangles.shape[0] >= decimate_min_tris:
                decimated = mesh.decimate(target_reduction)
                vertices = decimated.points
                triangles = decimated.faces.reshape(-1, 4)[:, 1:]
            else:
                # 统一取回 points/faces（PyVista 内部可能做了规范化）
                vertices = mesh.points
                triangles = mesh.faces.reshape(-1, 4)[:, 1:]

            # ---- 颜色 ----
            color_elem = obj.find(".//color"); r, g, b = 0.5, 0.5, 0.5  # 默认中性灰
            if color_elem is not None:
                r = float(color_elem.find("r").text.strip())
                g = float(color_elem.find("g").text.strip())
                b = float(color_elem.find("b").text.strip())

            result.append(MeshData(
                vertices=np.asarray(vertices),
                triangles=np.asarray(triangles, dtype=np.int64),
                color=np.array([r, g, b], dtype=float)
            ))

            # 可视化（如开启）
            if show and visualizer:
                visualizer.add_mesh(
                    mesh=result[-1],
                    opacity=0.5
                )
        except Exception as e:
            # 对单个 object 的错误容忍，继续下一个 object
            print(f"Error: processing object {index}. Details: {e}")
            continue

    if show and visualizer:
        visualizer.show()

    return result

def test_read_amf():
    file_name="merge"
    path=Path(__file__).parent.parent / "amf"/file_name/"source.AMF"
    
    print(path)
    read_amf_objects(uploaded_file=path, show=True)
if __name__ == "__main__":
    test_read_amf()
