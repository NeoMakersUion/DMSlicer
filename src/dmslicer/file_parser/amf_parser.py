"""AMF Parser
解析 AMF (Additive Manufacturing File) 文件为内部 Model 数据结构，并提供
缓存与可视化集成的入口函数。

主要接口:
- read_amf_objects: 读取 AMF 文件并返回 Model

依赖:
- 标准库: pathlib, typing, xml.etree.ElementTree
- 第三方: numpy, tqdm
- 本地模块: MeshData, Model, VisualizerType, workspace_utils

作者: QilongJiang
创建日期: 2026-02-14
"""

from pathlib import Path
from typing import Any, Optional, Union
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

from .mesh_data import MeshData
from .model import Model
from .workspace_utils import find_workspace_entry, sha256_of_file
from ..visualizer.visualizer_type import VisualizerType


def _read_vertices(obj_xml: ET.Element, progress: bool, index: int) -> np.ndarray:
    """Parse vertex list from an AMF object XML node.

    Args:
        obj_xml: The XML element representing an AMF object.
        progress: Whether to show progress bar for large inputs.
        index: Object index for progress bar labeling.

    Returns:
        A float32 numpy array of shape (N, 3) representing vertex coordinates.
    """
    verts_xml = obj_xml.findall(".//vertex")
    vertices: list[list[float]] = []
    it_v = tqdm(
        verts_xml,
        desc=f"[obj {index}] vertices",
        unit="v",
        leave=False,
        disable=not progress or len(verts_xml) < 2000,
    )
    for vertex in it_v:
        coords = vertex.find("coordinates")
        if coords is None:
            continue
        x = float(coords.find("x").text.strip())  # type: ignore[union-attr]
        y = float(coords.find("y").text.strip())  # type: ignore[union-attr]
        z = float(coords.find("z").text.strip())  # type: ignore[union-attr]
        vertices.append([x, y, z])
    return np.asarray(vertices, dtype=np.float32)


def _read_triangles(obj_xml: ET.Element, progress: bool, index: int) -> np.ndarray:
    """Parse triangle index list from an AMF object XML node.

    Args:
        obj_xml: The XML element representing an AMF object.
        progress: Whether to show progress bar for large inputs.
        index: Object index for progress bar labeling.

    Returns:
        An int64 numpy array of shape (M, 3) representing triangle indices.
    """
    tris_xml = obj_xml.findall(".//triangle")
    triangles: list[list[int]] = []
    it_t = tqdm(
        tris_xml,
        desc=f"[obj {index}] triangles",
        unit="tri",
        leave=False,
        disable=not progress or len(tris_xml) < 2000,
    )
    for triangle in it_t:
        v1 = int(triangle.find("v1").text.strip())  # type: ignore[union-attr]
        v2 = int(triangle.find("v2").text.strip())  # type: ignore[union-attr]
        v3 = int(triangle.find("v3").text.strip())  # type: ignore[union-attr]
        triangles.append([v1, v2, v3])
    return np.asarray(triangles, dtype=np.int64)


def _read_color(obj_xml: ET.Element) -> np.ndarray:
    """Parse optional RGB color from an AMF object XML node.

    Args:
        obj_xml: The XML element representing an AMF object.

    Returns:
        A float numpy array of shape (3,) representing RGB values in [0, 1].
    """
    color_elem = obj_xml.find(".//color")
    r, g, b = 0.5, 0.5, 0.5
    if color_elem is not None:
        r = float(color_elem.find("r").text.strip())  # type: ignore[union-attr]
        g = float(color_elem.find("g").text.strip())  # type: ignore[union-attr]
        b = float(color_elem.find("b").text.strip())  # type: ignore[union-attr]
    return np.asarray([r, g, b], dtype=float)


def read_amf_objects(
    uploaded_file: Union[str, Path],
    progress: bool = True,
    show: bool = False,
    visualizer_type: Optional[VisualizerType] = None,
    **kwargs: Any,
) -> Model:
    """Read an AMF file and construct a Model with mesh objects.

    功能:
        - 解析 AMF XML 内容，提取顶点、三角形与颜色
        - 基于文件哈希启用工作区缓存（优先加载）
        - 可选集成可视化器进行展示

    Args:
        uploaded_file: Path to AMF file (str or Path).
        progress: Whether to show progress bars during parsing.
        show: Whether to visualize after parsing/load.
        visualizer_type: Visualizer type for IVisualizer factory.
        **kwargs: Reserved for future options (unused).

    Returns:
        Model: Parsed or loaded model. Returns empty Model on error.

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the XML cannot be parsed.
        Exception: For unexpected runtime errors.

    Examples:
        >>> from pathlib import Path
        >>> from dmslicer.file_parser.amf_parser import read_amf_objects
        >>> model = read_amf_objects(Path(\"sample.amf\"), progress=False, show=False)
        >>> assert model.count >= 0
    """
    try:
        from ..visualizer import IVisualizer as _IVisualizer  # runtime factory
    except Exception:
        _IVisualizer = None

    try:
        sha256_hash = sha256_of_file(uploaded_file)
        _, cached_flag = find_workspace_entry(sha256_hash)

        if cached_flag:
            cached_model = Model.load(sha256_hash)
            if cached_model:
                if show and _IVisualizer is not None:
                    try:
                        vis_cached = _IVisualizer.create(visualizer_type)  # type: ignore[attr-defined]
                        if vis_cached:
                            cached_model.show(vis_cached)
                            vis_cached.show()
                    except Exception:
                        pass
                return cached_model

        model = Model()
        model.hash_id = sha256_hash
        with open(uploaded_file, "r", encoding="utf-8") as f:
            xml_content = f.read()
        xml_content = xml_content.replace("amf:", "").replace("ns:", "").replace(":", "")
        root = ET.fromstring(xml_content)
    except FileNotFoundError:
        return Model()
    except ET.ParseError:
        return Model()
    except Exception:
        return Model()

    objects_xml = root.findall(".//object")
    for index, obj_xml in enumerate(
        tqdm(objects_xml, desc="AMF objects", unit="obj", disable=not progress)
    ):
        try:
            vertices = _read_vertices(obj_xml, progress, index)
            triangles = _read_triangles(obj_xml, progress, index)
            color = _read_color(obj_xml)

            mesh = MeshData(vertices=vertices, triangles=triangles, color=color)
            model.add_mesh(mesh)
        except Exception:
            continue

    model.save()

    vis: Optional[Any] = None
    if show and _IVisualizer is not None:
        try:
            vis = _IVisualizer.create(visualizer_type)  # type: ignore[attr-defined]
        except Exception:
            vis = None
    if show and vis:
        model.show(vis)
        vis.show()

    return model
