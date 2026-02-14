# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from .visualizer_type import VisualizerType
from ..file_parser.mesh_data import MeshData
from typing import Optional, Union, TYPE_CHECKING, List, Tuple, Dict, Any
from .config import DEFAULT_VISUALIZER

if TYPE_CHECKING:
    from ..geometry_kernel import Object, Triangle
else:
    # Runtime placeholder or deferred import could be used if strictly needed,
    # but for Union checks `Object` must be defined.
    # If Object is only used for type hinting, we can use string forward reference or conditional import.
    # However, if isinstance() is used, we need the actual class.
    # Let's check if isinstance is used in subclasses. IVisualizer is abstract.
    pass

class IVisualizer(ABC):
    """
    Abstract Base Class for Visualization.
    Allows decoupling the parser from specific visualization libraries (e.g., PyVista).
    """
 
    def __init__(self, **kwargs):
        """
        Initialize the visualizer.
        
        Args:
            **kwargs: Arguments passed to the visualizer constructor.
        """
        pass

    @abstractmethod
    def addObj(
        self,
        object: "Object",
        include_triangles_ids: Optional[List[int]] = None,
        exclude_triangles_ids: Optional[List[int]] = None,
        opacity: float = 0.5,
        color: Optional[Any] = None
    ):
        """
        Add a specific Object to the visualizer, optionally filtering triangles.

        Args:
            object: The Object instance to visualize.
            include_triangles_ids: List of triangle IDs to include.
            exclude_triangles_ids: List of triangle IDs to exclude.
            opacity: Opacity value (0.0-1.0).
            color: Color of the object.
        """
        pass

    @abstractmethod
    def addTriangle(
        self,
        triangle: "Triangle",
        opacity: float = 0.5,
        color: Optional[Any] = None
    ):
        """
        Add a single Triangle to the visualizer.

        Args:
            triangle: The Triangle instance to visualize.
            opacity: Opacity value (0.0-1.0).
            color: Color of the triangle.
        """
        pass

    @abstractmethod
    def addContactTriangles(
        self,
        objects: Dict[int, "Object"],
        obj_pair: Tuple[int, int],
        tris_ids_pairs: List[Tuple[int, int]],
        select: Optional[Union[int, List[int]]] = None,
        opacity: float = 0.5
    ):
        """
        Add contact triangles between two objects to the visualizer.

        Args:
            objects: Dictionary of all objects.
            obj_pair: Tuple of (obj1_id, obj2_id).
            tris_ids_pairs: List of intersecting triangle ID pairs.
            select: Optional ID(s) to filter which object's contact triangles to show.
            opacity: Opacity value (0.0-1.0).
        """
        pass

    @abstractmethod
    def add(
        self,
        obj:Union[MeshData,"Object",np.ndarray],
        opacity: float = 1.0,
        **kwargs
    ):
        """
        Add a mesh object to the visualizer scene.

        Args:
            mesh: MeshData object containing vertices, triangles, and color.
            opacity: Opacity value (0.0-1.0).
            **kwargs: Additional renderer-specific arguments.
        """
        pass

    @abstractmethod
    def show(self, **kwargs):
        """
        Render the scene and display the window.
        
        Args:
            **kwargs: Additional show options (e.g., interactive, window_size).
        """
        pass

    @abstractmethod
    def save(self, file_path: Optional[str] = None, **kwargs):
        """
        Render the scene and save to a file.
        
        Args:
            file_path: Target file path (e.g., 'output.png'). 
                       If None, a default filename with timestamp will be generated in a default directory.
            **kwargs: Additional options (e.g., window_size, transparent_background).
        """
        pass




    @staticmethod
    def create(visualizer_type: Optional[VisualizerType] = DEFAULT_VISUALIZER, **kwargs) -> "IVisualizer":
        """
        Factory method to create a visualizer instance based on the specified type.

        Args:
            visualizer_type: Type of the visualizer to initialize.
            **kwargs: Arguments passed to the visualizer constructor.
        
        Returns:
            An instance of a concrete IVisualizer implementation.
        """
        if visualizer_type is None:
            visualizer_type = DEFAULT_VISUALIZER
            
        if visualizer_type == VisualizerType.PyVista:
            from .pyvista_visualizer import PyVistaVisualizer
            return PyVistaVisualizer(**kwargs)
        else:
            raise ValueError(f"Unsupported visualizer type: {visualizer_type}")
