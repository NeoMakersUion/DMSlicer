# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from .visualizer_type import VisualizerType
from ..file_parser.mesh_data import MeshData
from typing import Optional, Union, TYPE_CHECKING
from .config import DEFAULT_VISUALIZER,AMF_TARGET_REDUCTION,AMF_DECIMATE_MIN_TRIS

if TYPE_CHECKING:
    from ..geometry_kernel import Object
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
