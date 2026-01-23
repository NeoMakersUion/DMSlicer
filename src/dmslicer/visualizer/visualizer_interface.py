# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from .visualizer_type import VisualizerType
from ..file_parser.mesh_data import MeshData
from typing import Optional
from .config import DEFAULT_VISUALIZER,AMF_TARGET_REDUCTION,AMF_DECIMATE_MIN_TRIS
    
class IVisualizer(ABC):
    """
    Abstract Base Class for Visualization.
    Allows decoupling the parser from specific visualization libraries (e.g., PyVista).
    """
 

    @abstractmethod
    def add_mesh(
        self,
        mesh:MeshData,
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
            
        if visualizer_type == VisualizerType.PyVistaVisualizer:
            from .pyvista_visualizer import PyVistaVisualizer
            return PyVistaVisualizer(**kwargs)
        else:
            raise ValueError(f"Unsupported visualizer type: {visualizer_type}")
