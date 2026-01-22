# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np

from file_parser.mesh_data import MeshData

class IVisualizer(ABC):
    """
    Abstract Base Class for Visualization.
    Allows decoupling the parser from specific visualization libraries (e.g., PyVista).
    """
 

    @abstractmethod
    def add_mesh(
        self,
        object,
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
