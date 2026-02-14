# -*- coding: utf-8 -*-
"""
Slicer Orchestrator
===================
Main entry point for the slicing pipeline.
Coordinating GeometryKernel -> ContourOps -> PathPlanner.
"""

import logging
from typing import List

# Placeholder imports for components not yet fully implemented
from .geometry_kernel.geom_kernel import GeometryKernel
from .geometry_kernel.contour_ops import LoopNormalizer
# from .path_planner import PathPlanner 

class Slicer:
    """
    Slicer Orchestrator.
    
    Attributes:
        geom (GeometryKernel): The geometry backend.
        layer_height (float): Slice layer height in mm.
    """
    
    def __init__(self, geom: GeometryKernel):
        self.geom = geom
        self.logger = logging.getLogger("Slicer")
        
    def slice_model(self, layer_height: float = 0.2):
        """
        Execute full slicing pipeline.
        
        Args:
            layer_height: Layer height in mm.
        """
        self.logger.info(f"Starting slicing with layer_height={layer_height}mm")
        
        # 1. Determine Z bounds
        # TODO: Get Z range from geom.spatial or geom.bbox
        z_min = 0.0
        z_max = 100.0 # Placeholder
        
        current_z = z_min + layer_height / 2.0 # First layer
        
        layers = []
        
        while current_z < z_max:
            layer_loops = self.slice_layer(current_z)
            layers.append(layer_loops)
            current_z += layer_height
            
        self.logger.info(f"Slicing complete. Generated {len(layers)} layers.")
        return layers

    def slice_layer(self, z: float) -> List:
        """
        Process a single layer.
        
        Returns:
            List of normalized loops (List[np.ndarray]).
        """
        # 1. Query Triangles
        tri_ids = self.geom.query_triangles_by_plane(z)
        if not tri_ids:
            return []
            
        # 2. Intersect & Generate Segments
        segments = []
        for tid in tri_ids:
            res = self.geom.intersect_triangle_with_plane(tid, z)
            if res:
                # intersect_triangle_with_plane returns list of segments or single segment
                # Assuming it returns a list of tuples (p1, p2)
                if isinstance(res, list):
                    segments.extend(res)
                else:
                    segments.append(res)
        
        # 3. Contour Operations (Normalize & Dedup)
        # Convert float segments -> Normalized Loops
        loops = LoopNormalizer.process(segments)
        
        # 4. (Future) Path Planning
        # paths = PathPlanner.plan(loops)
        
        return loops

# Directory Structure Suggestion:
# src/dmslicer/
#   slicer.py           <-- This file
#   geometry_kernel/
#     geom_kernel.py
#     contour_ops.py
#     ...
#   path_planner/       <-- Future module
#     __init__.py
#     planner.py
