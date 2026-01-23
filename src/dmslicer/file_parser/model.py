from dataclasses import dataclass, field
from typing import List
from .mesh_data import MeshData
from ..visualizer.visualizer_interface import IVisualizer

@dataclass
class Model:
    """
    Data class representing a collection of 3D mesh objects.
    
    Attributes:
        meshes: List of MeshData objects.
    """
    meshes: List[MeshData] = field(default_factory=list)
    
    # 移除 explicit count 字段，避免状态不同步
    # count: int = field(default_factory=lambda: 0)

    @property
    def count(self) -> int:
        """Dynamic property returning the number of meshes."""
        return len(self.meshes)

    def add_mesh(self, mesh: MeshData):
        """Add a MeshData object to the model."""
        self.meshes.append(mesh)
        # 不需要手动维护 count
    
    def dump(self):
        """Dump the model to a dictionary."""
        # 注意：这里需要处理 numpy 数组的序列化，或者仅 dump 结构
        # 为了保持与原逻辑一致，我们仅 dump __dict__，但在实际应用中可能需要更复杂的序列化
        return {
            "meshes": [
                {k: v for k, v in mesh.__dict__.items() if k != 'count'} 
                for mesh in self.meshes
            ],
            "count": self.count
        }

    def load(self, data: dict):
        """Load the model from a dictionary."""
        # 清空当前 mesh
        self.meshes = []
        
        for mesh_data in data.get("meshes", []):
            # 过滤掉 id，因为 MeshData 重建时会自动生成新 ID
            # 或者如果需要保留原 ID，则需要修改 MeshData 允许传入 id
            # 这里假设重新生成 ID 是可接受的，或者数据中包含 vertices/triangles/color
            
            # 构造参数过滤 (排除 id，因为它现在是 init=False)
            init_args = {
                k: v for k, v in mesh_data.items() 
                if k in ['vertices', 'triangles', 'color']
            }
            new_mesh = MeshData(**init_args)
            self.meshes.append(new_mesh)
        
        # 恢复类计数器状态 (可选，取决于业务需求)
        # MeshData.count = data.get("count", 0) 
    def show(self, visualizer: IVisualizer):
        """Show the model using the provided visualizer."""
        for mesh in self.meshes:
            visualizer.add_mesh(
                mesh=mesh,
                opacity=0.5
            )
