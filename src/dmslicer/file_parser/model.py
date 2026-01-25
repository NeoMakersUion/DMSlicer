import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union, ClassVar, TYPE_CHECKING
from .mesh_data import MeshData
from .workspace_utils import get_workspace_dir
if TYPE_CHECKING:
    from ..visualizer.visualizer_interface import IVisualizer

@dataclass
class Model:
    """
    Data class representing a collection of 3D mesh objects.
    
    Attributes:
        meshes: List of MeshData objects.
    """
    meshes: List[MeshData] = field(default_factory=list)
    hash_id: str = field(default_factory=str)
    
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

    def load_from_dict(self, data: dict):
        """
        Load the model from a dictionary.
        Note: Renamed from 'load' to avoid conflict with persistence load method.
        """
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

    def save(self) -> bool:
        """
        Save the model instance to disk using pickle serialization.
        
        The model is saved to: data/workspace/{hash_id}/{hash_id}_model.pkl
        
        Returns:
            bool: True if save successful, False otherwise.
        """
        if not self.hash_id:
            print("Error: Model hash_id is not set. Cannot save.")
            return False
            
        try:
            ws_dir = get_workspace_dir()
            save_dir = ws_dir / self.hash_id
            
            # Ensure directory exists
            save_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = save_dir / f"{self.hash_id}_model.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
                
            print(f"Info: Model saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error: Failed to save model: {e}")
            return False

    @classmethod
    def load(cls, hash_id: str) -> Optional['Model']:
        """
        Load a model instance from disk using its hash_id.
        
        Args:
            hash_id: The unique identifier for the model.
            
        Returns:
            Optional[Model]: The loaded Model instance, or None if not found/error.
        """
        try:
            ws_dir = get_workspace_dir()
            file_path = ws_dir / hash_id / f"{hash_id}_model.pkl"
            
            if not file_path.exists():
                # print(f"Warning: Model file not found at {file_path}")
                return None
                
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            
            if not isinstance(model, cls):
                print(f"Error: Loaded object is not an instance of {cls.__name__}")
                return None
            
            # Verify hash_id consistency (optional)
            if model.hash_id != hash_id:
                print(f"Warning: Loaded model hash_id ({model.hash_id}) does not match requested ({hash_id})")
                model.hash_id = hash_id # Auto-correct?

            print(f"Info: Model loaded Successfully!")
            return model
        except Exception as e:
            print(f"Error: Failed to load model: {e}")
            return None

    def show(self, visualizer: "IVisualizer"):
        """Show the model using the provided visualizer."""
        for mesh in self.meshes:
            visualizer.add(
                obj=mesh,
                opacity=0.5
            )
