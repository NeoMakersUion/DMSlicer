import hashlib
import os
from pathlib import Path
from typing import Union, Optional, Tuple

def sha256_of_file(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # 逐块读取文件，适用于大文件
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_project_root() -> Path:
    """
    获取项目根目录路径。
    
    优先级：
    1. 环境变量 DMSLICER_ROOT
    2. 向上查找 pyproject.toml 文件
    3. 回退到基于当前文件位置的相对路径
    """
    # 1. 检查环境变量
    env_root = os.environ.get("DMSLICER_ROOT")
    if env_root:
        return Path(env_root).resolve()
        
    # 2. 向上查找标志文件 (pyproject.toml)
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
            
    # 3. 回退策略 (原有逻辑)
    # src/dmslicer/file_parser/workspace_utils.py -> .../DMSlicer
    # 注意：如果被安装到 site-packages，这个回退可能不准确，
    # 但如果没有 pyproject.toml 且无环境变量，这是最后的尝试。
    if len(current_path.parents) > 3:
        return current_path.parents[3]
    return current_path.parent # 极其糟糕的情况，防止越界


def get_workspace_dir() -> Path:
    """
    获取 workspace 目录路径。
    优先级：环境变量 DMSLICER_WORKSPACE > 项目根目录/data/workspace
    """
    env_ws = os.environ.get("DMSLICER_WORKSPACE")
    if env_ws:
        return Path(env_ws).resolve()
    
    return get_project_root() / "data" / "workspace"


def find_workspace_entry(
    folder_name: Union[str, Path],
    parent_dir: Optional[Union[str, Path]] = None
) -> Tuple[Optional[Path], bool]:
    """
    检查指定名称的文件夹或文件是否存在于项目的 workspace 目录下（支持递归搜索）。

    所有路径计算均基于 workspace 根目录（默认为 /data/workspace）进行标准化。
    
    Args:
        folder_name (Union[str, Path]): 要搜索的目标名称（文件或文件夹名）。
        parent_dir (Optional[Union[str, Path]]): 搜索的起始目录。
            - 如果为 None，则从 workspace 根目录开始。
            - 如果是绝对路径，必须位于 workspace 内部。
            - 如果是相对路径，则被视为相对于 workspace 根目录的路径。

    Returns:
        Optional[Path]: 如果找到目标，返回相对于 workspace 根目录的相对路径；
                        如果未找到或路径无效，返回 None。
    """
    try:
        workspace_dir = get_workspace_dir().resolve()
        
        # 1. 路径标准化：确定相对于 workspace 的起始搜索路径
        if parent_dir is None:
            rel_start_dir = Path(".")
        else:
            p_dir = Path(parent_dir)
            if p_dir.is_absolute():
                # 如果是绝对路径，尝试转换为相对于 workspace 的路径
                try:
                    # resolve() 处理符号链接，确保比较准确
                    p_dir_resolved = p_dir.resolve()
                    if workspace_dir not in p_dir_resolved.parents and workspace_dir != p_dir_resolved:
                         print(f"Warning: Access denied. Absolute path {p_dir} is outside workspace.")
                         return None
                    rel_start_dir = p_dir_resolved.relative_to(workspace_dir)
                except ValueError:
                    print(f"Warning: Absolute path {p_dir} is not relative to workspace {workspace_dir}")
                    return None
            else:
                # 如果是相对路径，直接视为相对于 workspace
                rel_start_dir = p_dir

        # 2. 构建实际搜索的绝对路径
        # 安全检查：防止相对路径包含 ".." 导致逃逸
        # (workspace_dir / rel_start_dir).resolve() 会解析 ".."
        abs_start_dir = (workspace_dir / rel_start_dir).resolve()

        # 再次验证最终路径是否在 workspace 内
        if workspace_dir not in abs_start_dir.parents and workspace_dir != abs_start_dir:
            print(f"Warning: Access denied. Path {rel_start_dir} resolves outside workspace.")
            return None,False
             
        if not abs_start_dir.exists():
            print(f"Warning: Path {rel_start_dir} does not exist in workspace.")
            return None,False

        # 3. 递归搜索
        target_name_str = str(folder_name)
        
        # 3.1 优先检查直接子项 (快速路径)
        direct_path = abs_start_dir / target_name_str
        if direct_path.exists():
            return direct_path, True
        else:
            return direct_path, False
    except Exception as e:
        print(f"Error: {e}")
        return None, False

# Backward-compatible alias
check_workspace_folder_exists = find_workspace_entry
