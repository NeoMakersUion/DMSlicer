# DMSlicer

DMSlicer 是一个用于 3D 增材制造文件（AMF）解析、几何归一化与可视化的工具集。

## 环境准备
- Python >= 3.10
- 建议启用虚拟环境并安装开发依赖（见 requirements-dev.txt）

## 安装步骤
```bash
pip install -r requirements-dev.txt
```

## 快速使用
```python
from pathlib import Path
from dmslicer.file_parser import parse_file, read_amf_objects

model = parse_file(Path("sample.amf"), progress=False, show=False)
# 兼容旧接口
model2 = read_amf_objects(Path("sample.amf"), progress=False, show=False)
print(model.count)
```

## 运行测试
```bash
pytest --maxfail=1 --disable-warnings -q
```

## 数学变换工具
- 位置：`src/dmslicer/math/rotation.py`
- 依赖：`numpy`
- 用法：
```python
import numpy as np
from dmslicer.math import rotate_z_to_vector

R = rotate_z_to_vector([1.0, 1.0, 1.0])
z = np.array([0.0, 0.0, 1.0])
assert np.allclose(R @ z, np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))
```

## 代码质量与静态检查
```bash
black .
isort .
flake8 .
mypy src
```

## MCP 对接说明
- 未来将通过 MCP (Model Context Protocol) 暴露 AMF 解析与几何处理接口
- 提供标准化的上下文检索与可视化操作端点，便于上层代理调用

## 目录结构
- `src/dmslicer/file_parser/amf_parser.py`: AMF 解析入口
- `src/dmslicer/file_parser/model.py`: Model 数据结构与持久化
- `src/dmslicer/file_parser/mesh_data.py`: MeshData 数据类
- `tests/`: 核心功能的单元测试
