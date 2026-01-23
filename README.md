# DMSlicer
```bash
AMF
  ↓
file_parser → Model
  ↓
geometry_kernel
  → unified vertices
  → face adjacency
  → object adjacency
  → contact surfaces
  ↓
build_prep
  → assign materials
  → detect boundaries
  → infer gradient zones
  → mark support / glue / embed
  ↓
slicer


```

```bash
DMSLICER/
│
├── src/                    # 你的代码（你现在已经有了）
│   ├── FILE_PARSER/
│   ├── PART/
│   ├── VISUALIZER/
│   └── slicer_pipeline.py
│
├── data/
│   ├── amf/                # 原始输入
│   │   ├── cube.amf
│   │   ├── ball.amf
│   │   └── beam.amf
│   │
│   ├── workspace/          # 每一个模型一个工作空间
│   │   ├── cube_multi_separate/
│   │   │   ├── source.amf
│   │   │   ├── hash.json
│   │   │   └── cache/
│   │   │       ├── stage1_geom.npz
│   │   │       ├── stage2_interfaces.json
│   │   │       ├── stage3_loops.npz
│   │   │       ├── stage4_gradients.npz
│   │   │       └── stage5_toolpath.json
│   │   │
│   │   └── ball/
│   │       ├── source.amf
│   │       └── cache/...
│   │
│   └── temp/               # 临时文件
│
└── config/
    └── pipeline.yaml       # 切片参数
```
## 导入文件`file_parser`包
核心是按照amf文件解析来解析导入文件的。导入的文件会经由`file_parser`包中的`parse_file`函数解析为`amf`文件。最终的`amf`文件会被`file_parser`包中的`amf_parser.py`解析成一个`MeshData`对象。
