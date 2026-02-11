# 精度依赖分析报告

本目录包含对 `src/dmslicer/default_config.py` 配置的递归依赖分析产物：

- dependency_graph.mmd：Mermaid 依赖图（节点为文件，边为引用关系；红色边为精度相关）
- report.csv：变量级引用清单与风险摘要

运行与复现：
- 建议安装 ripgrep（rg）用于快速代码扫描
- 关键命令示例：
  - `rg -n "from ..default_config import DEFAULTS" -g \"**/*.py\"`
  - `rg -n "\\b(GEOM_ACC|GEOM_PARALLEL_ACC|SOFT_NORMAL_GATE_ANGLE|INITIAL_GAP_FACTOR|MAX_GAP_FACTOR|OVERLAP_RATIO_THRESHOLD)\\b" -g \"**/*.py\"`
- 扫描范围：项目根目录（默认 Windows 路径 `D:\\DMSlicer`）
- 性能：rg 对万行代码扫描通常 < 1s，内存峰值远低于 300MB

解读建议：
- 若需统一调整计算精度，建议集中到单一配置源（如 `default_config.py` 或 `tolerance.yaml`），并通过 `geometry_kernel/config.py` 暴露常量
- 配置热加载可采用 `pydantic.BaseSettings`，并配合单元测试快照确保回归一致性
- 变更 GEOM_ACC 时需同时评估整数化比例（10^GEOM_ACC）、EPSILON 派生阈值与缓存键变化
