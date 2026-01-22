# AMF 示例说明

- 示例说明文档
  - 位置：<mcfile name="ReadMe.md" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\amf\ReadMe.md"></mcfile>
  - 作用：用样例解释 AMF 层如何组织数据与缓存，帮助你快速理解“AMF 解析 → Part 构建 → Slicer 切片”的整体流程。
  - 对应架构要点：
    - AMF 层只负责解析与缓存，输出 DTO（如顶点、三角形、block_dict 等）供 Part 层使用。
    - 下游 Part/Slicer 不直接依赖 AMF 源文件，而是通过统一接口与数据类交互。

- 球体样例数据（用于“球体包含/嵌套”关系验证）
  - 位置：<mcfolder name="ball" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\amf\ball\"></mcfolder>
  - 内容与产物：
    - source.AMF：原始 AMF 模型（如同心球壳等结构）。
    - cache\stage1_geom.npz：AMF 解析后的几何数据缓存（顶点/三角形/对象映射等）。
    - cache\stage2_interfaces.json：对象间界面候选与精判结果（pairs_by_objpair、pair_for_focus 等）。
  - 用途：验证“近共面但分离”的误判是否被正确消除，特别适用于 concentric shells 的干涉检测与阈值调校（如 eps、法线平行阈值）。

- 多立方体分离样例（用于“非标准包含/部分重叠/相邻边界为空/内部空腔”等复杂关系验证）
  - 位置：<mcfolder name="cube_multi_separate" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\amf\cube_multi_separate\"></mcfolder>
  - 内容与产物：
    - source.AMF：复杂相邻或部分重叠结构的 AMF 模型。
    - cache\stage1_geom.npz / cache\stage2_interfaces.json / cache\stage3_slices.json：分别对应“几何解析/界面检测/切片与轮廓输出”的各阶段缓存。
  - 用途：覆盖更多边界情形，如非标准化相邻、内部空腔等，帮助验证 BVH 加速与几何精判在复杂结构中的稳定性。

如何在当前项目中使用这些样例进行验证
- 驱动脚本
  - 位置：<mcfile name="main_neo.py" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\main_neo.py"></mcfile>
  - 你可以在这里选择要运行的 AMF 样例、配置 eps 与 keep_all，并触发“AMF 解析 → 界面检测 → 切片与导出”的流水线。
- 核心数据结构与检测逻辑
  - 位置：<mcfile name="Part.py" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\part\Part.py"></mcfile>
  - 说明：
    - 我已将法线近似平行判定阈值从硬编码改为实例属性 normal_parallel_cos_tol（默认 1e-2），便于你按模型尺度调节。
    - BVH 加速构建位于 BVH 模块：<mcfile name="bvh.py" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\BVH\bvh.py"></mcfile>，用于快速筛选三角候选对，减少精判开销。
- 可视化调试工具
  - 位置：<mcfile name="visual_debug.py" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\debug_tools\visual_debug.py"></mcfile>
  - 导出入口：<mcfile name="__init__.py" path="d:\git_or_gitee\pySliceBot\dmslicer\app_trae\debug_tools\__init__.py"></mcfile>
  - 说明：你可以直接导入并调用 debug_show_triangles(part, tri_ids, …) 来在 PyVista 中高亮指定三角形，辅助定位与验证界面候选与误判。

参数调校建议（针对上述两类样例）
- 球体包含/同心壳结构
  - eps：建议用小值作为“最终几何距离闸门”，例如 1e-4 到 1e-6（依模型对角线缩放），确保只有“真正接触”才判命中。
  - normal_parallel_cos_tol（Part 实例属性）：建议更严格一些，如 1e-3 或 5e-4，避免近似平行漏判或噪声。
  - keep_all：若你希望保留全部候选对以便后续过滤与审查，可开启；若只关心最终命中的界面对，可关闭以减少输出。
- 非标准包含/部分重叠/空腔结构
  - eps：可略放宽（如 1e-3 左右），以更稳健地覆盖边界重合与接触面带厚度的情况。
  - normal_parallel_cos_tol：根据结构复杂度与网格质量微调（1e-2 到 1e-3），兼顾近似共面筛选与复杂接触角度。
  - 建议开启基础网格显示与边线高亮（在 debug_show_triangles 中可配置），便于定位重叠区域与空腔出口。

与项目架构的契合点（确保后续扩展一致性）
- AMF 层只负责解析与缓存，不直接服务 Slicer；数据通过 Part 层传递。
- Part 层提供统一索引与几何变换接口，内部调用 BVH 加速，但不直接依赖更高层。
- Slicer 层只依赖 Part 与 geobase，必要时调用 BVH；输出标准化的每层 ContoursDTO/GradientContoursDTO，并写入 JSON。
- 阈值与判定参数尽量通过实例属性或 DTO 配置，避免跨层全局状态。

如果你愿意，我可以：
- 在 main_neo.py 中添加针对 ball 与 cube_multi_separate 的“参数预设切换”，便于一键切换样例并跑通全流程。
- 扩展 debug_show_triangles 支持叠加界面候选对的自动高亮（例如根据 stage2_interfaces.json 自动高亮 tri 对），提升定位效率。

你更倾向于先在球体样例上调校 eps 与 normal_parallel_cos_tol，还是先在 cube_multi_separate 上检查非标准包含的输出 JSON（stage2/3）？我可以按你的选择设置好参数并指导你跑一遍验证。
        