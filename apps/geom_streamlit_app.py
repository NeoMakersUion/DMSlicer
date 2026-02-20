# apps/geom_streamlit_app.py

import sys
from pathlib import Path

import streamlit as st

# 让 Python 能找到 src/dmslicer（和你的 test 脚本做法类似）
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from dmslicer.file_parser import read_amf_objects, Model
from dmslicer.geometry_kernel.geom_kernel import GeometryKernel


def load_model(amf_path: Path) -> Model:
    """Load AMF model from path."""
    return read_amf_objects(str(amf_path))


def build_geom_kernel(model: Model, acc: int = 5) -> GeometryKernel:
    """Build GeometryKernel from model."""
    return GeometryKernel(model, acc=acc)


def main() -> None:
    st.set_page_config(page_title="DMSlicer Geometry Explorer", layout="wide")
    st.title("DMSlicer Geometry Explorer")
    st.write("在这里选择 AMF 模型，并基于 GeometryKernel 配置 Objects。")

    # 1. 模型文件选择（复用 data/amf 目录）
    data_dir = ROOT_DIR / "data" / "amf"
    amf_files = sorted(data_dir.glob("*.AMF"))
    if not amf_files:
        st.error(f"在 {data_dir} 下没有找到 .AMF 文件")
        return

    amf_name = st.selectbox(
        "选择 AMF 模型文件 / Select AMF model file",
        [f.name for f in amf_files],
    )
    amf_path = data_dir / amf_name

    # 2. 可调精度参数（对应 GeometryKernel 的 acc）
    acc = st.sidebar.slider("几何精度 acc", min_value=3, max_value=8, value=5, step=1)

    # 3. 构建 GeometryKernel（按下按钮再算，避免每次交互都重建）
    if st.button("构建 GeometryKernel / Build GeometryKernel"):
        with st.spinner(f"正在加载模型 {amf_name} 并构建几何内核..."):
            model = load_model(amf_path)
            geom_kernel = build_geom_kernel(model, acc=acc)
        st.success("GeometryKernel 构建完成。")

        # 使用 session_state 保留内存对象，避免重复构建
        st.session_state["geom_kernel"] = geom_kernel

    geom_kernel: GeometryKernel | None = st.session_state.get("geom_kernel")
    if geom_kernel is None:
        st.info("请先选择模型并点击“构建 GeometryKernel”。")
        return

    geom = geom_kernel.geom

    # 4. Object 级别的设置界面
    st.header("Object 配置 / Object settings")

    object_ids = sorted(geom.objects.keys())
    if not object_ids:
        st.warning("当前模型中没有检测到 Objects。")
        return

    # 多选要操作 / 观察的 Object
    selected_objs = st.multiselect(
        "选择要操作的 Object ID / Select object IDs",
        object_ids,
        default=object_ids[:1],
    )

    st.write("当前可用 Objects：", object_ids)
    st.write("已选择 Objects：", selected_objs)

    # 5. 示例：展示 patch_info 中与选中 Objects 相关的 pair
    st.subheader("Patch-level 信息 / Patch-level information")

    related_pairs = [
        pair for pair in geom.patch_info.keys()
        if any(obj_id in pair for obj_id in selected_objs)
    ]

    if not related_pairs:
        st.info("当前选择的 Objects 没有对应的 patch_info。")
    else:
        st.write("相关对象对 / Related object pairs：", related_pairs)

        # 展示第一个 pair 的 patch 信息作为示例
        pair_to_show = st.selectbox(
            "选择一个对象对查看 patch 信息 / Select a pair to inspect patch",
            related_pairs,
        )
        patch_obj = geom.patch_info[pair_to_show]

        # 显示组件对（我们在 GeometryKernel 里已经把 patch 展平成 [{obj1: comp1, obj2: comp2}, ...]）
        st.write("组件配对列表 / Component pairs：")
        st.json(patch_obj.patch)


if __name__ == "__main__":
    main()