import os
from io import StringIO

import panel as pn
import pyvista as pv
import streamlit as st
import streamlit.components.v1 as components

pn.extension("vtk", sizing_mode="stretch_both")


def ensure_xvfb():
    """
    在非 Windows 环境下，确保为 PyVista 启动虚拟帧缓冲（Xvfb），以支持无界面渲染。
    """
    if os.name != "nt":
        if "IS_XVFB_RUNNING" not in st.session_state:
            if hasattr(pv, "start_xvfb"):
                pv.start_xvfb()
                st.session_state["IS_XVFB_RUNNING"] = True


def panel_html(plotter, use_container_width: bool = False) -> str:
    """
    将 PyVista 的 Plotter 对象渲染为 HTML 字符串，以便在 Streamlit 中嵌入显示。

    参数:
        plotter: PyVista 的 Plotter 实例，包含 3D 场景。
        use_container_width: 是否使用容器宽度。若为 True，则宽度自适应；否则使用 plotter 原始宽度。

    返回:
        str: 包含完整 HTML 的字符串，可直接嵌入前端展示。
    """
    # 获取 plotter 当前窗口尺寸
    width, height = plotter.window_size
    # 构造 Panel VTK 面板的后端参数
    backend_kwargs = {"height": height}
    if not use_container_width:
        backend_kwargs["width"] = width
    # 创建 Panel 的 VTK 面板，绑定渲染窗口
    vtk_pane = pn.pane.VTK(plotter.ren_win, **backend_kwargs)
    # 将 VTK 面板导出为 HTML 字符串
    with StringIO() as model_bytes:
        vtk_pane.save(model_bytes, title="PyVista Panel")
        html_plotter = model_bytes.getvalue()
    return html_plotter


def create_plotter(geometry, size, height, color, opacity, position):
    """
    根据当前控件参数构造并返回一个 PyVista Plotter 场景。
    """
    plotter = pv.Plotter(window_size=[700, 700])

    if geometry == "Sphere":
        mesh = pv.Sphere(radius=size, center=position)
    elif geometry == "Cube":
        mesh = pv.Cube(x_length=size, y_length=size, z_length=size, center=position)
    else:
        mesh = pv.Cone(radius=size * 0.6, height=height, center=position, direction=(0, 1, 0))

    plotter.add_mesh(mesh, color=color, opacity=opacity)
    plotter.set_background("white")
    plotter.camera_position = "iso"
    return plotter


def main():
    """
    Streamlit 应用入口，负责布局控件区和 3D 视图区，并完成交互绑定。
    """
    ensure_xvfb()

    st.set_page_config(page_title="PyVista + stpyvista Playground", layout="wide")

    st.title("PyVista + stpyvista 3D Playground")
    st.markdown("Interactive test page for PyVista, stpyvista and Streamlit integration.")

    controls, viewport = st.columns([1, 2])

    with controls:
        st.subheader("Controls")

        geometry = st.selectbox("Geometry", ["Sphere", "Cube", "Cone"], index=0)

        size = st.slider("Size", min_value=0.1, max_value=2.0, value=0.8, step=0.1)

        height = st.slider("Cone height", min_value=0.2, max_value=3.0, value=1.5, step=0.1)

        color = st.color_picker("Color", value="#1f77b4")

        opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

        x = st.slider("Position X", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        y = st.slider("Position Y", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        z = st.slider("Position Z", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

        reset_view = st.button("Reset view")

        st.markdown(
            f"Current state: geometry={geometry}, size={size:.2f}, height={height:.2f}, "
            f"opacity={opacity:.2f}, position=({x:.2f}, {y:.2f}, {z:.2f}), color={color}"
        )

    with viewport:
        st.subheader("3D View")

        position = (x, y, z)
        plotter = create_plotter(geometry, size, height, color, opacity, position)

        st.session_state["last_plotter_bounds"] = plotter.bounds

        if reset_view:
            plotter.camera_position = "iso"

        html = panel_html(plotter, use_container_width=True)
        # 将生成的 HTML 字符串嵌入到 Streamlit 页面中，高度固定为 720 像素，禁用滚动条
        components.html(html, height=720, scrolling=False)


if __name__ == "__main__":
    main()

