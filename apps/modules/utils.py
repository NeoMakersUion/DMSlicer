import os
import sys
import contextlib
from io import StringIO
import streamlit as st
import pyvista as pv
import panel as pn

@contextlib.contextmanager
def StreamlitLogCapture(placeholder):
    """
    捕获 stdout/stderr 并实时更新到 Streamlit 的 placeholder 中。
    """
    log_buffer = StringIO()
    
    class MultiWriter:
        def __init__(self, original, buffer):
            self.original = original
            self.buffer = buffer
            
        def write(self, text):
            self.original.write(text)
            self.buffer.write(text)
            # 实时更新 UI
            placeholder.code(self.buffer.getvalue(), language="text")
            
        def flush(self):
            self.original.flush()
            
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = MultiWriter(original_stdout, log_buffer)
    sys.stderr = MultiWriter(original_stderr, log_buffer)
    
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


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
    # 创建 Panel 的 VTK 面板，绑定渲染窗口，并启用交互
    vtk_pane = pn.pane.VTK(plotter.ren_win, **backend_kwargs, enable_keybindings=True)
    
    # 强制重新渲染，确保背景色生效
    plotter.render()
    
    # 将 VTK 面板导出为 HTML 字符串
    with StringIO() as model_bytes:
        vtk_pane.save(model_bytes, title="PyVista Panel")
        html_plotter = model_bytes.getvalue()
    return html_plotter
