import os
import sys
import streamlit as st
import panel as pn
import nest_asyncio

# 获取当前脚本所在目录 (d:\DMSlicer\apps)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (d:\DMSlicer)
project_root = os.path.dirname(current_dir)
# 获取 src 目录 (d:\DMSlicer\src)
src_dir = os.path.join(project_root, "src")

# 将项目根目录和 src 目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
if src_dir not in sys.path:
    sys.path.append(src_dir)


# 导入拆分的模块
# 注意：由于我们将 project_root 添加到了 sys.path，我们可以使用绝对导入
from apps.modules.state import init_session_state, get_global_resources
from apps.modules.utils import ensure_xvfb
from apps.modules.sidebar import render_sidebar
from apps.modules.controls import Materialize
from apps.modules.viewport import render_viewport

# 解决 Streamlit 的异步循环冲突
nest_asyncio.apply()
pn.extension("vtk", sizing_mode="stretch_both")

def main():
    """
    Streamlit 应用入口，负责布局控件区和 3D 视图区，并完成交互绑定。
    """
    ensure_xvfb()

    st.set_page_config(page_title="DMSlicer", layout="wide")
    
    # 确保全局资源初始化
    get_global_resources()
    init_session_state()

    st.title("DMSlicer")

    # 渲染侧边栏
    with st.sidebar:
        render_sidebar()
    try:
        file_name = st.session_state.uploaded_file.name
        st.markdown(f"**Uploaded File:** {file_name}")
    except:
        st.markdown("DMSlicer is a tool for slicing 3D models.")
    # 布局主体区域
    controls, viewport = st.columns([1, 2])

    with controls:
        Materialize()
        
    
    with viewport:
        render_viewport()

if __name__ == "__main__":
    main()
