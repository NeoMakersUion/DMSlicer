import time
import streamlit as st
import panel as pn
import streamlit.components.v1 as components
from copy import deepcopy
from io import StringIO
from dmslicer.geometry_kernel.geom_kernel import GeometryKernel
from dmslicer.visualizer.visualizer_interface import IVisualizer
from dmslicer.file_parser import Model

def render_viewport():
    """
    渲染 3D 视图区域，包括 PyVista 绘图和 HTML 缓存机制。
    """
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.subheader("3D View")

    with col_btn:
        force_refresh = st.button("Force Refresh")

    # 初始化 Session State (虽然在 state.py 中有，但这里作为局部保险也可以保留，或者移除)
    # 此处假设 state.py 已经正确初始化，直接使用 st.session_state
    try:
        st.session_state.scale_update_flag
    except:
        st.session_state.scale_update_flag = False
    try:
        st.session_state.html_content
    except:
        st.session_state.html_content = None
    # 判断是否需要刷新
    need_refresh = (
        st.session_state.recal_flag 
        or st.session_state.scale_update_flag 
        or force_refresh 
        or st.session_state.html_content is None
    )

    if need_refresh:
        start_time = time.time()
        try:
            if "cal_store" not in st.session_state:
                st.session_state.cal_store = {}
            
            # 获取几何对象并进行缩放处理
            objects = {}
            if "material" in st.session_state.cal_store:
                objects = st.session_state.cal_store["material"].geom_kernel.geom.objects
            visualizer = IVisualizer.create()
            
            scale = st.session_state.scale
            
            # 只有在有对象时才处理
            if objects:
                for idx, obj in objects.items():
                    tmp = deepcopy(obj)
                    tmp.vertices = tmp.vertices * scale
                    visualizer.addObj(tmp)
            
            plotter = visualizer.plotter
            
            # 设置背景色和坐标轴
            plotter.set_background("lightgray", top="white")
            plotter.add_axes()
            
            # 生成 Panel VTK 内容
            vtk_pane = pn.pane.VTK(plotter.ren_win, width=600, height=400)
            with StringIO() as model_bytes:
                vtk_pane.save(model_bytes, title="PyVista Minimal")
                html_content = model_bytes.getvalue()
            
            # 更新缓存
            st.session_state.html_content = html_content
            st.session_state.last_calc_time = time.time() - start_time
            st.session_state.last_update_ts = time.strftime("%H:%M:%S")
            
        except Exception as e:
            st.error(f"Error updating 3D view: {str(e)}")
            # import traceback
            # st.error(traceback.format_exc())
    
    # 渲染缓存的内容
    if st.session_state.html_content:
        components.html(st.session_state.html_content, height=720, scrolling=False)
        
        # 显示性能监控信息
        st.caption(f"Last updated: {st.session_state.last_update_ts} | Calculation time: {st.session_state.last_calc_time:.4f}s")
    else:
        st.info("No 3D content available. Please load a file.")
        
    st.session_state.recal_flag = False
    st.session_state.scale_update_flag = False
