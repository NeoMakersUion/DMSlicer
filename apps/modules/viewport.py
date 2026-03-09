import time
import streamlit as st
import panel as pn
import streamlit.components.v1 as components
from copy import deepcopy
from io import StringIO
from dmslicer.geometry_kernel.geom_kernel import GeometryKernel
from dmslicer.visualizer.visualizer_interface import IVisualizer
from dmslicer.file_parser import Model
from dmslicer.tools.size_of import get_total_size
from dmslicer.default_config import DEFAULTS

def html_generator(*args,**kwargs):
    start_time = time.time()
    objects = st.session_state.cal_store["material"].geom_kernel.geom.objects
    t_start_block = time.time()
    visualizer = IVisualizer.create()            
    # 只有在有对象时才处理
    if len(args)==0 and len(kwargs)==0:
        for idx, obj in objects.items():
            visualizer.addObj(obj)
    else:
        pass
    print(f"[DEBUG] Block execution time: {time.time() - t_start_block:.6f} seconds")
    plotter = visualizer.plotter
    # 设置背景色和坐标轴
    plotter.set_background("lightgray", top="white")
    plotter.add_axes()
    plotter.camera.zoom(10**(-DEFAULTS["GEOM_ACC"]))
    
    # 生成 Panel VTK 内容
    vtk_pane = pn.pane.VTK(plotter.ren_win, width=600, height=400)
    with StringIO() as model_bytes:
        vtk_pane.save(model_bytes, title="PyVista Minimal")
        html_content = model_bytes.getvalue()
    
    st.session_state.last_calc_time = time.time() - start_time
    return html_content

def render_viewport():
    """
    渲染 3D 视图区域，包括 PyVista 绘图和 HTML 缓存机制。
    """

    st.subheader("3D View")


    force_refresh = st.button("Force Refresh")

    # 初始化 Session State (虽然在 state.py 中有，但这里作为局部保险也可以保留，或者移除)
    # 此处假设 state.py 已经正确初始化，直接使用 st.session_state
    try:
        st.session_state.html_content
    except:
        st.session_state.html_content = None
    # 判断是否需要刷新
    need_refresh = (
        st.session_state.recal_flag 
        or force_refresh 
        or st.session_state.html_content is None
    )

    if need_refresh:
        try:
            if "cal_store" not in st.session_state:
                st.session_state.cal_store = {}
                return
            objects = st.session_state.cal_store["material"].geom_kernel.geom.objects
            # 获取几何对象并进行缩放处理            
            size_in_bytes = get_total_size(objects)
            if size_in_bytes < 1024:
                size_str = f"{size_in_bytes} bytes"
            elif size_in_bytes < 1024 * 1024:
                size_str = f"{size_in_bytes / 1024:.2f} KB"
            elif size_in_bytes < 1024 * 1024 * 1024:
                size_str = f"{size_in_bytes / (1024 * 1024):.2f} MB"
            else:
                size_str = f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"
            print(f"[DEBUG] objects size: {size_str}")
            t_start_render = time.time()

            html_content=html_generator()
            # 更新缓存
            st.session_state.html_content = html_content
            st.session_state.last_update_ts = time.strftime("%H:%M:%S")
            
            print(f"[DEBUG] Render execution time: {time.time() - t_start_render:.6f} seconds")
            
        except Exception as e:
            st.error(f"Error updating 3D view: {str(e)}")
            # import traceback
            # st.error(traceback.format_exc())
    
    # 渲染缓存的内容
    if st.session_state.html_content:
        # 显示性能监控信息
        components.html(st.session_state.html_content, height=720, scrolling=False)
        st.caption(f"Last updated: {st.session_state.last_update_ts} | Calculation time: {st.session_state.last_calc_time:.4f}s")
    else:
        st.info("No 3D content available. Please load a file.")
        
    st.session_state.recal_flag = False
