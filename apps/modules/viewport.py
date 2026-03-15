import time
import streamlit as st
import panel as pn
import streamlit.components.v1 as components
from io import StringIO
from dmslicer.visualizer.visualizer_interface import IVisualizer
from dmslicer.tools.size_of import get_total_size
from dmslicer.default_config import DEFAULTS

def html_generator(*args,**kwargs):
    start_time = time.time()
    objects = st.session_state.cal_store["material"].geom_kernel.geom.objects
    rest_set = set(objects.keys())
    t_start_block = time.time()
    visualizer = IVisualizer.create()            
    # 只有在有对象时才处理
    center_opacity = kwargs.get("center_opacity", 0.9)
    nbrs_opacity = kwargs.get("nbrs_opacity", 0.1)
    other_opacity = kwargs.get("other_opacity", 0.01)

    def normalize_ids(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, (int, float)):
            return [value]
        raise ValueError("value must be a list, tuple, set, int, or float.")

    center = args[0] if len(args) > 0 else None
    nbrs = args[1] if len(args) > 1 else None

    center_ids = normalize_ids(center)
    nbrs_ids = normalize_ids(nbrs)

    if not center_ids and not nbrs_ids:
        for _, obj in objects.items():
            visualizer.addObj(obj,show_edges=False)
    else:
        for oid in center_ids:
            obj = objects.get(oid)
            if obj is None:
                continue
            visualizer.addObj(obj, opacity=center_opacity, show_edges=False)
            rest_set.discard(oid)

        for oid in nbrs_ids:
            obj = objects.get(oid)
            if obj is None:
                continue
            visualizer.addObj(obj, opacity=nbrs_opacity,show_edges=False)
            rest_set.discard(oid)

        for oid in rest_set:
            obj = objects.get(oid)
            if obj is None:
                continue
            visualizer.addObj(obj, opacity=other_opacity,show_edges=False)


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
    if force_refresh:
        st.session_state.viewport_center = None
        st.session_state.viewport_nbrs = None
        st.session_state.viewport_dirty = True
        st.session_state.viewport_signature = None

    # 初始化 Session State (虽然在 state.py 中有，但这里作为局部保险也可以保留，或者移除)
    # 此处假设 state.py 已经正确初始化，直接使用 st.session_state
    if "html_content" not in st.session_state:
        st.session_state.html_content = None
    if "viewport_dirty" not in st.session_state:
        st.session_state.viewport_dirty = False
    if "viewport_signature" not in st.session_state:
        st.session_state.viewport_signature = None
    # 判断是否需要刷新
    center_opacity = st.session_state.get("viewport_center_opacity", 0.9)
    nbrs_opacity = st.session_state.get("viewport_nbrs_opacity", 0.1)
    other_opacity = st.session_state.get("viewport_other_opacity", 0.01)
    show_oids = tuple(
        sorted({int(oid) for oid in (st.session_state.get("show_oids") or [])})
    )
    current_signature = (
        show_oids,
        st.session_state.get("viewport_center"),
        tuple(st.session_state.get("viewport_nbrs") or []),
        center_opacity,
        nbrs_opacity,
        other_opacity,
    )
    need_refresh = (
        st.session_state.recal_flag 
        or force_refresh 
        or st.session_state.html_content is None
        or st.session_state.viewport_dirty
        or st.session_state.viewport_signature != current_signature
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

            center = st.session_state.get("viewport_center")
            nbrs = st.session_state.get("viewport_nbrs")
            if show_oids:
                html_content = html_generator(
                    list(show_oids),
                    center_opacity=0.8,
                    nbrs_opacity=0.1,
                    other_opacity=0.1,
                )
            else:
                html_content = (
                    html_generator(
                        center,
                        nbrs,
                        center_opacity=center_opacity,
                        nbrs_opacity=nbrs_opacity,
                        other_opacity=other_opacity,
                    )
                    if center is not None
                    else html_generator(
                        center_opacity=center_opacity,
                        nbrs_opacity=nbrs_opacity,
                        other_opacity=other_opacity,
                    )
                )
            # 更新缓存
            st.session_state.html_content = html_content
            st.session_state.last_update_ts = time.strftime("%H:%M:%S")
            st.session_state.viewport_dirty = False
            st.session_state.viewport_signature = current_signature
            
            print(f"[DEBUG] Render execution time: {time.time() - t_start_render:.6f} seconds")
            
        except Exception as e:
            st.error(f"Error updating 3D view: {str(e)}")
            # import traceback
            # st.error(traceback.format_exc())
    
    # 渲染缓存的内容
    if st.session_state.html_content:
        # 显示性能监控信息
        components.html(st.session_state.html_content, height=720, scrolling=False)
        # st.caption(f"Last updated: {st.session_state.last_update_ts} | Calculation time: {st.session_state.last_calc_time:.4f}s")
    else:
        st.info("No 3D content available. Please load a file.")
        
    st.session_state.recal_flag = False
