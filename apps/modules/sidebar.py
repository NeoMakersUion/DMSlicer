import streamlit as st
from dmslicer.file_parser import read_amf_objects, Model
from dmslicer.geometry_kernel.geom_kernel import GeometryKernel
from .state import DEFAULT_SIZE
from dmslicer.materials.materials import Materializer
def render_sidebar():
    """
    渲染侧边栏内容，包括文件上传和缩放滑块。
    """
    st.subheader("Controls")
    try:
        st.session_state.recal_flag
    except:
        st.session_state.recal_flag = False
        
    if st.session_state.recal_flag == False:
        uploaded_file = st.file_uploader("请选择文件", type=["amf"])
        
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.recal_flag = True
            st.session_state.uploaded_file = uploaded_file
            model: Model = read_amf_objects(uploaded_file)
            geom_kernal=GeometryKernel(model) 
            material=Materializer(geom_kernal)
            st.session_state.cal_store["material"] = material  
            st.session_state.material["Pending"]=material.pending_obj

            
    size = st.slider("Size", min_value=-10.0, max_value=10.0, value=DEFAULT_SIZE, step=0.1)
    scale = 10**size
    
    try:
        st.session_state.scale
    except:
        st.session_state.scale = 10**DEFAULT_SIZE
        
    if st.session_state.scale != scale:
        st.session_state.scale = scale
        st.session_state.scale_update_flag = True
