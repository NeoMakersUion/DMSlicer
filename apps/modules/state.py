import streamlit as st

# Default size for the model scale
DEFAULT_SIZE = -6.5
@st.cache_resource
def get_global_resources():
    """
    加载全局资源（如模型缓存等）。
    使用 st.cache_resource 装饰器确保只加载一次。
    """
    print(">>> 正在加载全局资源（如模型缓存等）...")
    # 这里可以放置真正需要缓存的重资源加载，比如大模型权重
    pass

def init_session_state():
    """初始化会话状态，每次脚本运行都需要检查"""
    if "recal_flag" not in st.session_state:
        st.session_state.recal_flag = False
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "cal_store" not in st.session_state:
        st.session_state.cal_store = {}
        st.session_state.cal_store['material']=None
    if "material" not in st.session_state:
        st.session_state.material = {
            "Pending": [],
            "InProgress": [],
            "Completed": []
        }
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    
    # 3D View related states
    if "html_content" not in st.session_state:
        st.session_state.html_content = None
    if "last_calc_time" not in st.session_state:
        st.session_state.last_calc_time = 0.0
    if "last_update_ts" not in st.session_state:
        st.session_state.last_update_ts = ""
    if "last_processed_id" not in st.session_state:
        st.session_state.last_processed_id = None

    if "material_properties" not in st.session_state:
        st.session_state.material_properties = {}
    if "temp_mat_dict" not in st.session_state:
        st.session_state.temp_mat_dict = {}
