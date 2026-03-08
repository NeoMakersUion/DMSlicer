import streamlit as st
import pandas as pd

def render_controls():
    """
    渲染控制面板，包括 Pending, InProgress, Complete 列表的操作。
    """
    st.subheader("Materialize")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    pending_obj_ids = None
    
    with col1:
        try:
            pending_obj_ids = st.radio("Pending", st.session_state.material["Pending"])
            if st.button("Process") and pending_obj_ids is not None:
                st.session_state.material["InProgress"].append(pending_obj_ids)
                st.session_state.material["InProgress"].sort()
                try:
                    st.session_state.material["Pending"].remove(pending_obj_ids)
                except Exception:
                    None
                # 记录刚刚处理的 ID，以便在 col2 中默认选中
                st.session_state.last_processed_id = pending_obj_ids
                st.rerun()
        except Exception:
            st.radio("Pending", [], key="pending_empty_fallback")
    with col2:
        # 确定 col2 的默认选中项索引
        try:
            inprogress_list = st.session_state.material["InProgress"]
            
            # 尝试查找上次处理的 ID 是否在列表中
            if "last_processed_id" in st.session_state and st.session_state.last_processed_id in inprogress_list:
                default_index = inprogress_list.index(st.session_state.last_processed_id)
            else:
                default_index = 0
            
            inprogress_obj_ids = st.radio(
                "InProgress", 
                inprogress_list, 
                index=default_index if inprogress_list else None
            )
            
        except Exception:
            st.radio("InProgress", [], key="inprogress_empty_fallback")
            inprogress_obj_ids = None
            
    with col3:
        try:
            complete_obj_ids = st.radio("Complete", st.session_state.material["Complete"])
            if st.button("Redo"):
                pass
        except Exception:
            st.radio("Complete", [], key="complete_empty_fallback")
    
    # 确保 inprogress_obj_ids 已定义
    if 'inprogress_obj_ids' not in locals():
        inprogress_obj_ids = None


    # Processing Table
    if inprogress_obj_ids is not None:
        try:
            # 尝试从 material store 中获取对象信息
            if "material" in st.session_state.cal_store:
                material_manager = st.session_state.cal_store['material']
                # 注意：这里假设 geom_kernel 结构，如果不行则降级处理
                if hasattr(material_manager, "geom_kernel") and hasattr(material_manager.geom_kernel, "geom"):
                    obj = material_manager.geom_kernel.geom.objects.get(inprogress_obj_ids)
                    
                    if obj:
                        st.write(f"### Processing Object {inprogress_obj_ids}")
                        # 相邻的对象ID
                        nbr_obj_ids=obj.nbr_objects
                        # 构建属性字典
                        nbr_obj_material_dict = {}
                        for nbr_obj_id in nbr_obj_ids:
                            nbr_obj = material_manager.geom_kernel.geom.objects.get(nbr_obj_id)
                            if nbr_obj:
                                nbr_obj_material_dict[nbr_obj_id] = type(nbr_obj.material).__name__
                        data = {
                            "Neighbor_Object_ID": list(nbr_obj_ids),
                            "Material_Type":[
                                    nbr_obj_material_dict[nbr_obj_id] for nbr_obj_id in nbr_obj_ids
                            ]
                        }                            
                        # 创建 DataFrame
                        df = pd.DataFrame(data)
                        # 使用 dataframe 展示，隐藏索引
                        st.dataframe(df, hide_index=True)
                        st.divider()
                        st.caption("Edit Visual Properties")
        except Exception as e:
            st.warning(f"Could not load object details: {e}")
