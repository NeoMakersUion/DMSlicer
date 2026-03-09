from pandas._libs import properties
import streamlit as st
from dmslicer.file_parser import read_amf_objects, Model
from dmslicer.geometry_kernel.geom_kernel import GeometryKernel
from .state import DEFAULT_SIZE
from dmslicer.materials.materials import Materializer,Material_Property
import colorsys
import pandas as pd
import copy
import ast
import math
import json

def clean_dict(data):
    cleaned = {}
    for key, value in data.items():
        # 1. 处理 NaN 值 (转为 None)
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
            continue
            
        # 2. 处理字符串类型的字典/列表
        if isinstance(value, str):
            # 尝试判断是否看起来像字典或列表 (以 { 或 [ 开头)
            stripped = value.strip()
            if (stripped.startswith('{') and stripped.endswith('}')) or \
               (stripped.startswith('[') and stripped.endswith(']')):
                try:
                    # 使用 ast.literal_eval 安全地转换字符串为 Python 对象
                    # 它能处理单引号 {'a': 1} 和双引号 {"a": 1}
                    cleaned[key] = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    # 如果解析失败，保留原字符串
                    cleaned[key] = value
            else:
                cleaned[key] = value
        else:
            # 其他类型直接保留
            cleaned[key] = value
            
    return cleaned

def adjust_unit(property_name:str):
    if "density" in property_name.lower():
        return ["kg/m3","g/cm3"]
    elif "modulus" in property_name.lower():
        return ["Pa","KPa","MPa"]
    elif "ratio" in property_name.lower():
        return ["-"]
    elif "temp" in property_name.lower():
        return ["C","F","K"]
    elif "color" in property_name.lower():
        return ["RGB","HSL","HEX"]
    else:
        return ["kg/m3","Pa","-","C"]

def hex2rgb(hex_color:str):
    hex_color=hex_color.lstrip("#")
    r,g,b=[int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return (r,g,b)

def hex2hsl(hex_color:str):
    r,g,b=hex2rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    return (h,l,s)

def rgb2hex(rgb:tuple):
    r,g,b=rgb
    return "#{:02x}{:02x}{:02x}".format(r, g, b)
def hsl2hex(hsl:tuple):
    h,l,s=hsl
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return rgb2hex((r,g,b))
def save_draft_cb():
    temp_mat_name=st.session_state.new_mat_name
    if not temp_mat_name: return
    st.session_state.temp_mat_dict[temp_mat_name] = copy.deepcopy(st.session_state.material_properties)
    st.session_state.material_properties = {}
    st.session_state.new_mat_name = ""
def load_temporary_material_cb():
    temp_mat_name=st.session_state.temp_mat_name
    if temp_mat_name in st.session_state.temp_mat_dict:
        st.session_state.material_properties=copy.deepcopy(st.session_state.temp_mat_dict[temp_mat_name])
        st.session_state.new_mat_name=temp_mat_name
        
def render_sidebar():
    """
    渲染侧边栏内容，包括文件上传和缩放滑块。
    """
    st.subheader("Upload and Recalculate")
    try:
        st.session_state.recal_flag
    except:
        st.session_state.recal_flag = False
        
    if st.session_state.recal_flag == False:
        uploaded_file = st.file_uploader("请选择文件", type=["amf","AMF"])
        
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.recal_flag = True
            st.session_state.uploaded_file = uploaded_file
            model: Model = read_amf_objects(uploaded_file)
            geom_kernal=GeometryKernel(model) 
            material=Materializer(geom_kernal)
            st.session_state.cal_store["material"] = material  
            st.session_state.material["Pending"]=material.pending_obj
            st.session_state.material["InProgress"]=[]
            st.session_state.material["Completed"]=[]

    st.markdown("---")
    with st.expander("Properties Editor", expanded=True):
        if "mat_caption_msg" not in st.session_state:
            st.session_state.mat_caption_msg = "Manage custom material properties"
        st.caption(st.session_state.mat_caption_msg)

        def add_material_property_cb():
            property_name=st.session_state.new_property_name
            property_unit=st.session_state.new_property_unit
            property_value=st.session_state.new_property_value
            st.session_state.material_properties[property_name]={"value":property_value,"unit":property_unit}
        # Callback for adding material
        def register_material_to_library_cb():
            try:
                name = st.session_state.new_mat_name
                if not name: return
                data = st.session_state.material_properties
                data["name"]=st.session_state.new_mat_name
                Material_Property.add_material(data)
                st.session_state.new_mat_name = "" # Reset name
                st.session_state.mat_caption_msg = f"✅ Material '{name}' added successfully"
                st.toast(f"Material '{name}' added")
                del st.session_state.temp_mat_dict[name]
                st.session_state.material_properties={}
            except ValueError as e:
                st.session_state.mat_caption_msg = f"❌ {str(e)}"
                st.error(str(e))
        
        def validate_material_cb():
            try:
                name = st.session_state.new_mat_name
                if not name: 
                    st.session_state.mat_caption_msg = "Manage custom material properties"
                    return
                
                if Material_Property.validate_material(name):
                    st.session_state.mat_caption_msg = f"✅ Material '{name}' is valid"
                else:
                    st.session_state.mat_caption_msg = f"❌ Material '{name}' is not valid (already exists)"
            except Exception as e:
                st.session_state.mat_caption_msg = f"❌ Error: {str(e)}"
        # Input Section
        c1, c2= st.columns([1, 1])
        temp_names = sorted(list(st.session_state.temp_mat_dict.keys()))
        c2.selectbox("Temporary Material", [""] + temp_names, key="temp_mat_name", index=0, on_change=load_temporary_material_cb)
        if (not st.session_state.get("new_mat_name")) and st.session_state.get("temp_mat_name"):
            st.session_state["new_mat_name"] = st.session_state["temp_mat_name"]
        c1.text_input("Material Name", key="new_mat_name", placeholder="Name (Enter to add)", on_change=validate_material_cb)
        c2_1,c2_2,c2_3=st.columns([2,1,1])
        property_list=list(Material_Property.__init__.__code__.co_varnames)
        remove_list=["id","name","self","kwargs"]
        for item in remove_list:
            property_list.remove(item)
        c2_1.selectbox("Property Name",property_list,key="new_property_name")
        unit_list=adjust_unit(st.session_state.new_property_name)
        c2_2.selectbox("Unit",unit_list,key="new_property_unit")
        if "color" in st.session_state.new_property_name.lower():
            hex_color=c2_3.color_picker("Value",key="property_color_picker")
            if "rgb" in st.session_state.new_property_unit.lower():
                hex_color=hex_color.lstrip("#")
                st.session_state.new_property_value=tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif "hex" in st.session_state.new_property_unit.lower():
                st.session_state.new_property_value=hex_color
            elif "hsl" in st.session_state.new_property_unit.lower():
                hex_color=hex_color.lstrip("#")
                r,g,b=[int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
                h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
                st.session_state.new_property_value=(h,l,s)
        else:
            st.session_state.new_property_value=c2_3.number_input("Value",value=1.0, min_value=0.01, step=0.1, key="property_value_input", format="%.2f")
        c3_1,c3_2,c3_3=st.columns([1,1,1])
        c3_1.button("Add Property", on_click=add_material_property_cb)
        c3_2.button("Save as Draft", on_click=save_draft_cb)
        c3_3.button("Save to Library", on_click=register_material_to_library_cb)
        props = st.session_state.material_properties
        if props:
            df = pd.DataFrame.from_dict(props, orient="index")
            if "value" not in df.columns: df["value"] = ""
            if "unit" not in df.columns: df["unit"] = ""
            
            df = df.reset_index()
            df["value"] = df["value"].astype(str)
            df['del'] = False
            
            df = df.rename(columns={"index": "property"})
            # Ensure column order
            df = df[["del","property", "value", "unit"]]
            
            edited_df_mat_tmp=st.data_editor(df,disabled=["property","unit"],hide_index=True,key="material_properties_df",width='stretch')
            del_properties=[df.iloc[i]['property'] for i in range(len(edited_df_mat_tmp)) if edited_df_mat_tmp.iloc[i]['del']==True]
            if del_properties:
                st.warning(f"Delete {len(del_properties)} item(s)?")
                col_d1, col_d2 = st.columns(2)
                if col_d1.button("Confirm Delete", key="btn_confirm_del_prop"):
                    for p in del_properties:
                        del st.session_state.material_properties[p]
                    st.rerun()
                if col_d2.button("Cancel", key="btn_cancel_del_prop"):
                    st.rerun()
        else:
            st.info("No properties yet.")

    
    with st.expander("Material Library", expanded=True):
        # Table Section
        materials = Material_Property.get_all_materials()
        
        current_data = [m.to_dict() for m in materials]
        for d in current_data:
             d['Select'] = False
        
        if len(current_data) > 0:
            all_keys = list(current_data[0].keys())
            fixed_order = ["id", "Select", "name"]
            fixed_order = [col for col in fixed_order if col in all_keys]
            remaining_cols = [col for col in all_keys if col not in fixed_order]
            column_order = fixed_order + remaining_cols
        else:
            column_order = ["id", "Select", "name"]

        all_cols = [c for c in column_order if c != "Select"]
        edited_df = st.data_editor(
            current_data,
            column_order=column_order,
            disabled=all_cols,
            column_config={
                "id": None
            },
            hide_index=True,
            width='stretch',
            key="material_editor"
        )
        selected_list=[e for e in edited_df if e.get('Select')]
        if selected_list!=[]:
             st.warning(f"{len(selected_list)} item(s) selected")
             col_d1, col_d2, col_d3 = st.columns(3)
             if col_d1.button("Edit", key="btn_edit_mat"):
                for e in selected_list:
                    Material_Property.remove_material(e['id'])
                    mat_dict=copy.deepcopy(clean_dict(e))
                    del mat_dict['id']
                    del mat_dict['name']
                    mat_name=e['name']
                    st.session_state.temp_mat_dict[mat_name]=mat_dict
                    st.rerun()
             if col_d2.button("Delete", key="btn_del_mat"):
                 for e in selected_list:
                    edited_df.remove(e)
                    Material_Property.remove_material(e['id'])
                 st.rerun()
             if col_d3.button("Cancel", key="btn_cancel_mat"):
                 st.rerun()

        changes_detected = False
        def is_diff(a, b):
            if a is None and b is None: return False
            if a is None or b is None: return True
            try:
                return abs(a - b) > 1e-6
            except TypeError:
                return True

        for row in edited_df:
            obj = next((m for m in materials if m.id == row['id']), None)
            if obj:
                if (obj.name != row['name'] or 
                    is_diff(obj.density, row['density']) or
                    is_diff(obj.elastic_modulus, row['elastic_modulus']) or
                    is_diff(obj.poisson_ratio, row['poisson_ratio'])):
                    
                    obj.name = row['name']
                    obj.density = row['density']
                    obj.elastic_modulus = row['elastic_modulus']
                    obj.poisson_ratio = row['poisson_ratio']
                    changes_detected = True
        
        if changes_detected:
            Material_Property.save_to_json()
    
