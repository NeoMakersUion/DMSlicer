import streamlit as st
from apps.modules.material_assignment_store import load_material_assignment
from dmslicer.file_parser import read_amf_objects, Model
from dmslicer.geometry_kernel.geom_kernel import GeometryKernel
from dmslicer.materials.materials import Materializer,Material_Property
from dmslicer.tools import hex2rgb,hex2hsl
import pandas as pd
import copy
from dmslicer.tools import clean_dict
from dmslicer.tools import adjust_unit
from typing import Dict

def save_draft_cb():
    temp_mat_name=st.session_state.new_mat_name
    if not temp_mat_name:
        return
    st.session_state.temp_mat_dict[temp_mat_name] = copy.deepcopy(st.session_state.material_properties)
    st.session_state.temp_mat_name = temp_mat_name
    st.session_state.new_mat_name = temp_mat_name
    st.session_state.material_properties = copy.deepcopy(st.session_state.temp_mat_dict[temp_mat_name])
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
    except Exception:
        st.session_state.recal_flag = False
        
    if not st.session_state.recal_flag:
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
            st.session_state.processing_active = False
            st.session_state.processing_obj_id = None
            st.session_state.selected_oid = None
            st.session_state.show_oids = []
            st.session_state.reset_show_editor = True

            try:
                objects = getattr(material.geom_kernel.geom, "objects", {}) or {}
                load_result = load_material_assignment(material, objects)
                restored_oid = load_result.get("processing_obj_id")
                if restored_oid is not None:
                    st.session_state.processing_active = True
                    st.session_state.processing_obj_id = restored_oid
                    st.session_state.selected_oid = restored_oid
                    st.session_state.processing_obj_select = str(restored_oid)
                else:
                    st.session_state.processing_obj_select = ""
                if load_result.get("found") and load_result.get("loaded_count"):
                    st.toast(f"Loaded saved progress from {load_result['path'].name}")
            except Exception as e:
                st.warning(f"Failed to load saved progress: {e}")

    st.markdown("---")
    with st.expander("Properties Editor", expanded=True):
        if "mat_caption_msg" not in st.session_state:
            st.session_state.mat_caption_msg = "Manage custom material properties"
        st.caption(st.session_state.mat_caption_msg)

        def add_material_property_cb():
            property_name=st.session_state.new_property_name
            property_unit=st.session_state.new_property_unit
            property_value=st.session_state.new_property_value
            if property_name == "composition" and property_value is None:
                st.session_state.mat_caption_msg = "❌ Composition sum must be 1"
                return
            st.session_state.material_properties[property_name]={"value":property_value,"unit":property_unit}
        # Callback for adding material
        def register_material_to_library_cb():
            try:
                name = st.session_state.new_mat_name
                if not name:
                    return
                data = {}
                for k, v in st.session_state.material_properties.items():
                    if isinstance(v, dict) and "value" in v:
                        data[k] = v.get("value")
                    else:
                        data[k] = v
                data["name"] = name
                Material_Property.add_material(data)
                st.session_state.new_mat_name = "" # Reset name
                st.session_state.mat_caption_msg = f"✅ Material '{name}' added successfully"
                st.toast(f"Material '{name}' added")
                if name in st.session_state.temp_mat_dict:
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
        preferred_props = ["melting_temperature", "soft_temperature", "composition"]
        property_list = [p for p in preferred_props if p in property_list] + [
            p for p in property_list if p not in preferred_props
        ]
        c2_1.selectbox("Property Name",property_list,key="new_property_name")
        unit_list=adjust_unit(st.session_state.new_property_name)
        c2_2.selectbox("Unit",unit_list,key="new_property_unit")

        def _format_composition_label(value) -> str:
            comp = value
            if isinstance(comp, dict) and "value" in comp and set(comp.keys()).issubset({"value", "unit"}):
                comp = comp.get("value")

            if isinstance(comp, str):
                parts = [p.strip() for p in comp.split(",") if p.strip()]
                parsed: Dict[str, float] = {}
                ok = True
                for part in parts:
                    if ":" not in part:
                        ok = False
                        break
                    part_key, part_val = part.split(":", 1)
                    part_key = part_key.strip()
                    if not part_key:
                        ok = False
                        break
                    try:
                        parsed[part_key] = float(part_val)
                    except Exception:
                        ok = False
                        break
                if ok and parsed:
                    comp = parsed
                else:
                    return comp

            if isinstance(comp, dict):
                normalized: Dict[str, object] = {}
                for raw_key, raw_val in comp.items():
                    name = str(raw_key).strip()
                    if not name:
                        continue
                    try:
                        normalized[name] = float(raw_val)
                    except Exception:
                        normalized[name] = str(raw_val)

                items = sorted(normalized.items(), key=lambda kv: kv[0])
                out = []
                for name, ratio in items:
                    if isinstance(ratio, (int, float)):
                        out.append(f"{name}:{float(ratio):.6g}")
                    else:
                        out.append(f"{name}:{ratio}")
                return ",".join(out)

            return "" if comp is None else str(comp)

        def _coerce_composition(value) -> Dict[str, float]:
            comp: Dict[str, float] = {}
            if value is None:
                return comp
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return comp
                if ":" in text:
                    parts = [p.strip() for p in text.split(",") if p.strip()]
                    parsed: Dict[str, float] = {}
                    for part in parts:
                        if ":" not in part:
                            continue
                        part_key, part_val = part.split(":", 1)
                        part_key = part_key.strip()
                        if not part_key:
                            continue
                        try:
                            parsed[part_key] = float(part_val)
                        except Exception:
                            continue
                    if parsed:
                        return parsed
                comp[text] = 1.0
                return comp
            if isinstance(value, dict):
                for k, v in value.items():
                    if k is None:
                        continue
                    name = str(k).strip()
                    if not name:
                        continue
                    try:
                        ratio = float(v)
                    except Exception:
                        continue
                    comp[name] = comp.get(name, 0.0) + ratio
                return comp
            return comp

        if st.session_state.new_property_name == "composition":
            default_component = st.session_state.get("new_mat_name") or "Material_name"
            init_comp = None
            if st.session_state.get("new_property_value") is not None:
                init_comp = _coerce_composition(st.session_state.get("new_property_value"))
            if init_comp is None or init_comp == {}:
                init_comp = {default_component: 1.0}
            comp_df = pd.DataFrame(
                [{"material": k, "ratio": float(v)} for k, v in init_comp.items()]
            )
            edited_comp_df = st.data_editor(
                comp_df,
                num_rows="dynamic",
                hide_index=True,
                key="composition_editor",
                width="stretch",
            )
            comp: Dict[str, float] = {}
            for _, row in edited_comp_df.iterrows():
                name = str(row.get("material", "")).strip()
                if not name:
                    continue
                try:
                    ratio = float(row.get("ratio", 0.0))
                except Exception:
                    ratio = 0.0
                comp[name] = comp.get(name, 0.0) + ratio
            total = sum(comp.values())
            if len(comp) == 0:
                comp = {default_component: 1.0}
                total = 1.0
            if abs(total - 1.0) <= 1e-6:
                st.session_state.new_property_value = comp
                c2_3.caption(_format_composition_label(comp))
            else:
                st.session_state.new_property_value = None
                c2_3.error(f"sum={total:.6g} (need 1)")

        elif "color" in st.session_state.new_property_name.lower():
            hex_color=c2_3.color_picker("Value",key="property_color_picker")
            if "rgb" in st.session_state.new_property_unit.lower():
                st.session_state.new_property_value=hex2rgb(hex_color)
            elif "hex" in st.session_state.new_property_unit.lower():
                st.session_state.new_property_value=hex_color
            elif "hsl" in st.session_state.new_property_unit.lower():
                st.session_state.new_property_value=hex2hsl(hex_color)
        else:
            st.session_state.new_property_value=c2_3.number_input("Value",value=1.0, min_value=0.01, step=0.1, key="property_value_input", format="%.2f")
        c3_1,c3_2,c3_3=st.columns([1,1,1])
        c3_1.button("Add Property", on_click=add_material_property_cb,width="stretch")
        c3_2.button("Save as Draft", on_click=save_draft_cb,width="stretch")
        c3_3.button("Save to Library", on_click=register_material_to_library_cb,width="stretch")
        props = st.session_state.material_properties
        if props:
            normalized_props = {}
            for prop_name, entry in props.items():
                if prop_name == "Select":
                    continue
                if isinstance(entry, dict) and "value" in entry and "unit" in entry:
                    normalized_props[prop_name] = entry
                    continue
                unit_candidates = adjust_unit(str(prop_name))
                unit_default = unit_candidates[0] if unit_candidates else "-"
                value = entry
                if prop_name == "composition":
                    coerced = _coerce_composition(entry)
                    value = coerced if coerced else entry
                    unit_default = "-"
                normalized_props[prop_name] = {"value": value, "unit": unit_default}

            st.session_state.material_properties = normalized_props
            df = pd.DataFrame.from_dict(normalized_props, orient="index")
            if "value" not in df.columns:
                df["value"] = ""
            if "unit" not in df.columns:
                df["unit"] = ""
            
            df = df.reset_index()
            df["value"] = df["value"].astype(str)
            df['del'] = False
            
            df = df.rename(columns={"index": "property"})
            # Ensure column order
            df = df[["del","property", "value", "unit"]]
            
            edited_df_mat_tmp=st.data_editor(df,disabled=["property","unit"],hide_index=True,key="material_properties_df",width='stretch')
            del_properties=[
                df.iloc[i]["property"]
                for i in range(len(edited_df_mat_tmp))
                if edited_df_mat_tmp.iloc[i]["del"]
            ]
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
            d["Select"] = False
            comp = d.get("composition")
            if isinstance(comp, dict):
                d["composition"] = _format_composition_label(comp)
        
        if len(current_data) > 0:
            key_set = set()
            for row in current_data:
                key_set.update(row.keys())

            preferred_order = [
                "id",
                "Select",
                "name",
                "melting_temperature",
                "soft_temperature",
                "composition",
                "density",
                "elastic_modulus",
                "poisson_ratio",
                "color",
            ]
            column_order = [c for c in preferred_order if c in key_set]
            remaining_cols = [c for c in sorted(key_set) if c not in set(column_order)]
            column_order = column_order + remaining_cols
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
             col_d1, col_d2 = st.columns(2)
             if col_d1.button("Edit", key="btn_edit_mat",width="stretch"):
                for e in selected_list:
                    Material_Property.remove_material(e['id'])
                    mat_dict=copy.deepcopy(clean_dict(e))
                    del mat_dict['id']
                    del mat_dict['name']
                    if "Select" in mat_dict:
                        del mat_dict["Select"]
                    mat_name=e['name']
                    normalized_edit_props = {}
                    for prop_name, value in mat_dict.items():
                        unit_candidates = adjust_unit(str(prop_name))
                        unit_default = unit_candidates[0] if unit_candidates else "-"
                        if prop_name == "composition":
                            coerced = _coerce_composition(value)
                            value = coerced if coerced else value
                            unit_default = "-"
                        normalized_edit_props[prop_name] = {"value": value, "unit": unit_default}
                    st.session_state.temp_mat_dict[mat_name]=normalized_edit_props
                    st.rerun()
             if col_d2.button("Delete", key="btn_del_mat",width="stretch"):
                 for e in selected_list:
                    edited_df.remove(e)
                    Material_Property.remove_material(e['id'])
                 st.rerun()


        changes_detected = False
        def is_diff(a, b):
            if a is None and b is None:
                return False
            if a is None or b is None:
                return True
            try:
                return abs(a - b) > 1e-6
            except TypeError:
                return True

        for row in edited_df:
            obj = next((m for m in materials if m.id == row['id']), None)
            if obj:
                row_density = row.get("density")
                row_elastic_modulus = row.get("elastic_modulus")
                row_poisson_ratio = row.get("poisson_ratio")

                if (
                    obj.name != row.get("name")
                    or is_diff(obj.density, row_density)
                    or is_diff(obj.elastic_modulus, row_elastic_modulus)
                    or is_diff(obj.poisson_ratio, row_poisson_ratio)
                ):
                    
                    obj.name = row.get("name")
                    obj.density = row_density
                    obj.elastic_modulus = row_elastic_modulus
                    obj.poisson_ratio = row_poisson_ratio
                    changes_detected = True
        
        if changes_detected:
            Material_Property.save_to_json()
    
