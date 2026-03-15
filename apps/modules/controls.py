import pandas as pd
import streamlit as st
import urllib.parse

from apps.modules.material_assignment_store import (
    get_assignment_path,
    get_material_properties_path,
    save_material_assignment,
)
from dmslicer.materials import (
    GradientMaterial,
    IsolationMaterial,
    PendingMaterial,
    SourceMaterial,
)
from dmslicer.materials.materials import Material_Property


def normalize_rgb(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) not in (3, 4):
        return None
    rgb = [float(value[0]), float(value[1]), float(value[2])]
    if all(0.0 <= x <= 1.0 for x in rgb):
        rgb255 = [int(round(x * 255)) for x in rgb]
    else:
        rgb255 = [int(round(x)) for x in rgb]
    rgb255 = [max(0, min(255, x)) for x in rgb255]
    return tuple(rgb255)


def rgb_to_hex(rgb) -> str:
    if rgb is None:
        return ""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def format_rgbarray(value) -> str:
    rgb = normalize_rgb(value)
    if rgb is None:
        return "None"
    r, g, b = rgb
    return f"rgbarray([{r/255:.3g}, {g/255:.3g}, {b/255:.3g}])"


def get_material_type_label(material) -> str:
    if isinstance(material, GradientMaterial):
        return "GradientMaterial"
    if isinstance(material, IsolationMaterial):
        return "IsolationMaterial"
    if isinstance(material, SourceMaterial):
        return "SourceMaterial"
    if isinstance(material, PendingMaterial):
        return "PendingMaterial"
    return type(material).__name__ if material is not None else "PendingMaterial"


def _parse_composition(value):
    if value is None:
        return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                out[key] = float(v)
            except Exception:
                continue
        return out or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            out = {}
            for part in parts:
                if ":" not in part:
                    continue
                part_key, part_val = part.split(":", 1)
                part_key = part_key.strip()
                if not part_key:
                    continue
                try:
                    out[part_key] = float(part_val)
                except Exception:
                    continue
            return out or None
        return text
    return None


def _format_composition(value) -> str:
    comp = _parse_composition(value)
    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    items = sorted(comp.items(), key=lambda kv: kv[0])
    return ",".join([f"{k}:{v:.6g}" for k, v in items])


def _object_id_badge(hex_color: str, oid) -> str:
    color = hex_color if isinstance(hex_color, str) and hex_color.startswith("#") else "#cccccc"
    text = str(oid)
    text_width = max(10, 7 * len(text))
    width = 6 + text_width + 6 + 16 + 6
    rect_x = 6 + text_width + 6
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="22" viewBox="0 0 {width} 22">'
        f'<text x="6" y="16" font-family="sans-serif" font-size="12" fill="#111">{text}</text>'
        f'<rect x="{rect_x}" y="3" width="16" height="16" rx="2" fill="{color}" stroke="#888"/>'
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def _get_material_name(material) -> str:
    name = getattr(material, "material_name", None)
    if name is None:
        return ""
    return str(name)


def _get_obj_row(obj, oid):
    if obj is None:
        hex_color = ""
        material = None
    else:
        hex_color = rgb_to_hex(normalize_rgb(getattr(obj, "color", None)))
        material = getattr(obj, "material", None)
    return {
        "Select": False,
        "oid": oid,
        "Object_ID": _object_id_badge(hex_color, oid),
        "Material_Name": _get_material_name(material),
        "Material_Type": type(material).__name__ if material is not None else "",
    }


def _normalize_oid_list(values):
    normalized = []
    seen = set()
    for value in values or []:
        try:
            oid = int(value)
        except Exception:
            continue
        if oid in seen:
            continue
        seen.add(oid)
        normalized.append(oid)
    return normalized


def _clear_show_selection():
    st.session_state.show_oids = []
    st.session_state.reset_show_editor = True


def _is_pending_material(material) -> bool:
    if material is None or isinstance(material, PendingMaterial):
        return True
    return _get_material_name(material).strip() == "Pending"


def _get_object_status(obj, objects) -> str:
    if obj is None:
        return "InProgress"

    if _is_pending_material(getattr(obj, "material", None)):
        return "InProgress"

    nbr_obj_ids = list(getattr(obj, "nbr_objects", []) or [])
    for nbr_obj_id in nbr_obj_ids:
        nbr_obj = objects.get(nbr_obj_id) if objects is not None else None
        if nbr_obj is None or _is_pending_material(getattr(nbr_obj, "material", None)):
            return "InProgress"
    return "Completed"


def _sync_material_state(objects) -> None:
    pending = []
    inprogress = []
    completed = []

    for oid in sorted((objects or {}).keys()):
        obj = objects.get(oid)
        status = _get_object_status(obj, objects)
        if status == "Completed":
            completed.append(oid)
        else:
            inprogress.append(oid)

    st.session_state.material["Pending"] = pending
    st.session_state.material["InProgress"] = inprogress
    st.session_state.material["Completed"] = completed


def _initialize_selected_materials(selected_oids, objects) -> int:
    updated = 0
    for oid in _normalize_oid_list(selected_oids):
        obj = objects.get(oid) if objects is not None else None
        if obj is None:
            continue
        obj.material = PendingMaterial()
        updated += 1
    return updated


def _enforce_gradient_include_minimum(include_key, valid_include_key, exclude_key, nbr_obj_ids) -> None:
    selected = _normalize_oid_list(st.session_state.get(include_key))
    valid_options = _normalize_oid_list(nbr_obj_ids)
    selected = [oid for oid in selected if oid in valid_options]

    if len(selected) >= 2:
        st.session_state[valid_include_key] = selected
        exclude_values = _normalize_oid_list(st.session_state.get(exclude_key))
        st.session_state[exclude_key] = [oid for oid in exclude_values if oid in valid_options and oid not in set(selected)]
        st.session_state.pop(f"{include_key}_warning", None)
        return

    fallback = _normalize_oid_list(st.session_state.get(valid_include_key))
    fallback = [oid for oid in fallback if oid in valid_options]
    if len(fallback) < 2:
        fallback = valid_options[:2]

    st.session_state[include_key] = fallback
    exclude_values = _normalize_oid_list(st.session_state.get(exclude_key))
    st.session_state[exclude_key] = [oid for oid in exclude_values if oid in valid_options and oid not in set(fallback)]
    st.session_state[valid_include_key] = fallback
    st.session_state[f"{include_key}_warning"] = "Include Neighbors must keep at least 2 values."


def process(inprogress_obj_ids=None):
    with st.expander("Processing"):
        process_candidates = (
            (st.session_state.material.get("Pending", []) or [])
            + (st.session_state.material.get("InProgress", []) or [])
        )
        process_candidates = sorted(set(x for x in process_candidates if x is not None))

        current_process_oid = st.session_state.get("processing_obj_id")
        if inprogress_obj_ids is not None:
            current_process_oid = inprogress_obj_ids

        process_options = [""] + [str(x) for x in process_candidates]
        current_value = "" if current_process_oid is None else str(current_process_oid)
        if current_value not in set(process_options):
            current_value = ""

        selected_value = st.selectbox(
            "Process",
            process_options,
            index=process_options.index(current_value),
            key="processing_obj_select",
        )

        next_oid = None if selected_value == "" else int(selected_value)
        if next_oid != st.session_state.get("processing_obj_id"):
            _clear_show_selection()
            if next_oid is None:
                st.session_state.processing_active = False
                st.session_state.processing_obj_id = None
            else:
                st.session_state.processing_active = True
                st.session_state.processing_obj_id = next_oid
                st.session_state.selected_oid = next_oid
            st.rerun()

        inprogress_obj_ids = st.session_state.get("processing_obj_id")
        if inprogress_obj_ids is None:
            if (
                st.session_state.get("viewport_center") is not None
                or st.session_state.get("viewport_nbrs") is not None
            ):
                st.session_state.viewport_center = None
                st.session_state.viewport_nbrs = None
                st.session_state.viewport_dirty = True
            return

        material_manager = st.session_state.get("cal_store", {}).get("material")
        if (
            material_manager is None
            or not hasattr(material_manager, "geom_kernel")
            or not hasattr(material_manager.geom_kernel, "geom")
        ):
            return

        objects = getattr(material_manager.geom_kernel.geom, "objects", {}) or {}
        obj = objects.get(inprogress_obj_ids)
        if obj is None:
            return

        processing_obj_id = f"### Processing Object ID:{inprogress_obj_ids}"
        obj_rgb = normalize_rgb(getattr(obj, "color", None))
        obj_hex = rgb_to_hex(obj_rgb)
        if obj_hex:
            st.markdown(
                processing_obj_id
                + f'<span style="display:inline-block;width:0.9em;height:0.9em;'
                f'border:1px solid #888;background:{obj_hex};vertical-align:middle;"></span> ',
                unsafe_allow_html=True,
            )
        else:
            st.write(processing_obj_id + f": {format_rgbarray(getattr(obj, 'color', None))}")

        current_material = getattr(obj, "material", None)
        current_type_label = get_material_type_label(current_material)
        nbr_obj_ids = list(getattr(obj, "nbr_objects", []) or [])
        has_gradient_option = len(nbr_obj_ids) >= 2
        type_options = ["PendingMaterial", "SourceMaterial", "IsolationMaterial"] + (
            ["GradientMaterial"] if has_gradient_option else []
        )
        selected_type = st.selectbox(
            "Material Type",
            type_options,
            index=type_options.index(current_type_label) if current_type_label in type_options else 0,
            key=f"material_type_{inprogress_obj_ids}",
        )

        library_materials = Material_Property.get_all_materials()
        library_names = sorted({m.name for m in library_materials if getattr(m, "name", None)})
        default_name = getattr(current_material, "material_name", None) or str(inprogress_obj_ids)
        material_name = None
        selected_library_mat = None

        if selected_type == "PendingMaterial":
            st.caption("PendingMaterial is the initial state.")
        elif selected_type == "GradientMaterial":
            grad_name = f"Grad_{inprogress_obj_ids}"
            material_name = st.selectbox(
                "Material Name",
                [grad_name],
                index=0,
                key=f"material_name_{inprogress_obj_ids}",
                disabled=True,
            )
        else:
            material_name_options = (
                [default_name] + [n for n in library_names if n != default_name]
                if library_names
                else [default_name]
            )
            material_name = st.selectbox(
                "Material Name",
                material_name_options,
                index=0,
                key=f"material_name_{inprogress_obj_ids}",
            )
            selected_library_mat = next(
                (m for m in library_materials if getattr(m, "name", None) == material_name),
                None,
            )

        include_list = None
        exclude_list = []
        gradient_valid = True
        if selected_type == "GradientMaterial":
            include_key = f"gradient_include_{inprogress_obj_ids}"
            valid_include_key = f"{include_key}_valid"
            exclude_key = f"gradient_exclude_{inprogress_obj_ids}"

            raw_exclude = st.session_state.get(exclude_key) or []
            raw_exclude = [x for x in raw_exclude if x in nbr_obj_ids]
            include_options = [x for x in nbr_obj_ids if x not in set(raw_exclude)]
            raw_include = st.session_state.get(include_key)
            if raw_include is None:
                raw_include = include_options
            raw_include = [x for x in raw_include if x in include_options]
            if len(raw_include) < 2:
                previous_valid = _normalize_oid_list(st.session_state.get(valid_include_key))
                previous_valid = [x for x in previous_valid if x in include_options]
                raw_include = previous_valid if len(previous_valid) >= 2 else include_options[:2]
            st.session_state[include_key] = raw_include
            st.session_state[valid_include_key] = raw_include

            include_list = st.multiselect(
                "Include Neighbors",
                options=include_options,
                default=raw_include,
                key=include_key,
                on_change=_enforce_gradient_include_minimum,
                args=(include_key, valid_include_key, exclude_key, nbr_obj_ids),
            )
            include_list = _normalize_oid_list(st.session_state.get(include_key))

            exclude_options = [x for x in nbr_obj_ids if x not in set(include_list)]
            raw_exclude = [x for x in raw_exclude if x in exclude_options]
            st.session_state[exclude_key] = raw_exclude
            exclude_list = st.multiselect(
                "Exclude Neighbors",
                options=exclude_options,
                default=raw_exclude,
                key=exclude_key,
            )

            overlap = set(include_list).intersection(set(exclude_list))
            if overlap:
                gradient_valid = False
                st.warning(f"Include/Exclude overlap: {sorted(overlap)}")
            include_warning = st.session_state.pop(f"{include_key}_warning", None)
            if include_warning:
                st.warning(include_warning)
            if include_list is None or len(include_list) < 2:
                gradient_valid = False
                st.warning("Include Neighbors must keep at least 2 values.")

        material_composition_value = None
        restriction_value = None
        if selected_type == "SourceMaterial":
            material_composition_value = (
                getattr(selected_library_mat, "composition", None) if selected_library_mat else None
            )
            st.text_input(
                "Material Composition",
                value=_format_composition(material_composition_value),
                key=f"material_composition_{inprogress_obj_ids}",
                disabled=True,
            )
            restriction_options = ["None"] + [n for n in library_names if n != material_name]
            restriction_value = st.selectbox(
                "Restricted Materials",
                restriction_options,
                index=0,
                key=f"restriction_{inprogress_obj_ids}",
            )

        if st.button("Apply Material", key=f"apply_material_{inprogress_obj_ids}"):
            if selected_type == "PendingMaterial":
                obj.material = PendingMaterial()
            elif selected_type == "SourceMaterial":
                parsed_comp = _parse_composition(material_composition_value)
                if parsed_comp is None:
                    parsed_comp = material_name
                restricted_materials = (
                    None if restriction_value in (None, "None") else [restriction_value]
                )
                obj.material = SourceMaterial(
                    material_name=material_name,
                    material_composition=parsed_comp,
                    restricted_materials=restricted_materials,
                )
            elif selected_type == "IsolationMaterial":
                obj.material = IsolationMaterial(material_name=material_name)
            else:
                if not gradient_valid:
                    st.warning("Fix GradientMaterial Include/Exclude selection before applying.")
                    return
                obj._geom_objects = objects
                center_include_exclude_dict = {
                    "center": inprogress_obj_ids,
                    "include": include_list if include_list else None,
                    "exclude": exclude_list or [],
                }
                obj.material = GradientMaterial(
                    material_name or str(inprogress_obj_ids),
                    obj,
                    center_include_exclude_dict,
                )
                obj.material._center_include_exclude_dict = {
                    "center": inprogress_obj_ids,
                    "include": list(include_list) if include_list else None,
                    "exclude": list(exclude_list or []),
                }
            st.session_state.viewport_dirty = True
            st.rerun()

        rows = []
        for nbr_obj_id in nbr_obj_ids:
            nbr_obj = objects.get(nbr_obj_id)
            if nbr_obj is None:
                continue
            rows.append(_get_obj_row(nbr_obj, nbr_obj_id))
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(
                df[["Object_ID", "Material_Name", "Material_Type"]],
                hide_index=True,
                column_config={
                    "Object_ID": st.column_config.ImageColumn("Object_ID"),
                    "Material_Name": st.column_config.TextColumn("Name"),
                    "Material_Type": st.column_config.TextColumn("Type"),
                },
            )

        next_center = inprogress_obj_ids
        next_nbrs = list(nbr_obj_ids)
        if (
            st.session_state.get("viewport_center") != next_center
            or st.session_state.get("viewport_nbrs") != next_nbrs
        ):
            st.session_state.viewport_center = next_center
            st.session_state.viewport_nbrs = next_nbrs
            st.session_state.viewport_dirty = True


def render_controls():
    """
    娓叉煋鎺у埗闈㈡澘锛屽寘鎷?Pending, InProgress, Complete 鍒楄〃鐨勬搷浣溿€?
    """
    st.subheader("Materialize")
    with st.expander("Object Selection"):
        uploaded_file = st.session_state.get("uploaded_file")
        uploaded_file_name = getattr(uploaded_file, "name", None)
        material_manager = st.session_state.get("cal_store", {}).get("material")
        objects = {}
        if st.session_state.get("reset_show_editor"):
            if "object_selection_editor" in st.session_state:
                del st.session_state["object_selection_editor"]
            st.session_state.reset_show_editor = False
        if (
            material_manager is not None
            and hasattr(material_manager, "geom_kernel")
            and hasattr(material_manager.geom_kernel, "geom")
        ):
            objects = getattr(material_manager.geom_kernel.geom, "objects", {}) or {}

        col1, col2, col3 = st.columns(3)
        with col1:
            initialize_selection = st.button("Initialize", width="stretch")
        with col2:
            show_selection = st.button("Show", width="stretch")
        with col3:
            save_progress = st.button(
                "Save",
                disabled=not uploaded_file_name or not objects,
                width="stretch",
            )

        _sync_material_state(objects)

        rows = []
        for oid in sorted(objects.keys()):
            obj = objects.get(oid)
            row = _get_obj_row(obj, oid)
            row["Status"] = _get_object_status(obj, objects)
            rows.append(row)

        df = pd.DataFrame(rows)
        show_oids = _normalize_oid_list(st.session_state.get("show_oids"))
        if show_oids and not df.empty:
            df.loc[df["oid"].isin(show_oids), "Select"] = True

        if df.empty:
            st.data_editor(
                pd.DataFrame(
                    columns=["Select", "Object_ID", "Status", "Material_Name", "Material_Type", "oid"]
                ),
                hide_index=True,
                disabled=["Select", "Object_ID", "Status", "Material_Name", "Material_Type", "oid"],
                column_order=["Select", "Object_ID", "Status", "Material_Name", "Material_Type", "oid"],
                column_config={
                    "Select": st.column_config.CheckboxColumn("Sel"),
                    "oid": None,
                    "Object_ID": st.column_config.ImageColumn("ID"),
                    "Material_Name": st.column_config.TextColumn("Name"),
                    "Material_Type": st.column_config.TextColumn("Type"),
                },
                key="object_selection_editor",
                width="stretch",
            )
            st.session_state.selected_oid = None
            st.session_state.show_oids = []
        else:
            edited = st.data_editor(
                df,
                hide_index=True,
                disabled=["oid", "Object_ID", "Status", "Material_Name", "Material_Type"],
                column_order=["Select", "Object_ID", "Status", "Material_Name", "Material_Type", "oid"],
                column_config={
                    "Select": st.column_config.CheckboxColumn("Sel"),
                    "oid": None,
                    "Object_ID": st.column_config.ImageColumn("ID"),
                    "Material_Name": st.column_config.TextColumn("Name"),
                    "Material_Type": st.column_config.TextColumn("Type"),
                },
                key="object_selection_editor",
                width="stretch",
            )
            selected = _normalize_oid_list(edited.loc[edited["Select"], "oid"].tolist())
            if initialize_selection:
                updated = _initialize_selected_materials(selected, objects)
                if updated == 0:
                    st.warning("Select at least one object to initialize.")
                else:
                    _clear_show_selection()
                    st.session_state.viewport_dirty = True
                    st.rerun()
            if show_selection:
                prev_show_oids = _normalize_oid_list(st.session_state.get("show_oids"))
                st.session_state.show_oids = selected
                if selected != prev_show_oids or selected:
                    st.session_state.viewport_dirty = True

        if save_progress:
            try:
                save_path = save_material_assignment(
                    material_manager,
                    objects,
                    processing_obj_id=st.session_state.get("processing_obj_id"),
                    source_file=uploaded_file_name,
                )
                st.toast(f"Saved progress to {save_path.name}")
            except Exception as exc:
                st.error(f"Failed to save progress: {exc}")

        save_path = get_assignment_path(material_manager)
        props_path = get_material_properties_path(material_manager)
        if save_path is not None:
            st.caption(f"Assignment file: {save_path.name}")
        if props_path is not None:
            st.caption(f"Material properties file: {props_path.name}")

    process(st.session_state.get("processing_obj_id"))
