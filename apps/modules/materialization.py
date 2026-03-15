
import pandas as pd
import streamlit as st

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
from dmslicer.tools import (
    normalize_rgb,
    rgb_to_hex,
    format_rgbarray,
    object_id_badge_svg,
    color_swatch_html,
)


# 颜色与调试渲染相关工具函数已迁移至 dmslicer.tools
# Color and debug rendering utilities are imported from dmslicer.tools.


# 根据 material 实例返回对应的类型字符串
def get_material_type_label(material) -> str:
    """根据材料实例返回对应的类型标签。
    Return the UI label for a material instance.

        Parameters
        ----------
        material : Any
            Material instance to classify.

        Returns
        -------
        str
            One of ``PendingMaterial``, ``SourceMaterial``,
            ``IsolationMaterial``, ``GradientMaterial``, or the runtime class name.

        Raises
        ------
        None
            Type checks are defensive and do not raise intentionally.

        Examples
        --------
        >>> get_material_type_label(None)
        'PendingMaterial'
        """
    if isinstance(material, GradientMaterial):
        return "GradientMaterial"
    if isinstance(material, IsolationMaterial):
        return "IsolationMaterial"
    if isinstance(material, SourceMaterial):
        return "SourceMaterial"
    if isinstance(material, PendingMaterial):
        return "PendingMaterial"
    return type(material).__name__ if material is not None else "PendingMaterial"


# 将前端输入的 composition 解析为字典或字符串
def _parse_composition(value):
    """将 composition 输入解析为规范的字典或名称字符串。
    Parse composition input into canonical dict or string.

        Parameters
        ----------
        value : Any
            Composition value from UI/model. Supports dict form, plain name string,
            or ratio string like ``"PLA:0.5,TPU:0.5"``.

        Returns
        -------
        dict[str, float] | str | None
            Parsed composition dictionary, plain string material name, or ``None``
            if parsing fails.

        Raises
        ------
        None
            Invalid key/value entries are skipped instead of raising.

        Examples
        --------
        >>> _parse_composition("PLA:0.5,TPU:0.5")
        {'PLA': 0.5, 'TPU': 0.5}
        >>> _parse_composition("PLA")
        'PLA'
        """
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
        # 支持 "Al:0.5,Mg:0.5" 格式
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


# 将解析后的 composition 字典格式化为字符串
def _format_composition(value) -> str:
    """将 composition 值格式化为紧凑的显示字符串。
    Format composition value to a compact display string.

        Parameters
        ----------
        value : Any
            Any value accepted by :func:`_parse_composition`.

        Returns
        -------
        str
            Empty string when unavailable, original string for name-only values, or
            normalized ratio string sorted by key.

        Raises
        ------
        None
            Parsing failures are normalized to empty output.

        Examples
        --------
        >>> _format_composition({"TPU": 0.5, "PLA": 0.5})
        'PLA:0.5,TPU:0.5'
        """
    comp = _parse_composition(value)
    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    items = sorted(comp.items(), key=lambda kv: kv[0])
    return ",".join([f"{k}:{v:.6g}" for k, v in items])


# 安全获取 material 的 material_name 属性
def _get_material_name(material) -> str:
    """安全读取材料对象的 ``material_name`` 属性。
    Safely read ``material_name`` from a material-like object.

        Parameters
        ----------
        material : Any
            Material instance or any object that may expose ``material_name``.

        Returns
        -------
        str
            ``material_name`` converted to string; empty string when unavailable.

        Raises
        ------
        None
            Uses ``getattr`` fallback and does not raise intentionally.

        Examples
        --------
        >>> _get_material_name(None)
        ''
        """
    name = getattr(material, "material_name", None)
    if name is None:
        return ""
    return str(name)


# 构造表格单行数据：Select/oid/Object_ID(徽章)/Material_Name/Material_Type
def _get_obj_row(obj, oid):
    """构造对象选择表格的一行数据。
    Construct one object-selection table row.

        Parameters
        ----------
        obj : Any | None
            Geometry object containing at least ``color`` and ``material`` fields.
            ``None`` is allowed and produces empty material fields.
        oid : int | str
            Object identifier for display and internal selection.

        Returns
        -------
        dict[str, Any]
            Row dictionary with ``Select``, ``oid``, ``Object_ID``,
            ``Material_Name``, and ``Material_Type``.

        Raises
        ------
        None
            Missing attributes are tolerated via safe access.

        Examples
        --------
        >>> row = _get_obj_row(None, 1)
        >>> row["oid"]
        1
        """
    if obj is None:
        hex_color = ""
        material = None
    else:
        hex_color = rgb_to_hex(normalize_rgb(getattr(obj, "color", None)))
        material = getattr(obj, "material", None)
    return {
        "Select": False,
        "oid": oid,
        "Object_ID": object_id_badge_svg(hex_color, oid),
        "Material_Name": _get_material_name(material),
        "Material_Type": type(material).__name__ if material is not None else "",
    }


# 将任意列表归一化为唯一且有序的 OID 列表
def _normalize_oid_list(values):
    """将候选 OID 值归一化为唯一的整数列表。
    Normalize candidate OID values into unique integer list.

    Parameters
    ----------
    values : Sequence[Any] | None
        Candidate values from UI/editor states.

    Returns
    -------
    list[int]
        Order-preserving, de-duplicated integer OID list.

    Raises
    ------
    None
        Non-convertible values are skipped.

    Examples
    --------
    >>> _normalize_oid_list(["1", 2, 2, "x"])
    [1, 2]
    """
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


# 清空右侧 3D 高亮选中缓存，并标记表格需要重置
def _clear_show_selection():
    """清空当前的 Show 选择并请求重置编辑器。
    Clear current Show selection and request editor reset.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Updates ``st.session_state`` in place.

    Raises
    ------
    None
        Session-state assignments are direct and not expected to raise under
        normal Streamlit execution.

    Examples
    --------
    >>> _clear_show_selection()  # doctest: +SKIP
    """
    st.session_state.show_oids = []
    st.session_state.reset_show_editor = True


# 判断当前 material 是否为 Pending 状态
def _is_pending_material(material) -> bool:
    """判断一个材料是否应被视为 Pending 状态。
    Determine whether a material should be treated as pending.

    Parameters
    ----------
    material : Any
        Material instance to test.

    Returns
    -------
    bool
        ``True`` when material is ``None``, ``PendingMaterial``, or has
        ``material_name == "Pending"``.

    Raises
    ------
    None
        Safe attribute access prevents exceptions for unknown types.

    Examples
    --------
    >>> _is_pending_material(None)
    True
    """
    if material is None or isinstance(material, PendingMaterial):
        return True
    return _get_material_name(material).strip() == "Pending"


# 根据对象自身及邻居材料状态计算行级状态：InProgress / Completed
def _get_object_status(obj, objects) -> str:
    """基于自身及邻居材料计算对象状态。
    Compute object status based on self and neighbor materials.

    Parameters
    ----------
    obj : Any | None
        Target object.
    objects : dict[int, Any] | None
        Object map used to resolve neighbor objects.

    Returns
    -------
    str
        ``"Completed"`` when object and all neighbors are non-pending;
        otherwise ``"InProgress"``.

    Raises
    ------
    None
        Missing objects/attributes are handled defensively.

    Examples
    --------
    >>> _get_object_status(None, {}) in {"InProgress", "Completed"}
    True
    """
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


# 扫描全部对象，将 OID 分类到 Pending/InProgress/Completed
def _sync_material_state(objects) -> None:
    """根据对象状态同步会话中的材料分组（Pending/InProgress/Completed）。
    Synchronize session material buckets from object statuses.

    Parameters
    ----------
    objects : dict[int, Any] | None
        Geometry-object mapping.

    Returns
    -------
    None
        Writes normalized OID lists into ``st.session_state.material``.

    Raises
    ------
    KeyError
        If ``st.session_state.material`` has not been initialized.

    Examples
    --------
    >>> _sync_material_state({})  # doctest: +SKIP
    """
    pending: list[int] = []
    inprogress: list[int] = []
    completed: list[int] = []

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


# 将表格中选中的对象初始化为 PendingMaterial，返回实际更新数量
def _initialize_selected_materials(selected_oids, objects) -> int:
    """将选中的对象初始化为 ``PendingMaterial``。
    Initialize selected objects to ``PendingMaterial``.

    Parameters
    ----------
    selected_oids : Sequence[Any]
        Selected OID values from the UI table.
    objects : dict[int, Any] | None
        Object mapping to update.

    Returns
    -------
    int
        Number of objects actually updated.

    Raises
    ------
    None
        Invalid OIDs and missing objects are skipped.

    Examples
    --------
    >>> _initialize_selected_materials([], {})  # doctest: +SKIP
    0
    """
    updated = 0
    for oid in _normalize_oid_list(selected_oids):
        obj = objects.get(oid) if objects is not None else None
        if obj is None:
            continue
        obj.material = PendingMaterial()
        updated += 1
    return updated


# GradientMaterial 必须至少包含 2 个邻居，否则强制回退；同时同步 exclude 列表
def _enforce_gradient_include_minimum(include_key, valid_include_key, exclude_key, nbr_obj_ids) -> None:
    """为 GradientMaterial 的界面强制至少两个 Include 邻居并同步 Exclude。
    Enforce minimum include-neighbor constraint for GradientMaterial UI.

    Parameters
    ----------
    include_key : str
        Session-state key for include neighbor multiselect.
    valid_include_key : str
        Session-state key storing last valid include selection.
    exclude_key : str
        Session-state key for exclude neighbor multiselect.
    nbr_obj_ids : Sequence[int]
        Full candidate neighbor list.

    Returns
    -------
    None
        Mutates related session-state keys in place.

    Raises
    ------
    None
        Invalid values are sanitized; function does not raise intentionally.

    Examples
    --------
    >>> _enforce_gradient_include_minimum("a", "b", "c", [1, 2, 3])  # doctest: +SKIP
    """
    selected = _normalize_oid_list(st.session_state.get(include_key))
    valid_options = _normalize_oid_list(nbr_obj_ids)
    selected = [oid for oid in selected if oid in valid_options]

    if len(selected) >= 2:
        st.session_state[valid_include_key] = selected
        exclude_values = _normalize_oid_list(st.session_state.get(exclude_key))
        st.session_state[exclude_key] = [oid for oid in exclude_values if oid in valid_options and oid not in set(selected)]
        st.session_state.pop(f"{include_key}_warning", None)
        return

    # 回退到上次合法列表，否则取前 2 个
    fallback = _normalize_oid_list(st.session_state.get(valid_include_key))
    fallback = [oid for oid in fallback if oid in valid_options]
    if len(fallback) < 2:
        fallback = valid_options[:2]

    st.session_state[include_key] = fallback
    exclude_values = _normalize_oid_list(st.session_state.get(exclude_key))
    st.session_state[exclude_key] = [oid for oid in exclude_values if oid in valid_options and oid not in set(fallback)]
    st.session_state[valid_include_key] = fallback
    st.session_state[f"{include_key}_warning"] = "Include Neighbors must keep at least 2 values."

def object_selection():
    with st.expander("Object Selection"):
        material_manager, objects, uploaded_file_name, save_progress = _render_object_selection_panel()

        # Save 逻辑：持久化材料分配与属性
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

        # 底部提示：展示当前保存路径（若存在）
        save_path = get_assignment_path(material_manager)
        props_path = get_material_properties_path(material_manager)
        if save_path is not None:
            st.caption(f"Assignment file: {save_path.name}")
        if props_path is not None:
            st.caption(f"Material properties file: {props_path.name}")

# 右侧折叠面板：单对象材料处理主界面
def process(inprogress_obj_ids=None):
    """渲染并处理单对象的 Processing 面板。
    Render and handle the per-object Processing panel.

    Parameters
    ----------
    inprogress_obj_ids : int | None, default=None
        Preferred object ID to process. When provided, it overrides the
        current ``st.session_state.processing_obj_id`` selection.

    Returns
    -------
    None
        Renders Streamlit widgets and updates session/object state in place.

    Raises
    ------
    ValueError
        Raised implicitly if the Process selectbox value cannot be converted to
        ``int`` (e.g., corrupted session value).
    streamlit.errors.StreamlitAPIException
        May be raised by Streamlit widget/state operations when called outside
        a valid Streamlit runtime context.

    Examples
    --------
    >>> process()  # doctest: +SKIP
    >>> process(3)  # doctest: +SKIP
    """
    with st.expander("Processing"):
        # 候选列表 = Pending + InProgress 去重排序
        process_candidates = (
            (st.session_state.material.get("Pending", []) or [])
            + (st.session_state.material.get("InProgress", []) or [])
        )
        process_candidates = sorted(set(x for x in process_candidates if x is not None))

        # 若外部指定 OID 则优先使用
        current_process_oid = st.session_state.get("processing_obj_id")
        if inprogress_obj_ids is not None:
            current_process_oid = inprogress_obj_ids

        # 构造 selectbox 选项
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

        # 切换对象时清理右侧高亮并重置状态
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

        # 未选中任何对象时清理 3D 高亮并返回
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

        # 获取 geometry 与对象字典
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

        # 顶部标题：Processing Object ID + 颜色块
        processing_obj_id = f"### Processing Object ID:{inprogress_obj_ids}"
        obj_rgb = normalize_rgb(getattr(obj, "color", None))
        obj_hex = rgb_to_hex(obj_rgb)
        if obj_hex:
            swatch = color_swatch_html(obj_hex, size=14)
            st.markdown(processing_obj_id + swatch + " ", unsafe_allow_html=True)
        else:
            st.write(processing_obj_id + f": {format_rgbarray(getattr(obj, 'color', None))}")

        # 材料类型下拉框
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

        # 材料库列表
        library_materials = Material_Property.get_all_materials()
        library_names = sorted({m.name for m in library_materials if getattr(m, "name", None)})
        default_name = getattr(current_material, "material_name", None) or str(inprogress_obj_ids)
        material_name = None
        selected_library_mat = None

        # 根据类型展示不同控件
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

        # GradientMaterial 专用：Include/Exclude 邻居多选框
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

            # 合法性检查
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

        # SourceMaterial 专用：展示只读 composition 与受限材料下拉框
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

        # Apply 按钮：根据类型实例化具体材料并附加到 obj
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

        # 底部邻居表格：展示当前对象的所有邻居 ID、名称、类型
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

        # 同步 3D 视图中心与邻居高亮列表
        next_center = inprogress_obj_ids
        next_nbrs = list(nbr_obj_ids)
        if (
            st.session_state.get("viewport_center") != next_center
            or st.session_state.get("viewport_nbrs") != next_nbrs
        ):
            st.session_state.viewport_center = next_center
            st.session_state.viewport_nbrs = next_nbrs
            st.session_state.viewport_dirty = True


def _render_object_selection_panel():
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

    return material_manager, objects, uploaded_file_name, save_progress


def Materialize():
    """渲染 Materialize 面板与对象选择工作流。
    Render the Materialize panel and object-selection workflow.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Builds Object Selection UI, updates selection state, persists
        assignments, and delegates object-level editing to :func:`process`.

    Raises
    ------
    streamlit.errors.StreamlitAPIException
        May be raised by Streamlit widget APIs if called outside Streamlit app
        execution.
    Exception
        Persistence failures from ``save_material_assignment`` are caught and
        surfaced through ``st.error``; unexpected runtime failures may still
        propagate.

    Examples
    --------
    >>> Materialize()  # doctest: +SKIP
    """
    # 面板标题
    with st.expander("Materialize"):
    # “Object Selection”折叠面板：展示对象列表与批量操作
    
        object_selection()
        # Processing 面板：只在 processing_obj_id 不为 None 时显示具体对象的处理控件
        process(st.session_state.get("processing_obj_id"))