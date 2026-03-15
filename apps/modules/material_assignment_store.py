import json
from datetime import datetime

from dmslicer.file_parser.workspace_utils import get_workspace_dir
from dmslicer.materials import (
    GradientMaterial,
    IsolationMaterial,
    PendingMaterial,
    SourceMaterial,
)
from dmslicer.materials.materials import Material_Property


SCHEMA_VERSION = 1
MATERIAL_DIR_NAME = "material"
ASSIGNMENT_FILE_NAME = "material_assignment.json"
MATERIAL_PROPERTIES_FILE_NAME = "material_properties.json"


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


def _get_model_hash_id(material_manager):
    geom_kernel = getattr(material_manager, "geom_kernel", None)
    geom = getattr(geom_kernel, "geom", None)
    model = getattr(geom, "model", None)
    hash_id = getattr(model, "hash_id", None)
    if hash_id is None:
        return None
    hash_id = str(hash_id).strip()
    return hash_id or None


def get_assignment_path(material_manager):
    hash_id = _get_model_hash_id(material_manager)
    if not hash_id:
        return None
    return get_workspace_dir() / hash_id / MATERIAL_DIR_NAME / ASSIGNMENT_FILE_NAME


def get_material_properties_path(material_manager):
    hash_id = _get_model_hash_id(material_manager)
    if not hash_id:
        return None
    return get_workspace_dir() / hash_id / MATERIAL_DIR_NAME / MATERIAL_PROPERTIES_FILE_NAME


def _get_legacy_assignment_path(material_manager):
    hash_id = _get_model_hash_id(material_manager)
    if not hash_id:
        return None
    return get_workspace_dir() / hash_id / ASSIGNMENT_FILE_NAME


def _get_legacy_material_properties_path(material_manager):
    hash_id = _get_model_hash_id(material_manager)
    if not hash_id:
        return None
    return get_workspace_dir() / hash_id / MATERIAL_PROPERTIES_FILE_NAME


def _get_material_name(material):
    name = getattr(material, "material_name", None)
    if name is None:
        return ""
    return str(name)


def _get_gradient_config(material, obj, oid):
    nbr_obj_ids = _normalize_oid_list(getattr(obj, "nbr_objects", []) or [])
    config = getattr(material, "_center_include_exclude_dict", None)
    include = []
    exclude = []

    if isinstance(config, dict):
        include = [x for x in _normalize_oid_list(config.get("include")) if x in nbr_obj_ids]
        exclude = [x for x in _normalize_oid_list(config.get("exclude")) if x in nbr_obj_ids]

    if not include:
        composition = getattr(getattr(material, "composition", None), "composition", None)
        if isinstance(composition, dict):
            include = [x for x in _normalize_oid_list(composition.keys()) if x in nbr_obj_ids]

    if not include:
        include = list(nbr_obj_ids)

    exclude = [x for x in exclude if x not in set(include)]
    if not exclude:
        exclude = [x for x in nbr_obj_ids if x not in set(include)]

    return {
        "center": int(oid),
        "include": include if include else None,
        "exclude": exclude,
    }


def _serialize_material(material, obj, oid):
    material_name = _get_material_name(material) or "Pending"

    if material is None or isinstance(material, PendingMaterial) or material_name == "Pending":
        return {
            "material_type": "PendingMaterial",
            "material_name": "Pending",
        }

    if isinstance(material, GradientMaterial):
        config = _get_gradient_config(material, obj, oid)
        return {
            "material_type": "GradientMaterial",
            "material_name": material_name or f"Grad_{oid}",
            "include": config.get("include"),
            "exclude": config.get("exclude"),
        }

    if isinstance(material, IsolationMaterial):
        return {
            "material_type": "IsolationMaterial",
            "material_name": material_name,
        }

    if isinstance(material, SourceMaterial):
        restricted_materials = getattr(material, "restricted_materials", None)
        return {
            "material_type": "SourceMaterial",
            "material_name": material_name,
            "restricted_materials": list(restricted_materials or []),
        }

    return {
        "material_type": type(material).__name__,
        "material_name": material_name,
    }


def _collect_referenced_material_names(objects):
    names = set()
    for obj in (objects or {}).values():
        material = getattr(obj, "material", None)
        material_name = _get_material_name(material).strip()
        if material_name and material_name != "Pending":
            names.add(material_name)
        restricted = getattr(material, "restricted_materials", None)
        if isinstance(restricted, list):
            for item in restricted:
                name = str(item).strip()
                if name and name != "Pending":
                    names.add(name)
    return names


def save_material_properties(material_manager, objects, source_file=None):
    path = get_material_properties_path(material_manager)
    if path is None:
        raise ValueError("Model hash_id is not available.")

    library_materials = Material_Property.get_all_materials()
    library_by_name = {
        str(getattr(item, "name", "")).strip(): item
        for item in library_materials
        if str(getattr(item, "name", "")).strip()
    }
    referenced_names = sorted(_collect_referenced_material_names(objects))
    material_records = []
    for name in referenced_names:
        material_property = library_by_name.get(name)
        if material_property is None:
            continue
        material_records.append(material_property.to_dict())

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_file": str(source_file) if source_file else None,
        "model_hash_id": _get_model_hash_id(material_manager),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "material_count": len(material_records),
        "referenced_material_names": referenced_names,
        "materials": material_records,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_material_properties(material_manager):
    path = get_material_properties_path(material_manager)
    result = {
        "path": path,
        "found": False,
        "loaded_count": 0,
    }
    if path is None:
        return result
    if not path.exists():
        legacy_path = _get_legacy_material_properties_path(material_manager)
        if legacy_path is None or not legacy_path.exists():
            return result
        path = legacy_path
        result["path"] = path
    if not path.exists():
        return result

    payload = json.loads(path.read_text(encoding="utf-8"))
    materials = payload.get("materials", [])
    current_materials = list(Material_Property.get_all_materials() or [])
    existing_by_name = {
        str(getattr(item, "name", "")).strip(): item
        for item in current_materials
        if str(getattr(item, "name", "")).strip()
    }

    for record in materials:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "").strip()
        if not name:
            continue
        existing = existing_by_name.get(name)
        if existing is None:
            current_materials.append(Material_Property(**record))
            existing_by_name[name] = current_materials[-1]
            result["loaded_count"] += 1
            continue
        for key, value in record.items():
            if key.startswith("_"):
                continue
            setattr(existing, key, value)
        result["loaded_count"] += 1

    Material_Property._materials_db = current_materials
    result["found"] = True
    return result


def save_material_assignment(material_manager, objects, processing_obj_id=None, source_file=None):
    path = get_assignment_path(material_manager)
    if path is None:
        raise ValueError("Model hash_id is not available.")

    path.parent.mkdir(parents=True, exist_ok=True)
    save_material_properties(material_manager, objects, source_file=source_file)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_file": str(source_file) if source_file else None,
        "model_hash_id": _get_model_hash_id(material_manager),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "processing_obj_id": processing_obj_id,
        "object_count": len(objects or {}),
        "objects": {},
    }

    for oid in sorted((objects or {}).keys()):
        obj = objects.get(oid)
        payload["objects"][str(int(oid))] = _serialize_material(getattr(obj, "material", None), obj, oid)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _build_material_from_payload(record, oid, obj, objects):
    if not isinstance(record, dict):
        return PendingMaterial()

    material_type = str(record.get("material_type") or "PendingMaterial")
    material_name = str(record.get("material_name") or "Pending").strip() or "Pending"

    if material_name == "Pending" or material_type == "PendingMaterial":
        return PendingMaterial(material_name)

    if material_type == "IsolationMaterial":
        return IsolationMaterial(material_name=material_name)

    if material_type == "SourceMaterial":
        restricted_materials = record.get("restricted_materials")
        if restricted_materials is not None and not isinstance(restricted_materials, list):
            restricted_materials = [restricted_materials]
        return SourceMaterial(
            material_name=material_name,
            material_composition=None,
            restricted_materials=restricted_materials,
        )

    if material_type == "GradientMaterial":
        nbr_obj_ids = _normalize_oid_list(getattr(obj, "nbr_objects", []) or [])
        include = [x for x in _normalize_oid_list(record.get("include")) if x in nbr_obj_ids]
        exclude = [x for x in _normalize_oid_list(record.get("exclude")) if x in nbr_obj_ids]

        if len(include) < 2 and len(nbr_obj_ids) >= 2:
            include = nbr_obj_ids[:2]
        elif not include:
            include = list(nbr_obj_ids)

        exclude = [x for x in exclude if x not in set(include)]
        center_include_exclude_dict = {
            "center": int(oid),
            "include": include if include else None,
            "exclude": exclude,
        }
        obj._geom_objects = objects
        material = GradientMaterial(material_name, obj, center_include_exclude_dict)
        material._center_include_exclude_dict = center_include_exclude_dict
        return material

    return PendingMaterial()


def load_material_assignment(material_manager, objects):
    path = get_assignment_path(material_manager)
    result = {
        "path": path,
        "found": False,
        "loaded_count": 0,
        "material_properties_loaded": 0,
        "processing_obj_id": None,
    }
    props_result = load_material_properties(material_manager)
    result["material_properties_loaded"] = props_result.get("loaded_count", 0)
    if path is None:
        return result
    if not path.exists():
        legacy_path = _get_legacy_assignment_path(material_manager)
        if legacy_path is None or not legacy_path.exists():
            return result
        path = legacy_path
        result["path"] = path
    if not path.exists():
        return result

    payload = json.loads(path.read_text(encoding="utf-8"))
    result["found"] = True

    object_records = payload.get("objects", {})
    for raw_oid, record in object_records.items():
        try:
            oid = int(raw_oid)
        except Exception:
            continue
        obj = (objects or {}).get(oid)
        if obj is None:
            continue
        obj.material = _build_material_from_payload(record, oid, obj, objects)
        result["loaded_count"] += 1

    try:
        processing_obj_id = int(payload.get("processing_obj_id"))
    except Exception:
        processing_obj_id = None

    if processing_obj_id in (objects or {}):
        result["processing_obj_id"] = processing_obj_id

    return result
