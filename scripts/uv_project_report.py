import json
import os
import subprocess
import tomllib
from pathlib import Path


def find_projects(root: Path):
    for p in root.rglob("pyproject.toml"):
        parent = p.parent
        sp = str(parent)
        if ".venv" in sp or "site-packages" in sp or "\\Lib\\" in sp:
            continue
        yield parent


def read_pyproject(path: Path):
    with open(path / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def count_dependencies(data: dict) -> int:
    deps = data.get("project", {}).get("dependencies", []) or []
    groups = data.get("dependency-groups", {})
    group_deps = sum(len(v or []) for v in groups.values())
    return len(deps) + group_deps


def uv_version() -> str:
    try:
        out = subprocess.check_output(["uv", "--version"], text=True).strip()
        return out
    except Exception as e:
        return f"uv unavailable: {e}"


def main():
    root = Path(".").resolve()
    rows = []
    for proj in find_projects(root):
        data = read_pyproject(proj)
        name = data.get("project", {}).get("name", proj.name)
        py_req = data.get("project", {}).get("requires-python", "")
        dep_count = count_dependencies(data)
        rows.append({"project": name, "path": str(proj), "python": py_req, "dependencies": dep_count})

    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "uv_project_inventory.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("project,path,python,dependencies\n")
        for r in rows:
            f.write(f"{r['project']},{r['path']},{r['python']},{r['dependencies']}\n")

    meta = {"uv": uv_version(), "project_count": len(rows)}
    with open(out_dir / "uv_project_inventory_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
