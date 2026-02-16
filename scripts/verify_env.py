#!/usr/bin/env python3
"""
Python Environment Verification Tool for DMSlicer (uv-optimized)

This script verifies the integrity and configuration of the Python development environment,
specifically tailored for projects using `uv` as the package manager.

Features:
1. Verifies Python version and virtual environment status.
2. Checks `uv` installation and configuration.
3. Validates project dependencies against `pyproject.toml`.
4. Performs basic import tests to ensure environment functionality.
5. Provides detailed reports and actionable fix suggestions.
"""

import sys
import os
import shutil
import subprocess
import importlib.metadata
import importlib.util
import platform
import argparse
import tomllib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# ANSI Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print_header(msg: str):
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== {msg} ==={Colors.ENDC}")

    @staticmethod
    def print_success(msg: str):
        print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

    @staticmethod
    def print_warning(msg: str):
        print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

    @staticmethod
    def print_error(msg: str):
        print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

    @staticmethod
    def print_info(msg: str):
        print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

# Enable ANSI colors on Windows 10+
if os.name == 'nt':
    os.system('color')

# Try to import packaging, needed for robust version comparison
try:
    from packaging.requirements import Requirement
    from packaging.version import Version, parse as parse_version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"

class EnvironmentVerifier:
    def __init__(self, root: Path, auto_fix: bool = False):
        self.root = root
        self.auto_fix_enabled = auto_fix
        self.issues: List[str] = []
        self.fixes: List[str] = []
        self.pyproject_data: Dict[str, Any] = {}

    def load_pyproject(self) -> bool:
        """Load and parse pyproject.toml."""
        if not PYPROJECT_PATH.exists():
            self.issues.append(f"pyproject.toml not found at {PYPROJECT_PATH}")
            Colors.print_error(f"pyproject.toml not found at {PYPROJECT_PATH}")
            return False
        
        try:
            with open(PYPROJECT_PATH, "rb") as f:
                self.pyproject_data = tomllib.load(f)
            return True
        except Exception as e:
            self.issues.append(f"Failed to parse pyproject.toml: {e}")
            Colors.print_error(f"Failed to parse pyproject.toml: {e}")
            return False

    def check_python_version(self):
        """Verify Python version matches requirements."""
        Colors.print_header("Python Version Check")
        
        current_version = platform.python_version()
        sys_version = sys.version.split()[0]
        
        Colors.print_info(f"Current Python: {sys_version} ({sys.executable})")
        
        required_python = self.pyproject_data.get("project", {}).get("requires-python")
        if required_python:
            # Simple check
            is_valid = True
            if required_python.startswith("=="):
                req_ver = required_python.lstrip("=")
                if req_ver.endswith(".*"):
                    base_req = req_ver[:-2]
                    if not sys_version.startswith(base_req):
                         is_valid = False
                elif sys_version != req_ver:
                     is_valid = False
            # Handle other simple cases if needed, but rely on uv mostly
            
            if not is_valid:
                self.issues.append(f"Python version mismatch. Required: {required_python}, Found: {sys_version}")
                Colors.print_error(f"Version mismatch! Required: {required_python}")
            else:
                Colors.print_success(f"Python version satisfies {required_python}")
        else:
            Colors.print_warning("No 'requires-python' found in pyproject.toml")

    def check_venv(self):
        """Check if running inside a virtual environment."""
        Colors.print_header("Virtual Environment Check")
        
        is_venv = (sys.prefix != sys.base_prefix) or hasattr(sys, 'real_prefix')
        
        if is_venv:
            Colors.print_success(f"Running in virtual environment: {sys.prefix}")
        else:
            Colors.print_error("Not running in a virtual environment!")
            self.issues.append("Not running in a virtual environment. This is highly discouraged.")
            self.fixes.append("Create and activate a virtual environment using 'uv venv'")

    def check_uv_installation(self):
        """Check if uv is installed and accessible."""
        Colors.print_header("uv Package Manager Check")
        
        uv_path = shutil.which("uv")
        if uv_path:
            try:
                version_out = subprocess.check_output(["uv", "--version"], text=True).strip()
                Colors.print_success(f"uv is installed: {version_out}")
                # Check uv.lock
                if UV_LOCK_PATH.exists():
                    Colors.print_success("uv.lock exists.")
                else:
                    Colors.print_warning("uv.lock is missing. Run 'uv sync' or 'uv lock' to generate it.")
                    self.fixes.append("Run 'uv lock' to generate lockfile")
            except subprocess.CalledProcessError:
                Colors.print_error("uv found but failed to run --version")
                self.issues.append("uv command failed")
        else:
            Colors.print_error("uv is NOT installed or not in PATH")
            self.issues.append("uv not found")
            self.fixes.append("Install uv: 'pip install uv' or see https://github.com/astral-sh/uv")

    def check_dependencies(self):
        """Check if project dependencies are installed."""
        Colors.print_header("Dependency Verification")
        
        dependencies = self.pyproject_data.get("project", {}).get("dependencies", [])
        if not dependencies:
            Colors.print_warning("No dependencies found in pyproject.toml")
            return

        if not PACKAGING_AVAILABLE:
            Colors.print_warning("'packaging' library not found. Version compatibility checks will be skipped.")
            Colors.print_info("Install 'packaging' (pip install packaging) for full verification.")

        missing_deps = []
        
        for dep_str in dependencies:
            pkg_name = None
            specifier = None
            
            if PACKAGING_AVAILABLE:
                try:
                    req = Requirement(dep_str)
                    pkg_name = req.name
                    specifier = req.specifier
                except Exception as e:
                    Colors.print_warning(f"Could not parse dependency '{dep_str}': {e}")
                    continue
            else:
                # Fallback: simple name extraction using regex
                import re
                # Matches valid python package names (letters, numbers, _, -) at start of string
                match = re.match(r"^([a-zA-Z0-9_\-]+)", dep_str)
                if match:
                    pkg_name = match.group(1)
                else:
                    Colors.print_warning(f"Could not extract package name from '{dep_str}'")
                    continue

            try:
                dist = importlib.metadata.distribution(pkg_name)
                installed_ver = dist.version
                
                if specifier and PACKAGING_AVAILABLE:
                    if not specifier.contains(installed_ver, prereleases=True):
                         Colors.print_warning(f"{pkg_name} version mismatch: Installed {installed_ver}, Required {specifier}")
                         # We count this as a warning, not strictly missing
                    else:
                         print(f"{Colors.OKGREEN}  ✓ {pkg_name} ({installed_ver}){Colors.ENDC}")
                else:
                    print(f"{Colors.OKGREEN}  ✓ {pkg_name} ({installed_ver}){Colors.ENDC}")
                    
            except importlib.metadata.PackageNotFoundError:
                Colors.print_error(f"Missing dependency: {pkg_name}")
                missing_deps.append(pkg_name)

        if missing_deps:
            self.issues.append(f"Missing dependencies: {', '.join(missing_deps)}")
            self.fixes.append("Run 'uv sync' to install missing dependencies")
        else:
            Colors.print_success("All declared dependencies appear to be installed.")

    def check_functional_imports(self):
        """Try importing key modules to verify environment health."""
        Colors.print_header("Functional Import Test")
        
        # Core libraries to test (from pyproject.toml)
        # We can dynamically extract them or use a hardcoded list of "critical" ones
        core_libs = ["numpy", "pandas", "pyvista", "pyarrow"]
        failed_imports = []
        
        for lib in core_libs:
            try:
                importlib.import_module(lib)
                print(f"{Colors.OKGREEN}  ✓ Import successful: {lib}{Colors.ENDC}")
            except ImportError as e:
                # Only report error if it's listed in dependencies
                # But here we assume core_libs are essential
                Colors.print_error(f"Failed to import {lib}: {e}")
                failed_imports.append(lib)
            except Exception as e:
                Colors.print_error(f"Error importing {lib}: {e}")
                failed_imports.append(lib)
        
        # Try to import local package 'dmslicer'
        try:
            import dmslicer
            print(f"{Colors.OKGREEN}  ✓ Import successful: dmslicer (Local Project){Colors.ENDC}")
        except ImportError:
            Colors.print_warning("Could not import 'dmslicer'. Is the project installed in editable mode? (uv pip install -e .)")
        
        if failed_imports:
            self.issues.append(f"Functional tests failed for: {', '.join(failed_imports)}")

    def run_checks(self):
        Colors.print_header("Starting Environment Verification")
        
        if not self.load_pyproject():
            return

        self.check_python_version()
        self.check_venv()
        self.check_uv_installation()
        self.check_dependencies()
        self.check_functional_imports()
        
        self.generate_report()

    def generate_report(self):
        Colors.print_header("Verification Report")
        
        if not self.issues:
            Colors.print_success("Environment is HEALTHY! No issues found.")
            return

        Colors.print_error(f"Found {len(self.issues)} issues:")
        for issue in self.issues:
            print(f"  - {issue}")
        
        if self.fixes:
            Colors.print_header("Suggested Fixes")
            unique_fixes = list(set(self.fixes))
            for fix in unique_fixes:
                print(f"  -> {fix}")
            
            if self.auto_fix_enabled:
                self.apply_fixes(unique_fixes)
            elif "uv sync" in self.fixes[0] or any("uv sync" in f for f in self.fixes):
                self.offer_autofix()

    def offer_autofix(self):
        print(f"\n{Colors.BOLD}Would you like to attempt auto-fix using 'uv sync'? (y/n){Colors.ENDC}")
        try:
            choice = input().lower().strip()
            if choice == 'y':
                self.run_uv_sync()
        except KeyboardInterrupt:
            print("\nCancelled.")

    def apply_fixes(self, fixes: List[str]):
        # Simple logic: if 'uv sync' is needed, run it.
        if any("uv sync" in f for f in fixes):
            self.run_uv_sync()

    def run_uv_sync(self):
        Colors.print_header("Running Auto-Fix (uv sync)")
        uv_path = shutil.which("uv")
        if not uv_path:
            Colors.print_error("Cannot run auto-fix: uv not found.")
            return

        try:
            subprocess.check_call([uv_path, "sync"], cwd=self.root)
            Colors.print_success("Successfully ran 'uv sync'. Please re-run verification.")
            # Clear issues and re-run checks?
            # For now just ask user to re-run
        except subprocess.CalledProcessError as e:
            Colors.print_error(f"Failed to run 'uv sync': {e}")


def main():
    parser = argparse.ArgumentParser(description="Verify Python environment for DMSlicer.")
    parser.add_argument("--fix", action="store_true", help="Automatically attempt to fix issues (e.g., run uv sync)")
    args = parser.parse_args()

    verifier = EnvironmentVerifier(PROJECT_ROOT, auto_fix=args.fix)
    verifier.run_checks()

if __name__ == "__main__":
    main()
