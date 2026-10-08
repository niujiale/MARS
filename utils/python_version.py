#!/usr/bin/env python3
"""
Check Python Package Versions

Display version information for all packages used by MARS.
Useful for troubleshooting and environment verification.

Usage:
    python python_version.py

Author: MARS Team
"""

import sys
import platform
import importlib
import importlib.metadata
from typing import Dict, Optional

# Packages used by MARS
REQUIRED_PACKAGES = [
    # Standard library (always available)
    ("os", "Standard Library"),
    ("sys", "Standard Library"),
    ("re", "Standard Library"),
    ("csv", "Standard Library"),
    ("time", "Standard Library"),
    ("glob", "Standard Library"),
    ("argparse", "Standard Library"),
    ("subprocess", "Standard Library"),
    ("multiprocessing", "Standard Library"),
    ("threading", "Standard Library"),
    ("concurrent.futures", "Standard Library"),
    ("logging", "Standard Library"),
    ("gc", "Standard Library"),

    # Third-party packages
    ("numpy", None),
    ("pandas", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("xgboost", None),
    ("sklearn", "scikit-learn"),  # Package name differs from import name
    ("scipy", None),
    ("pysam", None),
    ("tqdm", None),
    ("Bio", "biopython"),  # BioPython
]

OPTIONAL_PACKAGES = [
    ("torch", None),
    ("tensorflow", None),
    ("h5py", None),
]


def get_package_version(import_name: str, package_name: Optional[str] = None) -> str:
    """
    Get version of a package.

    Args:
        import_name: Name used for importing
        package_name: Name used in pip/conda (if different from import name)

    Returns:
        Version string or status message
    """
    actual_package = package_name or import_name

    # Handle standard library modules
    if actual_package == "Standard Library":
        return "Built-in"

    try:
        # Try to import the module
        module = importlib.import_module(import_name)

        # Try to get version from module attribute
        version = getattr(module, '__version__', None)
        if version:
            return version

        # Try importlib.metadata
        try:
            return importlib.metadata.version(actual_package)
        except importlib.metadata.PackageNotFoundError:
            return "Installed (version unknown)"

    except ImportError:
        return "Not installed"
    except Exception as e:
        return f"Error: {e}"


def print_system_info() -> None:
    """Print system and Python information."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"{'Platform:':<20} {platform.platform()}")
    print(f"{'Python Version:':<20} {platform.python_version()}")
    print(f"{'Python Path:':<20} {sys.executable}")
    print()


def print_package_versions(
    packages: list,
    title: str = "Package Versions"
) -> Dict[str, str]:
    """
    Print version information for a list of packages.

    Args:
        packages: List of (import_name, package_name) tuples
        title: Section title

    Returns:
        Dictionary of package names to versions
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'Package':<25} {'Version':<20}")
    print("-" * 45)

    versions = {}
    for import_name, package_name in packages:
        version = get_package_version(import_name, package_name)
        display_name = package_name or import_name
        print(f"{display_name:<25} {version:<20}")
        versions[display_name] = version

    print()
    return versions


def check_critical_packages() -> bool:
    """
    Check if all critical packages are installed.

    Returns:
        True if all critical packages are available
    """
    critical = ['numpy', 'pandas', 'xgboost', 'pysam']
    missing = []

    for pkg in critical:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("=" * 60)
        print("WARNING: Missing Critical Packages")
        print("=" * 60)
        print(f"The following critical packages are not installed: {', '.join(missing)}")
        print("Please install them using:")
        print(f"  pip install {' '.join(missing)}")
        print()
        return False

    return True


def main() -> None:
    """Main function."""
    print_system_info()
    print_package_versions(REQUIRED_PACKAGES, "Required Packages")
    print_package_versions(OPTIONAL_PACKAGES, "Optional Packages")

    all_ok = check_critical_packages()

    if all_ok:
        print("All critical packages are installed.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
