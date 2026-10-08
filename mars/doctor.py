#!/usr/bin/env python3
"""Environment diagnostics for MARS.

`mars doctor` prints everything needed to reproduce or debug a run: the MARS
version, interpreter, dependency versions, external tool paths, GPU status and
where the pretrained models were found. Paste its output into a bug report.
"""

import importlib
import importlib.metadata
import platform
import shutil
import sys
from typing import List, Optional, Tuple

from ._version import __version__
from .paths import describe_pretrained_lookup, project_root

# (import name, distribution name or None when identical)
CORE_DEPS: List[Tuple[str, Optional[str]]] = [
    ("numpy", None),
    ("scipy", None),
    ("pandas", None),
    ("pysam", None),
    ("xgboost", None),
    ("Bio", "biopython"),
    ("tqdm", None),
]

TRAIN_DEPS: List[Tuple[str, Optional[str]]] = [
    ("matplotlib", None),
    ("seaborn", None),
    ("sklearn", "scikit-learn"),
]

OPTIONAL_DEPS: List[Tuple[str, Optional[str]]] = [
    ("torch", None),
]

EXTERNAL_TOOLS = ["minimap2", "samtools"]


def _version_of(import_name: str, dist_name: Optional[str] = None) -> str:
    """Best-effort version lookup for an installed dependency."""
    try:
        module = importlib.import_module(import_name)
    except ImportError:
        return "NOT INSTALLED"
    except Exception as exc:  # a broken install should not abort the report
        return f"ERROR: {exc}"

    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib.metadata.version(dist_name or import_name)
    except importlib.metadata.PackageNotFoundError:
        return "installed (version unknown)"


def _print_group(title: str, deps: List[Tuple[str, Optional[str]]]) -> List[str]:
    """Print a dependency group; return the names that are missing."""
    print(f"\n{title}")
    print("-" * len(title))
    missing = []
    for import_name, dist_name in deps:
        version = _version_of(import_name, dist_name)
        label = dist_name or import_name
        print(f"  {label:<16} {version}")
        if version == "NOT INSTALLED":
            missing.append(label)
    return missing


def report() -> int:
    """Print the diagnostics report. Returns a process exit code."""
    print("=" * 66)
    print(f"MARS {__version__} - environment report")
    print("=" * 66)

    print("\nSystem")
    print("------")
    print(f"  platform         {platform.platform()}")
    print(f"  python           {platform.python_version()}")
    print(f"  interpreter      {sys.executable}")
    root = project_root()
    print(f"  source tree      {root if root else '(not an editable/source install)'}")

    missing_core = _print_group("Core dependencies", CORE_DEPS)
    _print_group("Training extras (needed by mars train)", TRAIN_DEPS)
    _print_group("Optional", OPTIONAL_DEPS)

    print("\nGPU")
    print("---")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  torch build      {torch.__version__}")
        print(f"  cuda available   {cuda_ok}")
        if cuda_ok:
            print(f"  device           {torch.cuda.get_device_name(0)}")
        else:
            print("  note             feature statistics will run on NumPy (CPU)")
    except ImportError:
        print("  torch            NOT INSTALLED -> statistics run on NumPy (CPU)")

    print("\nExternal tools")
    print("--------------")
    missing_tools = []
    for tool in EXTERNAL_TOOLS:
        path = shutil.which(tool)
        print(f"  {tool:<16} {path if path else 'NOT FOUND on PATH'}")
        if not path:
            missing_tools.append(tool)

    print("\nPretrained models")
    print("-----------------")
    print(describe_pretrained_lookup())

    print("\n" + "=" * 66)
    if missing_core or missing_tools:
        if missing_core:
            print(f"MISSING core dependencies: {', '.join(missing_core)}")
        if missing_tools:
            print(f"MISSING external tools:    {', '.join(missing_tools)}")
            print("  install with: conda install -c bioconda minimap2 samtools")
        print("=" * 66)
        return 1

    print("All required dependencies and tools are available.")
    print("=" * 66)
    return 0


def main(argv: Optional[List[str]] = None, prog: str = "mars-doctor") -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Report MARS version, dependencies, GPU status and model paths.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s (MARS) {__version__}"
    )
    parser.parse_args(argv)
    sys.exit(report())


def main_with_prog(argv: Optional[List[str]] = None) -> None:
    """Entry point for `mars doctor`, so --help shows the right prog name."""
    main(argv, prog="mars doctor")


if __name__ == "__main__":
    main()
