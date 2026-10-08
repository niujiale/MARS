"""Locate the resources MARS ships alongside its code.

The pretrained XGBoost models are ~60 MB of JSON and are deliberately kept
out of the Python wheel. They are present in every git checkout and in every
release tarball, so the resolution order below covers all documented install
paths and gives an actionable message when it cannot find them.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

PRETRAINED_ENV_VAR = "MARS_PRETRAINED_MODELS"

# Marker files that identify a MARS source tree / release tarball root.
_ROOT_MARKERS = ("pretrained_models", "pyproject.toml")


def package_dir() -> Path:
    """Directory containing the installed `mars` package."""
    return Path(__file__).resolve().parent


def project_root() -> Optional[Path]:
    """Root of the MARS source tree, or None for a non-editable install.

    `install.sh` performs an editable install (`pip install -e .`), so the
    package lives inside the checkout and the root is simply its parent.
    """
    candidate = package_dir().parent
    if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
        return candidate
    return None


def _candidate_pretrained_dirs() -> List[Tuple[str, Path]]:
    """(source, path) pairs to probe, in priority order."""
    candidates: List[Tuple[str, Path]] = []

    env_value = os.environ.get(PRETRAINED_ENV_VAR)
    if env_value:
        candidates.append((f"${PRETRAINED_ENV_VAR}", Path(env_value).expanduser()))

    # Bundled next to the package (if a future build ships them as package data).
    candidates.append(("package data", package_dir() / "pretrained_models"))

    root = project_root()
    if root is not None:
        candidates.append(("source tree", root / "pretrained_models"))

    # Data dir of the active environment, e.g. $CONDA_PREFIX/share/mars/...
    prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
    if prefix:
        candidates.append(
            ("environment share dir", Path(prefix) / "share" / "mars" / "pretrained_models")
        )

    return candidates


def find_pretrained_models_dir(override: Optional[str] = None) -> Path:
    """Return the pretrained-models directory.

    Args:
        override: explicit path from a `--pretrained-dir` flag; checked first
            and reported as an error if it does not exist.

    Raises:
        FileNotFoundError: with the full list of probed locations.
    """
    if override:
        path = Path(override).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"--pretrained-dir does not exist: {path}")
        return path

    probed = _candidate_pretrained_dirs()
    for _source, path in probed:
        if path.is_dir():
            return path

    listing = "\n".join(f"  - {source}: {path}" for source, path in probed)
    raise FileNotFoundError(
        "Could not locate the MARS pretrained_models directory.\n"
        f"Probed:\n{listing}\n\n"
        "Fix this by either:\n"
        "  * pointing MARS at the models explicitly:\n"
        f"      export {PRETRAINED_ENV_VAR}=/path/to/MARS/pretrained_models\n"
        "  * passing --pretrained-dir /path/to/MARS/pretrained_models\n"
        "  * installing from a release tarball or git checkout, which bundles them\n"
        "    (see the Installation section of the README)"
    )


def describe_pretrained_lookup() -> str:
    """Human-readable probe report, used by `mars doctor` and error paths."""
    lines = []
    for source, path in _candidate_pretrained_dirs():
        mark = "found" if path.is_dir() else "missing"
        lines.append(f"  [{mark:>7}] {source}: {path}")
    return "\n".join(lines)
