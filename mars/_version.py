"""Single source of truth for the MARS version.

`pyproject.toml` reads this via `[tool.setuptools.dynamic]`, and every CLI
exposes it through `--version`, so a release tag, the installed distribution
metadata and `mars version` can never drift apart.
"""

__version__ = "4.0.0"
