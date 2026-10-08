#!/usr/bin/env python3
"""
Backwards-compatible entry point for MARS prediction.

The implementation moved into the installed package as of MARS 4.0.0. Prefer
the installed commands, which work from any working directory:

    mars predict -i feature_dir --mod-type m6a --pretrained curlcake_DRACH_2244test
    mars-predict -i feature_dir -model model.json
    xgb_model_pred -i feature_dir -model model.json

This shim keeps `python scripts/model/xgb_model_pred.py ...` working for
existing scripts and for anyone following the MARS 3.x documentation.

Author: MARS Team
"""

import os
import sys

# Allow running straight from a checkout that has not been pip-installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mars.model.predict import cli_main  # noqa: E402

if __name__ == "__main__":
    print(
        "note: `python scripts/model/xgb_model_pred.py` is kept for compatibility; "
        "`mars predict` (or `xgb_model_pred`) is the supported entry point.",
        file=sys.stderr,
    )
    cli_main()
