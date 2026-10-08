#!/usr/bin/env python3
"""
Backwards-compatible entry point for the MARS pipeline.

The implementation moved into the installed package as of MARS 4.0.0. Prefer
the installed commands, which work from any working directory:

    mars run input.fastq reference.fa DRACH output_dir --num_threads 20
    mars-pipeline input.fastq reference.fa DRACH output_dir --gpu

This shim keeps `python MARS_main.py ...` working for existing scripts and
for anyone following the MARS 3.x documentation.

Author: MARS Team
"""

import os
import sys

# Allow running straight from a checkout that has not been pip-installed.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mars.pipeline import cli_main  # noqa: E402

if __name__ == "__main__":
    print(
        "note: `python MARS_main.py` is kept for compatibility; "
        "`mars run` is the supported entry point.",
        file=sys.stderr,
    )
    cli_main()
