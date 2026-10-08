#!/usr/bin/env python3
"""Allow `python -m mars ...` as an equivalent of the `mars` command.

Useful when a console script is shadowed on PATH, or for pinning a specific
interpreter: `python3.11 -m mars predict -i out/ ...`.
"""

from .cli import main

if __name__ == "__main__":
    main()
