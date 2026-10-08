#!/usr/bin/env python3
"""
NFL-Py Main Entry Point

Allows running the package as a module:
    python -m nfl_py prepdata ...
    python -m nfl_py train ...
    python -m nfl_py predict ...

Author: MARS Team
"""

from .cli import main

if __name__ == '__main__':
    main()
