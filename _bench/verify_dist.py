#!/usr/bin/env python3
"""
Build the distributions and check what ended up inside them.

A release is only reproducible if the artifact a user installs carries the
things it needs and none of the things it does not: the package itself and the
pretrained models, but not the benchmark harness, caches or pipeline outputs.
Version metadata has to agree with mars/_version.py too, since that is the
single source the CLI and the release workflow both read.

Invoked directly rather than through the `build` frontend so it works without
network access.
"""

import re
import sys
import shutil
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mars_dist")

# Paths that must never reach a distribution. An .egg-info directory is not on
# the list: setuptools writes its own metadata into every sdist by design.
FORBIDDEN = ("_bench/", "__pycache__/", ".pytest_cache/", "data/pred/",
             ".git/")
# Paths a user needs for the package to be usable.
REQUIRED_SDIST = ("mars/_version.py", "mars/cli.py", "nfl_py/boosting.py",
                  "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE",
                  "CITATION.cff")


def version_from_source():
    text = (REPO / "mars" / "_version.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    sys.path.insert(0, str(REPO))
    cwd = Path.cwd()
    try:
        import os
        os.chdir(REPO)
        from setuptools import build_meta
        sdist = build_meta.build_sdist(str(OUT))
        wheel = build_meta.build_wheel(str(OUT))
    finally:
        os.chdir(cwd)
    return OUT / sdist, OUT / wheel


def check(names, label, required=()):
    problems = []
    for bad in FORBIDDEN:
        hits = [n for n in names if bad in n]
        if hits:
            problems.append(f"{label} contains {bad} ({len(hits)} entries, "
                            f"e.g. {hits[0]})")
    for need in required:
        if not any(n.endswith(need) for n in names):
            problems.append(f"{label} is missing {need}")
    return problems


def main():
    version = version_from_source()
    sdist, wheel = build()

    with tarfile.open(sdist) as t:
        sdist_names = t.getnames()
    with zipfile.ZipFile(wheel) as z:
        wheel_names = z.namelist()

    problems = []
    problems += check(sdist_names, "sdist", REQUIRED_SDIST)
    problems += check(wheel_names, "wheel", ("mars/cli.py", "nfl_py/boosting.py"))

    for path, kind in ((sdist, "sdist"), (wheel, "wheel")):
        if version not in path.name:
            problems.append(f"{kind} filename {path.name} does not carry "
                            f"version {version}")

    models = [n for n in sdist_names if "pretrained_models/" in n
              and n.endswith(".json")]
    tests = [n for n in sdist_names if "/tests/" in n]

    print(f"version           {version}")
    print(f"sdist             {sdist.name}  "
          f"({sdist.stat().st_size / 1e6:.2f} MB, {len(sdist_names)} entries)")
    print(f"wheel             {wheel.name}  "
          f"({wheel.stat().st_size / 1e6:.2f} MB, {len(wheel_names)} entries)")
    print(f"pretrained models {len(models)} in sdist")
    print(f"tests             {len(tests)} in sdist")

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: no forbidden paths, all required files present, "
          "versions consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
