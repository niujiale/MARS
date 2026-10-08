"""
Tests for the installed command-line surface and version single-sourcing.

MARS 4.0.0 replaced "find the script and run it with python" with real console
scripts, and made `mars/_version.py` the single source of the version number.
Both are easy to break silently — a renamed function leaves a console script
that only fails when a user runs it, and a hand-edited version in one file
leaves `mars --version` disagreeing with the distribution metadata. These
tests fail fast on either.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mars._version import __version__

DISTRIBUTION = "mars-nanopore"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Every command that must exist after installation. `xgb_model_pred` and
# `xgb_model` are the MARS 3.x script names, kept so published protocols and
# the reviewers' own notes keep working.
EXPECTED_COMMANDS = [
    "mars",
    "mars-pipeline",
    "mars-predict",
    "mars-train",
    "mars-doctor",
    "xgb_model_pred",
    "xgb_model",
    "nfl-py",
]


def _console_scripts():
    """Entry points the installed distribution declares, keyed by name."""
    try:
        dist = md.distribution(DISTRIBUTION)
    except md.PackageNotFoundError:
        pytest.skip(
            f"{DISTRIBUTION} is not installed; run `pip install -e .` first"
        )
    return {
        ep.name: ep
        for ep in dist.entry_points
        if ep.group == "console_scripts"
    }


def test_version_is_single_sourced():
    """Distribution metadata, mars and nfl_py must all report one version."""
    try:
        dist_version = md.version(DISTRIBUTION)
    except md.PackageNotFoundError:
        pytest.skip(f"{DISTRIBUTION} is not installed")

    import mars
    import nfl_py

    assert dist_version == __version__
    assert mars.__version__ == __version__
    assert nfl_py.__version__ == __version__


def test_changelog_documents_current_version():
    """A release must be described in the changelog before it is tagged."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in changelog, (
        f"CHANGELOG.md has no '## [{__version__}]' section"
    )


def test_citation_matches_version():
    """CITATION.cff is what users copy into papers; keep it in step."""
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {__version__}" in citation


def test_all_expected_console_scripts_are_declared():
    declared = _console_scripts()
    missing = [name for name in EXPECTED_COMMANDS if name not in declared]
    assert not missing, f"console scripts missing from the distribution: {missing}"


@pytest.mark.parametrize("name", EXPECTED_COMMANDS)
def test_console_script_target_is_importable(name):
    """Each entry point must resolve to a real callable, not a stale path."""
    entry_point = _console_scripts().get(name)
    if entry_point is None:
        pytest.fail(f"console script '{name}' is not declared")
    target = entry_point.load()
    assert callable(target), f"entry point '{name}' does not resolve to a callable"


@pytest.mark.parametrize("name", EXPECTED_COMMANDS)
def test_console_script_is_on_path_and_reports_version(name):
    """The whole point of the change: the command works from any directory."""
    if shutil.which(name) is None:
        pytest.skip(f"'{name}' not on PATH (non-editable or partial install)")

    # Run from a directory that is not the repo, which is exactly the situation
    # the old `python scripts/model/xgb_model_pred.py` workflow could not handle.
    result = subprocess.run(
        [name, "--version"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=Path.home(),
    )
    assert result.returncode == 0, (
        f"`{name} --version` exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert __version__ in result.stdout + result.stderr


def test_mars_umbrella_help_lists_subcommands():
    from mars.cli import USAGE, _COMMANDS

    for command in _COMMANDS:
        assert command in USAGE


@pytest.mark.parametrize("command", ["run", "predict", "train"])
def test_mars_subcommand_help_exits_cleanly(command):
    """`mars <cmd> --help` must forward to the real parser, not the dispatcher."""
    result = subprocess.run(
        [sys.executable, "-m", "mars.cli", command, "--help"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert f"mars {command}" in result.stdout


def test_pretrained_models_are_discoverable():
    """Prediction with --pretrained is useless if the models can't be found."""
    from mars.model.predict import get_pretrained_models
    from mars.paths import describe_pretrained_lookup

    models = get_pretrained_models("m6a")
    assert models, (
        "no m6a pretrained models discovered\n" + describe_pretrained_lookup()
    )
    for path in models.values():
        assert Path(path).is_file()


def test_pretrained_dir_override_is_honoured(tmp_path):
    from mars.paths import find_pretrained_models_dir

    (tmp_path / "m6a").mkdir()
    assert find_pretrained_models_dir(str(tmp_path)) == tmp_path

    with pytest.raises(FileNotFoundError):
        find_pretrained_models_dir(str(tmp_path / "does-not-exist"))


@pytest.mark.parametrize(
    "module",
    ["mars.cli", "mars.paths", "mars.doctor", "mars.pipeline",
     "mars.model.predict", "mars.model.train"],
)
def test_package_modules_import(module):
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize(
    "script",
    ["MARS_main.py", "scripts/model/xgb_model_pred.py", "scripts/model/xgb_model.py"],
)
def test_legacy_scripts_still_expose_an_entry_point(script):
    """MARS 3.x invocations must keep working, just with a deprecation note."""
    path = REPO_ROOT / script
    assert path.is_file(), f"legacy shim removed: {script}"

    result = subprocess.run(
        [sys.executable, str(path), "--version"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=Path.home(),
    )
    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout + result.stderr
