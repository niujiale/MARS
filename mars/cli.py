#!/usr/bin/env python3
"""
MARS umbrella command-line interface.

Every stage of the pipeline is reachable both as a `mars <command>` subcommand
and as a standalone console script, so users never have to know where the
source tree lives:

    mars run       input.fastq ref.fa DRACH out/        (= mars-pipeline)
    mars predict   -i out/ --mod-type m6a --pretrained curlcake_DRACH_2244test
                                                        (= mars-predict,
                                                           xgb_model_pred)
    mars train     -i features/ -o model_out/           (= mars-train,
                                                           xgb_model)
    mars prepdata  reads.bam ref.fa loci.tsv out/       (= nfl-py prepdata)
    mars list-models --mod-type m6a
    mars doctor
    mars version

Subcommand arguments are forwarded verbatim to the underlying tool, so
`mars predict --help` shows exactly the same options as `mars-predict --help`.

Author: MARS Team
"""

import sys
from typing import List, Optional

from ._version import __version__

# command -> (one-line help, "module:function" resolved lazily)
_COMMANDS = {
    "run": ("Align reads and extract features (FASTQ -> NPZ features)",
            "mars.pipeline:main_with_prog"),
    "predict": ("Apply a trained/pretrained XGBoost model to extracted features",
                "mars.model.predict:main_with_prog"),
    "train": ("Train and evaluate a new XGBoost model",
              "mars.model.train:main_with_prog"),
    "prepdata": ("Extract features from an existing sorted BAM",
                 "mars.cli:_run_prepdata"),
    "list-models": ("List the bundled pretrained models",
                    "mars.cli:_run_list_models"),
    "doctor": ("Report versions, dependencies, GPU and model paths",
               "mars.doctor:main_with_prog"),
    "version": ("Print the MARS version", "mars.cli:_run_version"),
}

_ALIASES = {
    "pipeline": "run",
    "features": "run",
    "pred": "predict",
    "models": "list-models",
    "env": "doctor",
}

USAGE = f"""MARS {__version__} - RNA modification detection from Nanopore sequencing

Usage:
  mars <command> [options]

Commands:
{chr(10).join(f'  {name:<13} {help_text}' for name, (help_text, _) in _COMMANDS.items())}

Run `mars <command> --help` for the options of a single command.

Equivalent standalone commands (same options, no `mars` prefix needed):
  mars-pipeline, mars-predict, mars-train, xgb_model_pred, xgb_model, nfl-py

Documentation: https://github.com/MARS-lab/MARS#readme
"""


def _run_version(argv: Optional[List[str]] = None) -> None:
    print(f"MARS {__version__}")


def _run_list_models(argv: Optional[List[str]] = None) -> None:
    """`mars list-models` is `mars predict --list-models` without -i."""
    from .model.predict import build_parser, list_pretrained_models

    parser = build_parser(prog="mars list-models")
    args = parser.parse_args(list(argv or []) + ["--list-models"])
    list_pretrained_models(args.mod_type, args.pretrained_dir)


def _run_prepdata(argv: Optional[List[str]] = None) -> None:
    """Delegate to the nfl_py prepdata subcommand."""
    from nfl_py.cli import main as nfl_main

    nfl_main(["prepdata"] + list(argv or []))


def _resolve(target: str):
    """Import `module:function` lazily so `mars --help` stays fast."""
    module_name, func_name = target.split(":")
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def main(argv: Optional[List[str]] = None) -> None:
    """Dispatch to a subcommand."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return

    if args[0] in ("-V", "--version"):
        _run_version()
        return

    command = _ALIASES.get(args[0], args[0])
    rest = args[1:]

    if command not in _COMMANDS:
        print(f"mars: unknown command '{args[0]}'\n", file=sys.stderr)
        print(USAGE, end="", file=sys.stderr)
        sys.exit(2)

    _resolve(_COMMANDS[command][1])(rest)


if __name__ == "__main__":
    main()
