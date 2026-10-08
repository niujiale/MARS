"""
End-to-end alignment regression tests for the MARS prediction pipeline.

These tests exist because several refactors between MARS 2.x and 3.x moved
position metadata (chromosome, 1-based coordinate, strand, read depth) from
a sidecar `combined.txt` file into the NPZ feature archives and then into
the final `modification_pred.txt`. A small ordering bug anywhere along that
path would silently pair position A's coordinates with position B's
modification-rate prediction — the kind of error you'd only notice when a
downstream analyst asks why their m6A site keeps landing on a U.

What we verify (for every row of `modification_pred.txt`):

  1. `(trans, pos)` resolves to exactly one feature row across all NPZ
     batch files produced by feature extraction.
  2. `strand` and `depth` in the prediction file agree with the NPZ
     metadata at that position.
  3. Running the XGBoost model on that feature row in isolation reproduces
     the `y_pred` value printed in `modification_pred.txt` bit-for-bit.

We run this under both the default single-batch layout AND a forced
multi-batch layout (`--batch 5` yields 8 X_*.npz files for the example
data) so that any shuffle introduced by vstack/concatenation ordering
across NPZ files gets caught.

Run:   pytest tests/                 (from repo root, inside the MARS env)
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from scipy.sparse import csr_matrix


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_OUT = REPO_ROOT / "data" / "example" / "output"
PRED_FILE = EXAMPLE_OUT / "modification_pred" / "modification_pred.txt"
NPZ_GLOB = str(EXAMPLE_OUT / "feature" / "**" / "forward" / "Xdata" / "*.npz")
MODEL_PATH = (
    REPO_ROOT
    / "pretrained_models"
    / "m6a"
    / "curlcake_json"
    / "curlcake_DRACH_2244test_final_model.json"
)
RUN_EXAMPLE = REPO_ROOT / "run_example.sh"


def _require_binaries() -> None:
    """Skip end-to-end tests when the external tools are not installed."""
    missing = [name for name in ("minimap2", "samtools") if not shutil.which(name)]
    if missing:
        pytest.skip(f"Required tools not on PATH: {', '.join(missing)}")


def _run_pipeline(batch_size: int) -> None:
    """Execute run_example.sh with a specific batch size, clean state."""
    cmd = ["bash", str(RUN_EXAMPLE), "--clean", "--batch", str(batch_size)]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(
            f"run_example.sh failed (batch={batch_size}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def _load_npz_index() -> Dict[Tuple[str, int], dict]:
    """Build (chromosome, 1-based pos) -> feature-row lookup across all NPZ."""
    index: Dict[Tuple[str, int], dict] = {}
    npz_files = sorted(glob.glob(NPZ_GLOB, recursive=True))
    assert npz_files, f"No NPZ feature files found matching {NPZ_GLOB}"

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        X = csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=tuple(data["shape"]),
        ).toarray()
        chrom = str(data["chromosome"])
        strand = str(data["strand"])
        loci = np.asarray(data["loci"], dtype=np.int64)
        depth = np.asarray(data["depth"], dtype=np.int64)

        assert len(loci) == X.shape[0] == len(depth), (
            f"NPZ row count inconsistency in {npz_path}: "
            f"X.rows={X.shape[0]} loci={len(loci)} depth={len(depth)}"
        )

        for row_i, (pos, dep) in enumerate(zip(loci.tolist(), depth.tolist())):
            key = (chrom, int(pos))
            assert key not in index, (
                f"Duplicate (chrom, pos)={key}: appears in both "
                f"{index[key]['npz']} and {npz_path}"
            )
            index[key] = {
                "feat": X[row_i],
                "strand": strand,
                "depth": int(dep),
                "npz": npz_path,
                "row_in_npz": row_i,
            }
    return index


def _verify_alignment() -> None:
    """Assert every prediction row matches the NPZ-indexed feature at that pos."""
    assert PRED_FILE.is_file(), f"Prediction file not found: {PRED_FILE}"
    df = pd.read_csv(PRED_FILE, sep="\t")
    assert list(df.columns) == ["trans", "pos", "strand", "depth", "y_pred", "y_test"], (
        f"Unexpected prediction columns: {list(df.columns)}"
    )
    assert len(df) > 0, "Prediction file has no data rows"

    npz_index = _load_npz_index()
    assert len(df) == len(npz_index), (
        f"Row-count mismatch: prediction file has {len(df)} rows, "
        f"NPZ index has {len(npz_index)} (chrom, pos) entries"
    )

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    for idx, row in df.iterrows():
        key = (row["trans"], int(row["pos"]))
        assert key in npz_index, (
            f"Row {idx}: (trans, pos)={key} in prediction file but not in any NPZ"
        )
        info = npz_index[key]
        assert info["strand"] == row["strand"], (
            f"Row {idx} at {key}: strand pred='{row['strand']}' vs "
            f"NPZ '{info['strand']}' (from {info['npz']})"
        )
        assert info["depth"] == int(row["depth"]), (
            f"Row {idx} at {key}: depth pred={row['depth']} vs "
            f"NPZ {info['depth']} (from {info['npz']})"
        )
        y_pred_file = float(row["y_pred"])
        y_pred_indep = float(np.clip(model.predict(info["feat"][None, :])[0], 0, 1))
        assert np.isclose(y_pred_file, y_pred_indep, atol=1e-6), (
            f"Row {idx} at {key}: y_pred mismatch — file={y_pred_file:.6f} "
            f"independent re-prediction={y_pred_indep:.6f} (from {info['npz']})"
        )


@pytest.mark.parametrize(
    "batch_size, label",
    [
        (100, "single_batch_default"),
        (5, "multi_batch_forced_split"),
    ],
    ids=["default_100", "forced_5"],
)
def test_prediction_rows_align_with_feature_positions(batch_size, label):
    """
    Every row in modification_pred.txt must pair the correct NPZ feature
    row with the correct (trans, pos, strand, depth). Runs the full
    example pipeline end-to-end, then re-predicts each position in
    isolation and diffs the result.
    """
    _require_binaries()
    _run_pipeline(batch_size=batch_size)
    _verify_alignment()
