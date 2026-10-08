#!/usr/bin/env python3
"""
Compare two prepdata output trees field by field.

Reports whether the feature matrices are bit-identical, and if not, the worst
absolute difference and how many stored values disagree.

Usage: python compare_outputs.py <ref_dir> <test_dir> [--label NAME] [--tol T]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def npz_files(root):
    return sorted(
        p.relative_to(root) for p in Path(root).rglob("X_*.npz")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--label", default="")
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args()

    ref_root, test_root = Path(args.ref), Path(args.test)
    ref_files, test_files = npz_files(ref_root), npz_files(test_root)
    label = args.label or f"{ref_root.name} vs {test_root.name}"

    if ref_files != test_files:
        only_ref = set(ref_files) - set(test_files)
        only_test = set(test_files) - set(ref_files)
        print(f"[{label}] FAIL: file sets differ")
        for p in sorted(only_ref):
            print(f"    only in ref:  {p}")
        for p in sorted(only_test):
            print(f"    only in test: {p}")
        return 1
    if not ref_files:
        print(f"[{label}] FAIL: no X_*.npz found under {ref_root}")
        return 1

    worst = 0.0
    n_diff_total = 0
    n_values = 0
    meta_ok = True
    exact = True

    for rel in ref_files:
        a = np.load(ref_root / rel, allow_pickle=True)
        b = np.load(test_root / rel, allow_pickle=True)

        for key in ("shape", "loci", "chromosome", "strand", "depth",
                    "indptr", "indices"):
            if not np.array_equal(a[key], b[key]):
                meta_ok = False
                exact = False
                print(f"[{label}] {rel}: '{key}' differs")

        da, db = a["data"], b["data"]
        n_values += da.size
        if da.shape != db.shape:
            exact = False
            print(f"[{label}] {rel}: data length {da.shape} vs {db.shape}")
            continue
        if not np.array_equal(da, db):
            exact = False
            d = np.abs(da.astype(np.float64) - db.astype(np.float64))
            worst = max(worst, float(d.max()))
            n_diff_total += int((d > args.tol).sum())

    ok = exact or (worst <= args.tol and meta_ok)
    status = "IDENTICAL" if exact else ("WITHIN TOL" if ok else "MISMATCH")
    print(f"[{label}] {status}: {len(ref_files)} npz, {n_values} stored values"
          + (f", max|diff|={worst:.3e}, {n_diff_total} beyond tol={args.tol}"
             if not exact else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
