#!/usr/bin/env python3
"""
A/B the two joint-SR implementations on identical data.

The counting path is selected automatically, so measuring the gain needs the
dense path forced back on for the same input. This also checks the two agree
bit-for-bit, which is the claim that makes the switch safe.
"""

import os
import random
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from equiv_check import build_bam, make_reference  # noqa: E402
from nfl_py import stats as nfl_stats  # noqa: E402
from nfl_py.pileup import (  # noqa: E402
    pileup_matrix_bam, loci_keep_mask, close_cached_bams,
)
from nfl_py.boosting import pileup_matrix_to_boosting_data  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402


def timed(fn, repeat=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return (time.perf_counter() - t0) / repeat, out


def main():
    print(f"{'sub_rate':>9} {'reads':>7} {'loci':>6} {'reflen':>8} "
          f"{'dense':>9} {'auto':>9} {'gain':>7} {'nnz%':>6}  identical")

    for sub_rate in (0.07, 0.20, None):
        for n_reads, n_loci, reflen in ((800, 100, 20000),
                                        (2000, 100, 20000),
                                        (2000, 200, 60000)):
            rng = random.Random(99)
            refname = "synth_chr"
            refseq = make_reference(reflen, rng)
            tmpdir = tempfile.mkdtemp(prefix="mars_ab_")
            bam = os.path.join(tmpdir, "synth.bam")
            build_bam(bam, refname, reflen, n_reads, rng,
                      refseq=refseq, sub_rate=sub_rate)

            c = 10
            loci = sorted(rng.sample(range(c + 1, reflen - c), n_loci))
            loci_arr = np.asarray(loci, dtype=np.int64)
            keep = loci_keep_mask(loci[0] - c, loci[-1] + c, loci_arr, c)
            pm = pileup_matrix_bam(bam, refname, loci[0] - c, loci[-1] + c,
                                   reverse=False, keep_mask=keep)

            def features():
                return pileup_matrix_to_boosting_data(
                    pm, refseq, loci, context_size=c,
                    run_qv_mean=12.5, run_qv_sd=7.25, use_gpu=False)

            saved = nfl_stats._SPARSE_JOINT_PAIR_COST
            try:
                # A pair priced this high never beats the dense Gram matrix.
                nfl_stats._SPARSE_JOINT_PAIR_COST = 10 ** 18
                t_dense, X_dense = timed(features)
                nfl_stats._SPARSE_JOINT_PAIR_COST = saved
                t_auto, X_auto = timed(features)
            finally:
                nfl_stats._SPARSE_JOINT_PAIR_COST = saved

            same = np.array_equal(X_dense, X_auto, equal_nan=True)
            nnz = csr_matrix(np.round(X_auto, 4)).nnz
            frac = nnz / X_auto.size
            close_cached_bams()

            print(f"{str(sub_rate):>9} {n_reads:>7} {n_loci:>6} {reflen:>8} "
                  f"{t_dense * 1000:>8.1f}m {t_auto * 1000:>8.1f}m "
                  f"{t_dense / t_auto:>6.2f}x {frac:>5.1%}  {same}")


if __name__ == "__main__":
    main()
