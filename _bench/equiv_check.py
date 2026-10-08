#!/usr/bin/env python3
"""
Equivalence + speed check for the optimized prepdata path.

Builds a synthetic indexed BAM with a realistic CIGAR mix (soft clips,
insertions, deletions, reference skips), then compares the feature matrix
produced by the original dictionary-based pileup against the new matrix
pileup. Both must agree bit-for-bit.

Usage: python _bench/equiv_check.py [--reads N] [--loci M] [--reflen L]
"""

import argparse
import os
import random
import sys
import tempfile
import time

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfl_py.pileup import (  # noqa: E402
    pileup_bam, pileup_matrix_bam, extend_loci, loci_keep_mask,
    close_cached_bams,
)
from nfl_py.boosting import (  # noqa: E402
    pileup_to_boosting_data, pileup_matrix_to_boosting_data,
)

BASES = "ACGT"


def make_reference(reflen, rng):
    return "".join(rng.choice(list(BASES)) for _ in range(reflen))


def random_cigar(rng, ref_span):
    """A CIGAR covering roughly ref_span reference bases, with indels/skips."""
    ops = []
    remaining = ref_span
    # leading soft clip
    if rng.random() < 0.7:
        ops.append((4, rng.randint(1, 12)))
    while remaining > 0:
        m = min(remaining, rng.randint(20, 120))
        ops.append((0, m))
        remaining -= m
        if remaining <= 0:
            break
        r = rng.random()
        if r < 0.18:
            ops.append((1, rng.randint(1, 4)))          # insertion
        elif r < 0.34:
            d = min(remaining, rng.randint(1, 4))
            ops.append((2, d))                          # deletion
            remaining -= d
        elif r < 0.38:
            n = min(remaining, rng.randint(20, 60))
            ops.append((3, n))                          # reference skip
            remaining -= n
    if rng.random() < 0.7:
        ops.append((4, rng.randint(1, 12)))
    return ops


def query_len(cigar):
    return sum(l for op, l in cigar if op in (0, 1, 4, 7, 8))


def read_sequence(refseq, start, cigar, sub_rate, rng):
    """
    Query sequence for a read, either random or derived from the reference.

    With sub_rate set, the read follows the reference and mismatches at that
    rate. This matters for timing: nanopore basecalls agree with the reference
    at nearly every position, and any code path whose cost depends on how many
    substitutions there are will look far worse than it is when measured
    against fully random bases, which mismatch three times out of four.
    """
    if sub_rate is None:
        return "".join(rng.choice(list(BASES)) for _ in range(query_len(cigar)))

    out = []
    ref_pos = start
    for op, length in cigar:
        if op in (0, 7, 8):                      # consumes both
            for k in range(length):
                base = refseq[ref_pos + k]
                if rng.random() < sub_rate:
                    base = rng.choice([b for b in BASES if b != base])
                out.append(base)
            ref_pos += length
        elif op in (1, 4):                       # query only
            out.extend(rng.choice(list(BASES)) for _ in range(length))
        elif op in (2, 3):                       # reference only
            ref_pos += length
    return "".join(out)


def build_bam(path, refname, reflen, n_reads, rng, refseq=None, sub_rate=None):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"LN": reflen, "SN": refname}]}
    if sub_rate is not None and refseq is None:
        raise ValueError("sub_rate requires refseq")
    records = []
    for i in range(n_reads):
        span = rng.randint(300, 900)
        start = rng.randint(0, max(0, reflen - span - 1))
        cigar = random_cigar(rng, span)
        qlen = query_len(cigar)
        seq = read_sequence(refseq, start, cigar, sub_rate, rng)
        qual = [rng.randint(2, 40) for _ in range(qlen)]
        records.append((start, i, cigar, seq, qual, rng.random() < 0.5))
    records.sort(key=lambda r: r[0])

    unsorted = path + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for start, i, cigar, seq, qual, is_rev in records:
            a = pysam.AlignedSegment()
            a.query_name = f"read_{i:06d}"
            a.query_sequence = seq
            a.flag = 16 if is_rev else 0
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            a.cigartuples = cigar
            a.query_qualities = pysam.qualitystring_to_array(
                "".join(chr(q + 33) for q in qual)
            )
            out.write(a)
    pysam.sort("-o", path, unsorted)
    pysam.index(path)
    os.remove(unsorted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", type=int, default=400)
    ap.add_argument("--loci", type=int, default=100)
    ap.add_argument("--reflen", type=int, default=20000)
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--sub-rate", type=float, default=-1.0,
                    help="per-base mismatch rate; negative means fully "
                         "random reads, which mismatch ~75%% of bases and so "
                         "drive the dense statistics paths hardest")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    refname = "synth_chr"
    refseq = make_reference(args.reflen, rng)
    sub_rate = None if args.sub_rate < 0 else args.sub_rate

    tmpdir = tempfile.mkdtemp(prefix="mars_equiv_")
    bam = os.path.join(tmpdir, "synth.bam")
    build_bam(bam, refname, args.reflen, args.reads, rng,
              refseq=refseq, sub_rate=sub_rate)

    c = args.context
    span_lo, span_hi = c + 1, args.reflen - c
    loci = sorted(rng.sample(range(span_lo, span_hi), args.loci))

    print(f"synthetic BAM: {args.reads} reads, reflen={args.reflen}, "
          f"{args.loci} loci, context={c}, sub_rate={sub_rate}")

    ok = True
    for reverse in (False, True):
        tstart = loci[0] - c
        tend = loci[-1] + c
        loci_arr = np.asarray(loci, dtype=np.int64)

        # --- reference implementation: dicts, then fold to matrices
        loci_ext = extend_loci(loci, c)
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            pu = pileup_bam(bam, refname, tstart, tend, isqv=True,
                            isins=False, loci_ext=loci_ext)
            X_old = pileup_to_boosting_data(
                pu, refseq, loci, context_size=c, reverse=reverse,
                run_qv_mean=12.5, run_qv_sd=7.25, use_gpu=False)
        t_old = (time.perf_counter() - t0) / args.repeat

        # --- optimized implementation: straight to matrices
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            keep = loci_keep_mask(tstart, tend, loci_arr, c)
            pm = pileup_matrix_bam(bam, refname, tstart, tend,
                                   reverse=reverse, keep_mask=keep)
            X_new = pileup_matrix_to_boosting_data(
                pm, refseq, loci, context_size=c,
                run_qv_mean=12.5, run_qv_sd=7.25, use_gpu=False)
        t_new = (time.perf_counter() - t0) / args.repeat

        strand = "-" if reverse else "+"
        same_shape = X_old.shape == X_new.shape
        identical = same_shape and np.array_equal(
            X_old, X_new, equal_nan=True)
        rounded_same = same_shape and np.array_equal(
            np.round(X_old, 4), np.round(X_new, 4), equal_nan=True)
        if not identical:
            ok = False
            if same_shape:
                d = np.abs(X_old - X_new)
                print(f"  [{strand}] MISMATCH max|diff|={np.nanmax(d):.3e} "
                      f"n_diff={(d > 0).sum()}")
            else:
                print(f"  [{strand}] SHAPE MISMATCH {X_old.shape} vs {X_new.shape}")

        # Depth metadata must match too.
        depth_old = (pu.dn if reverse else pu.dp)[loci_arr - pu.blocus - 1]
        depth_new = pm.depth[loci_arr - pm.blocus - 1]
        depth_same = np.array_equal(depth_old, depth_new)
        if not depth_same:
            ok = False
            print(f"  [{strand}] DEPTH MISMATCH")

        print(f"  [{strand}] shape={X_old.shape} bit-identical={identical} "
              f"rounded-identical={rounded_same} depth-identical={depth_same}")
        print(f"  [{strand}] old={t_old * 1000:.1f} ms  new={t_new * 1000:.1f} ms  "
              f"speedup={t_old / t_new:.2f}x")

    close_cached_bams()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
