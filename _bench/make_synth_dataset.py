#!/usr/bin/env python3
"""
Emit a synthetic dataset in the shape stage_bench.py expects: an indexed BAM,
a reference FASTA, and a modification-proportions TSV.

Covers the axes the bundled example cannot: a long reference with many target
loci spread across it, which is where the per-batch reference serialization and
the whole-span dense matrices used to hurt most.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_check import build_bam, make_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reads", type=int, default=3000)
    ap.add_argument("--loci", type=int, default=2000)
    ap.add_argument("--reflen", type=int, default=200000)
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--sub-rate", type=float, default=0.07,
                    help="per-base mismatch rate against the reference; "
                         "pass a negative value for fully random reads")
    ap.add_argument("--seed", type=int, default=424242)
    args = ap.parse_args()
    sub_rate = None if args.sub_rate < 0 else args.sub_rate

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    refname = "synth_tx"
    refseq = make_reference(args.reflen, rng)

    fa = os.path.join(args.outdir, "synth.fa")
    with open(fa, "w") as fh:
        fh.write(f">{refname}\n")
        for i in range(0, len(refseq), 60):
            fh.write(refseq[i:i + 60] + "\n")

    bam = os.path.join(args.outdir, "synth.bam")
    build_bam(bam, refname, args.reflen, args.reads, rng,
              refseq=refseq, sub_rate=sub_rate)

    c = args.context
    loci = sorted(rng.sample(range(c + 1, args.reflen - c), args.loci))
    props = os.path.join(args.outdir, "synth_props.tsv")
    with open(props, "w") as fh:
        for pos in loci:
            # columns: chr, pos, base, strand, proportion, n_mod, n_unmod, depth
            fh.write(f"{refname}\t{pos}\tA\t+\t0.000000\t0\t0\t0\n")
        for pos in loci:
            fh.write(f"{refname}\t{pos}\tA\t-\t0.000000\t0\t0\t0\n")

    print(f"reads={args.reads} reflen={args.reflen} loci={args.loci} "
          f"sub_rate={sub_rate} (x2 strands)\n  {fa}\n  {bam}\n  {props}")


if __name__ == "__main__":
    main()
