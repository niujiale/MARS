#!/usr/bin/env python3
"""
Emit a transcriptome-shaped synthetic dataset: many short references, each with
only a handful of target loci.

This is the axis the single-reference generator cannot reach. Real MARS input is
mapped against a transcriptome, so prepdata is called once per
(transcript, strand) and most of those calls carry far fewer loci than one
batch. That makes the per-call cost -- pool construction, reference
serialization, CUDA context setup -- the thing that matters, rather than the
per-locus cost the wide/deep datasets measure.
"""

import argparse
import os
import random
import sys

import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_check import (  # noqa: E402
    make_reference, query_len, random_cigar, read_sequence,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--transcripts", type=int, default=400)
    ap.add_argument("--reflen", type=int, default=2000,
                    help="length of each transcript")
    ap.add_argument("--loci-per-tx", type=int, default=8)
    ap.add_argument("--reads-per-tx", type=int, default=60)
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--sub-rate", type=float, default=0.07,
                    help="per-base mismatch rate against the reference; "
                         "pass a negative value for fully random reads")
    ap.add_argument("--seed", type=int, default=515151)
    args = ap.parse_args()
    sub_rate = None if args.sub_rate < 0 else args.sub_rate

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    names = [f"tx{i:05d}" for i in range(args.transcripts)]
    seqs = {n: make_reference(args.reflen, rng) for n in names}

    fa = os.path.join(args.outdir, "synth.fa")
    with open(fa, "w") as fh:
        for n in names:
            fh.write(f">{n}\n")
            s = seqs[n]
            for i in range(0, len(s), 60):
                fh.write(s[i:i + 60] + "\n")

    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": n, "LN": args.reflen} for n in names]}

    bam = os.path.join(args.outdir, "synth.bam")
    unsorted = bam + ".unsorted.bam"
    read_id = 0
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for ref_id, n in enumerate(names):
            for _ in range(args.reads_per_tx):
                span = rng.randint(300, min(900, args.reflen - 1))
                start = rng.randint(0, max(0, args.reflen - span - 1))
                cigar = random_cigar(rng, span)
                qlen = query_len(cigar)
                seq = read_sequence(seqs[n], start, cigar, sub_rate, rng)
                qual = "".join(chr(rng.randint(2, 40) + 33) for _ in range(qlen))

                a = pysam.AlignedSegment()
                a.query_name = f"read_{read_id:07d}"
                a.query_sequence = seq
                a.flag = 16 if rng.random() < 0.5 else 0
                a.reference_id = ref_id
                a.reference_start = start
                a.mapping_quality = 60
                a.cigartuples = cigar
                a.query_qualities = pysam.qualitystring_to_array(qual)
                out.write(a)
                read_id += 1

    pysam.sort("-o", bam, unsorted)
    pysam.index(bam)
    os.remove(unsorted)

    c = args.context
    props = os.path.join(args.outdir, "synth_props.tsv")
    n_loci = 0
    with open(props, "w") as fh:
        for n in names:
            loci = sorted(rng.sample(range(c + 1, args.reflen - c),
                                     args.loci_per_tx))
            for strand in ("+", "-"):
                for pos in loci:
                    fh.write(f"{n}\t{pos}\tA\t{strand}\t0.000000\t0\t0\t0\n")
                    n_loci += 1

    print(f"transcripts={args.transcripts} reflen={args.reflen} "
          f"reads={read_id} loci_rows={n_loci} sub_rate={sub_rate} "
          f"jobs={args.transcripts * 2}\n  {fa}\n  {bam}\n  {props}")


if __name__ == "__main__":
    main()
