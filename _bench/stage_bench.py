#!/usr/bin/env python3
"""
Time the prepdata stage (bam_to_boosting_data) against a real BAM.

The source tree to import nfl_py from is given explicitly so the measurement
never depends on which editable install happens to win on sys.path.

Usage:
  python stage_bench.py --tree <MARS_dir> --bam <bam> --ref <fa>
                        --props <proportions.tsv> --out <dir>
                        [--gpu] [--workers N] [--repeat N] [--batch-size N]
"""

import argparse
import os
import shutil
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--props", required=True,
                    help="TSV: chr, pos, base, strand, proportion, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", dest="gpu", action="store_true", default=False)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--context-size", type=int, default=10)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    # Import nfl_py from the requested tree, ahead of anything else.
    tree = os.path.abspath(args.tree)
    sys.path.insert(0, tree)
    import numpy as np
    import pandas as pd
    import nfl_py
    from nfl_py.boosting import bam_to_boosting_data
    from nfl_py.iofastx import load_fasta_dna
    from nfl_py.pileup import get_mean_qv

    # MARS >= 4.0.0 pools workers across all jobs; older trees only expose the
    # per-job entry point. Use whichever the tree under test actually has, so
    # the measurement matches how that version's pipeline drives it.
    try:
        from nfl_py.boosting import FeatureJob, bam_to_boosting_jobs
        pooled = True
    except ImportError:
        pooled = False

    origin = os.path.dirname(os.path.abspath(nfl_py.__file__))
    if not origin.startswith(tree):
        raise SystemExit(
            f"nfl_py resolved to {origin}, expected inside {tree}"
        )

    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False

    props = pd.read_csv(args.props, sep="\t", header=None,
                        names=["chr", "pos", "base", "strand", "proportion",
                               "n_mod", "n_unmod", "depth"])
    refseq = load_fasta_dna(args.ref)
    run_qv_mean, run_qv_sd = get_mean_qv(args.bam)

    jobs = []
    for chrom, sub in props.groupby("chr", sort=True):
        for strand, name in (("+", "forward"), ("-", "backward")):
            sel = sub[sub["strand"] == strand]
            if len(sel) == 0:
                continue
            jobs.append((chrom, name, strand == "-", sel["pos"].tolist(),
                         sel["proportion"].values.astype(np.float32)))

    total_loci = sum(len(j[3]) for j in jobs)
    label = args.label or os.path.basename(tree)
    print(f"[{label}] nfl_py from {origin}")
    print(f"[{label}] gpu_requested={args.gpu} torch.cuda={cuda} "
          f"workers={args.workers} batch={args.batch_size} pooled={pooled}")
    print(f"[{label}] {len(jobs)} chrom/strand jobs, {total_loci} loci, "
          f"run_qv_mean={run_qv_mean:.4f} run_qv_sd={run_qv_sd:.4f}")

    best = None
    for rep in range(args.repeat):
        if os.path.isdir(args.out):
            shutil.rmtree(args.out)
        t0 = time.perf_counter()
        if pooled:
            specs = [
                FeatureJob(
                    chromosome=chrom,
                    loci=loci,
                    outdir=os.path.join(args.out, "feature", chrom, name,
                                        "Xdata"),
                    reverse=reverse,
                )
                for chrom, name, reverse, loci, _ in jobs
            ]
            bam_to_boosting_jobs(
                args.bam, refseq, specs,
                context_size=args.context_size,
                batch_size=args.batch_size,
                run_qv_mean=run_qv_mean,
                run_qv_sd=run_qv_sd,
                use_gpu=args.gpu,
                num_workers=args.workers,
                showprogress=False,
            )
        else:
            for chrom, name, reverse, loci, _ in jobs:
                outdir = os.path.join(args.out, "feature", chrom, name, "Xdata")
                bam_to_boosting_data(
                    args.bam, refseq, chrom, loci, outdir,
                    context_size=args.context_size,
                    batch_size=args.batch_size,
                    reverse=reverse,
                    run_qv_mean=run_qv_mean,
                    run_qv_sd=run_qv_sd,
                    use_gpu=args.gpu,
                    num_workers=args.workers,
                    showprogress=False,
                )
        for chrom, name, _, _, ydata in jobs:
            ydata_file = os.path.join(args.out, "feature", chrom, name,
                                      "ydata.tsv")
            np.savetxt(ydata_file, ydata, fmt="%.6f")
        elapsed = time.perf_counter() - t0
        print(f"[{label}] run {rep + 1}/{args.repeat}: {elapsed:.3f} s")
        best = elapsed if best is None else min(best, elapsed)

    print(f"[{label}] BEST {best:.3f} s  ({total_loci / best:.1f} loci/s)")
    print(f"TIMING\t{label}\t{best:.6f}\t{total_loci}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
