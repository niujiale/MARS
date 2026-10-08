#!/usr/bin/env python3
"""
NFL-Py Command Line Interface

Provides commands for:
- prepdata: Prepare feature data from BAM files
- train: Train XGBoost model
- predict: Make predictions with trained model
- get-mean-qv: Calculate mean quality values from BAM

Usage:
    nfl-py prepdata bamfile reffile locifile outdir
    nfl-py train Xdata_dir ydata_file outfile
    nfl-py predict modelfile Xdata_dir outfile

Also reachable as `mars prepdata ...` and `python -m nfl_py ...`.

Author: MARS Team
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import __version__

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_prepdata_parser(subparsers):
    """Setup prepdata command parser."""
    parser = subparsers.add_parser(
        'prepdata',
        help='Prepare boosting data from BAM file'
    )
    parser.add_argument('bamfile', help='Sorted and indexed BAM file')
    parser.add_argument('reffile', help='Reference FASTA file')
    parser.add_argument('locifile', help='Loci file (TSV: chr, pos, modtype, strand, proportion)')
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('-c', '--chr', default='', help='Process single chromosome')
    parser.add_argument('--context-size', type=int, default=10,
                        help='Bases of context on each side of a target site '
                             '(default: 10). Any value works; feature width '
                             'grows as its square, and sites whose window '
                             'overhangs the reference are dropped.')
    parser.add_argument('-b', '--batch-size', type=int, default=100, help='Batch size (default: 100)')
    parser.add_argument('-p', '--parallel', type=int, default=1,
                        help='Number of parallel workers (default: 1, sequential)')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite output')
    parser.add_argument('--run-level-normalize', action='store_true',
                        help='Normalize QV by run average')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    return parser


def setup_train_parser(subparsers):
    """Setup train command parser."""
    parser = subparsers.add_parser(
        'train',
        help='Train XGBoost model'
    )
    parser.add_argument('Xdata_dir', help='Feature data directory from prepdata')
    parser.add_argument('ydata_file', help='Labels file (one value per line)')
    parser.add_argument('outfile', help='Output model file')
    parser.add_argument('-e', '--eta', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('-n', '--num-round', type=int, default=200,
                        help='Number of trees (default: 200)')
    parser.add_argument('-m', '--max-depth', type=int, default=6, help='Tree depth (default: 6)')
    parser.add_argument('--objective', default='reg:squarederror',
                        help='Objective function (default: reg:squarederror)')
    parser.add_argument('-a', '--alpha', type=float, default=1e-8,
                        help='Logit transformation alpha (default: 1e-8)')
    return parser


def setup_predict_parser(subparsers):
    """Setup predict command parser."""
    parser = subparsers.add_parser(
        'predict',
        help='Make predictions with trained model'
    )
    parser.add_argument('modelfile', help='Trained model file')
    parser.add_argument('Xdata_dir', help='Feature data directory')
    parser.add_argument('outfile', help='Output predictions file')
    parser.add_argument('-e', '--esp', type=float, default=0.001,
                        help='Precision threshold (default: 0.001)')
    parser.add_argument('--objective', default='reg:squarederror',
                        help='Objective function used in training')
    return parser


def setup_get_mean_qv_parser(subparsers):
    """Setup get-mean-qv command parser."""
    parser = subparsers.add_parser(
        'get-mean-qv',
        help='Get mean QV from BAM file'
    )
    parser.add_argument('bamfile', help='BAM file')
    parser.add_argument('--progress', action='store_true', help='Show progress')
    return parser


def cmd_prepdata(args):
    """Execute prepdata command."""
    from .boosting import FeatureJob, bam_to_boosting_jobs
    from .iofastx import load_fasta_dna
    from .pileup import get_mean_qv
    import shutil

    logger.info("=" * 60)
    logger.info("NFL-Py PrepData")
    logger.info("=" * 60)

    logger.info(f"BAM file: {args.bamfile}")
    logger.info(f"Reference: {args.reffile}")
    logger.info(f"Loci file: {args.locifile}")
    logger.info(f"Output: {args.outdir}")
    logger.info(f"Context size: {args.context_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Parallel workers: {args.parallel}")
    logger.info(f"GPU enabled: {not args.no_gpu}")

    # Check/create output directory
    outdir = Path(args.outdir)
    if outdir.exists() and args.force:
        shutil.rmtree(outdir)
    if outdir.exists():
        raise ValueError(f"{outdir} already exists. Use --force to overwrite.")

    # Get run-level QV normalization
    run_qv_mean, run_qv_sd = 0.0, 1.0
    if args.run_level_normalize:
        logger.info("Computing run-level QV statistics...")
        run_qv_mean, run_qv_sd = get_mean_qv(args.bamfile)
        logger.info(f"Run QV mean: {run_qv_mean:.2f}, std: {run_qv_sd:.2f}")

    # Load loci data
    logger.info("Loading loci file...")
    loci_data = pd.read_csv(args.locifile, sep='\t', header=None)
    expected_cols = ['chr', 'pos', 'modtype', 'strand', 'proportion']
    if len(loci_data.columns) > len(expected_cols):
        # Handle extra columns by naming them col5, col6, etc.
        extra_cols = [f'col{i}' for i in range(len(expected_cols), len(loci_data.columns))]
        loci_data.columns = expected_cols + extra_cols
    else:
        loci_data.columns = expected_cols[:len(loci_data.columns)]

    if args.chr:
        loci_data = loci_data[loci_data['chr'] == args.chr]
        chromosomes = [args.chr]
    else:
        chromosomes = loci_data['chr'].unique().tolist()

    # Load reference
    logger.info("Loading reference sequences...")
    if args.chr:
        refseq = load_fasta_dna(args.reffile, args.chr)
    else:
        refseq = load_fasta_dna(args.reffile)

    # Collect all the work first: one worker pool then covers the whole run
    # instead of being rebuilt for every chromosome and strand.
    jobs = []
    ydata_of_job = {}
    for chrom in chromosomes:
        chrom_data = loci_data[loci_data['chr'] == chrom]

        for strand, name, reverse in (('+', 'forward', False),
                                      ('-', 'backward', True)):
            strand_data = chrom_data[chrom_data['strand'] == strand]
            if len(strand_data) == 0:
                continue

            strand_dir = outdir / chrom / name
            jobs.append(FeatureJob(
                chromosome=chrom,
                loci=strand_data['pos'].tolist(),
                outdir=str(strand_dir / 'Xdata'),
                reverse=reverse,
            ))
            # Keyed by position, because feature extraction may drop loci
            # whose context window overhangs the reference.
            ydata_of_job[(chrom, reverse)] = (strand_dir, dict(zip(
                strand_data['pos'].tolist(),
                strand_data['proportion'].values.astype(np.float32),
            )))

    logger.info(f"Processing {len(jobs)} chromosome/strand jobs...")
    extracted = bam_to_boosting_jobs(
        args.bamfile, refseq, jobs,
        context_size=args.context_size,
        batch_size=args.batch_size,
        run_qv_mean=run_qv_mean,
        run_qv_sd=run_qv_sd,
        use_gpu=not args.no_gpu,
        num_workers=args.parallel,
    )

    for job in extracted:
        strand_dir, prop_of_pos = ydata_of_job[(job.chromosome, job.reverse)]
        strand_dir.mkdir(parents=True, exist_ok=True)
        ydata = np.array([prop_of_pos[p] for p in job.loci], dtype=np.float32)
        np.savetxt(strand_dir / 'ydata.tsv', ydata, fmt='%.6f')

    logger.info("PrepData complete!")
    logger.info("=" * 60)


def cmd_train(args):
    """Execute train command."""
    from .boosting import (
        load_boosting_data, train_boosting_model,
        save_boosting_model, BoostingArgs
    )

    logger.info("=" * 60)
    logger.info("NFL-Py Train")
    logger.info("=" * 60)

    logger.info(f"Feature data: {args.Xdata_dir}")
    logger.info(f"Labels: {args.ydata_file}")
    logger.info(f"Output: {args.outfile}")
    logger.info(f"eta={args.eta}, num_round={args.num_round}, max_depth={args.max_depth}")

    # Load data
    logger.info("Loading feature data...")
    X = load_boosting_data(args.Xdata_dir, showprogress=True)
    X = X.toarray()

    logger.info("Loading labels...")
    y = np.loadtxt(args.ydata_file, dtype=np.float32)

    # Transform labels for logit
    if args.objective == 'reg:squarederror':
        y_t = np.log(y) - np.log(1 - y)
        y_t[y <= args.alpha] = np.log(args.alpha)
        y_t[y >= 1 - args.alpha] = -np.log(args.alpha)
    else:
        y_t = y

    # Train
    logger.info("Training model...")
    model_args = BoostingArgs(
        eta=args.eta,
        num_round=args.num_round,
        max_depth=args.max_depth,
        objective=args.objective
    )
    model = train_boosting_model(X, y_t, args=model_args)

    # Save
    save_boosting_model(model, args.outfile)
    logger.info("Training complete!")
    logger.info("=" * 60)


def cmd_predict(args):
    """Execute predict command."""
    from .boosting import (
        load_boosting_model, predict_boosting_model,
        load_boosting_data, load_boosting_data_info
    )

    logger.info("=" * 60)
    logger.info("NFL-Py Predict")
    logger.info("=" * 60)

    logger.info(f"Model: {args.modelfile}")
    logger.info(f"Feature data: {args.Xdata_dir}")
    logger.info(f"Output: {args.outfile}")

    # Load model
    logger.info("Loading model...")
    model = load_boosting_model(args.modelfile)

    # Load data
    logger.info("Loading feature data...")
    X = load_boosting_data(args.Xdata_dir, showprogress=True)
    X = X.toarray()

    # Predict
    logger.info("Making predictions...")
    y_pred_t = predict_boosting_model(model, X)

    # Transform predictions
    if args.objective == 'reg:squarederror':
        y_pred = np.exp(y_pred_t) / (1 + np.exp(y_pred_t))
        y_pred[y_pred <= args.esp] = 0
        y_pred[y_pred >= 1 - args.esp] = 1
    else:
        y_pred = y_pred_t

    # Save predictions
    np.savetxt(args.outfile, y_pred, fmt='%.6f')

    # Save BED file
    logger.info("Generating BED file...")
    info = load_boosting_data_info(args.Xdata_dir)

    bed_df = pd.DataFrame({
        'chr': info['chr'],
        'start': info['loci'] - 1,
        'end': info['loci'],
        'modtype': '5mC',
        'score': 1000,
        'strand': info['strand'],
        'thickStart': info['loci'] - 1,
        'thickEnd': info['loci'],
        'itemRgb': '0,0,0',
        'depth': info['depth'],
        'modprob': (100 * y_pred).astype(int),
        'ncano': (info['depth'] - np.round(info['depth'] * y_pred)).astype(int),
        'nmod': np.round(info['depth'] * y_pred).astype(int),
        'nfiltered': 0
    })

    bed_df.to_csv(args.outfile + '.bed', sep='\t', index=False, header=False)

    logger.info("Prediction complete!")
    logger.info("=" * 60)


def cmd_get_mean_qv(args):
    """Execute get-mean-qv command."""
    from .pileup import get_mean_qv

    mean_qv, std_qv = get_mean_qv(args.bamfile, showprogress=args.progress)
    print(f"{mean_qv:.4f}\t{std_qv:.4f}")


def main(argv: Optional[List[str]] = None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='nfl-py',
        description='NFL-Py: feature extraction core of MARS',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s (MARS) {__version__}'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    setup_prepdata_parser(subparsers)
    setup_train_parser(subparsers)
    setup_predict_parser(subparsers)
    setup_get_mean_qv_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command == 'prepdata':
        cmd_prepdata(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'get-mean-qv':
        cmd_get_mean_qv(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
