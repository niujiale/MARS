#!/usr/bin/env python3
"""
MARS Main Pipeline

A pipeline for processing FASTQ files, aligning to a reference genome,
identifying RNA modifications, and generating features for modification analysis.

Uses Python NFL (nfl_py) for feature extraction with optional GPU acceleration.

Installed as the `mars-pipeline` command, and as the `mars run` subcommand.

Usage:
    mars run input.fastq reference.fasta A output_dir --num_threads 20
    mars-pipeline input.fastq reference.fasta DRACH output_dir --gpu

Author: MARS Team
"""

# Suppress multiprocessing resource_tracker warnings
# This is a known issue with Python 3.12+ when using CUDA/GPU with multiprocessing
import sys
import os
import warnings

# Filter out the semaphore leak warning from resource_tracker
warnings.filterwarnings("ignore", message=".*resource_tracker.*leaked.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*leaked semaphore.*", category=UserWarning)

# Set environment variable to suppress resource_tracker warnings in subprocesses
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Use 'spawn' method for multiprocessing to avoid fork-related issues with CUDA
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import time
import csv
import logging
import argparse
import subprocess
import threading
import atexit
from typing import Set, Tuple, Dict, Optional
from multiprocessing import Pool, Lock, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import pysam
import numpy as np
from tqdm.auto import tqdm

from ._version import __version__
from .paths import project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global lock for thread-safe file operations
lock = Lock()

PROG = 'mars-pipeline'

# Constants
VALID_BASES = {'A', 'T', 'G', 'C'}
VALID_MODIFICATIONS = {'A', 'T', 'G', 'C', 'DRACH', 'm5C'}
EDGE_BUFFER = 10  # Positions to skip at sequence edges


def run_command(command: str, description: str = "") -> None:
    """
    Run an external command and capture output.

    Args:
        command: Shell command to execute
        description: Optional description for logging

    Raises:
        RuntimeError: If command fails
    """
    if description:
        logger.info(f"Running: {description}")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode().strip()
        logger.error(f"Command failed (exit code {process.returncode}): {error_msg}")
        raise RuntimeError(f"Command failed: {command}")

    output = stdout.decode().strip()
    if output:
        logger.debug(output)


def find_bases_in_genome(
    fasta_file: str,
    output_csv_file: str,
    target_base: str
) -> int:
    """
    Extract positions of a specific base in the reference genome.

    Args:
        fasta_file: Path to reference FASTA file
        output_csv_file: Path to output CSV file
        target_base: Target nucleotide base (A, T, G, or C)

    Returns:
        Number of positions found
    """
    logger.info(f"Finding '{target_base}' positions in genome...")
    count = 0
    target_upper = target_base.upper()

    with pysam.FastaFile(fasta_file) as fasta:
        with open(output_csv_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Chromosome', 'Position'])

            for chrom in fasta.references:
                sequence = fasta.fetch(chrom)
                seq_len = len(sequence)

                for pos, base in enumerate(sequence):
                    if pos < EDGE_BUFFER or pos >= seq_len - EDGE_BUFFER:
                        continue

                    if base.upper() == target_upper:
                        writer.writerow([chrom, pos + 1])  # 1-based index
                        count += 1

    logger.info(f"Found {count} '{target_base}' positions")
    return count


def find_drach_motifs(fasta_file: str, output_csv_file: str) -> int:
    """
    Extract positions of DRACH motifs (m6A modification sites) in the reference genome.

    DRACH motif: D=[A,G,T], R=[A,G], A, C, H=[A,C,T]
    Reports the position of the central 'A' in the motif.

    Args:
        fasta_file: Path to reference FASTA file
        output_csv_file: Path to output CSV file

    Returns:
        Number of DRACH motifs found
    """
    logger.info("Finding DRACH motif positions (m6A sites)...")

    drach_d = {'A', 'G', 'T'}  # D
    drach_r = {'A', 'G'}       # R
    drach_h = {'A', 'C', 'T'}  # H

    count = 0

    with pysam.FastaFile(fasta_file) as fasta:
        with open(output_csv_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Chromosome', 'Position'])

            for chrom in fasta.references:
                sequence = fasta.fetch(chrom).upper()
                seq_len = len(sequence)

                for pos in range(seq_len - 4):
                    if pos < EDGE_BUFFER or pos >= seq_len - EDGE_BUFFER:
                        continue

                    if (sequence[pos] in drach_d and
                        sequence[pos + 1] in drach_r and
                        sequence[pos + 2] == 'A' and
                        sequence[pos + 3] == 'C' and
                        sequence[pos + 4] in drach_h):
                        # Report position of 'A' (1-based index)
                        writer.writerow([chrom, pos + 3])
                        count += 1

    logger.info(f"Found {count} DRACH motifs")
    return count


def find_m5c_motifs(fasta_file: str, output_csv_file: str) -> int:
    """
    Extract positions of m5C modification sites (GCG motifs) in the reference genome.

    Reports the position of the central 'C' in GCG motifs.

    Args:
        fasta_file: Path to reference FASTA file
        output_csv_file: Path to output CSV file

    Returns:
        Number of m5C sites found
    """
    logger.info("Finding m5C motif positions (GCG sites)...")
    count = 0

    with pysam.FastaFile(fasta_file) as fasta:
        with open(output_csv_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Chromosome', 'Position'])

            for chrom in fasta.references:
                sequence = fasta.fetch(chrom).upper()
                seq_len = len(sequence)

                for pos in range(1, seq_len - 1):
                    if pos < EDGE_BUFFER or pos >= seq_len - EDGE_BUFFER:
                        continue

                    if (sequence[pos] == 'C' and
                        sequence[pos - 1] == 'G' and
                        sequence[pos + 1] == 'G'):
                        writer.writerow([chrom, pos + 1])  # 1-based index
                        count += 1

    logger.info(f"Found {count} m5C sites")
    return count


def read_reference_positions(reference_positions_file: str) -> Set[Tuple[str, int]]:
    """
    Read reference positions from CSV file.

    Args:
        reference_positions_file: Path to CSV file with Chromosome,Position columns

    Returns:
        Set of (chromosome, position) tuples
    """
    reference_positions = set()

    with open(reference_positions_file, 'r') as ref_file:
        next(ref_file)  # Skip header
        for line in ref_file:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                chrom, pos = parts[0], int(parts[1])
                reference_positions.add((chrom, pos))

    logger.info(f"Loaded {len(reference_positions)} reference positions")
    return reference_positions


def has_rg_tags(bam_file_path: str, max_reads: int = 1000) -> bool:
    """
    Check if BAM file reads have RG (Read Group) tags.

    Args:
        bam_file_path: Path to BAM file
        max_reads: Maximum number of reads to check

    Returns:
        True if RG tags are present
    """
    with pysam.AlignmentFile(bam_file_path, "rb") as bam:
        for i, read in enumerate(bam.fetch()):
            if i >= max_reads:
                break
            if read.has_tag('RG'):
                return True
    return False


def process_bam_file(args: Tuple) -> None:
    """
    Process a single BAM file and calculate modification ratios at each position.

    Args:
        args: Tuple of (bam_file_path, reference_positions, output_file, mod_type, base)
    """
    bam_file_path, reference_positions, output_txt_file_path, mod_type, base = args

    counts: Dict[Tuple[str, int], Dict[str, int]] = {}
    strand = '+'
    base_upper = base.upper()

    bam_has_tags = has_rg_tags(bam_file_path)

    with pysam.AlignmentFile(bam_file_path, "rb") as bam:
        for read in bam.fetch():
            if read.is_unmapped or read.query_sequence is None:
                continue

            chrom_name = read.reference_name
            ref_pos = read.reference_start
            query_pos = 0

            for cigartype, cigarlen in read.cigartuples:
                if cigartype == 0:  # M: match or mismatch
                    for i in range(cigarlen):
                        actual_pos = ref_pos + i + 1  # 1-based index
                        query_base = read.query_sequence[query_pos + i].upper()

                        if query_base == base_upper and (chrom_name, actual_pos) in reference_positions:
                            pos_key = (chrom_name, actual_pos)
                            if pos_key not in counts:
                                counts[pos_key] = {'mod': 0, 'unmod': 0}

                            if bam_has_tags:
                                rg_tag = read.get_tag('RG') if read.has_tag('RG') else 'unmod'
                                counts[pos_key][rg_tag] = counts[pos_key].get(rg_tag, 0) + 1
                            else:
                                counts[pos_key]['unmod'] += 1

                    ref_pos += cigarlen
                    query_pos += cigarlen
                elif cigartype in {1, 4}:  # I: insertion, S: soft clip
                    query_pos += cigarlen
                elif cigartype in {2, 3}:  # D: deletion, N: skipped region
                    ref_pos += cigarlen

    # Write results
    with lock:
        with open(output_txt_file_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            for (chrom, pos), counts_dict in sorted(counts.items()):
                mod_count = counts_dict.get('mod', 0)
                unmod_count = counts_dict.get('unmod', 0)
                total_count = mod_count + unmod_count

                if total_count > 0:
                    mod_ratio = mod_count / total_count if bam_has_tags else 0.0
                    writer.writerow([
                        chrom, pos, mod_type, strand,
                        f"{mod_ratio:.6f}", mod_count, unmod_count, total_count
                    ])


def process_all_bam_files(
    bam_file_path: str,
    reference_positions_file: str,
    output_directory: str,
    modification_base: str,
    base: str,
    num_processes: int = 64
) -> Optional[str]:
    """
    Process BAM file and calculate modification ratios at each position.

    Args:
        bam_file_path: Path to input BAM file
        reference_positions_file: Path to reference positions CSV
        output_directory: Output directory path
        modification_base: Modification type (A, T, G, C, DRACH, m5C)
        base: Target base to analyze
        num_processes: Number of parallel processes

    Returns:
        Path to output file, or None if no BAM file processed
    """
    os.makedirs(output_directory, exist_ok=True)

    start_time = time.time()
    reference_positions = read_reference_positions(reference_positions_file)

    # Normalize modification type
    if modification_base == 'm5C':
        modification_base = 'C'
        base = 'C'
    elif modification_base == 'DRACH':
        modification_base = 'A'
        base = 'A'

    output_csv_file = None

    if bam_file_path.endswith(".bam"):
        output_csv_file = os.path.join(
            output_directory,
            os.path.basename(bam_file_path).replace('.bam', '_modification_proportions.csv')
        )
        tasks = [(bam_file_path, reference_positions, output_csv_file, modification_base, base)]

        with Pool(min(num_processes, cpu_count())) as pool:
            pool.map(process_bam_file, tasks)

    elapsed_time = time.time() - start_time
    logger.info(f"BAM processing completed in {elapsed_time:.2f} seconds")

    return output_csv_file


def run_python_nfl_in_parallel(
    output_directory: str,
    reference_genome: str,
    num_threads: int,
    use_gpu: bool = True,
    context_size: int = 10,
    batch_size: int = 100,
    run_level_normalize: bool = False
) -> None:
    """
    Run Python NFL (nfl_py) for feature extraction with optional GPU acceleration.

    Supports:
    - GPU acceleration via PyTorch/CuPy for statistics computation
    - Multiprocessing for parallel batch processing

    Args:
        output_directory: Directory containing input files and for output
        reference_genome: Path to reference genome FASTA
        num_threads: Number of parallel workers for batch processing
        use_gpu: Whether to use GPU acceleration for statistics
        context_size: Context window size for features
        batch_size: Batch size for processing
        run_level_normalize: Whether to rescale the QV mean/covariance features
            by the run-level QV statistics. The bundled pretrained models were
            trained on unscaled QV, so this must stay off to use them.
    """
    # Import nfl_py. Normally it is installed alongside this package; the
    # fallback covers running from an un-installed source tree.
    try:
        from nfl_py.boosting import FeatureJob, bam_to_boosting_jobs
        from nfl_py.iofastx import load_fasta_dna
        from nfl_py.pileup import get_mean_qv
        from nfl_py.gpu_utils import is_gpu_available
    except ImportError:
        root = project_root()
        if root is None:
            raise
        sys.path.insert(0, str(root))
        from nfl_py.boosting import FeatureJob, bam_to_boosting_jobs
        from nfl_py.iofastx import load_fasta_dna
        from nfl_py.pileup import get_mean_qv
        from nfl_py.gpu_utils import is_gpu_available

    csv_file_path = os.path.join(
        output_directory,
        'reads-ref.sorted_filter_modification_proportions.csv'
    )
    bam_file = os.path.join(output_directory, "reads-ref.sorted_filter.bam")

    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"Required file not found: {csv_file_path}")

    # Check GPU availability
    gpu_available = is_gpu_available() if use_gpu else False
    if use_gpu and not gpu_available:
        logger.warning("GPU requested but not available. Falling back to CPU.")
    elif gpu_available:
        logger.info("GPU acceleration enabled")

    # Load loci data
    logger.info("Loading modification positions...")
    import pandas as pd
    loci_data = pd.read_csv(csv_file_path, sep='\t', header=None)
    loci_data.columns = ['chr', 'pos', 'modtype', 'strand', 'proportion',
                         'mod_count', 'unmod_count', 'total_count'][:len(loci_data.columns)]

    chromosomes = loci_data['chr'].unique().tolist()
    logger.info(f"Processing {len(chromosomes)} chromosomes/transcripts")

    # Get run-level QV statistics. Rescaling by them shifts qv_mean/qv_cov out
    # of the range the pretrained models were fit on, so it is opt-in.
    run_qv_mean, run_qv_sd = 0.0, 1.0
    if run_level_normalize:
        logger.info("Computing run-level QV statistics...")
        run_qv_mean, run_qv_sd = get_mean_qv(bam_file)
        logger.info(f"Run QV mean: {run_qv_mean:.2f}, std: {run_qv_sd:.2f}")
    else:
        logger.info(
            "Run-level QV normalization disabled "
            "(enable with --run-level-normalize; incompatible with the "
            "bundled pretrained models)"
        )

    # Load reference sequences
    logger.info("Loading reference sequences...")
    refseq = load_fasta_dna(reference_genome)

    # Collect every (transcript, strand) unit of work before extracting
    # anything, so that a single worker pool covers the whole run. Creating one
    # pool per transcript costs far more than the work a transcript carries.
    jobs = []
    ydata_of_job = {}
    for chrom in chromosomes:
        chrom_data = loci_data[loci_data['chr'] == chrom]

        for strand, name, reverse in (('+', 'forward', False),
                                      ('-', 'backward', True)):
            strand_data = chrom_data[chrom_data['strand'] == strand]
            if len(strand_data) == 0:
                continue

            strand_dir = os.path.join(output_directory, 'feature', chrom, name)
            jobs.append(FeatureJob(
                chromosome=chrom,
                loci=strand_data['pos'].tolist(),
                outdir=os.path.join(strand_dir, 'Xdata'),
                reverse=reverse,
            ))
            # Keyed by position, because feature extraction may drop loci
            # whose context window overhangs the reference.
            ydata_of_job[(chrom, reverse)] = (strand_dir, dict(zip(
                strand_data['pos'].tolist(),
                strand_data['proportion'].values.astype(np.float32),
            )))

    try:
        extracted = bam_to_boosting_jobs(
            bam_file, refseq, jobs,
            context_size=context_size,
            batch_size=batch_size,
            run_qv_mean=run_qv_mean,
            run_qv_sd=run_qv_sd,
            use_gpu=gpu_available,
            num_workers=num_threads,
            showprogress=True,
        )
    except Exception as e:
        tqdm.write(f"[ERROR] feature extraction: {e}")
        return

    # Labels are written only for the jobs that produced features, and only for
    # the loci those jobs kept, so a y vector never has rows that no X row
    # matches.
    for job in extracted:
        strand_dir, prop_of_pos = ydata_of_job[(job.chromosome, job.reverse)]
        os.makedirs(strand_dir, exist_ok=True)
        ydata = np.array([prop_of_pos[p] for p in job.loci], dtype=np.float32)
        np.savetxt(os.path.join(strand_dir, 'ydata.tsv'), ydata, fmt='%.6f')

    logger.info("Python NFL feature extraction complete!")


def main(
    fastq_file: str,
    reference_genome: str,
    modification_base: str,
    output_directory: str,
    num_threads: int,
    use_gpu: bool = True,
    context_size: int = 10,
    batch_size: int = 100,
    run_level_normalize: bool = False
) -> None:
    """
    Main pipeline controller.

    Args:
        fastq_file: Path to input FASTQ file
        reference_genome: Path to reference genome FASTA
        modification_base: Modification type to detect (A, T, G, C, DRACH, m5C)
        output_directory: Output directory path
        num_threads: Number of parallel threads
        use_gpu: Whether to use GPU acceleration
        context_size: Context window size for features
        batch_size: Batch size for processing
        run_level_normalize: Whether to rescale QV features by the run-level QV
            statistics; leave off for the bundled pretrained models
    """
    logger.info("="*60)
    logger.info("MARS Pipeline Started")
    logger.info("="*60)

    # Validate inputs
    if not os.path.exists(fastq_file):
        raise FileNotFoundError(f"FASTQ file not found: {fastq_file}")
    if not os.path.exists(reference_genome):
        raise FileNotFoundError(f"Reference genome not found: {reference_genome}")
    if modification_base not in VALID_MODIFICATIONS:
        raise ValueError(
            f"Invalid modification base: {modification_base}. "
            f"Must be one of: {VALID_MODIFICATIONS}"
        )

    os.makedirs(output_directory, exist_ok=True)

    # Define file paths
    sorted_bam_file = os.path.join(output_directory, 'reads-ref.sorted.bam')
    filtered_bam_file = os.path.join(output_directory, 'reads-ref.sorted_filter.bam')
    reference_positions_file = os.path.join(
        output_directory,
        f'output_positions_of_{modification_base}.csv'
    )

    # Step 1: Alignment with minimap2 and sorting with samtools
    logger.info("Step 1: Aligning reads and sorting BAM...")
    run_command(
        f"minimap2 -ax map-ont -t {num_threads} {reference_genome} {fastq_file} | "
        f"samtools sort -@ {num_threads} -o {sorted_bam_file} -T {output_directory}/reads.tmp",
        "Minimap2 alignment and sorting"
    )

    # Step 2: Filter BAM file
    logger.info("Step 2: Filtering BAM file...")
    run_command(
        f"samtools view -h -b -F 4079 {sorted_bam_file} > {filtered_bam_file}",
        "Filtering unmapped and secondary alignments"
    )
    run_command(
        f"samtools index {filtered_bam_file}",
        "Indexing filtered BAM"
    )

    # Step 3: Extract modification positions based on type
    logger.info(f"Step 3: Extracting {modification_base} positions...")
    if modification_base == 'DRACH':
        find_drach_motifs(reference_genome, reference_positions_file)
    elif modification_base == 'm5C':
        find_m5c_motifs(reference_genome, reference_positions_file)
    else:
        find_bases_in_genome(reference_genome, reference_positions_file, modification_base)

    # Step 4: Process BAM and calculate modification ratios
    logger.info("Step 4: Processing BAM and calculating modification ratios...")
    process_all_bam_files(
        filtered_bam_file,
        reference_positions_file,
        output_directory,
        modification_base,
        modification_base
    )

    # Step 5: Run NanoFreeLunch for feature extraction
    logger.info("Step 5: Running NFL for feature extraction...")
    run_python_nfl_in_parallel(
        output_directory, reference_genome, num_threads,
        use_gpu=use_gpu,
        context_size=context_size,
        batch_size=batch_size,
        run_level_normalize=run_level_normalize
    )

    logger.info("="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)


def build_parser(prog: str = PROG) -> argparse.ArgumentParser:
    """Build the argument parser for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='MARS: Pipeline for RNA modification analysis from Nanopore sequencing data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Detect m6A modifications (DRACH motifs)
    {prog} input.fastq reference.fasta DRACH output_dir --num_threads 20

    # Detect m5C modifications with GPU
    {prog} input.fastq reference.fasta m5C output_dir --gpu

    # Detect all adenine positions without GPU
    {prog} input.fastq reference.fasta A output_dir --no-gpu
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s (MARS) {__version__}'
    )
    parser.add_argument(
        'fastq_file',
        help='Path to the input FASTQ file containing sequencing reads'
    )
    parser.add_argument(
        'reference_genome',
        help='Path to the reference genome FASTA file'
    )
    parser.add_argument(
        'modification_base',
        choices=['A', 'T', 'G', 'C', 'DRACH', 'm5C'],
        help='Modification type to detect: A/T/G/C for single bases, '
             'DRACH for m6A motifs, m5C for 5-methylcytosine motifs'
    )
    parser.add_argument(
        'output_directory',
        help='Directory for output files'
    )
    parser.add_argument(
        '--num_threads', '-t',
        type=int,
        default=10,
        help='Number of threads for parallel processing (default: 10)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        dest='use_gpu',
        default=True,
        help='Enable GPU acceleration (default: enabled)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_false',
        dest='use_gpu',
        help='Disable GPU acceleration'
    )
    parser.add_argument(
        '--context-size',
        type=int,
        default=10,
        help='Bases of context on each side of a target site (default: 10). '
             'Any value works; feature width grows as its square, and sites '
             'whose window overhangs the reference are dropped. Pretrained '
             'models only accept the context they were trained with.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )
    parser.add_argument(
        '--run-level-normalize',
        action='store_true',
        help='Rescale the QV mean/covariance features by the run-level QV '
             'mean and standard deviation (default: off). The bundled '
             'pretrained models were trained on unscaled QV, so turning this '
             'on makes their predictions unusable'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser


def parse_args(argv: Optional[list] = None, prog: str = PROG) -> argparse.Namespace:
    """Parse command line arguments."""
    return build_parser(prog).parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Run the pipeline from parsed arguments."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(
        args.fastq_file,
        args.reference_genome,
        args.modification_base,
        args.output_directory,
        args.num_threads,
        use_gpu=args.use_gpu,
        context_size=args.context_size,
        batch_size=args.batch_size,
        run_level_normalize=args.run_level_normalize
    )


def cli_main(argv: Optional[list] = None, prog: str = PROG) -> None:
    """Console-script wrapper: report failures without a traceback."""
    try:
        run(parse_args(argv, prog))
    except KeyboardInterrupt:
        logger.error("Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def main_with_prog(argv: Optional[list] = None) -> None:
    """Entry point for `mars run`, so --help shows the right prog name."""
    cli_main(argv, prog='mars run')


if __name__ == "__main__":
    cli_main()
