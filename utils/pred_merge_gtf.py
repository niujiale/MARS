#!/usr/bin/env python3
"""
Merge Predictions with GTF Annotations

Match predictions with GTF-derived position information to add genomic coordinates.
Uses parallel processing for efficient handling of large files.

Usage:
    python pred_merge_gtf.py -g gtf_positions.csv -p predictions.csv -o output.csv

Author: MARS Team
"""

import os
import sys
import logging
import argparse
from typing import List, Tuple
import multiprocessing as mp

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default chunk size for reading large files
DEFAULT_CHUNK_SIZE = 100000


def process_chunk(
    chunk: pd.DataFrame,
    gtf_data: pd.DataFrame,
    chunk_index: int,
    total_chunks: int
) -> List[str]:
    """
    Process a chunk of GTF data and match with predictions.

    Args:
        chunk: DataFrame chunk from GTF file
        gtf_data: DataFrame with prediction data
        chunk_index: Current chunk index
        total_chunks: Total number of chunks

    Returns:
        List of matched result strings
    """
    result = []

    for _, row in chunk.iterrows():
        transcript_id = row['transcript_id']
        tx_pos = row['tx_pos']

        # Find matching rows in prediction data (tx_pos - 1 adjustment)
        matching_rows = gtf_data[
            (gtf_data['transcript_id'] == transcript_id) &
            (gtf_data['tx_pos'] == tx_pos - 1)
        ]

        for _, match in matching_rows.iterrows():
            # Combine GTF position info with prediction data
            result_line = (
                f"{row['transcript_id']},"
                f"{row['tx_pos']},"
                f"{row['g_pos']},"
                f"{row['chromosome']},"
                f"{row['strand']},"
                f"{match['tx_pos']},"
                f"{match.get('base', '')},"
                f"{match.get('value', '')}"
            )
            result.append(result_line)

    if (chunk_index + 1) % 10 == 0 or chunk_index == total_chunks - 1:
        logger.info(f"Processed chunk {chunk_index + 1}/{total_chunks}")

    return result


def merge_gtf_predictions(
    gtf_file: str,
    predictions_file: str,
    output_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    num_workers: int = None
) -> None:
    """
    Merge GTF position data with predictions.

    Args:
        gtf_file: Path to GTF positions CSV file
        predictions_file: Path to predictions CSV file
        output_file: Path to output file
        chunk_size: Number of rows per chunk
        num_workers: Number of parallel workers
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    logger.info(f"Loading predictions from: {predictions_file}")

    # Define column names for prediction file
    pred_columns = [
        'transcript_id', 'tx_pos', 'value', 'base', 'strand',
        'col5', 'col6', 'col7', 'col8', 'col9'
    ]

    predictions_data = pd.read_csv(
        predictions_file,
        header=0,
        names=pred_columns[:10]  # Adjust based on actual columns
    )
    predictions_data['tx_pos'] = predictions_data['tx_pos'].astype(int)

    logger.info(f"Loaded {len(predictions_data)} prediction rows")

    # Count total chunks
    logger.info(f"Counting chunks in GTF file...")
    total_chunks = sum(1 for _ in pd.read_csv(gtf_file, chunksize=chunk_size))
    logger.info(f"Total chunks to process: {total_chunks}")

    # Define column names for GTF file
    gtf_columns = ['transcript_id', 'tx_pos', 'g_pos', 'chromosome', 'strand']

    # Process with multiprocessing (using context manager to prevent semaphore leaks)
    results = []

    with mp.Pool(processes=num_workers) as pool:
        with pd.read_csv(
            gtf_file,
            header=0,
            names=gtf_columns,
            chunksize=chunk_size
        ) as reader:
            for chunk_index, chunk in enumerate(reader):
                chunk['tx_pos'] = chunk['tx_pos'].astype(int)
                async_result = pool.apply_async(
                    process_chunk,
                    args=(chunk, predictions_data, chunk_index, total_chunks)
                )
                results.append(async_result)

        # Wait for all results (pool.close() and pool.join() handled by context manager)
        results = [r.get() for r in results]

    # Write results
    logger.info(f"Writing results to: {output_file}")
    with open(output_file, 'w') as f:
        # Write header
        f.write("transcript_id,tx_pos,g_pos,chromosome,strand,pred_tx_pos,base,value\n")

        for result in results:
            for line in result:
                f.write(line + '\n')

    logger.info("Merge completed successfully")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge predictions with GTF position annotations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pred_merge_gtf.py -g gtf_positions.csv -p predictions.csv -o merged.csv
    python pred_merge_gtf.py -g gtf.csv -p pred.csv -o output.csv --chunk-size 50000
        """
    )

    parser.add_argument(
        '-g', '--gtf',
        required=True,
        help='GTF positions CSV file'
    )
    parser.add_argument(
        '-p', '--predictions',
        required=True,
        help='Predictions CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output merged CSV file'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f'Chunk size for processing (default: {DEFAULT_CHUNK_SIZE})'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("="*60)
    logger.info("GTF-Prediction Merger")
    logger.info("="*60)

    merge_gtf_predictions(
        args.gtf,
        args.predictions,
        args.output,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers
    )

    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
