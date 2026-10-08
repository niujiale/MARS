#!/usr/bin/env python3
"""
Merge Prediction Position Files

Merge multiple prediction files by transcript_id and position coordinates.
Useful for combining replicate experiments or comparing different conditions.

Usage:
    python merge_pred_position.py -i file1.txt file2.txt file3.txt -o merged.csv
    python merge_pred_position.py --input-list files.txt -o merged.csv

Author: MARS Team
"""

import sys
import logging
import argparse
from typing import List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def read_prediction_file(
    file_path: str,
    file_index: int,
    transcript_col: int = 0,
    position_col: int = 1,
    value_col: int = 2
) -> pd.DataFrame:
    """
    Read a prediction file and extract relevant columns.

    Args:
        file_path: Path to prediction file
        file_index: Index number for naming the value column
        transcript_col: Column index for transcript ID
        position_col: Column index for position
        value_col: Column index for prediction value

    Returns:
        DataFrame with transcript_id, tx_pos, and value columns
    """
    logger.info(f"Reading file {file_index + 1}: {file_path}")

    df = pd.read_csv(file_path, header=None)

    # Extract and rename columns
    result = pd.DataFrame({
        'transcript_id': df.iloc[:, transcript_col],
        'tx_pos': df.iloc[:, position_col],
        f'value_{file_index}': df.iloc[:, value_col]
    })

    logger.debug(f"  Loaded {len(result)} rows")
    return result


def merge_prediction_files(
    file_paths: List[str],
    transcript_col: int = 0,
    position_col: int = 1,
    value_col: int = 2,
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merge multiple prediction files by transcript_id and position.

    Args:
        file_paths: List of paths to prediction files
        transcript_col: Column index for transcript ID
        position_col: Column index for position
        value_col: Column index for prediction value
        how: Merge method ('inner', 'outer', 'left', 'right')

    Returns:
        Merged DataFrame
    """
    if not file_paths:
        raise ValueError("No input files provided")

    # Read first file as base
    merged_data = read_prediction_file(
        file_paths[0], 0,
        transcript_col, position_col, value_col
    )

    # Merge remaining files
    for i, file_path in enumerate(file_paths[1:], start=1):
        df = read_prediction_file(
            file_path, i,
            transcript_col, position_col, value_col
        )

        merged_data = pd.merge(
            merged_data,
            df,
            on=['transcript_id', 'tx_pos'],
            how=how
        )

        logger.info(f"  After merge: {len(merged_data)} rows")

    return merged_data


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge multiple prediction files by transcript and position.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge multiple files specified on command line
    python merge_pred_position.py -i rep1.txt rep2.txt rep3.txt -o merged.csv

    # Merge files listed in a text file (one path per line)
    python merge_pred_position.py --input-list file_paths.txt -o merged.csv

    # Specify column indices
    python merge_pred_position.py -i *.txt -o merged.csv --transcript-col 0 --position-col 1 --value-col 2

    # Use outer join to keep all positions
    python merge_pred_position.py -i rep1.txt rep2.txt -o merged.csv --merge-type outer
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--input',
        nargs='+',
        help='Input prediction files to merge'
    )
    input_group.add_argument(
        '--input-list',
        help='Text file containing list of input file paths (one per line)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output merged CSV file path'
    )
    parser.add_argument(
        '--transcript-col',
        type=int,
        default=0,
        help='Column index for transcript ID (0-indexed, default: 0)'
    )
    parser.add_argument(
        '--position-col',
        type=int,
        default=1,
        help='Column index for position (0-indexed, default: 1)'
    )
    parser.add_argument(
        '--value-col',
        type=int,
        default=2,
        help='Column index for prediction value (0-indexed, default: 2)'
    )
    parser.add_argument(
        '--merge-type',
        choices=['inner', 'outer', 'left', 'right'],
        default='inner',
        help='Merge method (default: inner)'
    )
    parser.add_argument(
        '--keep-keys',
        action='store_true',
        help='Keep transcript_id and tx_pos columns in output'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get input file list
    if args.input:
        file_paths = args.input
    else:
        with open(args.input_list, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]

    logger.info("="*60)
    logger.info("Prediction File Merger")
    logger.info("="*60)
    logger.info(f"Input files: {len(file_paths)}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Merge type: {args.merge_type}")

    # Validate input files
    import os
    for fp in file_paths:
        if not os.path.exists(fp):
            logger.error(f"File not found: {fp}")
            sys.exit(1)

    # Merge files
    merged_data = merge_prediction_files(
        file_paths,
        transcript_col=args.transcript_col,
        position_col=args.position_col,
        value_col=args.value_col,
        how=args.merge_type
    )

    # Optionally remove key columns
    if not args.keep_keys:
        output_data = merged_data.drop(columns=['transcript_id', 'tx_pos'])
    else:
        output_data = merged_data

    # Save output
    output_data.to_csv(args.output, index=False)
    logger.info(f"Merged file saved to: {args.output}")
    logger.info(f"Final shape: {output_data.shape}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
