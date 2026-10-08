#!/usr/bin/env python3
"""
Merge BED and CSV Files

Match and merge prediction results (CSV) with ground truth data (BED format)
based on chromosome and position coordinates.

Usage:
    python merge_bed.py -c predictions.csv -b ground_truth.bed -o output.csv

Author: MARS Team
"""

import csv
import sys
import logging
import argparse
from typing import List, Tuple, Dict, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def read_bed_file(bed_file_path: str) -> Dict[Tuple[str, int], List]:
    """
    Read BED file and index by chromosome and position.

    BED format: chrom, start, end, [additional fields...]
    Uses 'end' position (column 2, 0-indexed) for matching.

    Args:
        bed_file_path: Path to BED file

    Returns:
        Dictionary mapping (chromosome, position) to list of rows
    """
    bed_index: Dict[Tuple[str, int], List] = {}

    with open(bed_file_path, 'r') as bed_file:
        bed_reader = csv.reader(bed_file, delimiter='\t')
        for row in bed_reader:
            if len(row) < 3:
                continue

            chromosome = row[0]
            try:
                position = int(row[2])  # End position in BED format
            except ValueError:
                continue

            key = (chromosome, position)
            if key not in bed_index:
                bed_index[key] = []
            bed_index[key].append(row)

    logger.info(f"Loaded {len(bed_index)} unique positions from BED file")
    return bed_index


def read_csv_file(
    csv_file_path: str,
    chrom_col: int = 3,
    pos_col: int = 2,
    pos_offset: int = -2
) -> List[Tuple[str, int, List]]:
    """
    Read CSV file and extract chromosome, position, and full row data.

    Args:
        csv_file_path: Path to CSV file
        chrom_col: Column index for chromosome (0-indexed)
        pos_col: Column index for position (0-indexed)
        pos_offset: Offset to apply to position (for coordinate adjustment)

    Returns:
        List of (chromosome, adjusted_position, row) tuples
    """
    csv_data = []

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) <= max(chrom_col, pos_col):
                continue

            chromosome = row[chrom_col]
            try:
                position = int(row[pos_col]) + pos_offset
            except ValueError:
                continue

            csv_data.append((chromosome, position, row))

    logger.info(f"Loaded {len(csv_data)} rows from CSV file")
    return csv_data


def merge_files(
    csv_data: List[Tuple[str, int, List]],
    bed_index: Dict[Tuple[str, int], List]
) -> List[List]:
    """
    Merge CSV data with BED data based on matching chromosome and position.

    Args:
        csv_data: List of (chromosome, position, row) from CSV
        bed_index: Dictionary of BED data indexed by (chromosome, position)

    Returns:
        List of merged rows (CSV row + BED row)
    """
    merged_data = []

    for csv_chrom, csv_pos, csv_row in csv_data:
        key = (csv_chrom, csv_pos)
        if key in bed_index:
            for bed_row in bed_index[key]:
                merged_data.append(csv_row + bed_row)

    logger.info(f"Merged {len(merged_data)} matching rows")
    return merged_data


def write_output_file(
    output_file_path: str,
    merged_data: List[List],
    header: List[str] = None
) -> None:
    """
    Write merged data to output CSV file.

    Args:
        output_file_path: Path to output file
        merged_data: List of merged rows
        header: Optional header row
    """
    with open(output_file_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        if header:
            csv_writer.writerow(header)

        for row in merged_data:
            csv_writer.writerow(row)

    logger.info(f"Output saved to: {output_file_path}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge prediction CSV with ground truth BED file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python merge_bed.py -c predictions.csv -b ground_truth.bed -o merged.csv
    python merge_bed.py -c data.csv -b sites.bed -o output.csv --chrom-col 0 --pos-col 1
        """
    )

    parser.add_argument(
        '-c', '--csv',
        required=True,
        help='Path to input CSV file with predictions'
    )
    parser.add_argument(
        '-b', '--bed',
        required=True,
        help='Path to BED file with ground truth positions'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to output merged CSV file'
    )
    parser.add_argument(
        '--chrom-col',
        type=int,
        default=3,
        help='Column index for chromosome in CSV (0-indexed, default: 3)'
    )
    parser.add_argument(
        '--pos-col',
        type=int,
        default=2,
        help='Column index for position in CSV (0-indexed, default: 2)'
    )
    parser.add_argument(
        '--pos-offset',
        type=int,
        default=-2,
        help='Position offset for coordinate adjustment (default: -2)'
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
    logger.info("BED/CSV File Merger")
    logger.info("="*60)

    # Read input files
    bed_index = read_bed_file(args.bed)
    csv_data = read_csv_file(
        args.csv,
        chrom_col=args.chrom_col,
        pos_col=args.pos_col,
        pos_offset=args.pos_offset
    )

    # Merge data
    merged_data = merge_files(csv_data, bed_index)

    if not merged_data:
        logger.warning("No matching positions found between files")
    else:
        # Write output
        write_output_file(args.output, merged_data)

    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
