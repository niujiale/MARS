#!/usr/bin/env python3
"""
XGBoost Model Prediction for RNA Modification

Apply a trained XGBoost model to predict RNA modification levels on new data.
Supports parallel processing for large datasets.

Installed as the `mars-predict` / `xgb_model_pred` command, and as the
`mars predict` subcommand.

Usage:
    mars-predict -i /path/to/data -model /path/to/model.json
    mars-predict -i /path/to/data --mod-type m6a --list-models
    mars-predict -i /path/to/data --mod-type m6a --pretrained curlcake_DRACH_2244test

Author: MARS Team
"""

import os
import re
import sys
import glob
import time
import gc
import logging
import argparse
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count

import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, vstack
from tqdm.auto import tqdm

from .._version import __version__
from ..paths import (
    PRETRAINED_ENV_VAR,
    describe_pretrained_lookup,
    find_pretrained_models_dir,
)

MERGE_LABEL_PATTERN = re.compile(r'merge_(\d+)_percent')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROG = 'mars-predict'

# Available modification types and their pretrained models
AVAILABLE_MOD_TYPES = ['m6a', 'm5c', 'm1a', 'psu', 'ac4C', 'f5c', 'h5c', 'nanomud_psu']


def get_pretrained_models(mod_type: str, pretrained_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get available pretrained models for a modification type.

    Args:
        mod_type: Modification type (e.g., 'm6a', 'm5c')
        pretrained_dir: Optional explicit pretrained_models directory

    Returns:
        Dictionary mapping model name to model path
    """
    models = {}
    root = find_pretrained_models_dir(pretrained_dir)
    mod_dir = os.path.join(str(root), mod_type)

    if not os.path.exists(mod_dir):
        return models

    # Search for .json files recursively
    for root_dir, _dirs, files in os.walk(mod_dir):
        for f in files:
            if f.endswith('.json'):
                model_name = f.replace('_final_model.json', '').replace('_model.json', '')
                models[model_name] = os.path.join(root_dir, f)

    return models


def list_pretrained_models(
    mod_type: Optional[str] = None,
    pretrained_dir: Optional[str] = None,
) -> None:
    """
    List available pretrained models.

    Args:
        mod_type: Optional modification type to filter by
        pretrained_dir: Optional explicit pretrained_models directory
    """
    print("\n" + "=" * 60)
    print("Available Pretrained Models")
    print("=" * 60)

    try:
        root = find_pretrained_models_dir(pretrained_dir)
    except FileNotFoundError as exc:
        print(f"\n{exc}\n")
        return

    print(f"Models directory: {root}")

    mod_types = [mod_type] if mod_type else AVAILABLE_MOD_TYPES

    for mt in mod_types:
        models = get_pretrained_models(mt, pretrained_dir)
        if models:
            print(f"\n[{mt.upper()}] ({len(models)} models)")
            print("-" * 40)
            for name in sorted(models):
                print(f"  {name}")

    print("\n" + "=" * 60)
    print(f"Usage: {PROG} -i <data> --mod-type <type> --pretrained <model_name>")
    print("=" * 60 + "\n")


def build_parser(prog: str = PROG) -> argparse.ArgumentParser:
    """Build the argument parser for the prediction CLI."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Apply trained XGBoost model for RNA modification prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Use a custom model file
    {prog} -i /path/to/feature_dir -model /path/to/model.json

    # List available pretrained models
    {prog} --list-models
    {prog} --mod-type m6a --list-models

    # Use a pretrained model
    {prog} -i /path/to/data --mod-type m6a --pretrained curlcake_DRACH_2244test
    {prog} -i /path/to/data --mod-type m5c --pretrained eligos_m5c_C1test

    # Specify custom output path
    {prog} -i /path/to/data -model model.json -o /path/to/output/result.txt

Pretrained models are located automatically from a git checkout or release
tarball. Override with --pretrained-dir or ${PRETRAINED_ENV_VAR}.
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s (MARS) {__version__}'
    )
    parser.add_argument(
        '-i', '--input',
        help='Base path containing feature directory'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: <input_dir>/modification_pred/modification_pred.txt)'
    )
    parser.add_argument(
        '-model', '--model_path',
        help='Path to the trained XGBoost model file (.json)'
    )
    parser.add_argument(
        '--mod-type', '--mod_type',
        choices=AVAILABLE_MOD_TYPES,
        help=f'Modification type for pretrained models: {", ".join(AVAILABLE_MOD_TYPES)}'
    )
    parser.add_argument(
        '--pretrained',
        help='Name of pretrained model to use (use --list-models to see available)'
    )
    parser.add_argument(
        '--pretrained-dir', '--pretrained_dir',
        dest='pretrained_dir',
        help=f'Directory holding pretrained models '
             f'(default: auto-detected; override with ${PRETRAINED_ENV_VAR})'
    )
    parser.add_argument(
        '--list-models', '--list_models',
        action='store_true',
        help='List available pretrained models and exit'
    )
    parser.add_argument(
        '--output_name',
        default='modification_pred',
        help='Name for output subdirectory and file prefix when -o is not specified '
             '(default: modification_pred)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser


def parse_arguments(
    argv: Optional[List[str]] = None,
    prog: str = PROG,
) -> argparse.Namespace:
    """Parse command line arguments and resolve the model path."""
    parser = build_parser(prog)
    args = parser.parse_args(argv)

    # Handle --list-models
    if args.list_models:
        list_pretrained_models(args.mod_type, args.pretrained_dir)
        sys.exit(0)

    # Validate arguments for prediction
    if not args.input:
        parser.error("--input is required for prediction")

    # Resolve model path
    if args.pretrained:
        if not args.mod_type:
            parser.error("--mod-type is required when using --pretrained")

        try:
            models = get_pretrained_models(args.mod_type, args.pretrained_dir)
        except FileNotFoundError as exc:
            parser.error(str(exc))

        if args.pretrained not in models:
            available = ', '.join(sorted(models)[:5])
            parser.error(
                f"Pretrained model '{args.pretrained}' not found for {args.mod_type}. "
                f"Available: {available}... (use --list-models to see all)"
            )
        args.model_path = models[args.pretrained]
        logger.info(f"Using pretrained model: {args.pretrained}")

    if not args.model_path:
        parser.error("Either --model or (--mod-type and --pretrained) is required")

    return args


def get_file_list(base_path: str) -> List[str]:
    """
    Get list of NPZ feature files from feature directory.

    Args:
        base_path: Base directory path

    Returns:
        List of NPZ file paths

    Raises:
        FileNotFoundError: If no NPZ files found
    """
    root_dir = os.path.join(base_path, 'feature')
    pattern = os.path.join(root_dir, '**/forward/Xdata/*.npz')
    file_list = sorted(glob.glob(pattern, recursive=True))

    if not file_list:
        raise FileNotFoundError(
            f"No NPZ feature files found matching pattern: {pattern}\n"
            "Please check that the path contains a 'feature' directory "
            "with the expected structure."
        )

    logger.info(f"Found {len(file_list)} NPZ files")
    return file_list


def load_model(model_path: str) -> xgb.XGBRegressor:
    """
    Load a trained XGBoost model.

    Args:
        model_path: Path to model file

    Returns:
        Loaded XGBRegressor model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    logger.info(f"Model loaded from: {model_path}")

    return model


def process_file(
    filepath: str,
) -> Optional[Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load a single NPZ feature file.

    Args:
        filepath: Path to NPZ file (e.g. .../<transcript>/forward/Xdata/X_1.npz)

    Returns:
        Tuple (X_sparse, transcripts, loci, strand, depth, y_test) of length
        n_rows arrays aligned with the NPZ rows, or None on error. y_test is
        derived from the enclosing merge_X_percent directory when present
        (training-data convention), else 0.0.
    """
    try:
        # Structure: .../<transcript>/forward/Xdata/X_*.npz
        xdata_dir = os.path.dirname(filepath)
        transcript_dir = os.path.basename(
            os.path.dirname(os.path.dirname(xdata_dir))
        )

        label_value = 0.0
        ancestor = xdata_dir
        for _ in range(6):
            ancestor = os.path.dirname(ancestor)
            if not ancestor:
                break
            m = MERGE_LABEL_PATTERN.search(os.path.basename(ancestor))
            if m:
                label_value = int(m.group(1)) / 100
                break

        data = np.load(filepath, allow_pickle=True)
        X = csr_matrix(
            (data['data'], data['indices'], data['indptr']),
            shape=tuple(data['shape'])
        )
        n_rows = X.shape[0]
        if n_rows == 0:
            logger.debug(f"Skipping empty NPZ: {filepath}")
            return None

        # NPZ stores chromosome/strand as 0-d ndarray of Python strings and
        # loci/depth as 1-d arrays aligned with the CSR rows.
        chrom_name = str(data['chromosome']) if 'chromosome' in data.files else transcript_dir
        strand_val = str(data['strand']) if 'strand' in data.files else '+'
        loci_arr = np.asarray(data['loci'], dtype=np.int64) if 'loci' in data.files \
            else np.zeros(n_rows, dtype=np.int64)
        depth_arr = np.asarray(data['depth'], dtype=np.int64) if 'depth' in data.files \
            else np.zeros(n_rows, dtype=np.int64)

        transcripts = np.full(n_rows, chrom_name, dtype=object)
        strand_col = np.full(n_rows, strand_val, dtype=object)
        y_test = np.full(n_rows, label_value, dtype=np.float32)

        return X, transcripts, loci_arr, strand_col, depth_arr, y_test

    except Exception as e:
        logger.warning(f"Error processing {filepath}: {e}")
        return None


def write_results(
    output_path: str,
    results: List[Tuple[str, int, str, int, float, float]]
) -> None:
    """
    Write prediction results with position metadata.

    Columns: trans, pos, strand, depth, y_pred, y_test (tab-separated).
    `y_pred` is the predicted modification rate in [0, 1]; `y_test` is 0.0
    for inference data (non-zero only for merge_X_percent training dirs).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("trans\tpos\tstrand\tdepth\ty_pred\ty_test\n")
        for trans, pos, strand, depth, pred, actual in results:
            f.write(
                f"{trans}\t{pos}\t{strand}\t{depth}\t{pred:.6f}\t{actual:.6f}\n"
            )

    logger.info(f"Results saved to: {output_path}")


def main(argv: Optional[List[str]] = None, prog: str = PROG) -> None:
    """Main function for model prediction."""
    args = parse_arguments(argv, prog)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Pretrained model lookup:\n%s", describe_pretrained_lookup())

    base_path = args.input
    model_path = args.model_path
    output_name = args.output_name
    num_workers = args.num_workers or cpu_count()

    # Define output path
    if args.output:
        output_predictions_path = args.output
    else:
        output_predictions_path = os.path.join(
            base_path, f'{output_name}/{output_name}.txt'
        )

    logger.info("=" * 60)
    logger.info(f"MARS {__version__} - XGBoost Model Prediction")
    logger.info("=" * 60)
    logger.info(f"Input: {base_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_predictions_path}")
    logger.info(f"Workers: {num_workers}")

    # Get file list
    file_list = get_file_list(base_path)

    # Load model
    model = load_model(model_path)

    # Create output directory
    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)

    # Start timing
    start_time = time.time()

    # Process files in parallel
    logger.info("Loading NPZ feature files...")
    with Pool(num_workers) as pool:
        loaded = list(
            tqdm(
                pool.imap_unordered(process_file, file_list),
                total=len(file_list),
                desc="Combining NPZ",
                unit="file",
                dynamic_ncols=True,
            )
        )

    loaded = [item for item in loaded if item is not None]
    if not loaded:
        raise ValueError("No valid NPZ files found after processing")

    logger.info("Assembling feature matrix...")
    X_test = vstack([item[0] for item in loaded]).toarray()
    transcripts = np.concatenate([item[1] for item in loaded])
    loci = np.concatenate([item[2] for item in loaded])
    strand = np.concatenate([item[3] for item in loaded])
    depth = np.concatenate([item[4] for item in loaded])
    y_test = np.concatenate([item[5] for item in loaded])
    logger.info(f"Feature matrix shape: {X_test.shape}")

    del loaded
    gc.collect()

    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 1)

    results = list(zip(transcripts, loci, strand, depth, y_pred, y_test))

    del X_test
    gc.collect()

    elapsed_time = time.time() - start_time
    logger.info(f"Processing time: {elapsed_time:.2f} seconds")

    write_results(output_predictions_path, results)

    logger.info("=" * 60)
    logger.info("Prediction completed successfully!")
    logger.info("=" * 60)


def cli_main(argv: Optional[List[str]] = None, prog: str = PROG) -> None:
    """Console-script wrapper: report failures without a traceback."""
    try:
        main(argv, prog)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def main_with_prog(argv: Optional[List[str]] = None) -> None:
    """Entry point for `mars predict`, so --help shows the right prog name."""
    cli_main(argv, prog='mars predict')


if __name__ == "__main__":
    cli_main()
