#!/usr/bin/env python3
"""
XGBoost Model Training for RNA Modification Prediction

Train and evaluate an XGBoost regression model for predicting RNA modification levels.
Supports automatic train/test split by chromosome and generates evaluation metrics and plots.

Installed as the `mars-train` / `xgb_model` command, and as the `mars train`
subcommand.

Usage:
    mars-train -i /path/to/feature_dir -o /path/to/output_dir

Author: MARS Team
"""

import os
import sys
import glob
import random
import re
import logging
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, vstack
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from .._version import __version__

MERGE_LABEL_PATTERN = re.compile(r'merge_(\d+)_percent')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROG = 'mars-train'

# Constants
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2
MERGE_PATTERN = r'merge_\d+_percent_vs_\d+_percent'

_EXTRA_HINT = (
    "Install the training extras with:\n"
    "    pip install 'mars-nanopore[train]'\n"
    "or, inside the conda env:\n"
    "    pip install matplotlib seaborn scikit-learn"
)


def _require_sklearn():
    """Import scikit-learn metrics lazily; they are a training-only extra."""
    try:
        from sklearn.metrics import mean_squared_error, r2_score
    except ImportError as exc:
        raise ImportError(
            f"scikit-learn is required for model training/evaluation.\n{_EXTRA_HINT}"
        ) from exc
    return mean_squared_error, r2_score


def _require_plotting():
    """Import matplotlib/seaborn lazily; they are a training-only extra."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless servers have no display
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            f"matplotlib and seaborn are required to render training plots.\n{_EXTRA_HINT}"
        ) from exc
    return plt, sns


def get_chromosome_names(root_dir: str) -> Tuple[List[str], bool]:
    """
    Extract chromosome names from the directory structure.

    Args:
        root_dir: Root directory containing feature data

    Returns:
        Tuple of (chromosome names list, is_merge_pattern flag)
    """
    first_level_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    is_merge_pattern = all(re.match(MERGE_PATTERN, d) for d in first_level_dirs)

    if is_merge_pattern:
        # Nested directory structure: merge_X_percent_vs_Y_percent/chrom/
        chrom_dirs = []
        for sub_dir in first_level_dirs:
            sub_path = os.path.join(root_dir, sub_dir)
            if os.path.isdir(sub_path):
                for d in os.listdir(sub_path):
                    if os.path.isdir(os.path.join(sub_path, d)):
                        chrom_dirs.append(d)
    else:
        # Direct directory structure: chrom/forward/Xdata/*.npz
        chrom_dirs = []
        for chrom in first_level_dirs:
            xdata_dir = os.path.join(root_dir, chrom, 'forward/Xdata')
            if os.path.isdir(xdata_dir) and glob.glob(os.path.join(xdata_dir, '*.npz')):
                chrom_dirs.append(chrom)

    chrom_names = sorted(set(chrom_dirs))
    logger.info(f"Found {len(chrom_names)} chromosomes/transcripts")

    return chrom_names, is_merge_pattern


def split_chromosomes(
    chrom_names: List[str],
    is_merge_pattern: bool,
    seed: int = RANDOM_SEED
) -> Tuple[List[str], List[str]]:
    """
    Split chromosomes into training and testing sets.

    Args:
        chrom_names: List of chromosome names
        is_merge_pattern: Whether data uses merge pattern structure
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training chromosomes, testing chromosomes)
    """
    rng = random.Random(seed)

    if is_merge_pattern:
        # For merge pattern, randomly select one chromosome for testing
        test_chrom = rng.choice(chrom_names)
        train_chrom_names = [c for c in chrom_names if c != test_chrom]
        test_chrom_names = [test_chrom]
    else:
        # Standard split: 20% for testing
        shuffled = list(chrom_names)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * TEST_SPLIT_RATIO))
        test_chrom_names = shuffled[:split_idx]
        train_chrom_names = shuffled[split_idx:]

    return train_chrom_names, test_chrom_names


def generate_file_patterns(
    root_dir: str,
    train_chroms: List[str],
    test_chroms: List[str],
    is_merge_pattern: bool
) -> Tuple[List[str], List[str]]:
    """
    Generate file path patterns for training and testing data.

    Args:
        root_dir: Root directory path
        train_chroms: Training chromosome names
        test_chroms: Testing chromosome names
        is_merge_pattern: Whether data uses merge pattern structure

    Returns:
        Tuple of (training patterns, testing patterns)
    """
    if is_merge_pattern:
        train_patterns = [
            os.path.join(root_dir, f'**/{chrom}/forward/Xdata/*.npz')
            for chrom in train_chroms
        ]
        test_patterns = [
            os.path.join(root_dir, f'**/{chrom}/forward/Xdata/*.npz')
            for chrom in test_chroms
        ]
    else:
        train_patterns = [
            os.path.join(root_dir, f'{chrom}/forward/Xdata/*.npz')
            for chrom in train_chroms
        ]
        test_patterns = [
            os.path.join(root_dir, f'{chrom}/forward/Xdata/*.npz')
            for chrom in test_chroms
        ]

    return train_patterns, test_patterns


def define_output_paths(base_path: str, test_chrom: str) -> Dict[str, str]:
    """
    Define output file paths for results.

    Args:
        base_path: Base output directory
        test_chrom: Name of test chromosome (used in filenames)

    Returns:
        Dictionary of output path names to paths
    """
    return {
        'predictions_path': os.path.join(
            base_path, f'y_pred-y_test_file/{test_chrom}_predictions_vs_actuals.txt'
        ),
        'outliers_path': os.path.join(
            base_path, f'y_pred-y_test_file/{test_chrom}_outliers.txt'
        ),
        'model_path': os.path.join(
            base_path, f'json/{test_chrom}_model.json'
        ),
        'scatter_plot_path': os.path.join(
            base_path, f'plot/scatter_plot/{test_chrom}_scatter.png'
        ),
        'heatmap_path': os.path.join(
            base_path, f'plot/heatmap/{test_chrom}_heatmap.png'
        )
    }


def create_output_directories(output_paths: Dict[str, str]) -> None:
    """Create necessary directories for output files."""
    for path in output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _label_for_npz(filepath: str) -> float:
    """Extract merge_X_percent label from ancestor directories, else 0.0."""
    ancestor = os.path.dirname(filepath)
    for _ in range(6):
        ancestor = os.path.dirname(ancestor)
        if not ancestor:
            break
        m = MERGE_LABEL_PATTERN.search(os.path.basename(ancestor))
        if m:
            return int(m.group(1)) / 100
    return 0.0


def read_and_combine_data(patterns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read NPZ feature files matching the given glob patterns and stack them
    into a single (features, labels) pair.

    Labels are derived per-file from the enclosing merge_X_percent directory
    name and broadcast to every row of that NPZ.

    Raises:
        ValueError: if no valid data found.
    """
    X_blocks: List[csr_matrix] = []
    y_blocks: List[np.ndarray] = []

    all_files: List[str] = []
    for pattern in patterns:
        file_list = sorted(glob.glob(pattern, recursive=True))
        if not file_list:
            logger.warning(f"No files found matching pattern: {pattern}")
            continue
        all_files.extend(file_list)

    for filepath in tqdm(
        all_files,
        desc="Combining NPZ",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            data = np.load(filepath, allow_pickle=True)
            X = csr_matrix(
                (data['data'], data['indices'], data['indptr']),
                shape=tuple(data['shape'])
            )
            if X.shape[0] == 0:
                continue
            X_blocks.append(X)
            y_blocks.append(
                np.full(X.shape[0], _label_for_npz(filepath), dtype=np.float32)
            )
        except Exception as e:
            tqdm.write(f"[WARN] Error reading {filepath}: {e}")

    if not X_blocks:
        raise ValueError("No valid data found after processing all files")

    X_all = vstack(X_blocks).toarray()
    y_all = np.concatenate(y_blocks)
    logger.info(f"Combined data shape: {X_all.shape}")

    return X_all, y_all


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_path: str,
    seed: int = RANDOM_SEED,
    **kwargs
) -> xgb.XGBRegressor:
    """
    Initialize, train, and save an XGBoost regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_path: Path to save the trained model
        seed: Random seed passed to XGBoost as random_state
        **kwargs: Additional XGBoost parameters

    Returns:
        Trained XGBRegressor model
    """
    default_params = {
        'objective': 'reg:squarederror',
        'colsample_bytree': 0.3,
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 200,
        'random_state': seed
    }
    default_params.update(kwargs)

    logger.info("Training XGBoost model...")
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    model.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    return model


def evaluate_and_save_results(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    predictions_path: str,
    num_features: int
) -> Dict[str, float]:
    """
    Evaluate model performance and save predictions.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        predictions_path: Path to save predictions
        num_features: Number of features (for adjusted R²)

    Returns:
        Dictionary of evaluation metrics
    """
    mean_squared_error, r2_score = _require_sklearn()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr, p_value = pearsonr(y_pred, y_test)

    # Manual R² calculation for verification
    y_mean = np.mean(y_test)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Adjusted R²
    n = len(y_test)
    p = num_features
    adjusted_r2 = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

    metrics = {
        'mse': mse,
        'r2': r2,
        'adjusted_r2': adjusted_r2,
        'pearson_r': corr,
        'pearson_p': p_value
    }

    # Log metrics
    logger.info(f"Mean Squared Error: {mse:.6f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Adjusted R² Score: {adjusted_r2:.4f}")
    logger.info(f"Pearson Correlation: {corr:.4f} (p={p_value:.2e})")

    # Save predictions
    with open(predictions_path, 'w') as f:
        f.write("y_pred\ty_test\n")
        for pred, actual in zip(y_pred, y_test):
            f.write(f"{pred:.6f}\t{actual:.6f}\n")

    logger.info(f"Predictions saved to: {predictions_path}")

    return metrics


def identify_and_save_outliers(
    predictions_path: str,
    outliers_path: str,
    scatter_plot_path: str,
    heatmap_path: str,
    low_threshold: float = 0.2,
    high_threshold: float = 0.8,
    pred_threshold: float = 0.5
) -> None:
    """
    Identify outliers and generate visualization plots.

    Args:
        predictions_path: Path to predictions file
        outliers_path: Path to save outliers
        scatter_plot_path: Path to save scatter plot
        heatmap_path: Path to save heatmap
        low_threshold: Lower bound for y_test outlier detection
        high_threshold: Upper bound for y_test outlier detection
        pred_threshold: Threshold for y_pred in outlier detection
    """
    plt, sns = _require_plotting()

    data = pd.read_csv(predictions_path, sep='\t')

    # Clip predictions to [0, 1]
    data['y_pred'] = np.clip(data['y_pred'], 0, 1)

    # Calculate Pearson correlation
    pcc, _ = pearsonr(data['y_test'], data['y_pred'])
    logger.info(f"Pearson Correlation (clipped): {pcc:.4f}")

    # Generate scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(data['y_test'], data['y_pred'], alpha=0.3, s=10)
    plt.xlabel('Actual Values (y_test)', fontsize=12)
    plt.ylabel('Predicted Values (y_pred)', fontsize=12)
    plt.title(f'Actual vs Predicted (PCC = {pcc:.4f})', fontsize=14)

    # Add diagonal line
    min_val = min(data['y_test'].min(), data['y_pred'].min())
    max_val = max(data['y_test'].max(), data['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Scatter plot saved to: {scatter_plot_path}")

    # Identify outliers
    low_outliers = data[
        (data['y_test'] < low_threshold) & (data['y_pred'] > pred_threshold)
    ]
    high_outliers = data[
        (data['y_test'] > high_threshold) & (data['y_pred'] < pred_threshold)
    ]
    outliers = pd.concat([low_outliers, high_outliers])

    total_points = len(data)
    low_ratio = len(low_outliers) / total_points if total_points > 0 else 0
    high_ratio = len(high_outliers) / total_points if total_points > 0 else 0

    logger.info(f"Low-value outliers: {len(low_outliers)} ({low_ratio:.2%})")
    logger.info(f"High-value outliers: {len(high_outliers)} ({high_ratio:.2%})")

    outliers.to_csv(outliers_path, sep='\t', index=False)
    logger.info(f"Outliers saved to: {outliers_path}")

    # Generate heatmap
    bins = 101
    heatmap_data, _xedges, _yedges = np.histogram2d(
        data['y_test'], data['y_pred'], bins=(bins, bins)
    )
    log_heatmap = np.log1p(heatmap_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        log_heatmap.T,
        cmap="viridis",
        cbar_kws={'label': 'Log Frequency'}
    )
    plt.xlabel('Actual Values (y_test)', fontsize=12)
    plt.ylabel('Predicted Values (y_pred)', fontsize=12)
    plt.title('Density Heatmap of Actual vs Predicted', fontsize=14)

    # Set axis ticks
    tick_positions = np.linspace(0, bins - 1, 11).astype(int)
    tick_labels = np.linspace(0, 1, 11)
    plt.xticks(tick_positions, [f'{t:.1f}' for t in tick_labels])
    plt.yticks(tick_positions, [f'{t:.1f}' for t in tick_labels])
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Heatmap saved to: {heatmap_path}")


def run(args: argparse.Namespace) -> None:
    """Train and evaluate a model from parsed arguments."""
    root_dir = args.input_dir
    base_path = args.output_dir
    seed = args.seed

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info(f"MARS {__version__} - XGBoost Model Training")
    logger.info("=" * 60)
    logger.info(f"Random seed: {seed}")

    # Get chromosome names and determine data structure
    chrom_names, is_merge_pattern = get_chromosome_names(root_dir)

    if not chrom_names:
        raise ValueError(f"No valid chromosomes found in {root_dir}")

    # Split into train/test
    train_chroms, test_chroms = split_chromosomes(chrom_names, is_merge_pattern, seed=seed)

    logger.info(f"Training chromosomes: {len(train_chroms)}")
    logger.info(f"Testing chromosomes: {test_chroms}")

    # Generate file patterns
    train_patterns, test_patterns = generate_file_patterns(
        root_dir, train_chroms, test_chroms, is_merge_pattern
    )

    # Define output paths
    output_paths = define_output_paths(base_path, test_chroms[0])
    create_output_directories(output_paths)

    # Load data
    logger.info("Loading training data...")
    X_train, y_train = read_and_combine_data(train_patterns)
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    logger.info("Loading testing data...")
    X_test, y_test = read_and_combine_data(test_patterns)
    logger.info(f"Testing set: {X_test.shape[0]} samples")

    # Train model
    model = train_model(X_train, y_train, output_paths['model_path'], seed=seed)

    # Predict
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)

    # Evaluate and save results
    evaluate_and_save_results(
        y_test, y_pred,
        output_paths['predictions_path'],
        X_train.shape[1]
    )

    # Identify outliers and generate plots
    identify_and_save_outliers(
        output_paths['predictions_path'],
        output_paths['outliers_path'],
        output_paths['scatter_plot_path'],
        output_paths['heatmap_path']
    )

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


def build_parser(prog: str = PROG) -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Train and evaluate an XGBoost model for RNA modification prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    {prog} -i /path/to/feature_dir -o /path/to/output_dir
    {prog} -i /path/to/feature_dir -o /path/to/output_dir --seed 7
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s (MARS) {__version__}'
    )
    parser.add_argument(
        '-i', '--input_dir', '--input-dir',
        dest='input_dir',
        type=str,
        required=True,
        help='Root directory containing input feature data'
    )
    parser.add_argument(
        '-o', '--output_dir', '--output-dir',
        dest='output_dir',
        type=str,
        required=True,
        help='Directory for output files (models, predictions, plots)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser


def main(argv: Optional[List[str]] = None, prog: str = PROG) -> None:
    """Parse arguments and run training."""
    args = build_parser(prog).parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


def cli_main(argv: Optional[List[str]] = None, prog: str = PROG) -> None:
    """Console-script wrapper: report failures without a traceback."""
    try:
        main(argv, prog)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def main_with_prog(argv: Optional[List[str]] = None) -> None:
    """Entry point for `mars train`, so --help shows the right prog name."""
    cli_main(argv, prog='mars train')


if __name__ == "__main__":
    cli_main()
