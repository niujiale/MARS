#!/bin/bash
#
# MARS End-to-End Pipeline Script
#
# Runs alignment + feature extraction + prediction (with a pretrained
# m6A/DRACH model by default) on a single sample.
#
# Usage:
#   bash run_example.sh                              # run bundled example
#   bash run_example.sh <fastq> <ref.fa> <out_dir>   # run your own sample
#   bash run_example.sh <fastq> <ref.fa> <out_dir> [options]
#
# Options:
#   --mod TYPE       Candidate site type: A, T, G, C, DRACH, m5C (default: DRACH)
#   --pretrained N   Pretrained model name (default: curlcake_DRACH_2244test)
#   --mod-type T     Pretrained model family: m6a m5c m1a psu ac4C f5c h5c
#                    (default: m6a)
#   --threads N      Number of threads (default: 4)
#   --context N      Bases of context each side of a site (default: 10). Any
#                    value works, but pretrained models only accept the context
#                    they were trained with, so pair others with --skip-pred
#   --batch N        Batch size per NPZ (default: 100)
#   --gpu            Enable GPU acceleration (auto-fallback to CPU)
#   --run-level-normalize
#                    Rescale QV features by run-level QV stats (default: off;
#                    incompatible with the bundled pretrained models)
#   --clean          Clean output directory before running
#   --skip-pred      Stop after feature extraction; no prediction. Use when the
#                    bundled models cannot score the features (non-default
#                    --context or --run-level-normalize) or when building a
#                    training set
#   --help           Show this help
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example data directory
EXAMPLE_DIR="${SCRIPT_DIR}/data/example"

# Input files
FASTQ_FILE="${EXAMPLE_DIR}/example.fastq"
REFERENCE="${EXAMPLE_DIR}/example.fa"

# Output directory (inside example directory)
OUTPUT_DIR="${EXAMPLE_DIR}/output"

# Default parameters
MODIFICATION="DRACH"
THREADS=4
CONTEXT_SIZE=10
BATCH_SIZE=100
USE_GPU=false
SKIP_PRED=false
# The bundled pretrained models were trained on unscaled QV, so run-level
# normalization has to stay off for their predictions to mean anything.
RUN_LEVEL_NORMALIZE=false

# Default pretrained model mapping (matches pretrained_models/<mod-type>/)
PRED_MOD_TYPE="m6a"
PRED_PRETRAINED="curlcake_DRACH_2244test"

# =============================================================================
# Parse Arguments
# =============================================================================

show_help() {
    cat << EOF
MARS End-to-End Pipeline Script

Usage:
    bash run_example.sh                                    # bundled example
    bash run_example.sh <fastq> <ref.fa> <output_dir>      # your own sample
    bash run_example.sh <fastq> <ref.fa> <output_dir> [options]

Arguments (all or nothing — give zero or all three):
    fastq           Input FASTQ (reads)
    ref.fa          Reference genome / transcriptome FASTA
    output_dir      Output directory (will be created)

Options:
    --mod TYPE        Candidate site type: A, T, G, C, DRACH, m5C (default: DRACH)
    --pretrained N    Pretrained model name (default: curlcake_DRACH_2244test)
    --mod-type T      Pretrained model family: m6a m5c m1a psu ac4C f5c h5c
                      (default: m6a). Must be compatible with --pretrained.
    --threads N       Number of threads (default: 4)
    --context N       Bases of context on each side of a candidate site
                      (default: 10). Any value works; feature width grows as
                      its square and sites whose window overhangs the
                      reference are dropped. Pretrained models only accept the
                      context they were trained with, so pair any other value
                      with --skip-pred and a model of your own.
    --batch N         Batch size per NPZ (default: 100)
    --gpu             Enable GPU acceleration (requires torch-cuda)
    --run-level-normalize
                      Rescale the QV mean/covariance features by the run-level
                      QV mean and standard deviation (default: off). The
                      bundled pretrained models were trained on unscaled QV,
                      so turning this on makes their predictions unusable.
    --output DIR      Override output directory (if you didn't pass it positionally)
    --clean           Clean output directory before running
    --skip-pred       Stop after feature extraction; no prediction. Use it
                      when the bundled models cannot score what you extracted
                      (a non-default --context, or --run-level-normalize), or
                      when you want the features as a training set.
    --help            Show this help

Examples:
    # Bundled example data
    bash run_example.sh
    bash run_example.sh --gpu

    # Your own sample — the most common case
    bash run_example.sh sample.fastq ref.fa ./out
    bash run_example.sh sample.fastq ref.fa ./out --gpu --threads 16

    # Different modification type
    bash run_example.sh sample.fastq ref.fa ./out \\
        --mod m5C --mod-type m5c --pretrained eligos_m5c_C1test

    # Features only, then train your own model on them. Needed for any
    # context other than the 10 the pretrained models were trained with.
    bash run_example.sh sample.fastq ref.fa ./out --context 25 --skip-pred
    mars train -i ./out/feature -o ./my_model

EOF
    exit 0
}

CLEAN_OUTPUT=false

# Collect positional args separately so they can appear before or after flags.
# Zero positionals => use bundled example. Three positionals => fastq/ref/out.
# Any other count is a user error.
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --run-level-normalize)
            RUN_LEVEL_NORMALIZE=true
            shift
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --context)
            CONTEXT_SIZE="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mod)
            MODIFICATION="$2"
            shift 2
            ;;
        --mod-type)
            PRED_MOD_TYPE="$2"
            shift 2
            ;;
        --pretrained)
            PRED_PRETRAINED="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN_OUTPUT=true
            shift
            ;;
        --skip-pred)
            SKIP_PRED=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        --*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Resolve positional args: 0 = example, 3 = user sample, anything else = error.
case "${#POSITIONAL[@]}" in
    0)
        ;;
    3)
        FASTQ_FILE="${POSITIONAL[0]}"
        REFERENCE="${POSITIONAL[1]}"
        OUTPUT_DIR="${POSITIONAL[2]}"
        ;;
    *)
        echo "Error: expected 0 or 3 positional arguments (fastq ref output_dir), got ${#POSITIONAL[@]}:"
        printf '  %s\n' "${POSITIONAL[@]}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "========================================"
echo "MARS Example Run"
echo "========================================"
echo ""

# Check input files
echo "Checking input files..."
if [[ ! -f "$FASTQ_FILE" ]]; then
    echo "ERROR: FASTQ file not found: $FASTQ_FILE"
    exit 1
fi
if [[ ! -f "$REFERENCE" ]]; then
    echo "ERROR: Reference file not found: $REFERENCE"
    exit 1
fi
echo "  FASTQ:     $FASTQ_FILE"
echo "  Reference: $REFERENCE"
echo ""

# Prefer the installed console scripts (`pip install -e .` puts them on PATH),
# and fall back to in-tree execution so the example still runs from a bare
# checkout that has not been installed yet.
if command -v mars &> /dev/null; then
    MARS_PIPELINE_CMD=(mars run)
    MARS_PREDICT_CMD=(mars predict)
    MARS_INVOCATION="installed console scripts"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
    MARS_PIPELINE_CMD=("$PYTHON_BIN" "${SCRIPT_DIR}/MARS_main.py")
    MARS_PREDICT_CMD=("$PYTHON_BIN" "${SCRIPT_DIR}/scripts/model/xgb_model_pred.py")
    MARS_INVOCATION="in-tree scripts (run 'bash install.sh' to get the 'mars' command)"
fi

# Check dependencies
echo "Checking dependencies..."
if ! command -v minimap2 &> /dev/null; then
    echo "WARNING: minimap2 not found in PATH"
    echo "  Install with: conda install -c bioconda minimap2"
fi
if ! command -v samtools &> /dev/null; then
    echo "WARNING: samtools not found in PATH"
    echo "  Install with: conda install -c bioconda samtools"
fi
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi
echo ""

# Clean output if requested
if [[ "$CLEAN_OUTPUT" == true ]] && [[ -d "$OUTPUT_DIR" ]]; then
    echo "Cleaning output directory..."
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Run Pipeline
# =============================================================================

GPU_FLAG=$([[ "$USE_GPU" == true ]] && echo "--gpu" || echo "--no-gpu")

NORMALIZE_ARGS=()
if [[ "$RUN_LEVEL_NORMALIZE" == true ]]; then
    NORMALIZE_ARGS+=(--run-level-normalize)
fi

echo "========================================"
echo "Configuration:"
echo "========================================"
echo "  Modification:   $MODIFICATION"
echo "  Threads:        $THREADS"
echo "  Context size:   $CONTEXT_SIZE"
echo "  Batch size:     $BATCH_SIZE"
echo "  Use GPU:        $USE_GPU"
echo "  QV normalize:   $RUN_LEVEL_NORMALIZE"
echo "  Output:         $OUTPUT_DIR"
echo "  Skip pred:      $SKIP_PRED"
echo "  Invocation:     $MARS_INVOCATION"
echo ""
echo "========================================"
echo "Step 1/2: Alignment and feature extraction"
echo "========================================"
echo ""

START_TIME=$(date +%s)

"${MARS_PIPELINE_CMD[@]}" "$FASTQ_FILE" "$REFERENCE" "$MODIFICATION" "$OUTPUT_DIR" \
    --num_threads "$THREADS" --context-size "$CONTEXT_SIZE" --batch-size "$BATCH_SIZE" \
    "$GPU_FLAG" "${NORMALIZE_ARGS[@]}"

if [[ "$SKIP_PRED" != true ]]; then
    echo ""
    echo "========================================"
    echo "Step 2/2: Prediction (${PRED_MOD_TYPE} / ${PRED_PRETRAINED})"
    echo "========================================"
    echo ""

    "${MARS_PREDICT_CMD[@]}" \
        -i "$OUTPUT_DIR" \
        --mod-type "$PRED_MOD_TYPE" \
        --pretrained "$PRED_PRETRAINED"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "Pipeline completed!"
echo "========================================"
echo "  Elapsed time: ${MINUTES}m ${SECONDS}s"
echo "  Output:       $OUTPUT_DIR"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/ 2>/dev/null || echo "  (no files yet)"
echo ""
