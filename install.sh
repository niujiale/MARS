#!/bin/bash
#
# MARS Installation Script
#
# Usage:
#   bash install.sh [conda|venv] [--prefix=<path>] [--gpu [--cuda=<ver>]]
#
# Env location options:
#   (default)        Install as named env MARS under the tool's root
#                    (e.g. $MAMBA_ROOT_PREFIX/envs/MARS).
#   --prefix=<path>  Install the conda env at exactly <path>. Recommended on
#                    shared servers where $HOME is small — point it at a big
#                    public disk so the env files don't clog home quota.
#                    Example: --prefix=/public/share/$USER/env/MARS
#
# GPU / CUDA options (only relevant with --gpu):
#   --cuda=auto   Detect via nvidia-smi; pick the closest supported wheel
#   --cuda=cpu    Install CPU-only torch (no GPU acceleration, lightweight)
#   --cuda=118    PyTorch CUDA 11.8 wheel
#   --cuda=121    PyTorch CUDA 12.1 wheel
#   --cuda=124    PyTorch CUDA 12.4 wheel
#   --cuda=126    PyTorch CUDA 12.6 wheel
#   --cuda=128    PyTorch CUDA 12.8 wheel
#
# Default when --gpu is given without --cuda: auto.
# Default when neither --gpu nor --cuda is given: no torch is installed.
#
# Disk-space tip (shared servers with small $HOME):
#   Before running this script, redirect pip's cache and /tmp to a big disk:
#     export PIP_CACHE_DIR=/public/share/$USER/.pip-cache
#     export TMPDIR=/public/share/$USER/.tmp
#     mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
#

set -e

INSTALL_TYPE="conda"
GPU_SUPPORT=""
CUDA_CHOICE=""
ENV_PREFIX=""   # when set, use `-p $ENV_PREFIX` instead of `-n MARS`

# Parse arguments
for arg in "$@"; do
    case $arg in
        --gpu)
            GPU_SUPPORT="yes"
            ;;
        --cuda=*)
            CUDA_CHOICE="${arg#--cuda=}"
            GPU_SUPPORT="yes"
            ;;
        --prefix=*)
            ENV_PREFIX="${arg#--prefix=}"
            ;;
        conda|venv)
            INSTALL_TYPE="$arg"
            ;;
        --help|-h)
            cat <<EOF
MARS Installation Script

Usage:
  bash install.sh [conda|venv] [--prefix=<path>] [--gpu [--cuda=<ver>]]

Arguments:
  conda            Create environment via conda-compatible tool (default).
                   Auto-prefers: micromamba > mamba > conda. micromamba's
                   C++ solver is 10-50x faster than classic conda and is
                   strongly recommended on servers.
  venv             Create environment via Python venv (pip install only)

  --prefix=<path>  Install the conda env at exactly <path> instead of the
                   default named-env location. Recommended on shared
                   servers where \$HOME is small.
                   Example: --prefix=/public/share/\$USER/env/MARS

  --gpu            Enable GPU support (installs PyTorch)
  --cuda=auto      Detect CUDA via nvidia-smi and pick matching wheel (default with --gpu)
  --cuda=cpu       CPU-only torch (no GPU acceleration, ~200MB)
  --cuda=118|121|124|126|128   Install PyTorch for specific CUDA

Examples:
  bash install.sh                                       # CPU-only, no torch
  bash install.sh --gpu                                 # auto-detect CUDA
  bash install.sh --gpu --cuda=121                      # pin CUDA 12.1 wheel
  bash install.sh --prefix=/public/share/\$USER/env/MARS --gpu --cuda=auto
  bash install.sh venv --gpu --cuda=118                 # venv + CUDA 11.8

Disk-space tip (servers with small \$HOME):
  export PIP_CACHE_DIR=/public/share/\$USER/.pip-cache
  export TMPDIR=/public/share/\$USER/.tmp
  mkdir -p "\$PIP_CACHE_DIR" "\$TMPDIR"
  bash install.sh --prefix=/public/share/\$USER/env/MARS --gpu --cuda=auto
EOF
            exit 0
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="MARS"

# Read the single source of truth so the installer reports the same version as
# `mars --version` will after installation.
MARS_VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' \
    "$SCRIPT_DIR/mars/_version.py" 2>/dev/null)"
MARS_VERSION="${MARS_VERSION:-unknown}"

# Pick the fastest available conda-compatible tool. micromamba has a C++
# solver and is 10-50x faster than classic conda at env creation; mamba is
# a middle ground. All three share the same CLI surface we need:
# `env create/update -f`, `install -y -c`, `run -n/-p <env> <cmd>`.
pick_conda_tool() {
    if command -v micromamba &> /dev/null; then
        echo "micromamba"
    elif command -v mamba &> /dev/null; then
        echo "mamba"
    elif command -v conda &> /dev/null; then
        echo "conda"
    else
        echo ""
    fi
}

CONDA_TOOL="$(pick_conda_tool)"

# Build the flag used to address the env in every conda/mamba subcommand:
# -p <path> when --prefix was passed, otherwise -n MARS. Keeping this in a
# single variable means the rest of the script doesn't care which mode the
# user chose.
if [[ -n "$ENV_PREFIX" ]]; then
    ENV_SELECTOR=(-p "$ENV_PREFIX")
    ENV_DESC="$ENV_PREFIX"
else
    ENV_SELECTOR=(-n "$ENV_NAME")
    ENV_DESC="$ENV_NAME"
fi

# Detect whether the env already exists. For prefix envs we probe the
# directory directly because named-env listings won't contain a path. For
# named envs we use `env list` (portable across conda/mamba/micromamba).
env_exists() {
    if [[ -n "$ENV_PREFIX" ]]; then
        [[ -d "$ENV_PREFIX/conda-meta" ]]
    else
        "$CONDA_TOOL" env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"
    fi
}

# Resolve CUDA_CHOICE=auto to a concrete wheel tag by parsing nvidia-smi output.
# Rationale: PyTorch wheels only ship for a handful of CUDA versions, so we
# snap the detected driver/runtime version to the nearest supported wheel.
resolve_cuda_auto() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: --cuda=auto requested but nvidia-smi not found; falling back to cpu"
        echo "cpu"
        return
    fi
    local version
    version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    local cuda_line
    cuda_line=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version:[[:space:]]*[0-9]+\.[0-9]+" | head -1 || echo "")
    local cuda_ver="${cuda_line##*: }"
    cuda_ver="${cuda_ver// /}"

    if [[ -z "$cuda_ver" ]]; then
        echo "WARNING: could not parse CUDA version from nvidia-smi; falling back to cpu"
        echo "cpu"
        return
    fi
    local major="${cuda_ver%%.*}"
    local minor="${cuda_ver##*.}"
    local combined="${major}${minor}"

    if (( combined >= 128 )); then
        echo "128"
    elif (( combined >= 126 )); then
        echo "126"
    elif (( combined >= 124 )); then
        echo "124"
    elif (( combined >= 121 )); then
        echo "121"
    elif (( combined >= 118 )); then
        echo "118"
    else
        echo "WARNING: driver CUDA=${cuda_ver} is older than the oldest supported wheel (11.8); falling back to cpu"
        echo "cpu"
    fi
}

# Install torch with the selected CUDA variant. Picks the right PyTorch
# wheel index so the server's CUDA version matches the torch build.
install_torch() {
    local pip_cmd="$1"
    local choice="${CUDA_CHOICE:-auto}"

    if [[ "$choice" == "auto" ]]; then
        choice=$(resolve_cuda_auto)
        echo "  auto-detected CUDA choice: cu${choice} (or cpu)"
    fi

    local index_url
    case "$choice" in
        cpu)  index_url="https://download.pytorch.org/whl/cpu" ;;
        118)  index_url="https://download.pytorch.org/whl/cu118" ;;
        121)  index_url="https://download.pytorch.org/whl/cu121" ;;
        124)  index_url="https://download.pytorch.org/whl/cu124" ;;
        126)  index_url="https://download.pytorch.org/whl/cu126" ;;
        128)  index_url="https://download.pytorch.org/whl/cu128" ;;
        *)
            echo "ERROR: unknown --cuda=${choice}. Use one of: auto cpu 118 121 124 126 128"
            exit 1
            ;;
    esac

    echo "  installing torch from ${index_url}"
    $pip_cmd install --index-url "$index_url" torch
}

# MARS 3.x published this source tree under the distribution name `nfl-py`;
# 4.0.0 renamed it to `mars-nanopore`. pip therefore treats them as unrelated
# projects and will happily keep both records in one environment. That split
# install is silently broken: the 3.x editable path hook still resolves
# `import nfl_py` to the old source tree, so 4.x entry points end up calling
# 3.x library code. Remove the old distribution before installing 4.x.
#
# Arguments: the command prefix that runs inside the target environment
# (e.g. `micromamba run -n MARS`), or nothing when the env is already active.
remove_legacy_nfl_py() {
    if "$@" pip show nfl-py > /dev/null 2>&1; then
        echo "  found a MARS 3.x install (distribution 'nfl-py'); uninstalling it"
        if ! "$@" pip uninstall -y nfl-py; then
            echo "  WARNING: automatic uninstall failed. Run 'pip uninstall nfl-py'"
            echo "           inside the environment, then re-run this script."
        fi
    else
        echo "  no previous MARS 3.x install found"
    fi

    # An interrupted 3.x uninstall can leave the editable finder behind even
    # when pip no longer records the distribution, which reintroduces the same
    # shadowing problem. Sweep those orphans too.
    local purelib
    purelib=$("$@" python -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)
    if [[ -n "$purelib" && -d "$purelib" ]]; then
        local stale
        for stale in "$purelib"/__editable__*nfl_py* \
                     "$purelib"/nfl-py.egg-link \
                     "$purelib"/nfl_py.egg-link; do
            [[ -e "$stale" ]] || continue
            echo "  removing stale MARS 3.x path hook: $(basename "$stale")"
            rm -f "$stale"
        done
    fi
}

echo "========================================"
echo "MARS $MARS_VERSION - Installation Script"
echo "========================================"
echo "Installation method: $INSTALL_TYPE"
if [[ "$INSTALL_TYPE" == "conda" ]]; then
    echo "Conda tool:          ${CONDA_TOOL:-(none detected)}"
fi
echo "GPU support:         ${GPU_SUPPORT:-no}"
if [[ -n "$GPU_SUPPORT" ]]; then
    echo "CUDA choice:         ${CUDA_CHOICE:-auto}"
fi
echo "Installation dir:    $SCRIPT_DIR"
echo "========================================"

# Conda installation using the fastest available tool (prefers micromamba).
# Note: micromamba lists envs via `micromamba env list` like conda, and its
# env name space is shared with conda when $MAMBA_ROOT_PREFIX ==
# $CONDA_PREFIX's root; otherwise envs are separate. That's almost always
# what the user wants — one env per tool — so we don't override.
install_conda() {
    local tool="$CONDA_TOOL"
    echo "[1/5] Creating environment with $tool at: $ENV_DESC"

    if env_exists; then
        echo "Environment already exists; updating..."
        # micromamba doesn't accept -n with `env update` after the subcommand,
        # but -p works; for named envs all three tools accept -n.
        "$tool" env update "${ENV_SELECTOR[@]}" -f "$SCRIPT_DIR/environment.yml"
    else
        if [[ -n "$ENV_PREFIX" ]]; then
            mkdir -p "$(dirname "$ENV_PREFIX")"
            "$tool" env create -p "$ENV_PREFIX" -f "$SCRIPT_DIR/environment.yml"
        else
            "$tool" env create -f "$SCRIPT_DIR/environment.yml"
        fi
    fi

    echo "[2/5] Installing bioinformatics tools (minimap2, samtools, pysam)..."
    "$tool" install -y "${ENV_SELECTOR[@]}" -c bioconda minimap2 samtools pysam

    echo "[3/5] Installing PyTorch (for stats acceleration)..."
    if [[ -n "$GPU_SUPPORT" ]]; then
        install_torch "$tool run ${ENV_SELECTOR[*]} pip"
    else
        echo "  --gpu not requested; skipping torch install"
        echo "  (statistics stage will run on NumPy; still fast after vectorization)"
    fi

    echo "[4/5] Checking for a previous MARS 3.x install..."
    remove_legacy_nfl_py "$tool" run "${ENV_SELECTOR[@]}"

    echo "[5/5] Installing the MARS package and its commands..."
    # The [train] extra pulls matplotlib/seaborn/scikit-learn, needed by
    # `mars train`. The editable install puts the console scripts (mars,
    # mars-predict, xgb_model_pred, ...) on the env's PATH.
    "$tool" run "${ENV_SELECTOR[@]}" pip install -e "${SCRIPT_DIR}[train]"

    echo ""
    echo "========================================"
    echo "Installation complete!"
    echo "========================================"
    echo ""
    local activation_target
    if [[ -n "$ENV_PREFIX" ]]; then
        activation_target="$ENV_PREFIX"
    else
        activation_target="$ENV_NAME"
    fi
    echo "1) Activate the environment:"
    case "$tool" in
        micromamba) echo "     micromamba activate $activation_target" ;;
        mamba)      echo "     mamba activate $activation_target  (or: conda activate $activation_target)" ;;
        conda)      echo "     conda activate $activation_target" ;;
    esac
    print_cli_summary
}

# The point of the console scripts is that users never need to know where the
# source tree is, so spell them out at the end of every install.
print_cli_summary() {
    echo ""
    echo "2) Check the install (works from any directory):"
    echo "     mars --version"
    echo "     mars doctor"
    echo ""
    echo "3) Run the bundled example:"
    echo "     bash \"$SCRIPT_DIR/run_example.sh\""
    if [[ -n "$GPU_SUPPORT" ]]; then
        echo "     bash \"$SCRIPT_DIR/run_example.sh\" --gpu"
    fi
    echo ""
    echo "Available commands:"
    echo "     mars run <fastq> <ref.fa> <DRACH|m5C|A|T|G|C> <outdir>"
    echo "     mars predict -i <outdir> --mod-type m6a --pretrained curlcake_DRACH_2244test"
    echo "     mars train   -i <feature_dir> -o <model_out>"
    echo "     mars list-models | mars doctor | mars version"
    echo ""
    echo "     mars-pipeline / mars-predict / mars-train are the same commands"
    echo "     standalone; xgb_model_pred and xgb_model are kept as aliases."
    echo ""
}

# Venv installation
install_venv() {
    VENV_DIR="$SCRIPT_DIR/venv"

    echo "[1/6] Checking bioinformatics tools (minimap2, samtools)..."
    if ! command -v minimap2 &> /dev/null || ! command -v samtools &> /dev/null; then
        echo "Warning: minimap2 or samtools not installed"
        echo "Please install via conda: conda install -c bioconda minimap2 samtools"
        echo ""
        read -p "Continue installation? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "minimap2 and samtools are installed"
    fi

    echo "[2/6] Creating virtual environment..."
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment already exists: $VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
    fi

    echo "[3/6] Activating environment..."
    source "$VENV_DIR/bin/activate"

    echo "[4/6] Installing core Python dependencies..."
    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements.txt"

    if [[ -n "$GPU_SUPPORT" ]]; then
        echo "  Installing PyTorch (for stats acceleration)..."
        install_torch "pip"
    else
        echo "  --gpu not requested; skipping torch install"
    fi

    echo "[5/6] Checking for a previous MARS 3.x install..."
    remove_legacy_nfl_py

    echo "[6/6] Installing the MARS package and its commands..."
    pip install -e "${SCRIPT_DIR}[train]"

    echo ""
    echo "========================================"
    echo "Installation complete!"
    echo "========================================"
    echo ""
    echo "1) Activate the environment:"
    echo "     source $VENV_DIR/bin/activate"
    print_cli_summary
}

# Main logic
case "$INSTALL_TYPE" in
    conda)
        if [[ -z "$CONDA_TOOL" ]]; then
            echo "Error: no conda-compatible tool found on PATH."
            echo "Install one of: micromamba (fastest), mamba, or conda."
            exit 1
        fi
        install_conda
        ;;
    venv)
        if ! command -v python3 &> /dev/null; then
            echo "Error: python3 not found"
            exit 1
        fi
        install_venv
        ;;
    *)
        echo "Usage: bash install.sh [conda|venv] [--gpu [--cuda=<ver>]]"
        echo ""
        echo "Arguments:"
        echo "  conda          Create environment using Conda (default)"
        echo "  venv           Create environment using Python venv"
        echo "  --gpu          Enable GPU support (install PyTorch)"
        echo "  --cuda=auto    Detect CUDA version via nvidia-smi (default with --gpu)"
        echo "  --cuda=cpu     CPU-only torch (no GPU acceleration)"
        echo "  --cuda=118|121|124|126|128   Specific CUDA wheel"
        exit 1
        ;;
esac
