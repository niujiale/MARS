# MARS

**MARS** detects RNA modifications from Nanopore direct-RNA sequencing data:
from FASTQ to per-site modification rates, with pretrained XGBoost models for
m6A, m5C, m1A, Ψ, ac4C, f5C and hm5C.

[![CI](https://github.com/niujiale/MARS/actions/workflows/ci.yml/badge.svg)](https://github.com/niujiale/MARS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/niujiale/MARS?sort=semver)](https://github.com/niujiale/MARS/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Current release: **4.0.0** ([CHANGELOG](CHANGELOG.md), [Releases](https://github.com/niujiale/MARS/releases))

## Installation

```bash
VERSION=4.0.0
curl -L -o mars-${VERSION}.tar.gz \
    https://github.com/niujiale/MARS/archive/refs/tags/v${VERSION}.tar.gz
tar xzf mars-${VERSION}.tar.gz && cd MARS-${VERSION}

bash install.sh                      # CPU
bash install.sh --gpu --cuda=auto    # GPU (installs the matching torch wheel)
```

`install.sh` creates a `MARS` conda environment (micromamba, mamba or conda,
whichever is available) with minimap2 and samtools, and installs the `mars`
commands. Useful options:

- `--prefix=<path>`: install the environment somewhere other than `$HOME`.
- `venv`: use a Python venv instead of conda (`bash install.sh venv`);
  minimap2 and samtools must then be installed separately.

Check the installation:

```bash
conda activate MARS
mars --version
mars doctor          # dependencies, GPU, minimap2/samtools, model location
```

## Quick start

```bash
# Bundled example (first-time check)
bash run_example.sh

# Your own sample: alignment + feature extraction + m6A prediction
bash run_example.sh sample.fastq reference.fa ./out --threads 16 --gpu

# Another modification type
bash run_example.sh sample.fastq reference.fa ./out \
    --mod m5C --mod-type m5c --pretrained eligos_m5c_C1test
```

Results are written to `./out/modification_pred/modification_pred.txt`.
Run `bash run_example.sh --help` for all options.

## Step by step

```bash
# 1. Alignment + feature extraction  (mod_type: DRACH, m5C, A, T, G, C)
mars run sample.fastq reference.fa DRACH ./out --num_threads 16 --gpu

# 2a. Predict with a pretrained model
mars predict -i ./out --mod-type m6a --pretrained curlcake_DRACH_2244test

# 2b. Or train your own model on labeled features, then predict with it
mars train -i ./out/feature -o ./my_model
mars predict -i ./out -model ./my_model/json/<name>_model.json
```

Every command has `--help`. `mars prepdata` extracts features from an existing
sorted BAM.

**Pretrained models require the default feature settings** (`--context-size 10`,
no `--run-level-normalize`). With any other setting, stop after feature
extraction (`run_example.sh --skip-pred`) and train your own model.

`--gpu` falls back to CPU automatically if CUDA is unavailable. Combine it
with several threads; a single worker leaves the GPU mostly idle.

## Pretrained models

```bash
mars list-models                 # all models
mars list-models --mod-type m6a  # one modification type
```

| `--mod-type` | Modification |
|---|---|
| `m6a` | N6-methyladenosine (DRACH and NNANN models) |
| `m5c` | 5-methylcytosine |
| `m1a` | N1-methyladenosine |
| `psu`, `nanomud_psu` | Pseudouridine (Ψ) |
| `ac4C` | N4-acetylcytidine |
| `f5c` | 5-formylcytosine |
| `h5c` | 5-hydroxymethylcytosine |

If the models are not found (e.g. after installing only the wheel), set
`--pretrained-dir` or `export MARS_PRETRAINED_MODELS=/path/to/pretrained_models`.

## Output

`modification_pred.txt` is tab-separated, one row per candidate site:

| Column | Meaning |
|---|---|
| `trans` | transcript / chromosome |
| `pos` | 1-based position |
| `strand` | `+` or `-` |
| `depth` | read coverage |
| `y_pred` | predicted modification rate (0–1) |
| `y_test` | label from training data (0 for new samples) |

`mars run` also writes the sorted/filtered BAM, the candidate-site list and
the feature matrices under `feature/<chrom>/{forward,backward}/`.
`mars train` writes the model (`json/`), test-set predictions and plots.

## Development

```bash
git clone https://github.com/niujiale/MARS.git && cd MARS
bash install.sh
pytest tests/
```

Release steps are in [RELEASE.md](RELEASE.md).

## Citation

Please cite the version you used ([CITATION.cff](CITATION.cff); GitHub's
*Cite this repository* button generates the reference):

```
MARS Team. MARS: RNA modification detection from Nanopore direct-RNA
sequencing data. Version 4.0.0, 2026. https://github.com/niujiale/MARS
```

## License and feedback

MIT License, see [LICENSE](LICENSE). Questions and bug reports:
[GitHub Issues](https://github.com/niujiale/MARS/issues) (please attach the
output of `mars doctor`).
