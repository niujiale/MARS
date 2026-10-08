# Changelog

All notable changes to MARS are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and MARS adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Released versions are published as tagged GitHub Releases with attached
source archives and Python distributions; see [RELEASE.md](RELEASE.md) for the
process and the [Releases page](https://github.com/niujiale/MARS/releases) for
downloads.

## [Unreleased]

## [4.0.0] - 2026-10-08

First formally released version. Previously MARS was distributable only by
`git clone` of the development branch, which made it impossible to cite or
reproduce a specific state of the code. This release adds versioned artifacts
and a proper installed command-line interface, and includes the prediction and
feature-extraction fixes needed for pretrained models to behave correctly.

### Fixed

- **Run-level QV normalization is no longer applied unconditionally.**
  `mars run` rescaled the `qv_mean` and `qv_cov` feature blocks by the run's
  own QV mean and standard deviation, but every bundled pretrained model was
  trained on unscaled QV. The rescaling shifted those blocks outside the range
  the models split on, so a completely unmodified sample scored around 0.6
  instead of near 0 while a fully modified sample still looked correct — the
  failure was invisible unless a negative control was run. Normalization is now
  opt-in behind `--run-level-normalize` (`run_example.sh --run-level-normalize`)
  and off by default, which is what the pretrained models expect. Turning it on
  requires a model trained the same way, so pair it with `--skip-pred`.
- **The release archives could not run the bundled example.** `MANIFEST.in`
  omitted `MARS_main.py`, which `run_example.sh` executes when the console
  scripts are not on `PATH` — that is, in exactly the situation an unpacked
  sdist or source archive is in. Unpacking a release and running
  `bash run_example.sh` failed with `can't open file '.../MARS_main.py'`, and
  three tests that drive `run_example.sh` failed with it.
- **`--context-size` is no longer limited by the outermost site of a
  transcript.** A `(transcript, strand)` job was rejected outright when its
  first or last locus had no room for a full context window, so one site near a
  transcript end cost the run every other site on that transcript. The
  threshold was whatever the closest-to-the-edge locus allowed, which on a
  transcriptome is often far below the requested context: on the bundled
  Curlcake test data a context above 12 already started dropping transcripts
  and a context of 29 or more produced no features at all, leaving prediction
  to fail with `No NPZ feature files found`. Only the loci whose window
  actually overhangs the reference are dropped now, and the surviving loci are
  reported so labels stay aligned with the feature rows.

### Changed

- `--context-size` accepts any value. Feature width was already derived from
  the window (`3·w + 28·w²` for `w = 2·context + 1`), so a context of 100 gives
  1,131,831 features per site instead of the default 12,411. Feature
  extraction now logs the resulting width and per-batch memory, and warns past
  200,000 features that `--batch-size` and the worker count may need lowering.
  Pretrained models are fixed at the context they were trained with, so any
  other value needs `--skip-pred` and a model of your own.
- Label vectors (`ydata.tsv`) are written from the loci a job kept, in the
  same order as the feature rows. Previously they were written in input-CSV
  order while feature rows follow sorted locus order, so an unsorted candidate
  site list would silently mismatch labels to sites.

### Added

- **Formal release artifacts.** Every version is now a signed git tag plus a
  GitHub Release carrying a source archive, an sdist and a wheel, so an
  analysis can name the exact version it used. See [RELEASE.md](RELEASE.md).
- **Installed console commands.** MARS no longer has to be invoked through
  file paths from inside its own source directory. After installation the
  following commands are on `PATH` in the conda environment:

  | Command | Purpose |
  |---|---|
  | `mars` | umbrella CLI (`run`, `predict`, `train`, `prepdata`, `list-models`, `doctor`, `version`) |
  | `mars-pipeline` | alignment + feature extraction |
  | `mars-predict` | apply a trained or pretrained model |
  | `mars-train` | train and evaluate a new model |
  | `mars-doctor` | environment / dependency diagnostics |
  | `xgb_model_pred` | alias of `mars-predict` (MARS 3.x script name) |
  | `xgb_model` | alias of `mars-train` (MARS 3.x script name) |
  | `nfl-py` | feature-extraction core |

- `mars doctor` reports the MARS version, interpreter, dependency versions,
  `minimap2`/`samtools` locations, PyTorch CUDA status and where the
  pretrained models were found — paste it into bug reports.
- `--version` on every entry point, reported from a single source of truth
  (`mars/_version.py`), which `pyproject.toml` reads via
  `[tool.setuptools.dynamic]`.
- Pretrained models are now located automatically, with an explicit
  `--pretrained-dir` flag and a `MARS_PRETRAINED_MODELS` environment variable
  override. Failures list every path that was probed.
- `CITATION.cff`, `LICENSE` and this changelog.
- Continuous integration on Linux running the regression suite against
  Python 3.9–3.12, plus a release workflow that refuses to publish when the
  git tag and `mars/_version.py` disagree.
- `train` optional dependency group (`matplotlib`, `seaborn`,
  `scikit-learn`). These were listed in `requirements.txt` but missing from
  the package metadata and from `environment.yml`, so `xgb_model.py` failed on
  a fresh conda install.

### Changed

- The pipeline, prediction and training code moved into the installed `mars`
  package (`mars.pipeline`, `mars.model.predict`, `mars.model.train`). Public
  behaviour, inputs and outputs are unchanged.
- Distribution renamed from `nfl-py` 1.0.0 to `mars-nanopore` 4.0.0 so the
  package version matches the tool version. `nfl_py` remains importable and
  now reports the MARS version.
- `run_example.sh` calls the installed commands when they are available and
  falls back to in-tree execution otherwise.
- `mars train` honours `--seed` for the train/test split and for XGBoost
  (previously the flag was parsed and then ignored, so runs were only
  reproducible at the hard-coded default of 42). Default behaviour is
  unchanged.
- `mars train` selects a deterministic chromosome ordering before splitting;
  the previous `set()` iteration order made the split depend on Python's
  string hash seed.
- Training plots render through the `Agg` backend, so `mars train` works over
  SSH without an X display.
- Minimum supported Python raised to 3.9 (3.8 is end-of-life).

### Performance

- Feature extraction (`mars run` / `mars prepdata`) rewritten around dense
  NumPy matrices. Pileup now expands CIGAR operations in vectorised batches and
  emits base/quality/depth matrices directly, instead of building
  per-locus Python dictionaries and re-walking them for every feature; and the
  reference and sequence-context encodings use byte-level lookup tables.
- **`--num_threads` now helps instead of hurting.** MARS 3.x parallelised the
  batches *within* one (transcript, strand) unit and built a fresh worker pool
  for each such unit, so a run against a transcriptome paid to start a pool
  once per transcript while handing it far less work than that cost. With the
  default `--num_threads 10` this made feature extraction several times slower
  than running single-threaded. Parallelism now covers every batch of every
  transcript at once through a single pool created once per run, and worker
  processes inherit the reference sequences as copy-on-write memory instead of
  receiving a pickled copy each. Measured on 400 transcripts carrying 8 loci
  each, on 16 cores:

  | Workers | MARS 3.x | MARS 4.0.0 |
  |---|---|---|
  | 1 | 14.6 s | 4.5 s |
  | 2 | 42.9 s | 2.8 s |
  | 4 | 56.5 s | 1.9 s |
  | 8 | 87.6 s | 1.4 s |

  MARS 3.x gets 6x slower as workers are added; MARS 4.0.0 gets 3.1x faster.

  Results are unchanged by the worker count, which `tests/test_feature_parallelism.py`
  now checks directly.
- The joint substitution-rate tensor, the largest block of features, is now
  obtained by counting the substitutions that co-occur rather than by taking the
  Gram matrix of a one-hot tensor over every read, position and base. The dense
  form costs the same whether a read substitutes every base or none, which is a
  poor fit for sequencing data: basecalls agree with the reference at most
  positions, and a read overlaps only one of the windows in a batch. MARS
  measures the sparsity of each batch and falls back to the dense form when it
  would be cheaper, so the pathological cases do not regress. Feature
  extraction runs 1.5-1.7x faster on typical mismatch rates.

  Both forms return identical bits, and not by approximation: each entry is a
  count of reads, and float32 represents such integers exactly, so the order
  they are accumulated in cannot change the result.
  `tests/test_stats_equivalence.py` checks this across mismatch rates, coverage
  levels and both strands.
- The feature matrix is assembled directly in its final (locus, feature)
  layout. It was previously accumulated as (feature, locus) blocks and
  transposed at the end, which sent every block -- including the
  11025-column joint tensor -- on two extra strided passes through memory
  without changing a single value. Assembly alone is 6-8x faster.
- Measured against MARS 3.0.0 on the same machine and inputs, with
  bit-identical output in every case:

  | Dataset | MARS 3.0.0 | MARS 4.0.0 | Speedup |
  |---|---|---|---|
  | deep coverage, CPU | 100% | 13.0% | 7.7x |
  | deep coverage, GPU | 100% | 12.4% | 8.1x |
  | deep and wide, CPU | 100% | 24.0% | 4.2x |
  | deep and wide, GPU | 100% | 22.9% | 4.4x |
  | many loci, CPU | 100% | 35.9% | 2.8x |
  | many loci, GPU | 100% | 32.3% | 3.1x |

  "Deep coverage" is the bundled example replicated 60x (real reads, ~3400x
  depth); the other two are synthetic, at a mismatch rate near real nanopore
  data. Every row was checked for bit-identical output before being timed.

  Because the statistical kernels are no longer the bottleneck, `--gpu` now
  contributes a smaller share of the total (roughly 1.1-1.2x on top of the new
  CPU path) than it did in MARS 3.x.

### Deprecated

- `python MARS_main.py`, `python scripts/model/xgb_model_pred.py` and
  `python scripts/model/xgb_model.py` still work and produce identical
  results, but print a note pointing at the installed commands. They will be
  removed in MARS 5.0.0.

## [3.0.0] - prior development versions

Development-only versions distributed by `git clone`. See the git history for
details.

[Unreleased]: https://github.com/niujiale/MARS/compare/v4.0.0...HEAD
[4.0.0]: https://github.com/niujiale/MARS/releases/tag/v4.0.0
