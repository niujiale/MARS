# Benchmark and equivalence harness

These are the tools behind the performance figures in `CHANGELOG.md` and the
correctness claims that back them. They are development tools, not part of the
installed package — `MANIFEST.in` prunes this directory from the distribution.

Only the Python tools are tracked. The shell drivers used to produce the
published numbers hardcode local interpreter paths and dataset locations for
two side-by-side virtualenvs (one on a MARS 3.0.0 source tree, one on 4.0.0),
so they are gitignored; the Python tools below are self-contained and take the
tree and dataset as arguments.

## Correctness

MARS chooses between implementations at runtime — dict-based versus
matrix-based pileup, sparse versus dense joint statistics, any number of worker
processes. None of those choices may change a user's results, so equivalence is
checked rather than assumed.

| Tool | Checks |
|---|---|
| `equiv_check.py` | The original dict pileup and the matrix pileup produce bit-identical feature matrices and depths, on both strands. `--sub-rate` controls how much reads mismatch the reference; the default (negative) makes them fully random, which drives the dense paths hardest. |
| `compare_outputs.py` | Two `prepdata` output directories agree field by field across all `X_*.npz` archives. Used to compare MARS 3.0.0 against 4.0.0, CPU against GPU, and one worker count against another. |
| `ab_joint_path.py` | The sparse and dense joint substitution-rate paths agree on real pileup data, and measures the gain, by forcing each path over the same input. |

The equivalence properties that must hold permanently live in the test suite
instead — `tests/test_stats_equivalence.py` and
`tests/test_feature_parallelism.py` — so they run in CI.

## Profiling

| Tool | Reports |
|---|---|
| `profile_stages.py` | Old versus new per-batch timing split into pileup, window gather, QV statistics, joint statistics and the round/CSR tail. Pass `random` for fully random reads. |
| `profile_tail.py` | One level deeper: the one-hot build versus the matmul inside the joint statistics, and the assembly, rounding, CSR conversion and npz write that follow. |
| `probe_pool_cost.py` | Fixed cost of creating a `multiprocessing.Pool`, which is what made per-job pools untenable for transcriptome input. |
| `stage_bench.py` | Times feature extraction for a whole dataset against a given source tree, so 3.0.0 and 4.0.0 can be compared under one interpreter stack. |

## Packaging

`verify_dist.py` builds the sdist and wheel and inspects what landed inside:
that the package, pretrained models, tests and license are present, that the
benchmark harness, caches and pipeline outputs are not, and that the filenames
carry the version in `mars/_version.py`. It calls the setuptools backend
directly, so it works without network access.

## Calibration

`proto_sparse_joint.py` is where `_SPARSE_JOINT_PAIR_COST` in `nfl_py/stats.py`
comes from. It sweeps substitution rate against read coverage, times both joint
statistics implementations, and confirms they agree exactly. The constant is the
price of one counted pair relative to one BLAS multiply-add, chosen so the
cheaper path is selected at every point measured.

`proto_assembly.py` measures the feature-matrix assembly layout change on its
own, comparing the block-and-transpose form against writing straight into the
final layout.

Rerun `proto_sparse_joint.py` on new hardware if the threshold looks wrong:
different BLAS throughput moves the crossover.

## Datasets

Synthetic input, since real datasets are too large to keep here and the shapes
that matter are easy to construct:

| Tool | Shape |
|---|---|
| `make_synth_dataset.py` | One reference with configurable length, depth and locus count. Covers deep-and-narrow through deep-and-wide. |
| `make_multi_tx.py` | Many short references with a few loci each — the shape transcriptome input takes, and the one that exposed the per-job worker pool problem. |
| `make_scaled.py` | Replicates a FASTQ with fresh read names, to scale depth on the bundled example. |

Both generators default to `--sub-rate 0.07`, near a real nanopore mismatch
rate. This matters for timing: any code path whose cost depends on how many
bases disagree with the reference looks far worse than it is when measured
against random sequence, which mismatches three bases in four.
