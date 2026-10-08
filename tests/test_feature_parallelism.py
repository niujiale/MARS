"""
Feature extraction must not depend on how the work was scheduled.

MARS parallelises feature extraction over every batch of every
(transcript, strand) job at once, with one worker pool for the whole run. That
scheduling is invisible in the results only if each batch is genuinely
independent, so these tests pin the property down directly: the same input,
extracted with different worker counts and different batch sizes, must produce
byte-identical NPZ archives.

The tests also cover the shape of input that motivated the pooling in the first
place -- many short transcripts carrying only a handful of loci each -- because
that is the case where the old per-job pool did the most damage and where a
scheduling bug would be easiest to introduce.

Run:   pytest tests/test_feature_parallelism.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pysam = pytest.importorskip("pysam")

from nfl_py.boosting import FeatureJob, bam_to_boosting_jobs  # noqa: E402


BASES = "ACGT"
CONTEXT = 10


def _write_bam(path: Path, names, reflen, reads_per_ref, rng):
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": n, "LN": reflen} for n in names],
    }
    unsorted = path.with_suffix(".unsorted.bam")
    read_id = 0
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for ref_id, _ in enumerate(names):
            for _ in range(reads_per_ref):
                span = rng.randint(reflen // 4, reflen - 1)
                start = rng.randint(0, reflen - span - 1)
                # A few indels so the CIGAR decoding is exercised, not just
                # a run of matches.
                cigar, remaining = [], span
                while remaining > 0:
                    m = min(remaining, rng.randint(20, 80))
                    cigar.append((0, m))
                    remaining -= m
                    if remaining > 3 and rng.random() < 0.3:
                        d = min(remaining, rng.randint(1, 3))
                        cigar.append((2, d))
                        remaining -= d
                qlen = sum(n for op, n in cigar if op in (0, 1, 4))

                rec = pysam.AlignedSegment()
                rec.query_name = f"read_{read_id:06d}"
                rec.query_sequence = "".join(
                    rng.choice(BASES) for _ in range(qlen)
                )
                rec.flag = 16 if rng.random() < 0.5 else 0
                rec.reference_id = ref_id
                rec.reference_start = start
                rec.mapping_quality = 60
                rec.cigartuples = cigar
                rec.query_qualities = pysam.qualitystring_to_array(
                    "".join(chr(rng.randint(2, 40) + 33) for _ in range(qlen))
                )
                out.write(rec)
                read_id += 1

    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))
    unsorted.unlink()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A small multi-transcript BAM plus its reference and target loci."""
    rng = random.Random(20260825)
    root = tmp_path_factory.mktemp("parallel")
    names = [f"tx{i:03d}" for i in range(12)]
    reflen = 400

    refseq = {n: "".join(rng.choice(BASES) for _ in range(reflen))
              for n in names}
    bam = root / "reads.bam"
    _write_bam(bam, names, reflen, reads_per_ref=25, rng=rng)

    loci = {
        n: sorted(rng.sample(range(CONTEXT + 1, reflen - CONTEXT), 9))
        for n in names
    }
    return {"bam": str(bam), "refseq": refseq, "loci": loci, "names": names}


def _extract(dataset, outdir: Path, workers: int, batch_size: int):
    jobs = [
        FeatureJob(
            chromosome=name,
            loci=dataset["loci"][name],
            outdir=str(outdir / name / strand_name),
            reverse=reverse,
        )
        for name in dataset["names"]
        for strand_name, reverse in (("forward", False), ("backward", True))
    ]
    accepted = bam_to_boosting_jobs(
        dataset["bam"], dataset["refseq"], jobs,
        context_size=CONTEXT,
        batch_size=batch_size,
        use_gpu=False,
        showprogress=False,
        num_workers=workers,
    )
    assert len(accepted) == len(jobs)
    return jobs


def _read_all(outdir: Path):
    files = sorted(p.relative_to(outdir) for p in outdir.rglob("X_*.npz"))
    assert files, f"no feature files written under {outdir}"
    out = {}
    for rel in files:
        with np.load(outdir / rel, allow_pickle=True) as z:
            out[str(rel)] = {k: z[k] for k in z.files}
    return out


def _assert_same(reference, other, context: str):
    assert reference.keys() == other.keys(), f"{context}: file sets differ"
    for rel, ref_arrays in reference.items():
        for key, ref_val in ref_arrays.items():
            assert np.array_equal(ref_val, other[rel][key]), (
                f"{context}: {rel} field '{key}' differs"
            )


@pytest.mark.parametrize("workers", [2, 4, 7])
def test_results_do_not_depend_on_worker_count(dataset, tmp_path, workers):
    """Pooling batches across jobs must not perturb any output field."""
    seq_dir = tmp_path / "sequential"
    par_dir = tmp_path / f"workers{workers}"
    _extract(dataset, seq_dir, workers=1, batch_size=4)
    _extract(dataset, par_dir, workers=workers, batch_size=4)

    _assert_same(_read_all(seq_dir), _read_all(par_dir),
                 f"workers=1 vs workers={workers}")


def test_every_job_gets_its_own_batch_numbering(dataset, tmp_path):
    """Batch files are numbered within a job, not across the whole run."""
    outdir = tmp_path / "numbering"
    jobs = _extract(dataset, outdir, workers=4, batch_size=4)

    # 9 loci at batch_size=4 -> X_1, X_2, X_3 in every job directory.
    for job in jobs:
        produced = sorted(p.name for p in Path(job.outdir).glob("X_*.npz"))
        assert produced == ["X_1.npz", "X_2.npz", "X_3.npz"], (
            f"{job.chromosome} {'-' if job.reverse else '+'}: got {produced}"
        )


def test_unknown_chromosome_is_skipped_not_fatal(dataset, tmp_path):
    """One unusable job must not cost the run every other job."""
    outdir = tmp_path / "skip"
    good = dataset["names"][0]
    jobs = [
        FeatureJob("no_such_transcript", [50, 60], str(outdir / "bad"), False),
        FeatureJob(good, dataset["loci"][good], str(outdir / "good"), False),
    ]

    accepted = bam_to_boosting_jobs(
        dataset["bam"], dataset["refseq"], jobs,
        context_size=CONTEXT, batch_size=4, use_gpu=False,
        showprogress=False, num_workers=2,
    )

    assert [j.chromosome for j in accepted] == [good]
    assert list((outdir / "good").glob("X_*.npz"))
    assert not list((outdir / "bad").glob("X_*.npz"))
