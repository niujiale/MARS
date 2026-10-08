#!/usr/bin/env python3
"""
Isolate the fixed cost of the current parallelism: bam_to_boosting_data builds
and tears down a multiprocessing.Pool on every call, and the pipeline calls it
once per (transcript, strand).

Measures bare Pool construction against the real per-job work so the two can be
compared directly.
"""

import os
import sys
import time
from multiprocessing import Pool


def _noop(_):
    return 0


def pool_cycles(n_jobs, workers, tasks_per_job):
    t0 = time.perf_counter()
    for _ in range(n_jobs):
        with Pool(processes=workers) as pool:
            list(pool.imap_unordered(_noop, range(tasks_per_job)))
    return time.perf_counter() - t0


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    print(f"cpus={os.cpu_count()}  jobs={n_jobs}  1 task per job")
    print(f"{'workers':>8}  {'pool overhead':>14}  {'per job':>10}")
    for w in (2, 4, 8, 10):
        el = pool_cycles(n_jobs, w, 1)
        print(f"{w:>8}  {el:>11.3f} s  {el / n_jobs * 1000:>7.1f} ms")


if __name__ == "__main__":
    main()
