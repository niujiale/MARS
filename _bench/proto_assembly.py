#!/usr/bin/env python3
"""
Prototype: assemble the feature matrix in its final layout directly.

The current code accumulates each statistic into (feature, sample) arrays, then
stacks them and transposes the result to (sample, feature). The statistics
functions, however, already return (sample, ...) arrays, so every block is
transposed on the way in and the whole matrix is transposed again on the way
out. For the joint substitution tensor -- 11025 of the 12411 columns -- that is
two strided passes over the largest array in the batch plus a full copy in
vstack.

Assembling straight into (sample, feature) removes those passes. It is a pure
data-movement change: each block lands in the same column range with the same
values, so the result is bit-identical by construction, which this checks.
"""

import time

import numpy as np


def assemble_current(mean_b, cov_b, sk_b, kt_b, joint_b, enc, valid_idx,
                     ctx_idx, n_samples, block, run_qv_mean, run_qv_sd):
    qv_mean = np.zeros((block, n_samples), dtype=np.float32)
    qv_cov = np.zeros((block, block, n_samples), dtype=np.float32)
    qv_sk = np.zeros((block, block, n_samples), dtype=np.float32)
    qv_kt = np.zeros((block, block, n_samples), dtype=np.float32)
    sr_cov = np.zeros((25, block, block, n_samples), dtype=np.float32)
    context_encoded = np.zeros((2 * block, n_samples), dtype=np.float32)

    qv_mean[:, valid_idx] = mean_b.T
    qv_cov[:, :, valid_idx] = cov_b.transpose(1, 2, 0)
    qv_sk[:, :, valid_idx] = sk_b.transpose(1, 2, 0)
    qv_kt[:, :, valid_idx] = kt_b.transpose(1, 2, 0)
    sr_cov[:, :, :, valid_idx] = (
        joint_b.transpose(1, 2, 3, 4, 0).reshape(25, block, block,
                                                 valid_idx.size)
    )
    context_encoded[:, ctx_idx] = enc.reshape(ctx_idx.size, 2 * block).T

    qv_mean = (qv_mean - run_qv_mean) / run_qv_sd
    qv_cov = qv_cov / (run_qv_sd ** 2)

    return np.vstack([
        qv_mean,
        qv_cov.reshape(-1, n_samples),
        qv_sk.reshape(-1, n_samples),
        qv_kt.reshape(-1, n_samples),
        sr_cov.reshape(-1, n_samples),
        context_encoded,
    ]).T


def assemble_direct(mean_b, cov_b, sk_b, kt_b, joint_b, enc, valid_idx,
                    ctx_idx, n_samples, block, run_qv_mean, run_qv_sd):
    pair = block * block
    widths = (block, pair, pair, pair, 25 * pair, 2 * block)
    n_features = sum(widths)
    out = np.zeros((n_samples, n_features), dtype=np.float32)

    o = 0
    mean_v = out[:, o:o + block]
    mean_v[valid_idx] = mean_b
    # Invalid rows stay zero here and pick up the same offset the stacked
    # version gave them, since normalization runs over the whole block.
    mean_v -= run_qv_mean
    mean_v /= run_qv_sd
    o += block

    cov_v = out[:, o:o + pair]
    cov_v[valid_idx] = cov_b.reshape(valid_idx.size, pair)
    cov_v /= run_qv_sd ** 2
    o += pair

    out[:, o:o + pair][valid_idx] = sk_b.reshape(valid_idx.size, pair)
    o += pair
    out[:, o:o + pair][valid_idx] = kt_b.reshape(valid_idx.size, pair)
    o += pair
    out[:, o:o + 25 * pair][valid_idx] = joint_b.reshape(valid_idx.size,
                                                        25 * pair)
    o += 25 * pair
    out[:, o:o + 2 * block][ctx_idx] = enc.reshape(ctx_idx.size, 2 * block)
    return out


def timed(fn, repeat=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return (time.perf_counter() - t0) / repeat, out


def main():
    rng = np.random.default_rng(7)
    block = 21
    for n_samples, n_valid in ((100, 100), (100, 90), (500, 500)):
        mean_b = rng.random((n_valid, block), dtype=np.float32)
        cov_b = rng.random((n_valid, block, block), dtype=np.float32)
        sk_b = rng.random((n_valid, block, block), dtype=np.float32)
        kt_b = rng.random((n_valid, block, block), dtype=np.float32)
        joint_b = rng.random((n_valid, 5, 5, block, block), dtype=np.float32)
        enc = rng.random((n_samples, block, 2), dtype=np.float32)
        valid_idx = np.sort(rng.choice(n_samples, n_valid, replace=False))
        ctx_idx = np.arange(n_samples)

        args = (mean_b, cov_b, sk_b, kt_b, joint_b, enc, valid_idx, ctx_idx,
                n_samples, block, 18.82, 8.73)
        t_cur, a = timed(lambda: assemble_current(*args))
        t_new, b = timed(lambda: assemble_direct(*args))

        print(f"n_samples={n_samples} n_valid={n_valid}  "
              f"current {t_cur * 1000:6.2f} ms   direct {t_new * 1000:6.2f} ms"
              f"   {t_cur / t_new:5.2f}x   identical={np.array_equal(a, b)}"
              f"   layout {'C' if b.flags.c_contiguous else 'F'} vs "
              f"{'C' if a.flags.c_contiguous else 'F'}")


if __name__ == "__main__":
    main()
