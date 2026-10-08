#!/usr/bin/env python3
"""
Replicate a FASTQ N times with unique read names, to raise coverage depth.

The bundled example has 59 reads, which is far too shallow to time anything.
Copying the reads keeps the real base-calling error and quality profile (which
is what the features are computed from) while scaling the axis that dominates
prepdata cost.
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=60)
    args = ap.parse_args()

    with open(args.fastq) as fh:
        lines = fh.read().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 4:
        sys.exit(f"{args.fastq}: {len(lines)} lines is not a multiple of 4")
    records = [lines[i:i + 4] for i in range(0, len(lines), 4)]

    with open(args.out, "w") as out:
        for copy in range(args.copies):
            for name, seq, plus, qual in records:
                head = name.split(None, 1)
                rid = head[0] + f"_c{copy:04d}"
                rest = (" " + head[1]) if len(head) > 1 else ""
                out.write(f"{rid}{rest}\n{seq}\n{plus}\n{qual}\n")

    print(f"{len(records)} reads x {args.copies} copies = "
          f"{len(records) * args.copies} reads -> {args.out}")


if __name__ == "__main__":
    main()
