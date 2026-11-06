#!/usr/bin/env python3
"""Collect the last reported accuracy from DASSL logs and summarize across seeds."""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ACC_RE = re.compile(r"\* accuracy:\s*([\d.]+)%")
SEED_RE = re.compile(r"^seed(\d+)$")


def last_accuracy(path):
    value = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ACC_RE.search(line)
        if match:
            value = float(match.group(1))
    return value


def experiment_key(root, log_path):
    parts = list(log_path.relative_to(root).parts[:-1])
    seed = None
    for index, part in enumerate(parts):
        match = SEED_RE.match(part)
        if match:
            seed = int(match.group(1))
            parts[index] = "seed{seed}"
            break
    return "/".join(parts), seed


def find_logs(root):
    for path in sorted(root.rglob("log.txt")):
        value = last_accuracy(path)
        if value is not None:
            yield path, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="experiment output root")
    parser.add_argument("--csv", type=Path, default=None, help="optional CSV output")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")

    grouped = defaultdict(list)
    skipped = 0
    for path, accuracy in find_logs(args.root):
        key, seed = experiment_key(args.root, path)
        if seed is None:
            skipped += 1
            continue
        grouped[key].append((seed, accuracy))

    rows = []
    for key, values in sorted(grouped.items()):
        values.sort()
        accuracies = [value for _, value in values]
        rows.append(
            {
                "experiment": key,
                "seeds": ",".join(str(seed) for seed, _ in values),
                "n": len(values),
                "mean": statistics.fmean(accuracies),
                "std": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
            }
        )

    print("| experiment | seeds | n | accuracy (mean±std) |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['experiment']} | {row['seeds']} | {row['n']} | "
            f"{row['mean']:.2f}±{row['std']:.2f} |"
        )
    if skipped:
        print(f"\nSkipped {skipped} logs without a seed<N> path component.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("experiment", "seeds", "n", "mean", "std"))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
