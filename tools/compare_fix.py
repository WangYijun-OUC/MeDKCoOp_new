#!/usr/bin/env python3
"""汇总 base2new 对照实验：原版 vs 修复版。

用法：
    python tools/compare_fix.py

默认读取：
    原版    /data/wangyijun/ckpt/paper_base2new/base2new        (trainer=MeDKCoOp)
    fix_ctx /data/wangyijun/ckpt/MeDKCoOp_fixcmp/base2new/cmp_fix_ctx   (trainer=MeDKCoOpFix)
    fix_ctxvis 同上 /cmp_fix_ctxvis
"""

import glob
import os
import re
import statistics

ACC_RE = re.compile(r"\* accuracy:\s*([\d.]+)%")

OLD = "/data/wangyijun/ckpt/paper_base2new/base2new"
NEW = "/data/wangyijun/ckpt/MeDKCoOp_fixcmp/base2new"
DATASETS = ["BTMRI1", "CHMNIST", "covid", "ctkidney", "lungcolon", "retina", "octmnist", "kvasir"]
CFG = "vit_b16_0002_b50_randflip"
SEEDS = ["1", "2", "3"]
GROUPS = [
    ("orig", OLD, "MeDKCoOp"),
    ("fix_ctx", NEW + "/cmp_fix_ctx", "MeDKCoOpFix"),
    ("fix_ctxvis", NEW + "/cmp_fix_ctxvis", "MeDKCoOpFix"),
]


def last_acc(path):
    if not os.path.isfile(path):
        return None
    value = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ACC_RE.search(line)
            if m:
                value = float(m.group(1))
    return value


def split_acc(directory, split):
    if not os.path.isdir(directory):
        return None
    val = last_acc(os.path.join(directory, "log.txt"))
    if val is not None:
        return val
    for cand in sorted(glob.glob(os.path.join(directory, "log.txt-*")), reverse=True):
        val = last_acc(cand)
        if val is not None:
            return val
    return None


def run_acc(root, trainer, dataset, seed, split):
    d = os.path.join(root, dataset, "shots_16", trainer, CFG, "seed" + seed, split)
    return split_acc(d, split)


def mean_n(values):
    vals = [v for v in values if v is not None]
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def main():
    table = {}
    for ds in DATASETS:
        row = {}
        for tag, root, trainer in GROUPS:
            base = [run_acc(root, trainer, ds, s, "train_base") for s in SEEDS]
            new = [run_acc(root, trainer, ds, s, "test_new") for s in SEEDS]
            row[tag] = (mean_n(base), mean_n(new))
        table[ds] = row

    print("| 数据集 | orig base | fix_ctx base | fix_ctxvis base | orig new | fix_ctx new | fix_ctxvis new |")
    print("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        cells = []
        for tag in ("orig", "fix_ctx", "fix_ctxvis"):
            b, bn = table[ds][tag][0]
            cells.append(f"{b:.2f} ({bn})" if b is not None else f"- (0)")
        for tag in ("orig", "fix_ctx", "fix_ctxvis"):
            n, nn = table[ds][tag][1]
            cells.append(f"{n:.2f} ({nn})" if n is not None else f"- (0)")
        print(f"| {ds} | " + " | ".join(cells) + " |")

    print()
    for idx, name in ((0, "base"), (1, "new")):
        line = []
        for tag in ("orig", "fix_ctx", "fix_ctxvis"):
            per = []
            for ds in DATASETS:
                m, _ = table[ds][tag][idx]
                if m is not None:
                    per.append(m)
            line.append(f"{tag}={statistics.mean(per):.2f}" if per else f"{tag}=NA")
        print(f"平均 {name}: " + ", ".join(line))


if __name__ == "__main__":
    main()
