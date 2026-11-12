#!/usr/bin/env python3
"""汇总 base2new 复现结果，并与论文 Table 1 对比。

用法:
    python tools/collect_base2new.py --root ./output/paper_base2new
"""

import argparse
import glob
import os
import re
from collections import defaultdict

DATASETS = [
    ("BTMRI1", "BTMRI"),
    ("CHMNIST", "CHMNIST"),
    ("covid", "Covid-19"),
    ("ctkidney", "CTKIDNEY"),
    ("lungcolon", "LC25000"),
    ("retina", "RETINA"),
    ("octmnist", "OCTMNIST"),
    ("kvasir", "Kvasir"),
]

METHODS = [
    ("MeDKCoOp", "MeDKCoOp", "medkcoop"),
    ("BiomedCoOp_BiomedCLIP", "BiomedCoOp", "biomedcoop"),
]

# 论文 Table 1: (base, new, hm)
PAPER = {
    "MeDKCoOp": {
        "BTMRI": (85.57, 92.10, 87.71),
        "CHMNIST": (93.67, 60.17, 73.27),
        "Covid-19": (78.97, 91.07, 84.59),
        "CTKIDNEY": (90.70, 84.83, 87.67),
        "LC25000": (94.27, 96.27, 95.26),
        "RETINA": (88.60, 58.50, 70.47),
        "OCTMNIST": (83.27, 85.83, 84.53),
        "Kvasir": (88.00, 70.83, 78.49),
    },
    "BiomedCoOp": {
        "BTMRI": (83.73, 93.90, 88.52),
        "CHMNIST": (91.70, 43.63, 59.13),
        "Covid-19": (74.07, 90.47, 81.45),
        "CTKIDNEY": (85.50, 84.13, 84.81),
        "LC25000": (92.70, 96.53, 94.58),
        "RETINA": (90.57, 51.73, 65.85),
        "OCTMNIST": (76.07, 85.33, 80.43),
        "Kvasir": (86.93, 59.30, 70.50),
    },
}

ACC_RE = re.compile(r"\* accuracy:\s*([\d.]+)%")


def last_acc(path):
    if not os.path.isfile(path):
        return None
    val = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ACC_RE.search(line)
            if m:
                val = float(m.group(1))
    return val


def acc_from_dir(dirpath):
    """主日志为空时（DASSL 重跑会把旧日志改名），回退到最新的时间戳日志。"""
    main = os.path.join(dirpath, "log.txt")
    val = last_acc(main)
    if val is not None:
        return val
    for cand in sorted(glob.glob(os.path.join(dirpath, "log.txt-*")), reverse=True):
        val = last_acc(cand)
        if val is not None:
            return val
    return None


def hm(b, n):
    return 0.0 if (b + n) == 0 else 2 * b * n / (b + n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./output/paper_base2new")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    rows = []
    for trainer, pretty, _ in METHODS:
        per_ds = {}
        for ds, paper_name in DATASETS:
            base_vals, new_vals = [], []
            for seed in seeds:
                seed_dir = os.path.join(
                    args.root, "base2new", ds, "shots_16", trainer,
                    "vit_b16_0002_b50_randflip", f"seed{seed}")
                b = acc_from_dir(os.path.join(seed_dir, "train_base"))
                n = acc_from_dir(os.path.join(seed_dir, "test_new"))
                if b is not None:
                    base_vals.append(b)
                if n is not None:
                    new_vals.append(n)
            if not base_vals or not new_vals:
                per_ds[paper_name] = (None, None, None, len(base_vals), len(new_vals))
                continue
            b = sum(base_vals) / len(base_vals)
            n = sum(new_vals) / len(new_vals)
            per_ds[paper_name] = (b, n, hm(b, n), len(base_vals), len(new_vals))
        rows.append((pretty, per_ds))

    lines = []
    lines.append("# base2new 复现结果 vs 论文\n")
    lines.append(f"数据目录：`{args.root}`，seeds = {', '.join(seeds)}\n")
    for pretty, per_ds in rows:
        lines.append(f"## {pretty}\n")
        lines.append("| 数据集 | 复现 Base | 论文 Base | Δ | 复现 New | 论文 New | Δ | 复现 HM | 论文 HM | seeds |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        bs, ns, hms, pbs, pns, phms = [], [], [], [], [], []
        for _, paper_name in DATASETS:
            b, n, h, nb, nn = per_ds[paper_name]
            pb, pn, ph = PAPER[pretty][paper_name]
            if b is None:
                lines.append(f"| {paper_name} | - | {pb:.2f} | - | - | {pn:.2f} | - | - | {ph:.2f} | {nb}/{nn} |")
                continue
            bs.append(b); ns.append(n); hms.append(h)
            pbs.append(pb); pns.append(pn); phms.append(ph)
            lines.append(
                f"| {paper_name} | {b:.2f} | {pb:.2f} | {b-pb:+.2f} | "
                f"{n:.2f} | {pn:.2f} | {n-pn:+.2f} | {h:.2f} | {ph:.2f} | {nb}/{nn} |")
        if bs:
            mb, mn = sum(bs) / len(bs), sum(ns) / len(ns)
            lines.append(
                f"| **平均** | **{mb:.2f}** | {sum(pbs)/len(pbs):.2f} | {mb-sum(pbs)/len(pbs):+.2f} | "
                f"**{mn:.2f}** | {sum(pns)/len(pns):.2f} | {mn-sum(pns)/len(pns):+.2f} | "
                f"**{sum(hms)/len(hms):.2f}** | {sum(phms)/len(phms):.2f} | - |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
