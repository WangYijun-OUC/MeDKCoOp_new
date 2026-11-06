#!/usr/bin/env python3
"""汇总 cross-data 与 domain generalization 结果，与论文 Table 2/3 对比。"""

import os
import re
import statistics

ACC_RE = re.compile(r"\* accuracy:\s*([\d.]+)%")
ROOT = "/data/wangyijun/ckpt/MeDKCoOp_xd/cross_data/xd"
TRAINER = "MeDKCoOpFix"
SEEDS = ["1", "2", "3"]

# 论文 Table 2（cross-data）/ Table 3（DG）
PAPER_XD = {"CHMNIST": 32.60, "covid": 63.00, "ctkidney": 60.80,
            "lungcolon": 63.27, "retina": 37.70, "octmnist": 64.30, "kvasir": 58.77}
PAPER_DG = {"CHMNIST": 32.33, "covid": 63.37, "ctkidney": 60.77,
            "lungcolon": 63.07, "retina": 37.53, "octmnist": 64.33, "kvasir": 57.77}
PAPER_SRC = 84.33


def acc(path):
    if not os.path.isfile(path):
        return None
    val = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ACC_RE.search(line)
            if m:
                val = float(m.group(1))
    return val


def collect(target_key, suffix):
    vals = []
    for seed in SEEDS:
        d = os.path.join(ROOT, "evaluation", "BTMRI1",
                         target_key + suffix, "shots_16", TRAINER, "seed" + seed)
        vals.append(acc(os.path.join(d, "log.txt")))
    return [v for v in vals if v is not None]


def main():
    src = []
    for seed in SEEDS:
        src.append(acc(os.path.join(ROOT, "source", "BTMRI1", "shots_16",
                                    TRAINER, "seed" + seed, "log.txt")))
    src = [v for v in src if v is not None]
    print("## 源域（BTMRI1, 100 epoch, 16-shot）")
    print(f"seeds: {src}")
    if src:
        print(f"均值 {statistics.mean(src):.2f}，论文 {PAPER_SRC}，"
              f"Δ {statistics.mean(src) - PAPER_SRC:+.2f}\n")

    for title, paper, suffix in (("cross-data (Table 2)", PAPER_XD, ""),
                                 ("domain generalization (Table 3)", PAPER_DG, "v2")):
        print(f"## {title}")
        print("| 目标 | 复现 | 论文 | Δ | seeds |")
        print("|---|---|---|---|---|")
        means = []
        for ds, pv in paper.items():
            vals = collect(ds, suffix)
            if vals:
                m = statistics.mean(vals)
                means.append(m)
                print(f"| {ds} | {m:.2f} | {pv:.2f} | {m - pv:+.2f} | {len(vals)} |")
            else:
                print(f"| {ds} | - | {pv:.2f} | - | 0 |")
        if means:
            pm = statistics.mean([paper[d] for d in paper if collect(d, suffix)])
            print(f"| **平均** | **{statistics.mean(means):.2f}** | {pm:.2f} | "
                  f"{statistics.mean(means) - pm:+.2f} | |")
        print()


if __name__ == "__main__":
    main()
