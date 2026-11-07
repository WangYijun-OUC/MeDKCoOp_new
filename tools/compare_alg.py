#!/usr/bin/env python3
"""对比 base2new 三组：原版 / 修复版 / 算法改进版（方案 3）。

  orig      : /data/wangyijun/ckpt/paper_base2new/base2new            (MeDKCoOp)
  fix       : /data/wangyijun/ckpt/MeDKCoOp_fixcmp/base2new/cmp_fix_ctx (MeDKCoOpFix, GRAD_CTX)
  alg_v3    : /data/wangyijun/ckpt/MeDKCoOp_alg/base2new/alg_v3        (MeDKCoOpFix + 全开)
"""

import glob
import os
import re
import statistics

ACC_RE = re.compile(r"\* accuracy:\s*([\d.]+)%")
CFG = "vit_b16_0002_b50_randflip"
DATASETS = ["BTMRI1", "CHMNIST", "covid", "ctkidney", "lungcolon", "retina", "octmnist", "kvasir"]
SEEDS = ["1", "2", "3"]
GROUPS = [
    ("orig", "/data/wangyijun/ckpt/paper_base2new/base2new", "MeDKCoOp"),
    ("fix", "/data/wangyijun/ckpt/MeDKCoOp_fixcmp/base2new/cmp_fix_ctx", "MeDKCoOpFix"),
    ("alg_v3", "/data/wangyijun/ckpt/MeDKCoOp_alg/base2new/alg_v3", "MeDKCoOpFix"),
]


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


def split_acc(d):
    if not os.path.isdir(d):
        return None
    v = last_acc(os.path.join(d, "log.txt"))
    if v is not None:
        return v
    for c in sorted(glob.glob(os.path.join(d, "log.txt-*")), reverse=True):
        v = last_acc(c)
        if v is not None:
            return v
    return None


def mean_n(vals):
    ok = [v for v in vals if v is not None]
    return (statistics.mean(ok), len(ok)) if ok else (None, 0)


def main():
    rows = {}
    for ds in DATASETS:
        row = {}
        for tag, root, tr in GROUPS:
            base = [split_acc(os.path.join(root, ds, "shots_16", tr, CFG, "seed" + s, "train_base")) for s in SEEDS]
            new = [split_acc(os.path.join(root, ds, "shots_16", tr, CFG, "seed" + s, "test_new")) for s in SEEDS]
            row[tag] = (mean_n(base), mean_n(new))
        rows[ds] = row

    print("| 数据集 | orig base | fix base | alg_v3 base | orig new | fix new | alg_v3 new | Δnew(v3-orig) |")
    print("|---|---|---|---|---|---|---|---|")
    agg = {t: {"b": [], "n": []} for t, _, _ in GROUPS}
    for ds in DATASETS:
        cells = []
        for tag in ("orig", "fix", "alg_v3"):
            b, bn = rows[ds][tag][0]
            cells.append(f"{b:.2f}({bn})" if b is not None else "-")
            if b is not None:
                agg[tag]["b"].append(b)
        for tag in ("orig", "fix", "alg_v3"):
            n, nn = rows[ds][tag][1]
            cells.append(f"{n:.2f}({nn})" if n is not None else "-")
            if n is not None:
                agg[tag]["n"].append(n)
        d = None
        if rows[ds]["alg_v3"][1][0] is not None and rows[ds]["orig"][1][0] is not None:
            d = rows[ds]["alg_v3"][1][0] - rows[ds]["orig"][1][0]
        print(f"| {ds} | " + " | ".join(cells) + f" | {d:+.2f} |" if d is not None else f"| {ds} | " + " | ".join(cells) + " | - |")

    print()
    for key, name in (("b", "Base"), ("n", "New")):
        parts = []
        for tag in ("orig", "fix", "alg_v3"):
            vals = agg[tag][key]
            parts.append(f"{tag}={statistics.mean(vals):.2f}" if vals else f"{tag}=NA")
        print(f"平均 {name}: " + ", ".join(parts))
    if agg["orig"]["n"] and agg["alg_v3"]["n"]:
        print(f"New 提升（alg_v3 - orig）: {statistics.mean(agg['alg_v3']['n']) - statistics.mean(agg['orig']['n']):+.2f} pp")
    if agg["orig"]["b"] and agg["alg_v3"]["b"]:
        print(f"Base 变化（alg_v3 - orig）: {statistics.mean(agg['alg_v3']['b']) - statistics.mean(agg['orig']['b']):+.2f} pp")


if __name__ == "__main__":
    main()
