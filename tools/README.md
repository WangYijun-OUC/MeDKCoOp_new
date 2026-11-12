# tools

Utilities for aggregating experiment results from DASSL log files.

| File | Purpose |
|---|---|
| `collect_results.py` | Walk an output directory, group runs by `dataset/trainer/config` and seed, and report mean accuracy (optionally to CSV) |
| `collect_base2new.py` | Fixed-schedule aggregation of base-to-new runs |
| `compare_fix.py` | Compare the released implementation against the gradient-enabled variant |
| `compare_xd.py` | Compare cross-dataset and domain-generalization runs against the paper tables |
| `compare_alg.py` | Three-way comparison (released / gradient-fixed / larger-context variant) |
| `parse_test_res.py` | Read per-seed `log.txt` and print mean and standard deviation |

## Usage

```bash
# generic aggregation: any output tree containing seed*/log.txt
python tools/collect_results.py <output_root> --csv results.csv

# paper-specific comparisons (paths default to the released experiment layout)
python tools/compare_xd.py
python tools/compare_alg.py
```

Each script reads `* accuracy:` lines from DASSL logs and falls back to the timestamped
`log.txt-*` files that DASSL writes when a run is re-executed in the same directory.
