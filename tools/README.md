# tools

| 文件 | 用途 | 使用方式 |
|---|---|---|
| `parse_test_res.py` | 读取各 `seed*/log.txt` 中的 `* accuracy:`，输出均值/标准差 | `python tools/parse_test_res.py <实验目录> [--test-log]` |

## 待补（内容整改阶段）

- `collect_results.py`：遍历 `/data/wangyijun/ckpt` 下的实验目录，输出统一
  `results.csv`（`dataset,trainer,config,seed,split,acc,elapsed,log_path`），
  作为论文三张表的唯一数据源，避免手工抄数。
- 原 `heatmap*.py`、`test_blip*.py`、`interpret_prompt.py`、`combine_acc_columns.py`
  未在论文中使用，已从仓库移除（备份与 git 历史中仍可找回）。
