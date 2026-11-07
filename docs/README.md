# docs 导航

> 最后更新：2026-11-07

## 现行文档（会被持续维护）

| 文档 | 内容 | 什么时候看 |
|---|---|---|
| [STRUCTURE.md](STRUCTURE.md) | 目录结构、语义边界、trainer 类清单、开关说明 | 想知道"某个文件干什么" |
| [ENVIRONMENT.md](ENVIRONMENT.md) | 环境、权重、数据、单步运行命令、已知限制 | 换机器/重新搭环境 |
| [RESULTS.md](RESULTS.md) | **全部实验结果**：base2new、cross-data、DG、修复对照、算法改进 | 查数值、对比论文 |
| [FIXES.md](FIXES.md) | 代码问题清单与修复记录（含梯度证据） | 想知道改了什么、为什么改 |
| [WEIGHTS.md](WEIGHTS.md) | `/data` 权重与实验产物的清理记录 | 清理磁盘前 |

## 归档文档（历史快照，不再更新）

| 文档 | 说明 |
|---|---|
| [archive/REPRO_PLAN_2026-11-05.md](archive/REPRO_PLAN_2026-11-05.md) | 最初的备份与整理方案（含当时的审查结论） |
| [archive/AUDIT_2026-11-06.md](archive/AUDIT_2026-11-06.md) | 第一轮代码/结构审查（P0/P1/P2 清单） |
| [archive/REVIEW_codex_2026-11-06.md](archive/REVIEW_codex_2026-11-06.md) | codex 改动审查（含 R1 梯度问题发现） |
| [archive/VALIDATION_2026-11-06.md](archive/VALIDATION_2026-11-06.md) | codex 当时的运行验证记录 |

归档文档保留是为了追溯"为什么做某个决定"，其中的路径、目录名、数值可能已过时。

## 阅读顺序建议

1. 新接手项目：[STRUCTURE.md](STRUCTURE.md) → [ENVIRONMENT.md](ENVIRONMENT.md)
2. 只想复现数值：[ENVIRONMENT.md](ENVIRONMENT.md) → [RESULTS.md](RESULTS.md) 第 8 节
3. 想知道代码改了什么：[FIXES.md](FIXES.md) → [RESULTS.md](RESULTS.md) 第 4、5 节
