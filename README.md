# MeDKCoOp

Biomedical **D**ual **K**nowledge-guided graph prompt learning（ACM MM'25）代码仓库。

> 论文：*MeDKCoOp: Dual Knowledge-guided Graph Prompt Learning for Biomedical Vision-Language Models*
> 原项目：`/home/wangyijun/BiomedCoOp`（保持原样，未做任何修改）
> 本文档描述的是**已按"只用论文实际用到的东西"精简过**的结构。

## 目录结构（每一层都有明确用途）

```
MeDKCoOp/
├── train.py                       # 唯一入口：训练 / 评测（DASSL）
├── configs/
│   ├── datasets/                  # 论文 8 个数据集的 v1（base2new/few-shot）+ 7 个 v2（DG 目标域）
│   └── trainers/                  # MeDKCoOp + 8 个 baseline 的超参
├── datasets/                      # 与 configs/datasets 一一对应：8 个 v1 + 7 个 v2
├── trainers/
│   ├── medkcoop/
│   │   ├── reference/medkcoop.py  # 现存最接近论文的参考实现
│   │   ├── reference/paper_bytecode/  # 论文最终方法丢失源码后仅存的字节码（重建依据）
│   │   └── ablation/              # 论文 Table 4/5、Fig 4 用到的 16 个变体
│   └── baselines/                 # 论文对比的 8 个方法
├── scripts/
│   ├── medkcoop/                  # base2new / cross-data / DG / few-shot / 消融
│   └── baselines/                 # 8 个 baseline 的同类脚本
├── third_party/
│   ├── Dassl.pytorch/             # 训练框架（需在新环境 pip install -e）
│   ├── open_clip/                 # BiomedCLIP 加载
│   ├── clip/  clip1/              # 两套 CLIP（PubMed/PMC 系 vs MaPLe/DPLQ 系）
│   └── MedSAM/                    # 视觉知识分支
├── tools/parse_test_res.py        # 日志 → 均值/方差
└── docs/                          # 复现方案 + 结构映射
```

## 保留的东西为什么都在

| 位置 | 存在理由 |
|---|---|
| `trainers/medkcoop/reference/medkcoop.py` | 论文方法当前可读、可运行的最近版本；重建以它为骨架 |
| `reference/paper_bytecode/*.pyc` | 论文 "Ours"（`GRAPH_DCPL_NO_TXT_VIS_GRAPH`）源码丢失后的唯一残留 |
| `ablation/` 16 个文件 | 分别对应 Table 4（视觉/文本知识消融）、Table 5（KRGC/KRTL 组合）、Fig 4（三种 backbone） |
| `trainers/baselines/` 8 个 | 论文表格中的对比方法：BiomedCoOp、CoOp、CoCoOp、KgCoOp、MaPLe、DCPL、GraphAdapter、Zeroshot(BiomedCLIP) |
| `third_party/clip` 与 `clip1` | 前者供 PubMedCLIP/PMC-CLIP 系使用，后者供 MaPLe/DCPL-CLIP/GraphAdapter 使用，均被保留的 trainer 引用 |
| `datasets/*v2.py` | domain generalization 的目标域数据（整份数据直接评测） |

## 已删除的内容（可从备份恢复）

按"不用就删"的原则移除了：非论文 baseline（ClipAdapter/DPLQ/LP/LP2/ProGrad/TipAdapter）、
`main.py` 及其配套（`utils.py`、`utils/`、`datasets/utils.py`、`datasets/__init__.py`）、
非论文数据集（BTMRI/BTMRIv2/BTMRI1v2/BUSI/CCBTM*/DermaMNIST*/KneeXray*）、
34→16 之外的消融变体与 `hyper/`、BLIP 支线、qnn、descriptors、heatmap/可视化试跑脚本、
上游 README/assets、`data/` 旧 split、`trainers copy/`。

这些内容仍可从两处找回：

- git 历史：`git show 5e7ecc2:<path>`（基线提交包含整理前的全部代码文件）
- 备份：`/home/wangyijun/backup/BiomedCoOp_2026-11-05/`（全量 tar + `repo_archived/` 非代码大文件）

## 关键外部资源

| 资源 | 路径 |
|---|---|
| 数据集 | `/data/wangyijun/dataset/` |
| BiomedCLIP 权重 | `/data/wangyijun/model/BiomedCLIP/`、`/data/wangyijun/model/BiomedCLIP_local/` |
| MedSAM 权重 | `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` |
| 实验产物 | `/data/wangyijun/ckpt/Biomed_Graph_dcpl/` 等 |

## 当前状态

**结构已定稿，代码内容尚未适配**：`train.py` 的 import 仍指向旧路径，脚本内仍有旧目录与硬编码，
DASSL 也还是 editable 指向原项目。具体清单见 [docs/STRUCTURE.md](docs/STRUCTURE.md)，
这些问题在下一步统一修改，不改变本结构。

## 复现规划

见 [docs/REPRO_PLAN.md](docs/REPRO_PLAN.md)。
