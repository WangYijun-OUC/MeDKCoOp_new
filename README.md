# MeDKCoOp

Biomedical **D**ual **K**nowledge-guided graph prompt learning（ACM MM'25）代码仓库。

> 论文：*MeDKCoOp: Dual Knowledge-guided Graph Prompt Learning for Biomedical Vision-Language Models*
> 原项目：`/home/wangyijun/BiomedCoOp`（保持原样，未做任何修改）
> 本文档描述的是**已按"只用论文实际用到的东西"精简过**的结构。

## 目录结构（每一层都有明确用途）

```
MeDKCoOp/
├── train.py                       # 唯一入口：训练 / 评测（DASSL）
├── bootstrap.py                   # 路径引导 + 离线 BiomedCLIP 加载（train.py 第一行导入）
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
│   ├── clip/                      # CLIP + pmcclip（已合并 clip1 的 MaPLe/IVLP 分支）
│   └── MedSAM/                    # 视觉知识分支
├── tools/parse_test_res.py        # 日志 → 均值/方差
└── docs/                          # 环境与运行 / 复现方案 / 结构映射
```

## 保留的东西为什么都在

| 位置 | 存在理由 |
|---|---|
| `trainers/medkcoop/reference/medkcoop.py` | 论文方法当前可读、可运行的最近版本；重建以它为骨架 |
| `reference/paper_bytecode/*.pyc` | 论文 "Ours"（`GRAPH_DCPL_NO_TXT_VIS_GRAPH`）源码丢失后的唯一残留 |
| `ablation/` 16 个文件 | 分别对应 Table 4（视觉/文本知识消融）、Table 5（KRGC/KRTL 组合）、Fig 4（三种 backbone） |
| `trainers/baselines/` 8 个 | 论文表格中的对比方法：BiomedCoOp、CoOp、CoCoOp、KgCoOp、MaPLe、DCPL、GraphAdapter、Zeroshot(BiomedCLIP) |
| `third_party/clip` | 单一 CLIP 包：标准 CLIP + `pmcclip.py` + MaPLe/DCPL 的 IVLP 分支；所有 trainer 共用 |
| `datasets/*v2.py` | domain generalization 的目标域数据（整份数据直接评测） |

## 已删除的内容（可从备份恢复）

按"不用就删"的原则移除了：非论文 baseline（ClipAdapter/DPLQ/LP/LP2/ProGrad/TipAdapter）、
`main.py` 及其配套（`utils.py`、`utils/`、`datasets/utils.py`、`datasets/__init__.py`）、
非论文数据集（BTMRI/BTMRIv2/BTMRI1v2/BUSI/CCBTM*/DermaMNIST*/KneeXray*）、
34→16 之外的消融变体与 `hyper/`、BLIP 支线、qnn、descriptors、heatmap/可视化试跑脚本、
上游 README/assets、`data/` 旧 split、`trainers copy/`。

这些内容仍可从两处找回：

- 本地归档分支 `archive/pre-squash`（包含整理前的完整历史，未推送到远端）
- 备份：`/home/wangyijun/backup/BiomedCoOp_2026-11-05/`（全量 tar + `repo_archived/` 非代码大文件）

## 关键外部资源

| 资源 | 路径 |
|---|---|
| 数据集 | `/data/wangyijun/dataset/` |
| BiomedCLIP 权重 | `/data/wangyijun/model/BiomedCLIP/`、`/data/wangyijun/model/BiomedCLIP_local/` |
| MedSAM 权重 | `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` |
| 实验产物 | `/data/wangyijun/ckpt/Biomed_Graph_dcpl/` 等 |

## 快速开始

```bash
conda activate medkcoop
cd /home/wangyijun/MeDKCoOp
GPU=0 bash scripts/medkcoop/base2new.sh BTMRI1 1
```

完整环境、权重、单步命令与论文对照见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 当前状态

- 已适配并可跑通：`train.py`、离线 BiomedCLIP 加载、`configs/trainers/medkcoop/`、
  `scripts/medkcoop/base2new.sh`（训练 + 评测量化验证）。
- 已修复：跨目录 import、DASSL editable 指向、`--heatmap` 缺失提示、配置目录命名。
- 未适配：`scripts/medkcoop/` 下的其余脚本与 `scripts/baselines/*` 仍含旧 ckpt 路径与 `Device_id`，
  改写方式与 `base2new.sh` 相同；cross-data / DG 需重跑（旧产物不在磁盘）。

## 复现规划

见 [docs/REPRO_PLAN.md](docs/REPRO_PLAN.md)。
