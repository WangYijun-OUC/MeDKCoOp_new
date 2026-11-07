# MeDKCoOp

Biomedical **D**ual **K**nowledge-guided graph prompt learning（ACM MM'25）。

> 论文：*MeDKCoOp: Dual Knowledge-guided Graph Prompt Learning for Biomedical Vision-Language Models*
> 原项目：`/home/wangyijun/BiomedCoOp`（保持原样，未做任何修改，完整快照在
> `/home/wangyijun/backup/BiomedCoOp_2026-11-05/`）

## 快速开始

```bash
conda activate medkcoop
cd /home/wangyijun/MeDKCoOp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# base2new（单次：数据集 + seed）
GPU=0 bash scripts/medkcoop/base2new.sh BTMRI1 1
```

## 复现状态（3 seeds 均值）

| 任务 | 复现 | 论文 | 偏差 |
|---|---|---|---|
| base2new 平均 Base（8 数据集） | 87.50 | 87.88 | −0.38 |
| base2new 平均 New（8 数据集） | 79.75 | 79.95 | −0.20 |
| cross-data 平均（7 目标） | 52.78 | 54.35 | −1.56 |
| domain generalization 平均（7 目标） | 52.97 | 54.17 | −1.20 |
| BiomedCoOp base2new | 85.16 / 75.64 | 85.16 / 75.63 | ≤0.03 |

完整表格与结论见 [docs/RESULTS.md](docs/RESULTS.md)。

## 目录结构

```
MeDKCoOp/
├── train.py                     # 唯一训练/评测入口（DASSL）
├── bootstrap.py                 # sys.path 引导 + 离线 BiomedCLIP 加载
├── configs/
│   ├── datasets/                # 8 个 v1 + 7 个 v2
│   └── trainers/                # medkcoop + 8 个 baseline
├── datasets/                    # 与 configs/datasets 一一对应
├── trainers/
│   ├── medkcoop/
│   │   ├── reference/           # 论文原实现（对照）+ 丢失源码的字节码
│   │   ├── fixed/               # 修复版 + 算法开关
│   │   └── ablation/            # 16 个论文消融变体
│   └── baselines/               # 8 个对比方法
├── scripts/
│   ├── medkcoop/                # base2new / xd_train / xd_test / few_shot / 消融
│   ├── baselines/               # 各 baseline 的同名脚本
│   └── run_all_methods.sh       # 批量调度
├── third_party/                 # Dassl.pytorch open_clip clip MedSAM
├── tools/                       # 结果汇总（compare_fix / compare_xd / compare_alg）
├── tests/                       # 6 个冒烟测试
└── docs/                        # 见 docs/README.md
```

详细的语义边界与 trainer 类清单见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

## 关于论文实现的一个已知事实

原实现（`trainers/medkcoop/reference/medkcoop.py`）中，learnable context 因为
`torch.no_grad()` 切断了梯度，**从未被训练**；实际只有 `GCN_fuse` 在更新。
修复版（`fixed/medkcoop.py`，`GRAD_CTX=True` 默认）让 context 真正参与训练，
指标几乎不变（≤0.04 pp），说明 base2new 的收益主要来自 MedSAM 视觉偏置 + GCN 重加权。
在此基础上放大 prompt 学习能力（`N_CTX=8`、`lr=1e-2`）可再提升 New 类 +1.10 pp。
细节与数据见 [docs/FIXES.md](docs/FIXES.md) 与 [docs/RESULTS.md](docs/RESULTS.md)。

## 文档

从 [docs/README.md](docs/README.md) 开始：

| 文档 | 内容 |
|---|---|
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | 目录结构、语义边界、开关说明 |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | 环境、权重、数据、运行命令 |
| [docs/RESULTS.md](docs/RESULTS.md) | 全部实验结果 |
| [docs/FIXES.md](docs/FIXES.md) | 代码问题与修复记录 |
| [docs/WEIGHTS.md](docs/WEIGHTS.md) | 磁盘清理记录 |
| [docs/archive/](docs/archive/) | 历史审查与方案（不再更新） |

## 关键外部资源

| 资源 | 路径 |
|---|---|
| 数据集 | `/data/wangyijun/dataset/` |
| BiomedCLIP 权重 | `/data/wangyijun/model/BiomedCLIP/` |
| MedSAM 权重 | `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` |
| 复现产物 | `/data/wangyijun/ckpt/{paper_base2new,MeDKCoOp_xd,MeDKCoOp_alg,MeDKCoOp_fixcmp}` |

## 测试

```bash
python -m pytest tests -q   # 6 passed
```

## 已删除的历史内容

非论文 baseline（ClipAdapter/DPLQ/LP/LP2/ProGrad/TipAdapter）、旧 few-shot 管线
（`main.py`、根 `utils.py`）、非论文数据集（BTMRI/BUSI/CCBTM*/DermaMNIST*/KneeXray*）、
34→16 之外的消融变体、BLIP、qnn、descriptors、heatmap 脚本、`data/` 旧 split、`trainers copy/`。

可从两处恢复：本地 git 分支 `archive/pre-squash`，或备份目录
`/home/wangyijun/backup/BiomedCoOp_2026-11-05/`（全量 tar + `repo_archived/`）。
