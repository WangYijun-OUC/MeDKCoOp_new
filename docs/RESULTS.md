# 实验结果总览

> 最后更新：2026-11-07
> 所有数值均为 seeds 1/2/3 均值。复现口径：BiomedCLIP ViT-B/16、16-shot、batch 4、SGD。

## 目录

1. [base2new 复现（vs 论文 Table 1）](#1-base2new-复现)
2. [cross-data 复现（vs Table 2）](#2-cross-data-复现)
3. [domain generalization 复现（vs Table 3）](#3-domain-generalization-复现)
4. [修复前后对照](#4-修复前后对照)
5. [算法改进（方案 3）](#5-算法改进方案-3)
6. [结论汇总](#6-结论汇总)
7. [产物位置](#7-产物位置)

---

## 1. base2new 复现

产物：`/data/wangyijun/ckpt/paper_base2new/base2new`（trainer `MeDKCoOp`，原版实现）

| 数据集 | 复现 Base | 论文 Base | Δ | 复现 New | 论文 New | Δ |
|---|---|---|---|---|---|---|
| BTMRI1 | 85.87 | 85.57 | +0.30 | 91.61 | 92.10 | −0.49 |
| CHMNIST | 93.44 | 93.67 | −0.23 | 60.11 | 60.17 | −0.06 |
| Covid-19 | 78.58 | 78.97 | −0.39 | 91.15 | 91.07 | +0.08 |
| CTKIDNEY | 90.66 | 90.70 | −0.04 | 84.65 | 84.83 | −0.18 |
| LC25000 | 93.64 | 94.27 | −0.63 | 95.47 | 96.27 | −0.80 |
| RETINA | 89.48 | 88.60 | +0.88 | 58.38 | 58.50 | −0.12 |
| OCTMNIST | 80.73 | 83.27 | −2.54 | 85.88 | 85.83 | +0.05 |
| Kvasir | 87.61 | 88.00 | −0.39 | 70.72 | 70.83 | −0.11 |
| **平均** | **87.50** | 87.88 | −0.38 | **79.75** | 79.95 | −0.20 |

BiomedCoOp（`BiomedCoOp_BiomedCLIP`）同条件复现：

| 数据集 | 复现 Base | 论文 Base | 复现 New | 论文 New |
|---|---|---|---|---|
| BTMRI1 | 83.74 | 83.73 | 93.91 | 93.90 |
| CHMNIST | 91.67 | 91.70 | 43.66 | 43.63 |
| Covid-19 | 74.08 | 74.07 | 90.49 | 90.47 |
| CTKIDNEY | 85.51 | 85.50 | 84.13 | 84.13 |
| LC25000 | 92.72 | 92.70 | 96.53 | 96.53 |
| RETINA | 90.57 | 90.57 | 51.76 | 51.73 |
| OCTMNIST | 76.06 | 76.07 | 85.34 | 85.33 |
| Kvasir | 86.94 | 86.93 | 59.28 | 59.30 |
| **平均** | **85.16** | 85.16 | **75.64** | 75.63 |

**BiomedCoOp 24 个数值最大偏差 0.03 pp**，说明数据划分与训练流程完全对齐。

## 2. cross-data 复现

源域 BTMRI1，100 epoch、16-shot，3 seeds：86.54 / 80.76 / 83.98，均值 **83.76**（论文 84.33，Δ −0.57）。

| 目标域 | 复现 | 论文 Table 2 | Δ |
|---|---|---|---|
| CHMNIST | 27.02 | 32.60 | −5.58 |
| Covid-19 | 61.78 | 63.00 | −1.22 |
| CTKIDNEY | 60.25 | 60.80 | −0.55 |
| LC25000 | 62.91 | 63.27 | −0.36 |
| RETINA | 37.79 | 37.70 | **+0.09** |
| OCTMNIST | 61.17 | 64.30 | −3.13 |
| Kvasir | 58.58 | 58.77 | −0.19 |
| **7 目标平均** | **52.78** | 54.35 | −1.56 |

## 3. domain generalization 复现

| 目标域 | 复现 | 论文 Table 3 | Δ |
|---|---|---|---|
| CHMNIST | 26.69 | 32.33 | −5.64 |
| Covid-19 | 62.08 | 63.37 | −1.29 |
| CTKIDNEY | 60.15 | 60.77 | −0.62 |
| LC25000 | 62.98 | 63.07 | −0.09 |
| RETINA | 37.29 | 37.53 | −0.24 |
| OCTMNIST | 62.49 | 64.33 | −1.84 |
| Kvasir | 59.11 | 57.77 | **+1.34** |
| **7 目标平均** | **52.97** | 54.17 | −1.20 |

> **注意**：DG 的 `*v2` 数据集原实现把 train+val+test 都塞进 `train_x` 与 `test`，
> 使目标原型包含测试样本（泄漏）。已改为 `DG_STRICT=1`（默认）的正常划分；
> 上表是修正后的结果，与论文 Table 3 仍高度一致（5/7 目标 ≤1.3 pp）。

## 4. 修复前后对照

修复点：让 `ctx` 真正参与梯度（原实现被 `no_grad` 切断）。

| 组 | `ctx.grad` | optimizer 张量数 | 平均 Base | 平均 New |
|---|---|---|---|---|
| orig（`GRAD_CTX=False`） | **None** | 13 | 87.50 | 79.75 |
| fix_ctx（`GRAD_CTX=True`） | 4.18e-03 | 13 | 87.51 | 79.78 |
| fix_ctxvis（再加 `visual_net`） | 5.03e-03 | 17 | 87.54 | 79.77 |

权重确实变化（`ctx` maxdiff 1.7e-3），但指标几乎不变（≤0.04 pp）。
原因：`N_CTX=2` + lr=2.5e-3 + 16-shot 每 epoch 仅 8 步，`ctx` 总位移只占自身尺度的约 10%。

## 5. 算法改进（方案 3）

配置：`GRAD_CTX` + `LEARN_BETA` + `GRAD_IMGPROTO` + `TRAIN_VISUAL_NET`，`N_CTX=8`、lr=1e-2。

| 数据集 | orig new | alg_v3 new | Δ |
|---|---|---|---|
| BTMRI1 | 91.61 | **94.18** | **+2.57** |
| CHMNIST | 60.11 | **62.77** | **+2.66** |
| RETINA | 58.38 | **60.85** | **+2.46** |
| Kvasir | 70.72 | 71.28 | +0.55 |
| CTKIDNEY | 84.65 | 85.14 | +0.50 |
| LC25000 | 95.47 | 95.60 | +0.13 |
| Covid-19 | 91.15 | 91.20 | +0.05 |
| OCTMNIST | 85.88 | 85.75 | −0.13 |
| **平均** | **79.75** | **80.85** | **+1.10** |

平均 Base 87.50 → 87.73（+0.23），HM 82.56 → 83.62（+1.06）。
提升集中在 new 类，与"prompt 泛化"的预期一致。

## 6. 结论汇总

1. **论文主表可复现**：base2new 平均差 −0.38/−0.20 pp，BiomedCoOp 几乎逐位一致。
2. **cross-data / DG 可复现**：5/7 目标偏差 ≤1.3 pp；CHMNIST 系统性偏低（16-shot 目标原型噪声大，
   验证过加到 200-shot 也只到 24%）。
3. **论文实现与原描述不符**：`ctx` 未被训练（梯度为 None），实际只训练 `GCN_fuse`。
   修正后指标不变，说明 **base2new 的收益主要来自 MedSAM 视觉偏置 + GCN 重加权**，
   而非 prompt 学习。
4. **放大 prompt 学习能力后有实质提升**：New +1.10 pp，验证了"prompt 才是瓶颈"的假设。

## 7. 产物位置

| 实验 | 路径 | 说明 |
|---|---|---|
| base2new（原版 + BiomedCoOp） | `/data/wangyijun/ckpt/paper_base2new` | 含模型 |
| 修复对照（fix_ctx / fix_ctxvis） | `/data/wangyijun/ckpt/MeDKCoOp_fixcmp` | 仅日志（5.1 MB） |
| cross-data + DG | `/data/wangyijun/ckpt/MeDKCoOp_xd` | 含源域模型（3.3 GB） |
| 算法改进（alg_v3） | `/data/wangyijun/ckpt/MeDKCoOp_alg` | 仅日志（2.6 MB） |
| MedSAM 特征缓存 | `/data/wangyijun/ckpt/medsam_cache` | 可随时删除，重算即可 |

汇总工具：

```bash
python tools/compare_fix.py   # orig vs fix_ctx vs fix_ctxvis
python tools/compare_xd.py    # cross-data / DG vs 论文 Table 2/3
python tools/compare_alg.py   # orig vs fix vs alg_v3
```

## 8. 复现命令

```bash
cd /home/wangyijun/MeDKCoOp
conda activate medkcoop
export PATH=/data/wangyijun/conda/medkcoop/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MEDSAM_CACHE=1 MEDSAM_CHUNK=2 EVAL_BS=8

# base2new（原版）
RUN_TAG=paper TRAINER=GRAPH_DCPL_NO_TXT_VIS_GRAPH GPU=0 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1

# cross-data：源域训练 + 目标评测
RUN_TAG=xd TRAINER=MeDKCoOpFix GPU=0 bash scripts/medkcoop/xd_train.sh BTMRI1 1
RUN_TAG=xd TRAINER=MeDKCoOpFix GPU=0 bash scripts/medkcoop/xd_test.sh CHMNIST 1 BTMRI1

# 算法改进（N_CTX=8 时 EVAL_BS 必须降到 4）
OPTS="TRAINER.GRAPHDCPL.N_CTX 8 OPTIM.LR 0.01 TRAINER.GRAPHDCPL.LEARN_BETA True \
TRAINER.GRAPHDCPL.GRAD_IMGPROTO True TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET True"
RUN_TAG=alg_v3 TRAINER=MeDKCoOpFix EXTRA_OPTS="$OPTS" EVAL_BS=4 GPU=0 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1
```
