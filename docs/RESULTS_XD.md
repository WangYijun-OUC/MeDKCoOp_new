# cross-data 与 domain generalization 复现结果

> 实验日期：2026-11-06
> 条件：BiomedCLIP ViT-B/16、源域 BTMRI1 16-shot、100 epoch、batch 4、SGD lr=2.5e-3
> trainer：`MeDKCoOpFix`（`GRAD_CTX=True`，与论文实现数值等价，见 [RESULTS_FIX.md](RESULTS_FIX.md)）
> 产物：`/data/wangyijun/ckpt/MeDKCoOp_xd/cross_data/xd/`

## 1. 源域（BTMRI1，100 epoch，16-shot）

| seed | accuracy | macro-F1 |
|---|---|---|
| 1 | 86.54 | — |
| 2 | 80.76 | — |
| 3 | 83.98 | — |
| **均值** | **83.76** | |
| 论文 Table 2 源域 | 84.33 | Δ = **−0.57** |

## 2. cross-data（7 个目标域，3 seeds 均值）

| 目标域 | 复现 | 论文 Table 2 | Δ |
|---|---|---|---|
| CHMNIST | 27.02 | 32.60 | **−5.58** |
| Covid-19 | 61.78 | 63.00 | −1.22 |
| CTKIDNEY | 60.25 | 60.80 | −0.55 |
| LC25000 (lungcolon) | 62.91 | 63.27 | −0.36 |
| RETINA | 37.79 | 37.70 | **+0.09** |
| OCTMNIST | 61.17 | 64.30 | −3.13 |
| Kvasir | 58.58 | 58.77 | −0.19 |
| **7 个目标平均** | **52.81** | 55.06 | −2.25 |
| **含源域 8 个平均** | **55.99** | 58.10 | −2.11 |

5 个目标的偏差在 ±1.2 pp 内（含 3 个 ≤0.55），与论文高度一致。

## 3. 两个偏差较大的目标域

### CHMNIST（−5.58）

- 目标集只有 8 类、每类 16-shot（共 128 张）用于构建目标原型 `new_fc`；
- 源域可迁移的只有 `GCN_fuse`（源类分类头不能跨标签使用，见 `TRANSFER_EVAL`）；
- 之前做的对照：把目标 shots 提到 200（1600 张）只到 24.14%，说明瓶颈不在样本量；
- 结合 base2new 的结论（`ctx` 训练与否对结果几乎无影响），这里是**目标原型质量**的上限。

### OCTMNIST（−3.13）

- 测试集 32,792 张（最大），原型同样来自 16-shot/类；
- 该数据集在 base2new 里也是波动最大的（论文 3 seed base 80.8/85.5/83.5）。

> 这两个数据集偏差的方向一致（复现偏低），与"目标原型噪声大"的解释相容，
> 不是脚本或 checkpoint 加载的问题（已用同参数、同 seed、100 epoch 复核）。

## 4. domain generalization（未完成，被硬件中断）

DG 使用 `*v2` 数据集作为目标域。第一轮 21 个评测全部失败（`EVAL_BS=16` 时
MedSAM 在 v2 更大的测试集上 OOM，且 `xd_test.sh` 当时在失败后仍返回 0，掩盖了问题）。
修复脚本与 chunk 大小后重新排队时，**GPU5 报 Xid 79（GPU has fallen off the bus），
CUDA 驱动随后对所有进程不可用**，实验被迫中止。

已完成的部分：`covidv2` 2 个 seed（61.99，论文 63.37，Δ −1.38）。
其余 6 个目标域（CHMNISTv2 / ctkidneyv2 / lungcolonv2 / retinav2 / octmnistv2 / kvasirv2）
待机器重启后按下面的命令补跑。

**已知风险（实验语义，不是 bug）**：`datasets/*v2.py` 把同一份 split 同时赋给
`train_x` 与 `test`（`super().__init__(train_x=data, test=data)`）。而 MeDKCoOp 会用
`train_x` 构建视觉知识图，因此 **v2 目标域的原型包含了测试集样本**，DG 数值会偏高。
论文 Table 3 的数值大概率也是在这个设定下得到的（两张表数字几乎相同），
但若要作为严格 DG 结果使用，需要改成"只用目标域 train split 构建原型"。

## 5. 复现命令

```bash
cd /home/wangyijun/MeDKCoOp
export CKPT=/data/wangyijun/ckpt/MeDKCoOp_xd
export PATH=/data/wangyijun/conda/medkcoop/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 源域训练（每个 seed 一次）
RUN_TAG=xd TRAINER=MeDKCoOpFix GPU=8 bash scripts/medkcoop/xd_train.sh BTMRI1 1

# 目标域评测
RUN_TAG=xd TRAINER=MeDKCoOpFix GPU=8 bash scripts/medkcoop/xd_test.sh CHMNIST 1 BTMRI1
```

## 6. 备注

- 源域用了修复版 `MeDKCoOpFix`（`GRAD_CTX=True`），与原版 base2new 差异 ≤0.04 pp，
  因此源域 83.76 可直接与论文 84.33 比较；
- cross-data 的评测走 `test` 分支（`new_fc` 用目标域 16-shot 重建原型），
  这是原项目设计，不是 bug（见 `docs/ENVIRONMENT.md` 第 7 节）；
- 单次 OCTMNIST 评测需过 3.3 万张图的 MedSAM 前向（约 20 分钟），
  v2 版本需要 2.7 万张（约 1.5 小时）；已在 `fixed/medkcoop.py` 加入
  MedSAM 特征缓存（`MEDSAM_CACHE=1`，maxdiff 3e-5），后续重复评测可复用。
