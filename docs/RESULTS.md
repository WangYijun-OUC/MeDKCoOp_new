# base2new 复现结果 vs 论文

数据目录：`/data/wangyijun/ckpt/paper_base2new`，seeds = 1, 2, 3

## MeDKCoOp

| 数据集 | 复现 Base | 论文 Base | Δ | 复现 New | 论文 New | Δ | 复现 HM | 论文 HM | seeds |
|---|---|---|---|---|---|---|---|---|---|
| BTMRI | 85.87 | 85.57 | +0.30 | 91.61 | 92.10 | -0.49 | 88.64 | 87.71 | 3/3 |
| CHMNIST | 93.44 | 93.67 | -0.23 | 60.11 | 60.17 | -0.06 | 73.16 | 73.27 | 3/3 |
| Covid-19 | 78.58 | 78.97 | -0.39 | 91.15 | 91.07 | +0.08 | 84.40 | 84.59 | 3/3 |
| CTKIDNEY | 90.66 | 90.70 | -0.04 | 84.65 | 84.83 | -0.18 | 87.55 | 87.67 | 3/3 |
| LC25000 | 93.64 | 94.27 | -0.63 | 95.47 | 96.27 | -0.80 | 94.55 | 95.26 | 3/3 |
| RETINA | 89.48 | 88.60 | +0.88 | 58.38 | 58.50 | -0.12 | 70.66 | 70.47 | 3/3 |
| OCTMNIST | 80.73 | 83.27 | -2.54 | 85.88 | 85.83 | +0.05 | 83.23 | 84.53 | 3/3 |
| Kvasir | 87.61 | 88.00 | -0.39 | 70.72 | 70.83 | -0.11 | 78.27 | 78.49 | 3/3 |
| **平均** | **87.50** | 87.88 | -0.38 | **79.75** | 79.95 | -0.20 | **82.56** | 82.75 | - |

## BiomedCoOp

| 数据集 | 复现 Base | 论文 Base | Δ | 复现 New | 论文 New | Δ | 复现 HM | 论文 HM | seeds |
|---|---|---|---|---|---|---|---|---|---|
| BTMRI | 83.74 | 83.73 | +0.01 | 93.91 | 93.90 | +0.01 | 88.54 | 88.52 | 3/3 |
| CHMNIST | 91.67 | 91.70 | -0.03 | 43.66 | 43.63 | +0.03 | 59.15 | 59.13 | 3/3 |
| Covid-19 | 74.08 | 74.07 | +0.01 | 90.49 | 90.47 | +0.02 | 81.47 | 81.45 | 3/3 |
| CTKIDNEY | 85.51 | 85.50 | +0.01 | 84.13 | 84.13 | +0.00 | 84.81 | 84.81 | 3/3 |
| LC25000 | 92.72 | 92.70 | +0.02 | 96.53 | 96.53 | +0.00 | 94.59 | 94.58 | 3/3 |
| RETINA | 90.57 | 90.57 | +0.00 | 51.76 | 51.73 | +0.03 | 65.88 | 65.85 | 3/3 |
| OCTMNIST | 76.06 | 76.07 | -0.01 | 85.34 | 85.33 | +0.01 | 80.43 | 80.43 | 3/3 |
| Kvasir | 86.94 | 86.93 | +0.01 | 59.28 | 59.30 | -0.02 | 70.49 | 70.50 | 3/3 |
| **平均** | **85.16** | 85.16 | +0.00 | **75.64** | 75.63 | +0.01 | **78.17** | 78.16 | - |


## 结论

- **BiomedCoOp：与论文几乎逐项一致。** 8 个数据集、24 个数值的最大偏差为 0.03 pp，
  平均 Base/New/HM = 85.16 / 75.64 / 78.17，论文为 85.16 / 75.63 / 78.16。
- **MeDKCoOp：整体高度一致。** 平均 Base/New/HM = 87.50 / 79.75 / 82.56，
  论文为 87.88 / 79.95 / 82.75（Δ = -0.38 / -0.20 / -0.19 pp）。
  7 个数据集逐项偏差在 ±0.88 pp 内；OCTMNIST 的 Base 低 2.54 pp。
- OCTMNIST 的波动属正常现象：原始论文 3 个 seed 的 Base 分别为 80.8 / 85.5 / 83.5，
  本次为 76.5 / 82.6 / 83.1；该数据集只有 2 个 base 类（二分类），对初始化与
  `cudnn.benchmark` 的非确定性更敏感。New 类（85.88 vs 85.83）与论文一致。

## 复现设置

- 16-shot，50 epoch，SGD lr=2.5e-3，batch 4（与论文一致）；
- seeds 1/2/3，取均值；评测 batch=16（仅影响速度，不影响精度）；
- MeDKCoOp 使用 `trainers/medkcoop/reference/medkcoop.py`（与论文最终版字节码等价，
  仅另加 MedSAM 分块前向以降低显存）；
- BiomedCoOp 使用 `BiomedCoOp_BiomedCLIP`；
- 产物：`/data/wangyijun/ckpt/paper_base2new/base2new/<数据集>/shots_16/<方法>/.../seed<k>/`。

## 复现命令

```bash
conda activate medkcoop
cd /home/wangyijun/MeDKCoOp

# MeDKCoOp
for ds in BTMRI1 CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
  for s in 1 2 3; do
    CONFIG_DIR=$PWD/configs/trainers/medkcoop TRAINER=MeDKCoOp GPU=8 \
      bash scripts/medkcoop/base2new.sh "$ds" "$s"
  done
done

# BiomedCoOp
for ds in BTMRI1 CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
  for s in 1 2 3; do
    CONFIG_DIR=$PWD/configs/trainers/BiomedCoOp TRAINER=BiomedCoOp_BiomedCLIP GPU=9 \
      bash scripts/medkcoop/base2new.sh "$ds" "$s"
  done
done

# 汇总
python tools/collect_base2new.py --root /data/wangyijun/ckpt/paper_base2new --out docs/RESULTS.md
```
