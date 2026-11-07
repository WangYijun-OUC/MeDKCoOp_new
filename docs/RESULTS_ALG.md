# 算法改进（方案 3）实验结果

> 日期：2026-11-07
> 条件：BiomedCLIP ViT-B/16、16-shot base2new、50 epoch、batch 4、seeds 1/2/3
> 产物：`/data/wangyijun/ckpt/MeDKCoOp_alg/base2new/alg_v3/`（仅保留日志）

## 1. 三组对照

| 组 | trainer | 关键设置 | 有效可训练参数 |
|---|---|---|---|
| `orig` | `MeDKCoOp` | `GRAD_CTX=False`，`N_CTX=2`，lr=2.5e-3 | `GCN_fuse` |
| `fix` | `MeDKCoOpFix` | `GRAD_CTX=True`，`N_CTX=2`，lr=2.5e-3 | `GCN_fuse` + `ctx` + `meta_net` |
| `alg_v3` | `MeDKCoOpFix` | `GRAD_CTX` + `LEARN_BETA` + `GRAD_IMGPROTO` + `TRAIN_VISUAL_NET`，`N_CTX=8`，lr=1e-2 | 上述 + `beta` + `fc` + `visual_net` |

## 2. 结果（8 数据集 × 3 seeds 均值）

| 数据集 | orig base | fix base | alg_v3 base | orig new | fix new | alg_v3 new | Δnew |
|---|---|---|---|---|---|---|---|
| BTMRI1 | 85.87 | 85.70 | 85.87 | 91.61 | 91.61 | **94.18** | **+2.57** |
| CHMNIST | 93.44 | 93.44 | 93.79 | 60.11 | 60.15 | **62.77** | **+2.66** |
| Covid-19 | 78.58 | 78.57 | 78.98 | 91.15 | 91.16 | 91.20 | +0.05 |
| CTKIDNEY | 90.66 | 90.66 | 89.02 | 84.65 | 84.68 | 85.14 | +0.50 |
| LC25000 | 93.64 | 93.68 | 93.41 | 95.47 | 95.63 | 95.60 | +0.13 |
| RETINA | 89.48 | 89.53 | 90.52 | 58.38 | 58.44 | **60.85** | **+2.46** |
| OCTMNIST | 80.73 | 80.74 | 82.72 | 85.88 | 85.88 | 85.75 | −0.13 |
| Kvasir | 87.61 | 87.72 | 87.56 | 70.72 | 70.67 | 71.28 | +0.55 |
| **平均** | **87.50** | **87.51** | **87.73** | **79.75** | **79.78** | **80.85** | **+1.10** |

HM（8 数据集 HM 的平均）：orig 82.56 → fix 82.58 → **alg_v3 83.62**（+1.06）。

## 3. 结论

1. **只修 `ctx` 梯度（fix）几乎无效**（+0.03 pp）。原因：`N_CTX=2` + lr=2.5e-3 + 50 epoch
   （16-shot 下每 epoch 仅 8 步）总位移只有 `ctx` 尺度的约 10%，学不动。
2. **放大 prompt 学习能力后（alg_v3）才有实质提升**：New +1.10 pp，HM +1.06 pp。
   增益集中在 `new` 类，正是 prompt 泛化能力该起作用的方向。
3. 逐数据集看，提升最大的是 BTMRI1（+2.57）、CHMNIST（+2.66）、RETINA（+2.46）；
   OCTMNIST 略降 0.13，CTKIDNEY base 降 1.64 但 new 升 0.50（HM 基本持平）。
4. 这说明**论文原实现的瓶颈不在 GCN 融合，而在 learnable prompt 没被真正优化**：
   一旦给它足够的容量与步长，new 类泛化就能提升，与论文"prompt 学习带来泛化"的叙述一致。

## 4. 复现命令

```bash
cd /home/wangyijun/MeDKCoOp
export CKPT=/data/wangyijun/ckpt/MeDKCoOp_alg
export PATH=/data/wangyijun/conda/medkcoop/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MEDSAM_CACHE=1 MEDSAM_CHUNK=2 EVAL_BS=4

OPTS="TRAINER.GRAPHDCPL.N_CTX 8 OPTIM.LR 0.01 \
TRAINER.GRAPHDCPL.LEARN_BETA True TRAINER.GRAPHDCPL.GRAD_IMGPROTO True \
TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET True"

RUN_TAG=alg_v3 TRAINER=MeDKCoOpFix EXTRA_OPTS="$OPTS" GPU=0 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1
```

汇总：

```bash
python tools/compare_alg.py
```

## 5. 注意事项

- **显存**：`N_CTX=8` + `EVAL_BS=16` 会 OOM（实测），脚本需用 `EVAL_BS=4`、`MEDSAM_CHUNK=2`。
- **不是论文数值**：alg_v3 是改进方案，不能与论文表格直接比较；论文复现仍以 `orig` 列为准。
- 若要写进论文，需要再验证：不同 seed 的稳定性（本次 3 seeds 已跑）、
  以及 cross-data / DG 上是否同样受益（尚未跑）。
