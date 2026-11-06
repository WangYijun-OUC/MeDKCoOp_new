# MeDKCoOp 修复前后对照结果

> 实验日期：2026-11-06
> 条件：BiomedCLIP ViT-B/16、16-shot base2new、50 epoch、batch 4、SGD lr=2.5e-3、seeds 1/2/3
> 原版产物：`/data/wangyijun/ckpt/paper_base2new/base2new`（trainer `MeDKCoOp`）
> 修复版产物：`/data/wangyijun/ckpt/MeDKCoOp_fixcmp/base2new/cmp_fix_ctx{,_vis}`（trainer `MeDKCoOpFix`）

## 0. 结论

**修复让 `ctx` 与 `visual_net` 真正参与了训练（权重确实变化），但 base2new 指标几乎不变：**
平均 Base 87.50 → 87.51 / 87.54，平均 New 79.75 → 79.78 / 79.77，逐数据集偏差 ≤ 0.28 pp。

也就是说：**论文报告的 base2new 能力主要来自"冻结的 BiomedCLIP 特征 + MedSAM 视觉偏置 + GCN 重加权"，
而不是来自 learnable prompt 的优化**。原实现"只训 GCN"并不构成 base2new 复现的障碍，
但它与论文 3.4 节的描述不符。

## 1. 三个实验组

| 组 | trainer 类 | 关键开关 | 实际可训练参数 |
|---|---|---|---|
| `orig` | `MeDKCoOp` | `GRAD_CTX=False` | 仅 `GCN_fuse`（512×512 + bias） |
| `fix_ctx` | `MeDKCoOpFix` | `GRAD_CTX=True` | `GCN_fuse` + `ctx` + `meta_net` |
| `fix_ctxvis` | `MeDKCoOpFix` | `GRAD_CTX=True` + `TRAIN_VISUAL_NET=True` | 上述 + `visual_net` |

梯度探针（单 batch，BTMRI1）：

| 组 | `ctx.grad` | `GCN_fuse.grad` | `visual_net.grad` | optimizer 张量数 |
|---|---|---|---|---|
| orig | **None** | 3.03e-01 | 1.55e-02（有梯度但不更新） | 13 |
| fix_ctx | **4.18e-03** | 3.03e-01 | 1.55e-02 | 13 |
| fix_ctxvis | 5.03e-03 | 2.79e-01 | 1.52e-02 | **17** |

权重差异验证（BTMRI1 seed1 训练完成后，与 orig checkpoint 逐参数比对）：

| 组 | `ctx` maxdiff | `visual_net` maxdiff | `GCN_fuse` maxdiff |
|---|---|---|---|
| fix_ctx | 0.001717 | 0.000000 | 0.005201 |
| fix_ctxvis | 0.001708 | **0.002812** | 0.005343 |

`ctx` 从"完全不变"变成有微小更新（1.7e-3），`visual_net` 在 fix_ctxvis 下确实被更新（2.8e-3）。

## 2. 结果对照（8 数据集 × 3 seeds 均值）

| 数据集 | orig base | fix_ctx base | fix_ctxvis base | orig new | fix_ctx new | fix_ctxvis new |
|---|---|---|---|---|---|---|
| BTMRI1 | 85.87 | 85.70 | 85.74 | 91.61 | 91.61 | 91.61 |
| CHMNIST | 93.44 | 93.44 | 93.48 | 60.11 | 60.15 | 60.33 |
| covid | 78.58 | 78.57 | 78.59 | 91.15 | 91.16 | 91.14 |
| ctkidney | 90.66 | 90.66 | 90.66 | 84.65 | 84.68 | 84.66 |
| lungcolon | 93.64 | 93.68 | 93.65 | 95.47 | 95.63 | 95.60 |
| retina | 89.48 | 89.53 | 89.58 | 58.38 | 58.44 | 58.44 |
| octmnist | 80.73 | 80.74 | 80.98 | 85.88 | 85.88 | 85.87 |
| kvasir | 87.61 | 87.72 | 87.67 | 70.72 | 70.67 | 70.50 |
| **平均** | **87.50** | **87.51** | **87.54** | **79.75** | **79.78** | **79.77** |

差异（相对 orig）：

| 组 | Δ Base | Δ New |
|---|---|---|
| fix_ctx | +0.01 | +0.03 |
| fix_ctxvis | +0.04 | +0.02 |

单数据集最大偏差：base −0.17（BTMRI1 fix_ctx）、new +0.22（CHMNIST fix_ctxvis）。

## 3. 为什么修复几乎不改变结果

三方面原因：

1. **学习率与步数太小**：50 epoch、16-shot、batch 4 = 每 epoch 仅 8 步，共 400 步；
   `ctx` 的梯度量级只有 1e-3，SGD lr=2.5e-3 + cosine 衰减下总位移约 1.7e-3 —— 相对
   `ctx` 自身尺度（std≈0.0176）只有 10%，不足以改变文本分类器方向。
2. **主信号通路不依赖 ctx**：分类 logits = `image_features @ normalize(z_f)`，
   而 `z_f = 0.3*z_text_mean + 0.7*GCN_fuse(imgproto, edge)`。
   `imgproto` 来自 `fc.weight.data`（detach），`edge` 只影响边的权重分配；
   `ctx` 只通过 `edge_tt`（余弦相似度）间接影响融合系数，作用被 `beta=0.3` 的残差进一步稀释。
3. **MedSAM 视觉偏置主导**：`image_features += visual_net(MedSAM_feat)`，
   `image_features` 归一化后直接参与点积，是精度的主要来源。`visual_net` 更新后
   （fix_ctxvis）指标几乎同幅变化，说明它本身也不是瓶颈。

换句话说：**当前方法在 base2new 上的收益，大部分来自"用 MedSAM 特征改写图像表示"，
prompt learning 只是名义上的组件**。这也解释了 MeDKCoOp 与 BiomedCoOp 的差距（约 2 pp）。

## 4. 对论文的影响

| 论文表述 | 实际代码 |
|---|---|
| 3.4 节："only the GCN parameters and the learnable context are updated" | 原版：**只有 GCN 更新**，context 梯度为 None |
| 摘要/方法：learnable prompts 通过 relation learning 动态优化 | prompt 未优化，仅作为图节点输入 |

修复后（`GRAD_CTX=True`）才符合论文描述，但指标不变，说明"prompt 优化"在当前设置下无实质贡献。
三种处理方式：

1. **保持原版**（`GRAD_CTX=False`）：数值与论文完全对齐，但需修正论文措辞；
2. **采用修复版**（`GRAD_CTX=True`）：与论文描述一致，数值几乎不变（Δ≤0.04 pp），风险最低；
3. **增强 prompt 学习**（提高 `N_CTX` 与 lr、或用更多 shot）：需要重新调参，会改变全部数字。

建议采用 **方式 2**：`GRAD_CTX=True` 作为默认，既让代码符合论文叙述，又不破坏已复现的数值。

## 5. 复现命令

```bash
cd /home/wangyijun/MeDKCoOp
export CKPT=/data/wangyijun/ckpt/MeDKCoOp_fixcmp

# 修复 ctx
RUN_TAG="cmp_fix_ctx" TRAINER=MeDKCoOpFix \
  EXTRA_OPTS="TRAINER.GRAPHDCPL.GRAD_CTX True" GPU=8 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1

# 修复 ctx + visual_net
RUN_TAG="cmp_fix_ctxvis" TRAINER=MeDKCoOpFix \
  EXTRA_OPTS="TRAINER.GRAPHDCPL.GRAD_CTX True TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET True" GPU=9 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1

# 汇总（复用 paper_base2new 作为 orig）
python tools/compare_fix.py
```

## 6. 待办

- `GRAD_CTX` 默认值改为 True（`scripts/medkcoop/base2new.sh` 或 `train.py::extend_cfg`）；
- 若采用修复版，需同步更新 `docs/RESULTS.md` 的说明（数值不变，注明实现差异）；
- cross-data / DG 的对照尚未跑（本表只覆盖 base2new）；
- 若要验证"prompt 确实学得动"，需单独做 `N_CTX=16` + `lr=1e-2` 的敏感性实验。
