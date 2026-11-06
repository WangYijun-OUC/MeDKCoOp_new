# MeDKCoOp 代码修复记录

> 对应审查报告：[AUDIT.md](AUDIT.md)、[REVIEW_2026-11-06.md](REVIEW_2026-11-06.md)
> 原则：**原有实现保留不动**，修复版放进 `trainers/medkcoop/fixed/`，用开关逐项对照。

## 0. 一句话结论

原实现里 **只有 `GCN_fuse` 一个层在训练**：`ctx`（learnable context）因为 `torch.no_grad()`
被彻底切断梯度，`visual_net` 有梯度却不在 optimizer 里，`fc`/`GCN_tt`/`GCN_it` 属死参数。
这与论文 3.4 节"only the GCN parameters and the learnable context are updated"不符。

## 1. 代码布局

```
trainers/medkcoop/
├── reference/medkcoop.py    # 原实现，保持不动（对照组）
└── fixed/medkcoop.py        # 修复版，带开关（实验组）
```

| 注册类名 | 来源 | 用途 |
|---|---|---|
| `GRAPH_DCPL_NO_TXT_VIS_GRAPH` | reference | 论文原始行为（对照基线） |
| `MeDKCoOp` | reference | 同上，别名 |
| `MeDKCoOpFix` | fixed | 修复版主类 |
| `MeDKCoOpFixedImpl` | fixed | 修复版实现体 |
| `GRAPH_DCPL_NO_TXT_VIS_GRAPH_FIX` | fixed | 修复版的论文类名别名 |

## 2. 逐项修复

### 2.1 `ctx` 梯度被 `no_grad` 切断（核心问题）

**位置**：`MultiModalPromptLearner.build_prompt_graph`

**原代码**：

```python
with torch.no_grad():                       # ← 整块切断
    inputs_text = prompts.unsqueeze(dim=1)
    node_cluster_t = text_features.view(...)
    feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1).transpose(1, 2)
    edge_tt = edge_construct(feat_tt)
img_gfeature = self.relu(self.GCN_fuse(imgproto_features.transpose(0, 1), edge_tt))
```

**问题**：`prompts` 由 `ctx + meta_net 偏移` 得到，被包在 `no_grad` 里，
所以 `ctx` 与 `meta_net` 的梯度恒为 `None`，永远不会更新。

**修复**：把梯度开关交给 `TRAINER.GRAPHDCPL.GRAD_CTX`：

```python
_grad_ctx = getattr(self, "grad_ctx", True)
_ctx_scope = nullcontext() if _grad_ctx else torch.no_grad()
with _ctx_scope:
    ...
```

**验证**（单 batch backward，BTMRI1）：

| 配置 | `ctx.grad` | `GCN_fuse.grad` |
|---|---|---|
| `GRAD_CTX=False`（原版） | **None** | 3.03e-01 |
| `GRAD_CTX=True`（修复） | **4.18e-03** | 3.03e-01 |

### 2.2 `visual_net` 有梯度但不在 optimizer

**位置**：`MeDKCoOp.build_model`

**原代码**：

```python
if "visual_net" in name:
    param.requires_grad_(True)          # 允许梯度
...
self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)   # 但没有 visual_net
```

**问题**：`visual_net`（256→16→512 的 MLP，把 MedSAM 特征映射成图像 bias）每步都算梯度，
却从不被 optimizer 更新，纯属浪费算力。

**修复**：新增开关 `TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET`（默认 False = 原行为）：

```python
if _train_visual:
    self.optim = build_optimizer(
        [{"params": self.model.prompt_learner.parameters()},
         {"params": self.model.visual_net.parameters()}],
        cfg.OPTIM,
    )
else:
    self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
```

**验证**：optimizer 参数张量数 13 → 17（增加 visual_net 的 4 个）。

### 2.3 `fc.weight.data` 阻断了原型分支的梯度

**位置**：`build_prompt_graph`

**原代码**：`F.normalize(self.fc.weight.data[...])` —— `.data` 等价于 detach。

**影响**：`fc`（图像类原型）不参与梯度；`GCN_fuse` 只通过 `edge_tt` 拿到梯度。
这条**没有改**，因为：
- `fc` 的原型来自 `build_image_graph()` 的类均值，本来就不是"训练目标"；
- 若要让它可训练，等于改变了方法设计，超出"修复 bug"的范围。
只在此记录事实，留待你决定。

### 2.4 `GCN_tt` / `GCN_it` / `scale_mm` 是死参数

**证据**：前向里只有 `GCN_fuse` 被调用；`GCN_tt`/`GCN_it` 相关代码已被注释，`scale_mm` 只在注释中出现。
它们仍被 `register` 进 checkpoint，`requires_grad=True` 却永远拿不到梯度（已用探针确认 grad=None）。

**处理**：未删除（删了会破坏旧 checkpoint 兼容）。建议后续在文档/代码注释里标明"保留仅为 checkpoint 兼容"。

### 2.5 `alpha_it` 图像-文本融合被移除

**历史**：原 `trainers/MeDKCoOp/medkcoop.py` 有
`graph_o_t = base_img_features * alpha_it + (1-alpha_it) * fuse_features`；
`trainers/GRAPH_DCPL/medkcoop.py`（论文数字实际来源）已把它注释掉。

**结论**：论文数字对应**没有**这一项的版本。当前 `reference/` 与 `fixed/` 都保持注释状态，
与论文一致；原 `trainers/MeDKCoOp/medkcoop.py` 留在备份中作为历史。

### 2.6 其他已修复的工程问题

| 问题 | 修复 |
|---|---|
| MedSAM 权重路径是占位符 `/path/to/your/medsam.pth` | 改为 `MEDSAM_CKPT` 环境变量 |
| MedSAM 1024² 前向在 eval batch 16 时 OOM | 每 4 张分块，数值等价 |
| 缺 3 行 RNG 消耗代码（影响 shuffle 顺序） | 按字节码恢复 `mean`/`noise` |
| `reset_parameters` 名字被 GCN 复用 | 改名 `reset_parameters_txt` |

## 3. 怎么跑对照

```bash
cd /home/wangyijun/MeDKCoOp
export CKPT=/data/wangyijun/ckpt/MeDKCoOp_fixcmp

# 原版（对照组）
RUN_TAG="cmp_orig" TRAINER=GRAPH_DCPL_NO_TXT_VIS_GRAPH GPU=8 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1

# 修复 ctx
RUN_TAG="cmp_fix_ctx" TRAINER=MeDKCoOpFix \
  EXTRA_OPTS="TRAINER.GRAPHDCPL.GRAD_CTX True" GPU=9 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1

# 修复 ctx + visual_net
RUN_TAG="cmp_fix_ctxvis" TRAINER=MeDKCoOpFix \
  EXTRA_OPTS="TRAINER.GRAPHDCPL.GRAD_CTX True TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET True" GPU=5 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1
```

统一条件：BiomedCLIP ViT-B/16 backbone、16-shot、50 epoch、batch 4、SGD lr=2.5e-3、seeds 1/2/3。
输出：`$CKPT/base2new/cmp_<tag>/<数据集>/shots_16/<trainer>/<cfg>/seed<k>/`。

汇总：

```bash
python tools/collect_results.py /data/wangyijun/ckpt/MeDKCoOp_fixcmp --csv /tmp/cmp.csv
```

## 4. 对照结果

见 [RESULTS_FIX.md](RESULTS_FIX.md)（跑完后填写）。

## 5. 尚未处理（需要你决策）

1. **`fc` 是否加入优化器**：会改变原型分支的学习方式，影响所有数字。
2. **死参数是否清理**：`GCN_tt`/`GCN_it`/`scale_mm` 保留兼容，还是删掉简化。
3. **`MODEL.mode` 双语义**：目前 `--mode train|test` 同时决定"训练/评测"和"fc/new_fc 分支"，
   跨数据集评测依赖这个分支来重建目标原型。若拆开需要改 `load_model` 与数据集逻辑。
4. **论文叙述**：若保持 `GRAD_CTX=False`（原行为），论文 3.4 节应改为
   "only the GCN fusion layer is updated"；若采用修复版，需重跑并更新全部数值。
