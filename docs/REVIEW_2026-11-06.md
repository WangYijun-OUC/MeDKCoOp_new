# Codex 改动审查（Round 2）

> 审查对象：`origin/main` @ `10ce304`（codex 的 8 个提交，相对我上一轮 `6a216d7`）
> 方法：静态检查（bash -n / compileall / 注册表解析）+ 实际运行（4 次 probe）
> 结论：**方向正确、质量明显提升**，但有 3 个必须修的实质问题，其中 1 个影响论文可信度。

## 0. 结论摘要

codex 的这一轮做对了很多事：修了 `seed=0`/`device=0` 的 CLI bug、补了 `DONE` 式的断点校验、
把 73 处旧引用清到 0、删掉了死文件 `maple_biomedclip.py`、统一了 PubMed/PMC 本地权重路径、
加了 `--deterministic`、加了 `environment.yml` / `NOTICE.md` / `VALIDATION.md`。

但有三处需要立刻处理：

| # | 严重度 | 问题 | 证据 |
|---|---|---|---|
| **R1** | **P0 可信度** | 论文实现里 `ctx`、`meta_net`、`fc`、`visual_net` **根本没有被优化**，只有 `GCN_fuse` 在训练 | 2-epoch probe 与 50-epoch 论文 checkpoint 在这四个参数上**逐位相同**；梯度探针显示 `ctx.grad=None` |
| **R2** | P1 正确性 | `has_accuracy` 的 grep 模式写成 `'\\* accuracy:'`（双反斜杠），靠巧合匹配 | `od -c` 显示 8 个 baseline 脚本都是 `\ \ *` |
| **R3** | P1 一致性 | 三个方法入口的默认 trainer 不一致：`base2new.sh` 用 `MeDKCoOp`，`few_shot.sh`/`xd_*.sh` 用 `GRAPH_DCPL_NO_TXT_VIS_GRAPH` | 两个同名实现写进不同目录，结果无法横向比较 |

另外 cross-data 的 23.20% 与论文 32.60% 差 9.4 pp，我做了两个对照实验定位原因，见 R4。

---

## 1. 做得好的部分（建议保留）

| 改动 | 评价 |
|---|---|
| `train.py` `--seed`/`--device-id` 改 `is not None` | 修好了 `seed=0` 被忽略的 bug |
| `cfg.MODEL.mode = "train"` 显式声明 | 去掉了 yacs 动态加键的隐患 |
| `--deterministic` + `DETERMINISTIC` 环境变量 | 补齐了 AUDIT 里的 B7 |
| `base2new.sh` 断点：checkpoint + base 指标 + new 指标三者齐全才跳过 | 修好了我上一轮踩到的"残缺结果被跳过" |
| `GRAPH_DCPL_NO_TXT_VIS_GRAPH(MeDKCoOp)` 别名 | 旧脚本/旧 checkpoint 兼容，方案正确 |
| `TRANSFER_EVAL`：cross-data 丢弃源域 `fc`/`new_fc` | 语义正确（源类分类头不能跨标签迁移） |
| `FEWSHOT_CACHE` 环境变量 | 不再污染 `/data/wangyijun/dataset` |
| 删除 `maple_biomedclip.py` | 消除了 `MaPLe_CLIP` 重复注册 |
| `PMCCLIP_DIR` / `PMCCLIP_BERT_DIR` + `local_files_only` | 修好了 `clip/checkpoints` 失效路径 |
| `NOTICE.md` / `environment.yml` | 补齐 AUDIT 的 B12/B13 |
| 73 处旧引用清零、`bash -n` + `compileall` 全过 | 脚本层可运行性大幅提升 |

实测：`scripts/baselines/coop/base2new.sh BTMRI1 1` 跑通，base 87.63 / new 96.10，
与论文 CoOp 的 89.80/53.70 口径不同（见 R5），但脚本链路本身是通的。

---

## 2. R1（P0）：论文实现只训练了 `GCN_fuse`

### 2.1 证据

**实验一：2-epoch 探针 vs 论文 50-epoch checkpoint**

```
prompt_learner.ctx                    maxdiff_2ep_vs_50ep = 0.000000
visual_net.linear1.weight             maxdiff_2ep_vs_50ep = 0.000000
prompt_learner.fc.weight              maxdiff_2ep_vs_50ep = 0.000000
prompt_learner.GCN_fuse.gcn_weights   maxdiff_2ep_vs_50ep = 0.109600   ← 唯一变化
```

逐参数比对 2ep/50ep 的完整 state_dict，**只有 `GCN_fuse.gcn_weights` 与 `gcn_bias` 不同**。

**实验二：梯度探针（单 batch backward）**

| 参数 | requires_grad | grad |
|---|---|---|
| `prompt_learner.ctx` | True | **None** |
| `prompt_learner.meta_net.linear1.weight` | True | **None** |
| `prompt_learner.GCN_fuse.gcn_weights` | True | 3.077e-01 |
| `prompt_learner.GCN_tt.gcn_weights` | True | None |
| `prompt_learner.fc.weight` | True | None |
| `visual_net.linear1.weight` | True | 1.559e-02（有梯度，但**不在 optimizer 里**） |

**实验三：跨数据集比对（同一 seed 1）**

```
prompt_learner.ctx                covid == octmnist : True
visual_net.linear1.weight         covid == octmnist : True
prompt_learner.meta_net.linear2   covid == octmnist : True
prompt_learner.GCN_fuse.gcn_weights  covid == octmnist : False
prompt_learner.fc.weight          covid == octmnist : False
```

即 ctx / meta_net / visual_net 在不同数据集上完全相同——它们只是初始化后被原样存下。

### 2.2 原因（代码层面）

`trainers/medkcoop/reference/medkcoop.py` 里有一条被 `no_grad` 切断的路径：

```python
# build_prompt_graph, line ~442-460
if self.mode == "test":
    imgproto_features = F.normalize(self.new_fc.weight.data[...])   # .data → detach
else:
    imgproto_features = F.normalize(self.fc.weight.data[...])       # .data → detach

with torch.no_grad():                       # ← 切断 ctx → text_features → prompts
    inputs_text = prompts.unsqueeze(dim=1)
    ...
    edge_tt = edge_construct(feat_tt)
img_gfeature = self.relu(self.GCN_fuse(imgproto_features.transpose(0, 1), edge_tt))
```

后果：
1. `ctx` 产生的 `prompts` 被 `no_grad` 包住 → `ctx.grad` 为 None；
2. `fc.weight.data` 是 detach 的 → `fc`/`visual_net` 的梯度不会回传；
3. `GCN_tt`/`GCN_it` 在 `reference/medkcoop.py` 里已不参与前向（只剩死参数）；
4. optimizer 只给了 `self.model.prompt_learner`（`line 639`），`visual_net` 即使有梯度也永远不更新。

所以"MeDKCoOp"实际是一个 **bi-level：只训练一个 512×512 的 GCN 融合层**。

### 2.3 影响

- 论文声称"only the GCN parameters and the learnable context are updated"（正文 3.4 节），
  代码里 context 实际没更新 → **论文描述与代码不一致**；
- 这解释了为什么 MeDKCoOp 的 base2new 数字与 BiomedCoOp 只差 2–3 pp：
  learnable prompt 没训练，增益全部来自 GCN 重加权 + MedSAM bias；
- 不改变"能复现"这个事实（checkpoint 就是这么训出来的），但**方法叙述需要修正**，
  或者修好梯度让 prompt 真正可训练（那样论文数字会变）。

### 2.4 建议

1. **先不要改实现**。把它作为事实记录到 `docs/RESULTS.md` 与 `docs/ARCHITECTURE.md`：
   "MeDKCoOp 的有效可训练参数是 `GCN_fuse`（512×512 + bias）；`ctx`/`meta_net`/`fc`/`visual_net`
   在前向中被 detach/未进 optimizer"。
2. 如果目标是让论文方法名副其实，需要：
   - `build_prompt_graph` 去掉 `no_grad`（让 `ctx` 参与梯度）；
   - `fc` 不再用 `.data`，或改为 register_buffer；
   - 把 `visual_net` 加入 optimizer（`build_optimizer([prompt_learner, visual_net])`）；
   - 重跑 base2new 并与论文数值对照。
3. 两种口径都要在文档里分开标注，避免"论文数字 vs 修好后数字"混淆。

---

## 3. R2（P1）：`has_accuracy` 的 grep 模式错误

8 个 baseline 脚本与 1 个 `zeroshot` 脚本写的是：

```bash
has_accuracy() { [[ -s "$1" ]] && grep -q '\\* accuracy:' "$1"; }
```

`od -c` 确认字节是 `\ \ *`（两个反斜杠）。在 BRE 里 `\\*` 表示"零个或多个反斜杠"，
因此该模式实际等价于 `* accuracy:` 中的 `*` 前可有可无反斜杠——**靠巧合匹配**。
`scripts/medkcoop/base2new.sh` 写的是单反斜杠 `'\* accuracy:'`，是正确的。

风险：如果有人把 `* accuracy:` 改成别的写法，或日志里出现 `\* accuracy:` 之外的行，
判定就会失效；语义上也不可维护。

修复（一条命令）：

```bash
sed -i "s/grep -q '\\\\\\\\\\\\* accuracy:'/grep -q '\\\\* accuracy:'/" scripts/baselines/*/*.sh
```

然后 `bash -n` 全部复检。

---

## 4. R3（P1）：三个入口的默认 trainer 不一致

| 脚本 | 默认 `TRAINER` |
|---|---|
| `scripts/medkcoop/base2new.sh` | `MeDKCoOp` |
| `scripts/medkcoop/few_shot.sh` | `GRAPH_DCPL_NO_TXT_VIS_GRAPH` |
| `scripts/medkcoop/xd_train.sh` / `xd_test.sh` | `GRAPH_DCPL_NO_TXT_VIS_GRAPH` |

现在仓库里有**两个类对应同一个实现**：
- `trainers/medkcoop/reference/medkcoop.py::MeDKCoOp`
- `trainers/medkcoop/reference/medkcoop.py::GRAPH_DCPL_NO_TXT_VIS_GRAPH(MeDKCoOp)`

后果：
1. 同一方法在 `base2new` 下产物落在 `.../MeDKCoOp/...`，在 `few_shot`/`cross_data` 下落
   在 `.../GRAPH_DCPL_NO_TXT_VIS_GRAPH/...`，汇总工具无法按同一 key 聚合；
2. 论文日志用的是 `GRAPH_DCPL_NO_TXT_VIS_GRAPH`，而 base2new 复现用 `MeDKCoOp`——
   数值上等价（同名子类），但对比表格时会混淆。

建议：统一默认到 `GRAPH_DCPL_NO_TXT_VIS_GRAPH`（论文名），`MeDKCoOp` 作为别名保留；
或反过来，但必须一处固定，并在 `docs/STRUCTURE.md` 写明。

---

## 5. R4（P1）：cross-data 23.20% 与论文 32.60% 的差距

codex 跑出的 CHMNIST 23.20%，论文 Table 2 是 32.60%。我做了两个对照：

| 实验 | 设置 | 结果 |
|---|---|---|
| 复现 codex | `NUM_SHOTS=16, SUBSAMPLE=all, TRANSFER_EVAL=True` | **23.20%** |
| 去掉 TRANSFER_EVAL | 同上，不丢 `fc`/`new_fc` | **23.20%**（完全相同） |
| 加大目标 shots | `NUM_SHOTS=200`（train_x 1,600） | **24.14%** |

三个实验都在 23–24%，离 32.60% 很远。原因分析：

1. **目标域原型来自目标数据**：`test` 模式下 `new_fc.weight` 由 `build_image_graph(..., train_loader_x)`
   用目标集的 16-shot 特征算类均值。CHMNIST 有 8 类，16-shot/类 → 128 张图，
   类均值噪声很大，直接把分类器原型带偏。
2. **源域权重只贡献 `GCN_fuse` + `ctx`**（见 R1，`ctx` 未训练）→ 可迁移的信息极少，
   基本等于"用目标域 128 张图做原型分类"。
3. 论文 32.60% 大概率是另一套流程（例如 target 原型用更多样本，或源域用全量训练而非 16-shot）。

结论：**23.20% 是当前实现的真实行为，不是脚本 bug**；但它与论文口径不一致，
VALIDATION.md 的"权重链路可工作"结论是准确的，而"能替代 Table 2"则不能。

建议：
- 在 `VALIDATION.md` 补一句：23.20% 与论文 32.60% 的差距来自目标原型样本量（16-shot/类）；
- 若要贴近论文，先确认 `_get_base_image_features` 在 cross-data 时应使用目标集多大的子集；
- 在 `docs/ARCHITECTURE.md` 把"目标原型来源"写清楚，避免后续误判为 bug。

---

## 6. R5（P2）：baseline 的 base2new 与论文口径不一致

实测 `coop/base2new.sh BTMRI1 1` → base 87.63 / new 96.10；论文 CoOp 是 89.80 / 53.70。

- new 高 42 pp，说明 **new 类评测没有用 base 训练得到的 prompt**，而是重新初始化后才评测；
  或者 `SUBSAMPLE_CLASSES=new` 的路径与训练时类顺序不一致；
- CoOp 的 `vit_b16_base2new.yaml` 里 `MAX_EPOCH: 50`，脚本也传 `OPTIM.MAX_EPOCH 50`，这没问题；
- MaPLe 的 `vit_b16_c2_ep5_batch4_2ctx.yaml` 是 **5 epoch**（原版 MaPLe base2new 配置），
  脚本 `EPOCH=5` 与之匹配，但论文 MaPLe 是 5 epoch 吗？需要核对；
- 各 baseline `EPOCH` 默认 50（base2new）/100（few_shot），与各自配置一致，但 **MaPLe 是 5**，
  这是唯一一个与其他方法不同量级的，建议单独标注。

建议：先跑 CoOp 的 3 seeds，与论文 Table 1 的 CoOp 行（base 89.80 / new 53.70 / HM 67.21）
逐项对比；若 new 仍在 90+，说明 base2new 的 new 评测路径需要修（和 R8 同源）。

---

## 7. 其他仍需处理（延续 AUDIT）

| ID | 问题 | 状态 |
|---|---|---|
| A2 | 断点逻辑 | ✅ codex 已修（checkpoint + base + new 三者齐全） |
| A3 | 编排未进仓库 | ⚠️ 部分：`scripts/run_all_methods.sh` 只串行调用，无队列/重试/多卡 |
| A4 | 离线 monkeypatch 隐式 | ⚠️ 仍在 `bootstrap.py`，未改成显式 `load_biomedclip()` |
| A5 | DASSL 调试代码 | ✅ 已删除 `test1()`/pdb |
| B1 | 73 处旧引用 | ✅ 已清零（实测 0） |
| B2 | 配置重复 135 个 | ❌ 未动 |
| B3 | 16 个消融复制粘贴 | ❌ 未动 |
| B4 | 15 个数据集类重复 | ⚠️ 只加了 `FEWSHOT_CACHE`（且 `BTMRI1.py` 漏改），未抽基类 |
| B6 | CLI 语义 | ✅ seed/device 已修；`--mode` 仍双语义 |
| B7 | 确定性 | ✅ 已加 `--deterministic` |
| B8 | eval-only base 指标 | ⚠️ 仍无文档说明 |
| B10 | MaPLe 重复注册 | ✅ 已删死文件 |
| B11 | 文档漂移 | ⚠️ `STRUCTURE.md:69` 仍写 `clip1`；REPRO_PLAN 仍含旧路径 |
| B12 | 环境锁 | ⚠️ 有 `environment.yml`，无 lock |
| B13 | 许可证 | ✅ 新增 `NOTICE.md` |
| B14 | 特征缓存 | ⚠️ 只有 few-shot split 缓存，无 MedSAM/文本图特征缓存 |
| C5 | tests/CI | ❌ 仍无 |

### 7.1 `BTMRI1.py` 漏改 `FEWSHOT_CACHE`

其余 7 个数据集都加了 `FEWSHOT_CACHE`，`datasets/BTMRI1.py:34` 仍是
`self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")`。
BTMRI1 是 cross-data / DG 的源域，这条遗漏会让源域缓存继续写进 `/data/wangyijun/dataset`。

### 7.2 `collect_base2new.py` 与 `collect_results.py` 并存且口径不同

- `collect_base2new.py`：写死 2 个方法 + 论文 Table 1 数值 + 固定 `vit_b16_0002_b50_randflip` 路径层；
- `collect_results.py`：通用 rglob，按 `seed(\d+)` 分组，但**要求路径里有 `seed1` 这类目录**，
  而 baseline 脚本的产物路径是 `.../$TRAINER/seed1/`（有 seed 层），medkcoop 是
  `.../$TRAINER/$CFG/seed1/`（多一层）。两者都能被 `rglob` 处理，但 key 粒度不同。

建议：保留 `collect_results.py` 作为唯一通用工具，`collect_base2new.py` 改为只做
"论文表格对比"（读 `configs/paper/table1.json`，不写死数值）。

---

## 8. 新增的结构性风险（本轮仍未解决）

### 8.1 `cudnn.benchmark` 在评测阶段的差异

`--deterministic` 只影响训练。base2new 的 new 评测仍是 `benchmark=True`（默认），
同一 checkpoint 在不同卡上可能得到 ±0.1 pp 的结果。若要严格复评，评测也应带 `--deterministic`。

### 8.2 目标域原型样本量未参数化

`build_image_graph` 用的 `train_loader_x` 完全由 `DATASET.NUM_SHOTS` 控制；
cross-data / DG 场景下这个值直接决定原型质量，但脚本没有暴露为独立参数
（例如 `TARGET_SHOTS`），只能复用 `SHOTS`，语义容易被误解。

### 8.3 `scripts/run_all_methods.sh` 的 zeroshot 分支

Zeroshot 没有 `xd_train.sh`，`run_all_methods.sh` 的 `cross_data` 任务会调用
`scripts/baselines/zeroshot/cross_data_eval.sh`（存在，只做评测），这一步是对的；
但 `base2new` 任务对 Zeroshot 是"直接评测两个 split"，不产生 checkpoint，
与其它方法的行为不同，需在 README 标注。

---

## 9. 建议的处理顺序

| 阶段 | 动作 | 预估 |
|---|---|---|
| **立即** | 修 R2 的 grep 模式；统一 R3 的默认 trainer；补 `BTMRI1.py` 的 `FEWSHOT_CACHE` | 0.5 h |
| **立即** | 把 R1 写成 `docs/ARCHITECTURE.md` 的事实说明（不改代码） | 1 h |
| 短期 | 跑 CoOp 3 seeds 对比论文，定位 R5；若 new 异常，修 base2new new 路径 | 3 h GPU |
| 短期 | 确认 cross-data 目标原型口径（R4），必要时参数化 `TARGET_SHOTS` | 2 h |
| 中期 | 配置去重（B2）、消融单实现（B3）、数据集基类（B4） | 2–3 天 |
| 中期 | MedSAM/文本图特征缓存（B14） | 1 天 |
| 长期 | tests + CI（C5）、`load_biomedclip()` 显式化（A4） | 1–2 天 |

## 10. 验收清单

```bash
cd /home/wangyijun/MeDKCoOp
# 1) 脚本语法与 Python 编译
for f in $(git ls-files '*.sh'); do bash -n "$f" || echo "FAIL $f"; done
python -m compileall -q . -x third_party

# 2) grep 模式（应为单反斜杠）
grep -c "grep -q '\\\\* accuracy:'" scripts/baselines/*/*.sh   # 期望 0
grep -c "grep -q '\* accuracy:'" scripts/baselines/*/*.sh      # 期望 1

# 3) 默认 trainer 一致
grep -h 'TRAINER="${TRAINER:-' scripts/medkcoop/*.sh | sort -u

# 4) 无未注册引用
python - <<'EOF'
import os,re,glob,sys
sys.path.insert(0,os.getcwd()); import bootstrap,train
from dassl.engine import TRAINER_REGISTRY
reg=set(TRAINER_REGISTRY._obj_map)
for sh in glob.glob('scripts/**/*.sh', recursive=True):
    for m in re.finditer(r'TRAINER="\$\{TRAINER:-([^}]+)\}"', open(sh).read()):
        if m.group(1) not in reg: print('BAD', sh, m.group(1))
EOF

# 5) R1 回归：2 epoch 与 50 epoch 的 ctx 应仍然相同（当前事实）
#    若未来修好梯度，此处应变为不同
```
