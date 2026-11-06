# 代码与结构审查报告

> 审查范围：`/home/wangyijun/MeDKCoOp`（当前 HEAD `6912767` 之后的工作区）
> 审查方式：静态检查 + 实际运行验证（base2new 48 个 run 已完成）
> 本文只做诊断与建议，除清理缓存外未修改代码。

## 0. 摘要

项目现在**能稳定复现论文的 base-to-new 主表**：MeDKCoOp 平均 87.50/79.75/82.56（论文 87.88/79.95/82.75），
BiomedCoOp 平均 85.16/75.64/78.17（论文 85.16/75.63/78.16，逐项最大偏差 0.03 pp）。

但作为"可长期维护、可复现全部实验"的工程，还有 5 类明显短板：

1. **cross-data / domain generalization 完全跑不了**：论文最终方法的类 `GRAPH_DCPL_NO_TXT_VIS_GRAPH`
   不在注册表里（只剩字节码），相关脚本全部指向它；这两张表的产物也不在磁盘上。
2. **脚本层大面积失效**：73 处旧路径/旧类名/已删数据集引用（含注释），8 个 baseline 的
   `scripts/` 基本还是原始实验时的写法，硬编码 `/home/wangyijun/ckpt` 与 `Device_id`。
3. **配置与训练器高度重复**：135 个 YAML（同一方法内 4 份内容完全相同）、16 个消融 trainer
   共 1.66 万行（1.2 MB）复制粘贴、15 个数据集类 3450 行近似重复。
4. **复现编排没有进仓库**：48-run 的调度脚本在 `/tmp`，结果汇总脚本把论文表格与路径写死，
   换机器/换数据集无法直接复用。
5. **工程基建缺失**：无 tests、无 CI、无统一 CLI、无环境锁文件；DASSL 里还有 `pdb`/`test1()` 调试代码。

## 1. 现状基线

| 能力 | 状态 | 证据 |
|---|---|---|
| base2new（2 方法 × 8 数据集 × 3 seed） | ✅ 已跑通并对齐 | `docs/RESULTS.md` |
| 离线 BiomedCLIP 加载 | ✅ 可用 | `bootstrap.py` + `HF_HUB_OFFLINE=1` 验证 |
| clip / clip1 合并 | ✅ 位级一致 | 标准路径 maxdiff = 0 |
| cross-data | ❌ 无产物、脚本失效 | 见 A1/A2 |
| domain generalization | ❌ 同上 | 见 A1/A2 |
| few-shot（1/2/4/8/16） | ⚠️ 有脚本，未验证；脚本含错误覆盖 | 见 B1/B12 |
| 消融 Table 4/5 | ⚠️ 代码在，脚本多数失效 | 见 B1/B3 |
| 自动测试 / CI | ❌ 无 | 见 C5 |

规模：552 个跟踪文件，工作树 12 MB；`trainers/medkcoop/ablation` 16 个文件 16,653 行；
`configs` 135 个 YAML；脚本：medkcoop 560 行 + baselines 1,813 行。

---

## 2. 问题总览

| ID | 优先级 | 问题 | 影响 |
|---|---|---|---|
| A1 | P0 | 论文最终方法类缺失，cross-data/DG 脚本依赖它 | 两张主表无法复现 |
| A2 | P0 | 断点/跳过逻辑只看 new 日志 | 会复用"只有 checkpoint、没 base 指标"的残缺结果 |
| A3 | P0 | 复现编排与论文表格写死在 `/tmp` / 代码里 | 结果不可从仓库一键重跑 |
| A4 | P0 | 离线加载是全局 monkeypatch，且只在 `train.py` 生效 | 直接 import trainer 会联网失败 |
| A5 | P0 | DASSL 本地魔改 + 调试代码 | 环境不一致、潜在副作用 |
| B1 | P1 | 73 处旧引用：已删数据集/类、错误相对路径、硬编码 ckpt/GPU | 大部分脚本不可运行 |
| B2 | P1 | 配置重复与 epoch 不一致 | 维护成本高、结果口径不一 |
| B3 | P1 | 16 个消融 trainer 复制粘贴，24 个文件含 `pdb` | 改一处要改 16 处，易漏 |
| B4 | P1 | 15 个数据集类近似重复，5 个有未定义属性 | 修 bug 要改 15 份 |
| B5 | P1 | 类名 `_`→空格 的归一化不统一 | 同一数据集在不同方法的 prompt 文本不同 |
| B6 | P1 | `train.py` CLI 语义 bug（seed=0/device=0 被忽略，`--mode` 混淆） | 隐性配置错误 |
| B7 | P1 | 确定性不足 + 日志轮转导致 0 字节日志 | 复现实验靠运气/补丁 |
| B8 | P1 | eval-only 无法复现训练时的 base 指标 | 误用会得到接近随机的数（实测 53%） |
| B9 | P1 | `clip/checkpoints` 相对路径不存在，PubMed/PMC 权重缺失 | backbone 消融/zero-shot 会失败 |
| B10 | P1 | MaPLe 有两个同名注册类（其一为死文件） | 同时 import 会注册冲突 |
| B11 | P1 | 文档与代码漂移 | 新人按文档会踩坑 |
| B12 | P1 | 环境文件不完整、无锁 | 换机复现困难 |
| B13 | P1 | 许可证/归属不完整 | 发布风险 |
| B14 | P1 | MedSAM/文本图每次重算 | OCT 类评测 20+ 分钟，浪费 GPU |
| C1–C10 | P2 | 缓存、registry 噪声、命名、模板格式、测试/CI、统计口径等 | 整洁度与长期效率 |

---

## 3. P0（阻断复现，建议先做）

### A1. 论文最终方法类缺失，cross-data / DG 无法运行

- 证据：注册表中没有 `GRAPH_DCPL_NO_TXT_VIS_GRAPH`；只有
  `trainers/medkcoop/reference/paper_bytecode/vip_graph_dcpl_biomedclip_adapter_no_text_vis_graph.cpython-310.pyc`。
  `scripts/medkcoop/xd_test.sh:9`、`scripts/medkcoop/domain_genera.sh` 都引用该类；
  `scripts/medkcoop/xd_train.sh` 用的却是 `GRAPH_DCPL_NO_TXT_VIS_GRAPH1`（Table 5 的另一个消融）。
- 影响：论文 Table 2/3 无法产生；现有 cross-data 目录为空。
- 建议：
  1. 从字节码重建 `trainers/medkcoop/final/medkcoop_final.py`，类名保留
     `GRAPH_DCPL_NO_TXT_VIS_GRAPH`（兼容旧 checkpoint 与脚本），实现以
     `reference/medkcoop.py` 为骨架，恢复那 3 行 RNG 代码（已确认它们虽未被使用但影响 shuffle）；
  2. 加一次 `--eval-only` 回归测试：BTMRI1 seed1 应给出 93.5%–93.7%（原始日志 93.5%）；
  3. 重写 `xd_train.sh`/`xd_test.sh`/`cross_data_eval.sh`/`domain_genera.sh`，复用
     `base2new.sh` 的参数化方式；
  4. 用 3 个 seed 重跑 Table 2/3。

### A2. 断点/跳过逻辑不完整

- 证据：`scripts/medkcoop/base2new.sh:59` 只检查 `test_new/log.txt` 里是否有 `* accuracy:`；
  `:67` 只看 checkpoint 是否存在。训练结束时的 base 评测若被中断，脚本会认为"已完成"。
  本次复现确实发生过：5 个 seed 的 `train_base/log.txt` 没有 accuracy，但脚本直接跳过训练。
- 影响：结果表里出现缺项或错误复用；需要人工发现并补跑。
- 建议：在输出目录写 `DONE.json`（含 `train_base_acc`、`test_new_acc`、epoch、耗时、git sha），
  只有两个指标都存在才跳过；任何一步 `rc != 0` 立即非零退出并保留日志。

### A3. 复现编排没有进仓库

- 证据：48-run 的 `/tmp/run_worker.sh`、`/tmp/jobs.txt` 不在仓库；
  `tools/collect_base2new.py:31` 起把论文 Table 1 的数值写死在代码里，路径也默认写死。
- 影响：换机器/换数据集/换方法要重写调度；论文表格与代码耦合，无法审计。
- 建议：
  - `scripts/grid/run_grid.sh`：参数化 `METHODS/DATASETS/SEEDS/GPUS`，带队列、断点、失败重试；
  - `configs/paper/table1.json`：论文数值单独存放；
  - `tools/collect_results.py --tasks base2new,few_shot,cross_data,dg --paper configs/paper/table1.json`。

### A4. 离线加载依赖全局 monkeypatch

- 证据：`bootstrap.py` 在 import 时替换 `open_clip.src.open_clip.create_model_from_pretrained`
  和 `get_tokenizer`；只有 `train.py:5` 会 import 它。直接 `from trainers.medkcoop... import ...`
  或别人写的新脚本不会触发 patch，仍会访问 HuggingFace。
- 影响：离线环境下的"隐式联网失败"；调试时难定位。
- 建议：提供显式 `medkcoop/backbones.py::load_biomedclip(local_dir=None)`，
  trainer 调用它；`bootstrap.py` 只保留 `sys.path` 与路径常量。保留 monkeypatch 作为兼容层，
  并加 `assert os.path.isdir(BIOMEDCLIP_DIR)` 的显式报错。

### A5. DASSL 本地魔改与调试代码

- 证据：`third_party/Dassl.pytorch/dassl/engine/trainer.py:618-624` 的 `test1()` 里
  `print("1111111")` + `pdb.set_trace()`；`:589` 还有注释掉的 pdb；
  `dassl/config/defaults.py` 增加了 `DATASET.LABEL_MIN_MAX` 与整块 `DIFFUSION` 配置（本项目未用）。
- 影响：任何人误调 `test1()` 会卡在断点；升级 Dassl 时 patch 容易丢。
- 建议：删除 `test1()`/pdb；把必要 patch 收敛到 `third_party/patches/dassl.patch`
  并在 `docs/THIRD_PARTY.md` 记录；CI 断言 `dassl.__file__` 指向本仓库。

---

## 4. P1（正确性与可维护性）

### B1. 脚本层 73 处旧引用

| 类型 | 示例 | 说明 |
|---|---|---|
| 已删除的类 | `base2new_abla.sh:8`（`GRAPH_DCPL_b01`）、`base2new_fuse.sh:8`（`b02`）、`base2new_clip.sh:8`（`b07`）、`few_shot.sh:5`（`MeDKCoOp_Time`）、`dcpl/base2new_test_dcpl.sh:7`（`DCPL`）、`graph_adapter/base2new_test_dcpl.sh:5`（`GraphCLIP_v1`） | 运行即报 `KeyError` |
| 已删除的数据集 | `*/domain_genera.sh` 仍调用 `CCBTMv2`/`dermamnistv2`/`kneexrayv2`；`MaPLe/base2new.sh` 含 `BTMRI/CCBTM/dermamnist/kneexray` | 配置不存在 |
| 错误相对路径 | `scripts/baselines/dcpl/domain_genera.sh:4` 调 `scripts/coop/xd_test.sh` | 目录已改名 |
| 硬编码输出 | `xd_train.sh`/`xd_test.sh` 用 `/home/wangyijun/ckpt/...` | 该目录不存在 |
| 硬编码 GPU | 各脚本 `Device_id=5..9` | 多卡调度冲突 |
| 无效超参覆盖 | 31 处 `TRAINER.BIOMEDCOOP.*` 覆盖非 BiomedCoOp 方法 | 静默无效 |

建议：只保留一个通用 runner（`scripts/run.sh`：`task/method/dataset/seed/gpu`），
其它脚本变成 3–5 行的薄封装；删除所有 `Device_id` 与 `/home/.../ckpt`。

### B2. 配置重复与 epoch 不一致

- 135 个 YAML / 624 KB；同一方法内常见 4 份内容完全相同（例如 BiomedCoOp 有 4 个相同 md5）。
- DCPL / GraphCLIP 的 few-shot 配置：`BTMRI1.yaml` 是 50 epoch，其余 14 份是 100 epoch。
- 建议：`configs/methods/<method>.yaml`（唯一超参）+ `configs/datasets/<ds>.yaml`（只写 NAME/ROOT），
  运行时合并；把 epoch 统一到 100（few-shot）并在文档注明。
- 预计可从 135 个文件降到约 25 个。

### B3. 消融 trainer 复制粘贴

- `trainers/medkcoop/ablation/` 16 个文件、16,653 行、1.2 MB；差异通常只有
  `build_subgraph` 里几行开关。
- 24 个 trainer 文件 `import pdb`；还有中文调试 `print`。
- 建议：抽一个 `MeDKCoOpBase`，用 `cfg.TRAINER.GRAPHDCPL.ABLATION`（或构造参数）
  控制 `use_text_graph/use_visual_graph/use_kr_tl/use_medsam/text_knowledge`；
  为 16 个旧类名保留 `@TRAINER_REGISTRY.register()` 的空子类（兼容旧 checkpoint 与论文表格）。
- 验收：同一 seed 下重构前后 `eval-only` 指标完全一致。

### B4. 数据集类近似重复

- 15 个文件、3,450 行；每个类都复制了 `read_split/save_split/split_trainval/subsample_classes`。
- 5 个文件（`BTMRI1`、`kvasir`、`lungcolon`、`covid`、`octmnist`）的 `read_data()` 使用
  未定义的 `self.anno_dir`（当前未被调用，属死代码）。
- few-shot 缓存写入数据集目录：`datasets/<DS>/split_fewshot/`，会污染 `/data` 且未记录版本。
- 建议：`datasets/base.py::SplitJsonDataset` 统一实现；每个数据集只保留
  `dataset_dir/image_dir/split_path/classnames` 四个类属性；缓存目录改为
  `FEWSHOT_CACHE`（默认 `$CKPT/fewshot_cache`）。

### B5. 类名归一化不一致

- MeDKCoOp 系列在 `build_text_graph` 里做 `name.replace("_", " ")`；
  `CoOp/coop_biomedclip.py`、`BiomedCoOp/biomedcoop_biomedclip.py` 等没有统一处理。
- 影响：`glioma_tumor` vs `glioma tumor` 的 token 化不同，prompt 语义与结果会受影响。
- 建议：在数据集层统一输出规范化 `classnames`（或统一提供 `raw_classnames`），
  trainer 不再各自处理。

### B6. `train.py` CLI 语义问题

| 位置 | 问题 | 修复 |
|---|---|---|
| `train.py:111` | `if args.seed:` → `--seed 0` 被忽略 | `if args.seed is not None` |
| `train.py:126` | `if args.device_id:` → `--device-id 0` 被忽略 | 同上 |
| `train.py:135` | `--mode` 同时表示"训练/测试"与模型结构分支，语义混淆 | 拆成 `--split-mode {base,new,all}` 与 `--eval-only` |
| `train.py:136` | `cfg.MODEL.mode` 未在 `extend_cfg` 声明，靠 yacs 动态加键 | 显式声明默认值 |
| `train.py:293` | `--heatmap` 调不存在的 `heatmap_hook()` | 已改为报错；建议直接删掉该参数 |

### B7. 确定性与日志

- `train.py` 设置 `torch.backends.cudnn.benchmark = True` → 同一 seed 不同 GPU/进程会有微小差异。
- `edge_construct` 使用 `random.uniform(0.5, 0.6)` 作为边阈值，且不可配置。
- DASSL `setup_logger` 在重跑时把旧 `log.txt` 改名为 `log.txt-<时间>`；新进程若提前退出，
  主 `log.txt` 会是 0 字节（本次遇到，靠汇总脚本回退才恢复）。
- 建议：`--deterministic`（benchmark=False、固定 worker seed、固定阈值）；
  输出目录加 `RUN_ID`；`DONE.json` 里记录 seed/epoch/git sha/eval batch。

### B8. eval-only 无法复现训练时的 base 指标

- 实测：用原始论文 checkpoint 做 `--mode train --eval-only`（base split），得到 53.65%，
  而训练日志是 80.5%。说明 eval-only 的模型分支（`fc`/`new_fc`）与训练时不同。
- 影响：任何"用已有 checkpoint 复评 base"的脚本都会得到错误结论。
- 建议：在 `docs/ENVIRONMENT.md` 明确写"base 指标只能由训练流程产出或专用 base-eval 流程"；
  若确实需要 eval-only，需要在 `load_model` 后显式恢复 `fc` 并验证。

### B9. `clip/checkpoints` 相对路径与 PubMed/PMC 权重

- `trainers/baselines/Zeroshot/zeroshot.py:28`、`KgCoOp/kgcoop_pubmedclip.py:31`、
  `KgCoOp/kgcoop_pmcclip.py:30` 写死 `directory = "clip/checkpoints"`。
- 合并后该目录不存在（权重在 `/data/wangyijun/model/PubMedCLIP` 与 `model/PubMed`）。
- 建议：统一 `WEIGHTS_DIR` 环境变量，并在 `bootstrap.py` 提供 `get_weights_dir()`；
  zero-shot 的下载逻辑改为"本地优先，缺失时才提示下载"。

### B10. MaPLe 重复注册

- `trainers/baselines/MaPLe/maple.py:213` 与 `maple_biomedclip.py:178` 都注册 `MaPLe_CLIP`；
  当前 `train.py` 只 import 前者，后者是死文件；一旦有人 import 两个就会 `KeyError`。
- 建议：删除 `maple_biomedclip.py`（论文用的是 CLIP 版），或改名 `MaPLe_BiomedCLIP` 并在脚本中显式选择。

### B11. 文档漂移

- `docs/STRUCTURE.md:69` 仍写 `clip1`（已合并）；
- `README.md` "当前状态"仍说"只有 base2new 适配"，但结果已产出；
- `docs/ENVIRONMENT.md:111-112` 的单 seed 结果（88.91/92.20）与 `docs/RESULTS.md` 的 3-seed
  结果（85.87/91.61）并列，容易误读；
- `docs/REPRO_PLAN.md` 是最初的历史规划（已加注，但仍含已删除路径）。
- 建议：以 `RESULTS.md`（结果）、`STRUCTURE.md`（结构）、`ENVIRONMENT.md`（环境）为唯一事实源；
  REPRO_PLAN 移到 `docs/archive/`。

### B12. 环境文件不完整

- `requirements.txt` 缺 `torch/torchvision/dassl/numpy/pillow/tensorboard` 等实际依赖；
  `dcpl.yml` 是全量 `conda env export`，不可读且无锁。
- 没有 `pyproject.toml`/`setup.py`；`dassl` 必须手动 `pip install -e third_party/Dassl.pytorch`。
- 建议：提供 `environment.yml`（conda 依赖）+ `requirements-lock.txt`（pip freeze）；
  `README` 第一屏写清两步安装；CI 用同一文件建环境。

### B13. 许可证与归属

- `LICENSE` 是 Health-X Lab 的 MIT（继承自 BiomedCoOp）。
- 代码同时派生自 CoOp、MaPLe、DCPL、GraphAdapter、Dassl、open_clip、MedSAM、CLIP；
  仓库没有 NOTICE / 第三方许可证清单。
- 建议：保留各上游 LICENSE 于 `third_party/*/LICENSE`，新增 `NOTICE.md`
  列出每个上游与对应许可证；`prompt_templates*.py` 注明由 LLM 生成及使用范围。

### B14. MedSAM / 文本图重复计算

- 每次评测、每个 seed 都把测试集重新过一遍 MedSAM（1024²），OCTMNIST 单次约 20 分钟；
  50 条/类的 LLM 文本也每次都重新编码。
- 建议：按 `(image path, MedSAM ckpt)` 缓存特征到 `$CKPT/feature_cache/`（`.npz`/`.pt`）；
  文本图按 `(dataset, template md5)` 缓存。可把 8 数据集 × 3 seed 的评测从小时级降到分钟级，
  且不改变数值（特征只算一次）。

---

## 5. P2（整洁度与长期效率）

| ID | 问题 | 建议 |
|---|---|---|
| C1 | 43 个 `__pycache__`；`third_party/clip1/__pycache__` 残留 52 KB | 清理并保持 `.gitignore` 生效 |
| C2 | 注册表混入约 20 个 Dassl DA trainer（`ADDA/DANN/...`） | 只 import 本项目 trainer，或在 `train.py` 打印前过滤 |
| C3 | 命名不一致：`GraphCLIP` vs 论文 `GraphAdapter`；`DCPL_biomedclip` 大小写；`TRAINER.GRAPHDCPL` 段名 | 统一命名，旧名保留别名 |
| C4 | `prompt_templates*.py` 共 1.1 MB Python 字面量 | 改为 JSON + lazy load，减少 import 时间/内存 |
| C5 | 无 tests / CI / Makefile | 至少加：import smoke、数据集 split 断言、ckpt 键兼容、1-epoch smoke |
| C6 | 结果只报 mean，无 std / 每 seed 明细 | `collect_results.py` 输出 mean±std、CSV、Markdown |
| C7 | `tools/parse_test_res.py` 与 `collect_base2new.py` 功能重叠 | 合并为一个结果工具 |
| C8 | baseline 脚本 1.8k 行重复 | 统一 runner + 薄封装 |
| C9 | `*v2.py` 把 train/val/test 全塞进 train_x 与 test，语义隐晦 | 改名 `*_eval.py` 或加显式 flag/assert |
| C10 | `reference/` 名称含糊（实际是待重建的参考） | 重建完成后改为 `final/`，`reference/` 保留为对照 |

---

## 6. 结构性重构建议（目标形态）

不必一次性大改，建议按下面的边界收敛：

```
MeDKCoOp/
├── pyproject.toml            # 或 environment.yml + requirements-lock.txt
├── medkcoop/                 # 唯一 Python 包（避免 sys.path 魔法）
│   ├── cli.py                # train / eval / collect 三个子命令
│   ├── backbones.py          # load_biomedclip / load_medsam（显式、可注入路径）
│   ├── data/base.py          # SplitJsonDataset 基类
│   ├── trainers/final.py     # 论文最终方法（从字节码重建）
│   └── trainers/baselines/   # 8 个 baseline
├── configs/
│   ├── methods/<method>.yaml # 唯一超参（epoch/lr/batch/ablation 开关）
│   ├── datasets/<ds>.yaml    # 只有 NAME/ROOT
│   └── paper/table1.json     # 论文数值，供汇总对比
├── scripts/
│   ├── run.sh                # task/method/dataset/seed/gpu
│   └── grid.sh               # 多卡队列 + 断点 + 失败重试
├── tools/results.py          # 统一汇总（mean±std、CSV、Markdown）
├── tests/                    # 4 个 smoke 测试
├── third_party/…/LICENSE     # 保留上游许可证
└── docs/{RESULTS,STRUCTURE,ENVIRONMENT,WEIGHTS,AUDIT}.md
```

- `train.py` 保留为兼容入口（内部调用 `medkcoop.cli`），避免破坏已有脚本。
- 16 个消融类名保留为注册别名，指向同一个 `final.py` 实现。
- 旧 checkpoint 的键（`prompt_learner.*`）保持不变，确保 eval-only 兼容。

## 7. 建议路线图

| 阶段 | 内容 | 预估 | 验收 |
|---|---|---|---|
| S0 快速修复 | 清理缓存/文档漂移；修 `seed=0`/`device=0`；`base2new.sh` 加 DONE 标记；统一 few-shot epoch | 0.5 天 | `--seed 0` 生效；重跑不再跳过残缺结果 |
| S1 主表补齐 | 从字节码重建最终方法类；修 `xd_*`/`domain_genera`；跑 Table 2/3 | 1–2 天 | Table 2/3 有 3-seed 结果并与论文对比 |
| S2 去重 | 配置 135→~25；脚本收敛为 runner；数据集基类；消融单实现 | 2–3 天 | eval-only 指标与重构前逐项一致 |
| S3 工程化 | tests + CI、环境锁、确定性开关、特征缓存、结果 mean±std | 1–2 天 | 新机器按 README 一条命令跑通 smoke |

## 8. 可直接做的低风险修复（按顺序）

1. 删除 `third_party/clip1/`（仅剩 `__pycache__`）与所有 `__pycache__`；提交 `docs/WEIGHTS.md` 的空白改动。
2. `train.py:111/126` 改为 `is not None`。
3. `scripts/medkcoop/base2new.sh`：加 `DONE.json`（base+new 都有才算完成）。
4. 删除 `scripts/medkcoop/base2new_abla.sh|base2new_fuse.sh|base2new_clip.sh|few_shot*.sh` 里已删类，或改为调用通用 runner。
5. 删除 `trainers/baselines/MaPLe/maple_biomedclip.py`（死文件）。
6. 统一 `configs/trainers/DCPL|GraphCLIP/few_shot/BTMRI1.yaml` 的 epoch 为 100。
7. 更新 `docs/STRUCTURE.md:69` 与 `README` 状态段，`ENVIRONMENT.md` 结果段改为指向 `RESULTS.md`。
8. 删除 Dassl `test1()` 与 pdb 残留。

## 附录 A：cross-data / DG 相关失效点

| 文件 | 问题 |
|---|---|
| `scripts/medkcoop/xd_test.sh:9` | `TRAINER=GRAPH_DCPL_NO_TXT_VIS_GRAPH`（类不存在） |
| `scripts/medkcoop/xd_train.sh:8` | 用 `..._GRAPH1`（Table 5 的消融，不是最终方法） |
| `scripts/medkcoop/xd_test.sh:19` | `--model-dir /home/wangyijun/ckpt/...`（不存在） |
| `scripts/medkcoop/domain_genera.sh` | 目标列表含已删数据集（CCBTMv2/dermamnistv2/kneexrayv2） |
| `scripts/medkcoop/cross_data_eval.sh` | 同上（部分被注释） |
| `scripts/baselines/*/xd_*.sh` | 同类问题 + 目录名漂移（`scripts/coop` 等） |

## 附录 B：验证命令

```bash
cd /home/wangyijun/MeDKCoOp
conda activate medkcoop

# 当前可复现的主表
CONFIG_DIR=$PWD/configs/trainers/medkcoop TRAINER=MeDKCoOp GPU=8 \
  bash scripts/medkcoop/base2new.sh BTMRI1 1
python tools/collect_base2new.py --root /data/wangyijun/ckpt/paper_base2new
```
