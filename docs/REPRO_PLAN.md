# BiomedCoOp / MeDKCoOp 项目备份与整理复现方案

> 版本：v1（2026-11-05）
> 状态：**只读评审稿。本次只新增本文件，未修改、未移动、未删除任何既有文件。**
>
> 注：本文件是最初的审查与复现规划。仓库实际结构已按"只保留论文用到的文件"定稿，
> 以 [STRUCTURE.md](STRUCTURE.md) 为准；文中提到的 `legacy/` 目录已不再存在。
> 依据：`/home/wangyijun/BiomedCoOp` 全量代码、`/data/wangyijun/dataset`、`/data/wangyijun/ckpt`、
> 论文 *MeDKCoOp: Dual Knowledge-guided Graph Prompt Learning for Biomedical Vision-Language Models* (ACM MM'25)。

---

## 0. 结论速览

1. **论文最终方法在代码里"丢了一半"**：论文 "MeDKCoOp (Ours)" 的全部数字（Avg Base 87.88 / New 79.95 / HM 82.87，以及 8 个数据集的每一格）都精确对应 trainer 类 `GRAPH_DCPL_NO_TXT_VIS_GRAPH`，但该类唯一的源文件
   `trainers/GRAPH_DCPL/vip_graph_dcpl_biomedclip_adapter_no_text_vis_graph.py` **已经不存在**，只剩 `__pycache__` 里的 3.10 字节码；git 历史、`trainers copy/` 里都没有。**这是本方案的第一优先级**。
2. `trainers/MeDKCoOp/medkcoop.py` **不是**论文版本：它还保留着 `alpha_it=0.7` 的图像特征融合（论文最终版已去掉），MedSAM 权重路径是占位符 `/path/to/your/medsam.pth`，且 `train.py` 里对它的 import 是注释掉的——当前仓库按 `--trainer MeDKCoOp` 实际加载的是 `trainers/GRAPH_DCPL/medkcoop.py`。你的担心成立。
3. **base-to-new 实验产物完整**：`/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/{train_base,test_new}/<数据集>/shots_16/<变体>/<配置>/seed{1,2,3}/`，2060 份 `log.txt`、1088 个模型（共 939 GB）。论文 Table 1 与 Table 4/5 的 "Ours" 行可从日志逐格复算。
4. **cross-dataset（Table 2）和 domain generalization（Table 3）的结果在磁盘上完全不存在**。相关脚本指向 `/home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/...`，该目录不存在；`/data/wangyijun/ckpt` 下也没有任何 `CrossData`/`cross`/`domain` 目录。这两张表（含 6 个 baseline）必须重跑。
5. **环境可用但离线不可跑**：`conda dcpl`（Python 3.10.16 / torch 1.12.0+cu113）在 9×RTX 4090 上 CUDA 运算正常；但代码用 `create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-...')`，HF 缓存为空且当前无外网 → 直接报 `LocalEntryNotFoundError`。本地已有完整 BiomedCLIP 权重，整理版必须加"本地权重加载"适配。

---

## 1. 现状盘点

### 1.1 目录地图（`/home/wangyijun/BiomedCoOp`，含 .git 177 MB，不含 .git 52 MB）

| 路径 | 体积 | 作用 | 处置建议 |
|---|---|---|---|
| `train.py` | 14 KB | **论文主入口**（DASSL 框架）：import 数据/训练器触发注册，`--mode train/test`、`--eval-only` | 保留并重构 |
| `main.py` | 11 KB | 另一条独立 few-shot 管线（`configs/base.yaml` + `datasets/*.yaml`），只服务 `trainers/__init__.py` 里的 TipAdapter/ClipAdapter/LP/LP2 | 移入 `tools/legacy/`，论文主体不用 |
| `utils.py` | 8 KB | 文本分类器、特征预载等工具；`import clip` | 保留 |
| `datasets/` | 496 KB | 23 个数据集类：v1 11 个（`split json → few-shot → subsample base/new`）+ v2 12 个（整份 json 直接作为 train_x 与 test，不做 few-shot/subsample） | 只保留论文用到的 8+8 个 |
| `trainers/` | 7.3 MB | 16 个方法目录 + `prompt_templates*.py`（4 份，72 类 × 50 条 LLM 提示）；`__init__.py` 只注册 TipAdapter/ClipAdapter/LP/LP2 | 重组为 `medkcoop/ + baselines/ + legacy/` |
| `trainers copy/` | 7.3 MB | 与 `trainers/` 仅差一个 `prompt_templates1.py`（`diff -rq` 只有 1 处差异） | 确认后删除（备份中保留） |
| `configs/` | 1.1 MB | 232 个 yaml：25 个数据集 + 各方法 `base_to_novel/`、`few_shot/` | 归一化，补 `configs/trainers/GRAPH_DCPL` 缺失 |
| `scripts/` | 360 KB | 16 个 baseline 目录；`graph_dcpl/` 下 12 个脚本（base2new / cross_data / domain_genera / xd_train / xd_test / few_shot / 消融） | 按任务重组，路径参数化 |
| `Dassl.pytorch/` | 1.7 MB | 以 **editable** 方式装在 dcpl 环境（`dassl 0.5.0 -> /home/wangyijun/BiomedCoOp/Dassl.pytorch`）；有本地改动：`DATASET.LABEL_MIN_MAX`、`DIFFUSION` 配置块、`test1()` 里带 `pdb.set_trace()` | 进 `third_party/`，**新工作区必须重装** |
| `clip/` | 1.5 MB | OpenAI CLIP 改版 + `pmcclip.py`（PMC-CLIP 的 ResNet50） | 与 clip1 合并 |
| `clip1/` | 1.5 MB | CLIP + IVLP/MaPLe 的 `ResidualAttentionBlock_IVLP` 等；`DCPL_CLIP`、`GraphCLIP`、`MaPLe`、`DPLQ` 都 `from clip1 import clip` | 合并后作为唯一 `clip` 包 |
| `open_clip/` | 2.5 MB | 本地 open_clip（代码以 `open_clip.src.open_clip` 绝对路径导入） | 进 `third_party/`，**不要改目录名**或同步改 import |
| `MedSAM/` | 97 MB | MedSAM（含 `.git` 与 assets PDF）；论文的视觉知识分支 | 进 `third_party/` |
| `BLIP/` | 15 MB | BLIP base，仅 `medkcoop_blip.py` 用，论文未用 | 进 `third_party/` 或 legacy |
| `qnn/` | 3.0 MB | 与论文无关的量子网络支线（`rada.py` 等引用） | legacy |
| `data/` | 24 MB | 11 个 split json 副本；`CHMNIST/split_CHMNIST.json` 与 `/data/wangyijun/dataset` 版本 **md5 不同**（分别为 `97efb40c…` / `374295be…`） | 归档，禁止作为运行数据源 |
| `descriptors/` | 36 KB | 12 个 LLM 描述 json；只有 `vip_graph_dcpl.py` 读 `cfg.DESCRIPTOR_PATH`，该 trainer 未被 `train.py` 导入 | 归档 |
| `assets/` | 1.3 MB | 上游 BiomedCoOp 的 README 图片/DATASETS.md/RUN.md | 保留 |
| 顶层散件 | — | `cal.sh`、`heatmap.py/2/3/4/5.py`、`test_blip.py`、`test_blip2.py`、`rada.py`、`OragnAMNIST.py`、`interpret_prompt.py`、`parse_test_res.py`、`combine_acc_columns.py`、`grad_cam_result.png` | 归入 `tools/`、`legacy/` |
| `__pycache__/` | 70 个目录 | 含**已丢失源码的 pyc** | **备份后清理，但 pyc 必须单独保全** |

### 1.2 版本与工作区状态

- 远端：`https://github.com/HealthX-Lab/BiomedCoOp.git`，分支 `main`，最新提交 `7a8c459 Update README.md`。
- 工作区 **200 项未提交**：55 个已跟踪文件被改、8 个被删、137 个未跟踪（`trainers copy/`、`clip/`、`clip1/`、`BLIP/`、`MedSAM/`、`qnn/`、`configs/datasets/*v2.yaml`、`datasets/*v2.py`、`descriptors/`、`dcpl.yml`、`cal.sh` …）。
- `git fsck --lost-found` 无输出、无 stash、无该文件历史 → **丢失的源文件无法从 git 恢复**。
- 上游仓库本身是 BiomedCoOp 复现框架，MeDKCoOp 的改动全部停留在未提交状态，因此"提交历史"对复现没有参考价值，只能靠工作区快照。

### 1.3 环境与硬件（实测）

| 项 | 值 |
|---|---|
| conda env | `/data/wangyijun/conda/dcpl`，Python 3.10.16 |
| torch / torchvision | 1.12.0+cu113 / 0.13.0+cu113 |
| transformers / open-clip-torch / timm | 4.35.2 / 2.23.0（代码实际用本地 `open_clip/`）/ 1.0.15 |
| dassl | 0.5.0，editable 指向 `/home/wangyijun/BiomedCoOp/Dassl.pytorch` |
| GPU | 9 × RTX 4090 24 GB，driver 550.67；已实测 `matmul` 正常（sm_89） |
| 磁盘 | `/home` 1.6 T 已用 87%（剩 203 G）；`/data` 56 T 已用 78%（剩 12 T） |

### 1.4 数据与权重

| 资源 | 位置 | 状态 |
|---|---|---|
| 8 个论文数据集 | `/data/wangyijun/dataset/{BTMRI,CHMNIST,COVID_19,CTKidney,lungColon,OCTMNIST,RETINA,Kvasir}` | 完整；`BTMRI1` 指向 `BTMRI/brain-tumor-mri-dataset` 与 `split_BTMRI1.json` |
| v1 split（base2new/few-shot） | 上述目录下 `split_*.json`（+ `split_fewshot/`） | 完整，论文数值的来源 |
| v2 split（domain generalization） | 同一批 json，由 `*v2.py` 整份读入 | 完整 |
| 额外数据集 | `CCBTM`、`DermaMNIST`、`KneeXray`、`BUSI`、`miniBTMRI`、`miniBTMRI1` | 磁盘有，论文未用（kneexray 只在日志里多跑了一个） |
| BiomedCLIP | `/data/wangyijun/model/BiomedCLIP`（HF 仓库全量）+ `/data/wangyijun/model/BiomedCLIP_local`（单目录） | 完整，可离线 |
| MedSAM ViT-B | `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth`（375 MB） | 完整 |
| BLIP base | `/data/wangyijun/model/BLIP/model_base.pth`（2.1 GB） | 完整（论文未用） |
| PubMedCLIP / PMC-CLIP | `clip/checkpoints/` 为空 | **缺失**，做 backbone 消融需联网下载 |
| HF cache | `~/.cache/huggingface` 28 KB | 空，离线模式无法解析 `hf-hub:` |

---

## 2. 代码架构与调用链

### 2.1 主链路（论文实验）

```
scripts/*/*.sh
  └─ python train.py --trainer <类名> --dataset-config-file configs/datasets/<DS>.yaml
                    --config-file configs/trainers/<MODEL>/{base_to_novel,few_shot}/<DS>.yaml
                    --mode train|test --eval-only --load-epoch 50|100
       ├─ import datasets.*  → DATASET_REGISTRY 注册（数据集类名 = DATASET.NAME）
       ├─ import trainers.*  → TRAINER_REGISTRY 注册（类名 = --trainer）
       ├─ DASSL setup_cfg：get_cfg_default() + extend_cfg() + 数据集 yaml + 方法 yaml + opts
       ├─ build_trainer → TrainerX（DASSL）
       │    ├─ build_data → datasets/<DS>.py（split json → few-shot → subsample base/new）
       │    └─ build_model → trainer 自己的 CustomCLIP
       └─ train() / test()
```

- base-to-new 是两阶段：先在 `SUBSAMPLE_CLASSES=base` 上训练 50 epoch，再 `--eval-only --load-epoch 50 SUBSAMPLE_CLASSES=new`。
- few-shot / cross-data / DG 走 `few_shot/*.yaml`（`MAX_EPOCH: 100`），论文写明 100 epoch。
- 结果落在 `OUTPUT_DIR/log.txt`，模型落在 `OUTPUT_DIR/MultiModalPromptLearner/model.pth.tar-<epoch>`。
- 评估脚本 `parse_test_res.py` 从各 `seed*/log.txt` 抓 `* accuracy:` 做均值/方差；`cal.sh` 是手工批量调用。

### 2.2 论文方法（以已确认的 `GRAPH_DCPL_NO_TXT_VIS_GRAPH` 为准）

`CustomCLIP.forward` 的四个分支（参数只训练 `prompt_learner.*` 与 `visual_net.*`，BiomedCLIP 与 MedSAM 全冻结）：

1. **视觉知识分支**：`image` 插值到 1024² → MedSAM `image_encoder` → `[B,256,H,W]` → 空间均值 → `[B,256]` → `visual_net`（256→16→512）得到 `D_b^v`，加到 BiomedCLIP 图像特征后 L2 归一化。
2. **可学习提示 + 文本知识**：`ctx`（N_CTX=2，768 维，从 "a photo of a" 初始化）+ `meta_net(D_b)` 偏移 → 与 `token_prefix/token_suffix` 拼成 256 长度提示 → BiomedCLIP 文本编码器 → `[n_cls,512]`。
3. **KRGC（文本/视觉知识图）**：`build_text_graph` 用每类 50 条 LLM 描述（`BIOMEDCOOP_TEMPLATES`）编码取均值作节点；`build_image_graph` 用 few-shot 训练集的 BiomedCLIP(+视觉 bias) 特征取类均值作节点；边由余弦相似度 + `random.uniform(0.5,0.6)` 阈值掩码 + 自环构成。
4. **KRTL（关系迁移）**：把类别文本节点与当前 batch 的提示特征拼成 `[B, 1+C, 512]`，用文本图邻接做对称归一化 GCN（`GCN_fuse`，512×512）→ 取第 0 个节点作融合特征；最终 `z' = 0.3 * z_text_mean + 0.7 * z_fused`（`beta_it=0.3`），与图像特征做余弦 logits，交叉熵训练。

> 代码里 `GCN_tt`、`GCN_it`、`scale_mm`、`base_img_features`、`fc` 都仍在 checkpoint 中，但在最终版前向里是"死参数/仅用于初始化"，阅读时容易被误导。

### 2.3 两条易混淆的管线

- `main.py` + `trainers/__init__.py`：另一套 few-shot 评测（TipAdapter/ClipAdapter/LP/LP2），配置是 `configs/base.yaml` 风格，与论文表格无关。
- `train.py` + DASSL：论文全部实验。`trainers/__init__.py` **不参与** train.py 的注册。

---

## 3. 论文 ↔ 代码 ↔ 产物 对齐结论

### 3.1 最终方法定位（决定性证据）

用 2060 份 `log.txt` 聚合 `train_base`（base）与 `test_new`（new）后：

| trainer 类 | 8 数据集平均 Base | 平均 New | 论文对应行 |
|---|---|---|---|
| **`GRAPH_DCPL_NO_TXT_VIS_GRAPH`** | **87.88** | **79.95** | Table 1 平均、Table 4 "Ours"（87.88 / 79.95 / 82.87） |
| `GRAPH_DCPL_NO_TXT_VIS_FUSE` | 84.81 | 72.37 | Table 5 第 1 行（✗✗✗，84.81 / 72.36 / 76.30） |
| `GRAPH_DCPL_NO_VIS_FUSE_GRAPH` | 83.77 | 74.70 | Table 5 第 2 行（✗✓✗，83.77 / 74.70 / 77.44） |
| `GRAPH_DCPL_NO_VIS_GRAPH` | 87.53 | 71.04 | Table 5 第 7 行（✓✗✓，87.52 / 71.03 / 76.95） |
| `GRAPH_DCPL_NO_TXT_VIS_GRAPH1` | 87.52 | 51.92 | Table 5 第 3 行（✗✗✓，87.52 / 51.92 / 61.95） |

逐数据集核对（`GRAPH_DCPL_NO_TXT_VIS_GRAPH`，3 seed 均值 vs 论文 Table 1）：

| 数据集 | 代码 base | 论文 base | 代码 new | 论文 new |
|---|---|---|---|---|
| BTMRI1 | 85.57 | 85.57 | 92.10 | 92.10 |
| CHMNIST | 93.67 | 93.67 | 60.17 | 60.17 |
| Covid-19 | 78.97 | 78.97 | 91.07 | 91.07 |
| CTKIDNEY | 90.70 | 90.70 | 84.83 | 84.83 |
| LC25000 (lungcolon) | 94.27 | 94.27 | 96.27 | 96.27 |
| RETINA | 88.60 | 88.60 | 58.50 | 58.50 |
| OCTMNIST | 83.27 | 83.27 | 85.83 | 85.83 |
| Kvasir | 88.00 | 88.00 | 70.83 | 70.83 |

**16 个格子全部精确一致**，可判定论文结果就是该类跑出来的。
（HM 列 82.87 是"各数据集 HM 的平均"；若用平均 base/new 再算 HM 会得到 83.73，属正常差异，不是矛盾。）

### 3.2 丢失的源文件与可重建性

- 存在：`trainers/GRAPH_DCPL/__pycache__/vip_graph_dcpl_biomedclip_adapter_no_text_vis_graph.cpython-310.pyc`（2025-04-07，22 KB）。
- 不存在：对应 `.py`；git 全历史无；`trainers copy/` 无；全盘 `find` 无同名源码。
- 已用 `conda dcpl` 的 Python 3.10 成功 `marshal.loads` 并反汇编，与 `trainers/GRAPH_DCPL/medkcoop.py` 逐函数对比，差异只有：
  1. 类名不同（`GRAPH_DCPL_NO_TXT_VIS_GRAPH` vs `MeDKCoOp`）；
  2. `build_prompt_graph` 里辅助函数名 `cal_edge_emb` vs `edge_construct`（实现一致）；
  3. `_get_base_image_features` 里多出 3 行"图像特征加噪"（`mean`/`noise = randn * mean`），medkcoop.py 已删；
  4. `_get_base_text_features` 多一句 `print("构建文本子图......")`；
  5. pyc 的 `build_subgraph` 是 `beta*base_text + (1-beta)*fuse_features`，与 medkcoop.py **一致**（`alpha_it` 那行在 GRAPH_DCPL/medkcoop.py 已注释）。
- 结论：**可以在 2 小时内高保真重建该 trainer，并用现有 checkpoint 做 eval-only 等价性验证**（state_dict 键完全兼容，见 3.3）。

### 3.3 checkpoint 兼容性（已抽查 `CHMNIST/.../NO_TXT_VIS_GRAPH/seed1`）

- 单个 `model.pth.tar-50` 1.16 GB，`state_dict` 有 **838 个张量**：因为 `register_model("MultiModalPromptLearner", self.model, ...)` 注册的是整个 `CustomCLIP`，把冻结的 BiomedCLIP 一并存了。
- `prompt_learner` 键：`ctx(2,768)`、`scale_mm(1)`、`token_prefix/suffix`、`base_text_features(4,512)`、`base_img_features(4,512)`、`meta_net.*`、`fc(4,512)`、`GCN_tt/GCN_it/GCN_fuse.{gcn_weights(512,512), gcn_bias(4,512)}`；另有 `visual_net.{linear1,linear2}.*`。
- 该键集合与 `trainers/GRAPH_DCPL/medkcoop.py` 的模型结构**完全对应**（含 `fc`；`new_fc` 只在 test 模式存在，`load_model` 用 `strict=False` 且显式删除 `base_*`/`token_*`/`gcn_bias`）。
- 因此"重建源码 → 用旧 checkpoint 复评 → 与 log 数值逐格比对"是一条可靠的验证闭环。

### 3.4 cross-dataset 与 domain generalization：**产物为零**

- 脚本：`scripts/graph_dcpl/xd_train.sh` / `xd_test.sh` / `cross_data_eval.sh` / `domain_genera.sh`。
- 脚本期望：源域训练于 `/home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/BTMRI1/...`，评测于 `.../CrossData/evaluation/...`。
- 实际：`/home/wangyijun/ckpt` 不存在；`/data/wangyijun/ckpt` 全盘无 `CrossData`、`cross`、`domain` 目录；`Biomed_Graph_dcpl` 只有 `ablation/`、`few_shot/`、`heatmap/`。
- 含义：论文 Table 2、Table 3（含 6 个 baseline）**当前无法从盘上复算，只能重跑**。这也解释了为什么论文里两张表的数值几乎相同但又不完全相等（cross-data 用 v1 split，DG 用 v2 split，训练配置同为 100 epoch）。

### 3.5 few-shot 与消融

- few-shot：`/data/wangyijun/ckpt/Biomed_Graph_dcpl/few_shot/<DS>/shots_{1,2,4,8,16}/MeDKCoOp/nctx2_cscFalse_ctpend/seed{1,2,3}/`，共 41 组。
- rebuttal：`/data/wangyijun/ckpt/Biomed_Graph_dcpl_rebut/few-shot/<DS>/GRAPH_DCPL_NO_TXT_VIS_GRAPH1/vit_b16_base2new_{1,2,4,8,16}shots/MeDKCoOp/seed*`（120 份日志、130 GB 模型）。**目录名写的是 `_GRAPH1`，但这是 rebuttal 阶段重跑的产物**，引用时不能按目录名直接当"无图版本"。
- 计时：`/data/wangyijun/ckpt/Biomed_Graph_dcpl_time/`（1 组，`MeDKCoOp_Time`）。
- 消融：`ablation/train_base` 与 `test_new` 各有 41 个 trainer 变体（`NO_TXT_GRAPH`、`NO_VIS_GRAPH`、`NO_MEDSAM`、`NO_TXT_BIAS`、`NO_VIS_BIAS`、`NO_FUSE`、`hyper/b01..b08`、`a04b03..a08b03`、三套 backbone 等）。已确认 5 个与论文表格精确对应（见 3.1），其余 3 行需在整理阶段逐一对齐后写入 `EXPERIMENTS.md`。
- 论文正文没有 few-shot 表，这些属于补充实验/返修材料，优先级低于前三张表。

---

## 4. 问题清单（按优先级）

### P0 — 阻断复现

| # | 问题 | 证据 | 处理 |
|---|---|---|---|
| 1 | 论文最终方法源码丢失 | 仅剩 `.pyc`；git/副本均无 | 从 pyc 反汇编 + 对照 `medkcoop.py` 重建，并用旧 checkpoint 验证 |
| 2 | 离线无法加载 BiomedCLIP | `train.py --no-train` 实测 `LocalEntryNotFoundError` | 封装 `load_biomedclip()`：`add_model_config()` + 本地 `open_clip_pytorch_model.bin` + 本地 tokenizer 目录 |
| 3 | cross-data / DG 无任何产物 | 无 `CrossData` 目录 | 重建后按论文设置重跑（源域 BTMRI1，100 epoch） |
| 4 | DASSL 是 editable 安装到旧路径 | `pip list` 显示 `dassl 0.5.0 -> /home/wangyijun/BiomedCoOp/Dassl.pytorch` | 新工作区 `pip install -e third_party/Dassl.pytorch`，否则 import 到旧代码 |
| 5 | 脚本引用的配置目录不存在 | `configs/trainers/GRAPH_DCPL` 缺失，脚本 `MODEL=GRAPH_DCPL`；`--cfg` 名 `vit_b16_0002_b50_randflip` 与文件名路径不一致 | 统一到 `configs/trainers/medkcoop/{base_to_novel,few_shot}/<DS>.yaml`，脚本改用变量 |
| 6 | `trainers/MeDKCoOp/medkcoop.py` 不可用 | MedSAM 路径为 `/path/to/your/medsam.pth`；含多余 `alpha_it` 融合；`train.py` 中 import 被注释 | 以重建版替换，删除旧副本或降级为 legacy |

### P1 — 影响正确性/可维护性

| # | 问题 | 说明 |
|---|---|---|
| 7 | 注册名冲突风险 | `MeDKCoOp` 在两个文件里同名；`MaPLe/maple.py` 与 `maple_biomedclip.py` 都注册 `MaPLe_CLIP`；若同时 import 会 `KeyError`。当前靠"只 import 一个"规避 |
| 8 | `train.py` 只 import 了部分 trainer | 大量行被注释（41 个消融变体只开了一个 `NO_TXT_VIS_GRAPH1`；`MeDKCoOp/medkcoop.py` 被注释）；换方法要手改源码 |
| 9 | 随机性 | `edge_construct` 每次调用 `random.uniform(0.5,0.6)`；`train.py` 设 `cudnn.benchmark=True`；DataLoader worker 消耗 RNG → **逐位复现不可能**，只能按容差验收 |
| 10 | `data/` 中的 split 是旧副本 | CHMNIST 的 md5 与 `/data/wangyijun/dataset` 不同；若 `--root data` 会得到不同划分 |
| 11 | v2 数据集的语义 | `*v2.py` 把 train+val+test 全量塞进 `train_x` 与 `test`，不做 few-shot/subsample；只适用于"跨数据集直接评测"，不能用于 base2new |
| 12 | `main.py` 的 `dataset_list` 映射 | `"BTMRI1": BTMRI`（`miniBTMRI1` 子集），与 DASSL 用 `DATASET_REGISTRY.get("BTMRI1")` 得到的类不同；两条管线混用会拿错数据 |
| 13 | clip / clip1 分叉 | `clip1` 有 IVLP/MaPLe block 与 `pdb`；`clip` 独有 `pmcclip.py`。`DCPL_CLIP`/`GraphCLIP`/`MaPLe`/`DPLQ` 依赖 clip1，`main.py`/`utils.py`/PubMed 系依赖 clip |
| 14 | `--heatmap` 会崩 | `train.py` 调 `trainer.heatmap_hook()`，全仓库无该定义（`Dassl` 只有 `output_test/test1`） |
| 15 | few-shot 脚本覆盖了错误命名空间 | `few_shot.sh` 传 `TRAINER.BIOMEDCOOP.N_CTX` 等，但 `GRAPHDCPL` trainer 读的是 `TRAINER.GRAPHDCPL.*`，覆盖静默无效 |
| 16 | 基线配置不一致 | 例如 `configs/trainers/DCPL/few_shot/BTMRI1.yaml` 是 50 epoch，同目录 CHMNIST/covid 是 100 epoch |
| 17 | 模型文件浪费 20 倍空间 | 只训 `prompt_learner`+`visual_net`（约 50 MB），却整包保存 1.16 GB；939 GB 中约 900 GB 是冻结的 BiomedCLIP |
| 18 | 论文 8 数据集之外的 kneexray | `ablation` 里有 9 个数据集、`scripts` 里也列了 kneexray，论文表里没有；归档时标注"未入表" |
| 19 | PubMedCLIP/PMC-CLIP 权重缺失 | `clip/checkpoints/` 为空；backbone 消融（论文 Fig. 4）需联网补齐 |

### P2 — 整洁性

- `Dassl.pytorch/dassl/engine/trainer.py` 里 `test1()` 带 `pdb.set_trace()` 并打印 `"1111111"`；`clip1/model.py` import 了 `pdb`。
- `prompt_templates.py` / `prompt_templates1.py` / `_incorrect.py` / `_outofdomain.py` 四份并存（260–450 KB），只有 `prompt_templates.py` 被主链路使用。
- 中文 `print`、大量注释掉的实验代码、`trainers copy/`、`qnn/`、`grad_cam_result.png`、`OragnAMNIST.py` 等与论文无关的散件。
- `descriptors/` 未被主链路读取（只被未导入的 `vip_graph_dcpl.py` 使用）。

---

## 5. 备份方案

原则：**先冻结、再分层备份、最后才动原目录**。代码与日志很小（<400 MB），模型很大（939 GB），两者策略分开。

### 5.1 落盘位置与分层

```
/data/wangyijun/backup/BiomedCoOp_2026-11-05/
├── MANIFEST.md              # 备份说明、清单、校验结果
├── l0_snapshot/             # 代码全量快照（含 .git、含 pyc）
│   ├── BiomedCoOp_code.tar.gz
│   ├── BiomedCoOp_git.bundle
│   ├── git_status.txt
│   ├── git_worktree.diff
│   └── MD5SUMS.txt
├── l1_artifacts/            # 实验元数据（不含模型）
│   ├── logs_Biomed_Graph_dcpl.tar.gz
│   ├── logs_rebuttal_time.tar.gz
│   ├── tensorboard_index.txt
│   └── results_flat.csv     # dataset,trainer,cfg,seed,split,acc
├── l2_models/               # 精选模型（按需，见 5.3）
│   └── final_method/<DS>/seed<k>/model.pth.tar-50
└── l3_checksums/            # 大文件 sha256（可选，耗时）
```

选 `/data` 的原因：`/home` 仅剩 203 GB，而 L2 至少要几十 GB；`/data` 剩 12 TB。

### 5.2 L0 代码快照（约 180 MB → tar 后约 120 MB）

```bash
# 1) 冻结：只记录，不 chmod，避免影响后续使用
mkdir -p /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot
cd /home/wangyijun
git  -C BiomedCoOp status --porcelain > /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/git_status.txt
git  -C BiomedCoOp diff            > /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/git_worktree.diff
git  -C BiomedCoOp bundle create /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/BiomedCoOp_git.bundle --all

# 2) 全量 tar：保留权限/时间戳；__pycache__ 一并保留（里面有关键 pyc）
tar --numeric-owner -czf /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/BiomedCoOp_code.tar.gz \
    BiomedCoOp

# 3) 清单与校验
cd BiomedCoOp
find . -type f -not -path './.git/*' -print0 | sort -z | xargs -0 md5sum \
  > /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/MD5SUMS.txt
tar -tzf /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/BiomedCoOp_code.tar.gz | wc -l
```

**额外必做（P0）**：把丢失源码的 pyc 单独复制一份并提升为"最高保全级别"：

```bash
mkdir -p /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/lost_source
cp -a BiomedCoOp/trainers/GRAPH_DCPL/__pycache__/vip_graph_dcpl_biomedclip_adapter_no_text_vis_graph.cpython-310.pyc \
      /data/wangyijun/backup/BiomedCoOp_2026-11-05/l0_snapshot/lost_source/
```

### 5.3 L1 实验元数据（约 80 MB，优先做）

```bash
OUT=/data/wangyijun/backup/BiomedCoOp_2026-11-05/l1_artifacts
mkdir -p "$OUT"
# 只打包日志，不碰 1.16 GB 的模型
find /data/wangyijun/ckpt/Biomed_Graph_dcpl         -name log.txt -print0 | tar --null -czf "$OUT/logs_Biomed_Graph_dcpl.tar.gz" --files-from=-
find /data/wangyijun/ckpt/Biomed_Graph_dcpl_rebut   -name log.txt -print0 | tar --null -czf "$OUT/logs_rebuttal.tar.gz"   --files-from=-
find /data/wangyijun/ckpt/Biomed_Graph_dcpl_time    -name log.txt -print0 | tar --null -czf "$OUT/logs_time.tar.gz"       --files-from=-
# log 头部含完整 cfg 打印，可用于反推每次运行的超参
```

同时生成 `results_flat.csv`（整理版工具 `tools/collect_results.py` 的第一版即可复用本次评审脚本的逻辑）：
遍历 `ablation/{train_base,test_new}/<DS>/shots_16/<TRAINER>/<CFG>/seed*/log.txt`，抽取最后一个 `* accuracy: XX.X%`，输出
`dataset,trainer,config,seed,split,acc,elapsed,log_path`。这是后续所有表格的唯一数据源，避免再从日志里手工抄数。

### 5.4 L2 模型（按价值分级，不是全备）

| 级别 | 内容 | 体积 | 建议 |
|---|---|---|---|
| A（必留） | 最终方法 8 数据集 × 3 seed 的 base 训练模型（24 个） | ~28 GB | 必留，用于 eval-only 复评与 cross-data 起点 |
| B（建议） | 论文表格用到的 5 个消融变体 × 8 × 3（约 120 个） | ~140 GB | 建议留，便于复算 Table 4/5 |
| C（可选） | 全部 41 个变体 × 8 × 3（1088 个） | 939 GB | 不建议全留；保留 L1 日志即可复算指标 |
| D（基线） | `BiomedCoOp` 146 GB、`DCPL_BiomedCLIP` 133 GB、`MaPLe` 60 GB、`GraphAdapter` 2.9 GB、`coop` 1 GB | ~343 GB | 只留 `log.txt` + 需要复算的 seed 模型 |

```bash
# 级别 A 示例
DST=/data/wangyijun/backup/BiomedCoOp_2026-11-05/l2_models/final_method
for DS in BTMRI1 CHMNIST covid ctkidney lungcolon octmnist retina kvasir; do
  SRC=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/$DS/shots_16/GRAPH_DCPL_NO_TXT_VIS_GRAPH/vit_b16_0002_b50_randflip
  for S in 1 2 3; do
    mkdir -p "$DST/$DS/seed$S"
    cp -a "$SRC/seed$S/MultiModalPromptLearner/model.pth.tar-50" "$DST/$DS/seed$S/"
  done
done
```

若空间紧张，可只保留 `prompt_learner` + `visual_net` 的权重（约 50 MB/个，压缩 20 倍），但需要写一个"提取/回填"脚本，且会失去"整包直接 eval"的便利——建议整理阶段再做，不要现在动。

### 5.5 备份验收

1. `tar -tzf` 条目数 ≈ `find | wc -l`；`MD5SUMS.txt` 抽查 20 个文件一致。
2. 从 tar 解压到 `/tmp` 后 `python -c "import train"` 成功（证明快照自洽）。
3. 抽样 `torch.load` 三个 L2 模型，键集合与 3.3 一致。
4. `results_flat.csv` 行数 = 日志数（2060 + 120 + 1），且用它对 3.1 的表能完全重算出来。
5. `MANIFEST.md` 记录：备份时间、命令、体积、校验和、缺失项（cross-data/DG 产物、PubMedCLIP/PMC-CLIP 权重）。

---

## 6. 整理方案（新工作区）

### 6.1 目标结构

```
/home/wangyijun/MeDKCoOp/                 # 从 L0 解包复制，原目录保持不动
├── README.md                  # 项目说明 + 论文引用 + 快速开始
├── REPRO.md                   # 环境、权重、数据、逐表复现命令
├── EXPERIMENTS.md             # 论文表 ↔ trainer 类 ↔ 脚本 ↔ ckpt 路径 的映射表
├── RESULTS.md                 # 复现结果 vs 论文（含容差与差异说明）
├── CHANGELOG.md
├── configs/
│   ├── datasets/              # 只留论文 8 个 + 对应 v2 8 个
│   └── trainers/
│       ├── medkcoop/          # 唯一方法配置：base_to_novel/ few_shot/
│       └── baselines/{coop,cocoop,kgcoop,maple,dcpl,biomedcoop,graphadapter}/
├── datasets/                  # 8 个 v1 + 8 个 v2；删除 BUSI/CCBTM/DermaMNIST/KneeXray（移 legacy）
├── scripts/
│   ├── base2new.sh            # 参数：DATASET SEED GPU
│   ├── few_shot.sh
│   ├── cross_data.sh
│   ├── domain_generalization.sh
│   ├── ablation.sh
│   └── smoke/base2new_BTMRI1_1epoch.sh
├── trainers/
│   ├── medkcoop/medkcoop.py   # 重建后的论文方法（唯一权威实现）
│   ├── baselines/             # 6 个 baseline，每个 1 个 biomedclip 版本
│   └── legacy/                # 41 个历史变体 + 旧 medkcoop 副本，只读
├── third_party/
│   ├── Dassl.pytorch/         # 去掉 pdb/test1 调试代码后重装 editable
│   ├── open_clip/             # 保持目录名（绝对 import 依赖）
│   ├── clip/                  # clip1 为主体 + 并入 pmcclip.py
│   └── MedSAM/
├── tools/
│   ├── collect_results.py     # 日志 → results_flat.csv → 三张表
│   ├── parse_test_res.py
│   ├── heatmap/               # heatmap*.py 合并成一个 CLI
│   └── env/                   # dcpl.yml（导出精确版本）+ 权重下载清单
├── tests/
│   ├── test_imports.py        # 所有 trainer/dataset 可导入、注册名唯一
│   ├── test_dataset_split.py  # 8 数据集类别数/样本数断言
│   └── test_ckpt_compat.py    # 旧 checkpoint 键集合断言
└── legacy/                    # main.py 管线、qnn、BLIP 实验、descriptors、data/旧 split、png
```

### 6.2 分阶段执行（每步都有验收门槛，不通过不进入下一步）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **S0 冻结** | 只做盘点与记录；原目录不再新增/修改文件 | 本文件 + `git_status.txt` + `MD5SUMS.txt` 落盘 |
| **S1 备份** | 执行第 5 章 L0/L1，L2 级别 A | 5.5 的五项校验全过 |
| **S2 复制** | `cp -a`/解包到新工作区；重装 DASSL editable；本地权重适配 | `python -c "import train"` 通过；`pip list` 中 dassl 指向新路径 |
| **S3 清理** | 删 `__pycache__`（已备份）、`trainers copy/`、`data/`、`qnn/`、`BLIP/`；合并 `clip`+`clip1`；散件归 `tools/legacy/` | 仓库 < 100 MB；`tests/test_imports.py` 通过；无重复注册名 |
| **S4 归一化** | 配置只保留 `configs/trainers/medkcoop/`；脚本路径全部变量化（`DATA/CKPT/GPU/SEED`）；补 `collect_results.py`；统一 `OUTPUT_DIR` 命名 | `scripts/smoke/..._1epoch.sh` 在单卡跑通 |
| **S5 方法重建** | 依据 pyc 反汇编重写 `trainers/medkcoop/medkcoop.py`：恢复 `cal_edge_emb`、恢复 `_get_base_image_features` 的噪声行（作为开关 `NOISE=1`）、删除 `alpha_it` 死代码、把 `no_text/no_vis` 消融做成开关 | `test_ckpt_compat.py` 通过；`--eval-only` 加载旧 checkpoint 不报缺键 |
| **S6 复现验证** | ① 8 数据集 × 3 seed 的 base2new；② cross-data（源 BTMRI1，100 epoch，8 目标）；③ DG（v2）；④ few-shot 1/2/4/8/16；⑤ Table 4/5 消融 + 待对齐的 3 行 | `RESULTS.md` 三张表全部 ≤ 容差（见 7.4） |
| **S7 归档** | 写 `EXPERIMENTS.md` 全映射、`REPRO.md` 命令、`RESULTS.md` 差异说明；回填 `MANIFEST.md` | 新环境下按 REPRO.md 从零跑通 smoke 与 1 个数据集 |

### 6.3 clip / clip1 合并的具体做法（低风险顺序）

1. 以 `clip1/` 为主干（它包含 MaPLe 需要的 `ResidualAttentionBlock_IVLP` 等）。
2. 把 `clip/pmcclip.py` 原样复制进主干；`clip1` 缺的只有这个文件（`diff -rq` 证据）。
3. 删除主干里的 `import pdb` 与调试断点。
4. 包名统一为 `clip`；若担心存量脚本，保留 `clip1/__init__.py` 做 `from clip import *` 的兼容壳，下一阶段删除。
5. 数值等价性检查：同一张图 + 同一文本，对比合并前后 `encode_image`/`encode_text` 输出的 `allclose(atol=1e-6)`。注意 `clip1` 的 `model.py` 多了 IVLP 分支，但默认不启用，构造路径应完全一致；若不一致，以"MaPLe 能跑 + CLIP 数值一致"为准逐项定位。

### 6.4 数据层整理要点

- 运行数据一律用 `--root /data/wangyijun/dataset`；`repo/data/` 降级为 `legacy/` 并加 `README` 说明"旧副本，md5 与现用 split 不同"。
- `datasets/__init__.py` 的 `dataset_list` 只服务 `main.py`，整理时要么删掉这条管线，要么把 `"BTMRI1": BTMRI` 改成正确的 `BTMRI1` 类，避免再次拿错数据。
- v2 数据集在 docstring 里写清"train_x = train+val+test，仅用于跨数据集直接评测，不做 few-shot/subsample"。
- 每个数据集类加一个 `assert`：类别数、train/val/test 样本数与 `split_*.json` 一致（`tests/test_dataset_split.py`）。
- 论文只用 8 个数据集；`kneexray`、`CCBTM`、`BUSI`、`DermaMNIST` 移入 legacy 并在 `EXPERIMENTS.md` 标注"未入表"。

---

## 7. 复现路线

### 7.1 环境

```bash
conda create -n medkcoop python=3.10.16 -y      # 或直接复用 dcpl 并冻结导出
conda activate medkcoop
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt                  # open_clip_torch==2.23.0, transformers==4.35.2 ...
pip install -e third_party/Dassl.pytorch         # 关键：不要沿用旧路径的 editable 安装
conda env export --no-builds > tools/env/environment.yml   # 固化
```

`requirements.txt` 缺 `numpy/scipy/scikit-learn` 的固定版本，`dcpl.yml` 里有全量 pip 列表，建议以 `dcpl.yml` 为准导出并锁定。

### 7.2 权重（全部本地，不依赖网络）

| 权重 | 本地路径 | 加载方式（整理版） |
|---|---|---|
| BiomedCLIP ViT-B/16 | `/data/wangyijun/model/BiomedCLIP`（含 `open_clip_config.json`、`open_clip_pytorch_model.bin`、`BiomedBERT/`、tokenizer） | `add_model_config(<dir>)` + `create_model_from_pretrained('microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', pretrained=<...bin>)`；tokenizer 指向本地 `BiomedBERT` |
| MedSAM ViT-B | `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` | 环境变量 `MEDSAM_CKPT`，禁止硬编码 |
| BLIP base | `/data/wangyijun/model/BLIP/model_base.pth` | 仅 legacy |
| PubMedCLIP / PMC-CLIP | 缺失 | 需要时从 HF 下载（`sarahESL/PubMedCLIP`、`axiong/pmc_oa`），并写入 `MANIFEST.md` |

> 注意：`/data/wangyijun/model/BiomedCLIP_local` 只有 `config.json`、`open_clip_config.json`、权重、`vocab.txt`，没有 `tokenizer.json`；离线 tokenizer 需要用 `/data/wangyijun/model/BiomedCLIP/BiomedBERT/`（含 `vocab.txt`/`tokenizer.json`）。整理时先做一次"离线加载 + 与在线版本输出 allclose"的验证，再改全仓库。

### 7.3 命令模板（整理后）

```bash
export DATA=/data/wangyijun/dataset
export CKPT=/data/wangyijun/ckpt/MeDKCoOp
export GPU=0
export MEDSAM_CKPT=/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth

# 1) base2new（8 数据集 × 3 seed，50 epoch）
for SEED in 1 2 3; do
  for DS in BTMRI1 CHMNIST covid ctkidney lungcolon octmnist retina kvasir; do
    bash scripts/base2new.sh "$DS" "$SEED" "$GPU"
  done
done

# 2) cross-data（源 BTMRI1，100 epoch → 8 目标直接评测）
bash scripts/cross_data.sh BTMRI1 1 "$GPU"

# 3) domain generalization（v2 split）
bash scripts/domain_generalization.sh BTMRI1 1 "$GPU"

# 4) few-shot（1/2/4/8/16）
bash scripts/few_shot.sh 16 1 "$GPU"

# 5) 汇总
python tools/collect_results.py --root "$CKPT" --out RESULTS.md
```

### 7.4 预期数值与容差

- base2new 目标值见 3.1 的表（16 格精确值）。
- **验收容差**：单数据集 base/new 与论文差 ≤ 0.5 pp；3 seed 平均差 ≤ 0.3 pp；HM 用"各数据集 HM 的平均"口径复算。
- 由于 `edge_construct` 的随机阈值 + `cudnn.benchmark=True`，**逐位复现不可能**。建议整理版同时提供 `--deterministic`（固定阈值 0.55、`cudnn.deterministic=True`、`worker_init_fn` 固定），并单独记录该模式下的基线数值，不要与论文值混用。
- cross-data/DG：论文 Table 2/3 是唯一参照（无盘上产物），首次重跑以"同配置、同 seed"为准；若差异 > 1 pp，优先检查 v1/v2 split 与 `few_shot/*.yaml` 的 epoch。

### 7.5 时间预算（单卡 4090，基于现有日志实测）

| 任务 | 单次实测/推算 | 总量 |
|---|---|---|
| base2new 单数据集单 seed（训练 50 ep + new 评测） | BTMRI1 4m21s、CHMNIST 6m04s；OCTMNIST 评测更大 | 8×3 ≈ 6–10 h |
| cross-data（源训练 100 ep + 8 目标评测） | 目标测试集 1.2k–33k，MedSAM 1024² 前向为主 | 每 seed 1.5–3 h；3 seed 5–9 h；含 baseline 再 ×7 |
| DG（v2 全量评测） | 同上，测试集为整份数据 | 与 cross-data 同量级 |
| few-shot 1/2/4/8/16 × 8 数据集 × 3 seed | 16-shot CHMNIST 100 ep = 21m48s | 约 30–40 h |
| Table 4/5 消融（5 个已确认变体 + 3 个待对齐） | 复用 base2new 流程 | 约 15–25 h |

---

## 8. 需要你确认的 6 件事

1. **备份落盘位置**：建议 `/data/wangyijun/backup/BiomedCoOp_2026-11-05/`（`/home` 只剩 203 GB，放不下 L2）。是否同意？
2. **模型保留级别**：A（28 GB，最终方法）/ B（+140 GB，论文用到的消融）/ C（939 GB，全量）。建议 A+B。选哪个？
3. **新工作区**：建议 `/home/wangyijun/MeDKCoOp`（与原目录同级、互不干扰）。是否沿用这个名字？
4. **cross-data / DG 必须重跑**（盘上无任何产物，含 6 个 baseline）：是否按论文设置全部重跑，还是先只重跑 MeDKCoOp + BiomedCoOp（最关键对照）？
5. **clip 合并方向**：建议"以 clip1 为主干 + 并入 `pmcclip.py` + 包名统一为 `clip`"。是否同意？
6. **非论文支线**：`main.py` 管线（TipAdapter/ClipAdapter/LP/LP2）、BLIP、qnn、descriptors、`data/` 旧 split——建议移入 `legacy/` 只读保留，不删除。是否同意？

---

## 9. 交付物与工时估算

| 阶段 | 交付物 | 估算 |
|---|---|---|
| S1 备份 | 4 个 tar/bundle + `MD5SUMS.txt` + `MANIFEST.md` + `results_flat.csv` | 0.5 天（939 GB 全备则另加 1–2 天） |
| S2–S4 整理 | 新仓库骨架、依赖重装、离线权重适配、配置/脚本归一、导入与数据断言测试 | 1.5–2 天 |
| S5 方法重建 | 重建后的 `trainers/medkcoop/medkcoop.py` + 旧 checkpoint eval-only 等价性报告 | 0.5 天 |
| S6 复现 | `RESULTS.md`：Table 1 全绿、Table 4/5 对齐、Table 2/3 新跑 | 3–5 天 GPU 机时（可与整理并行） |
| S7 归档 | `README/REPRO/EXPERIMENTS/RESULTS/CHANGELOG` 五件套 | 0.5 天 |

**建议的立即动作（待你确认后执行）**：先做 S1 的 L0 + L1（约 1 小时、<400 MB），把丢失的 pyc 单独保全；这一步不改动任何既有代码，风险最低、收益最高。

---

## 附录 A：关键路径速查

| 用途 | 路径 |
|---|---|
| 论文主入口 | `/home/wangyijun/BiomedCoOp/train.py` |
| 论文方法（现状最接近版） | `trainers/GRAPH_DCPL/medkcoop.py`（类 `MeDKCoOp`） |
| 论文方法（论文数值对应版，源码丢失） | `trainers/GRAPH_DCPL/__pycache__/vip_graph_dcpl_biomedclip_adapter_no_text_vis_graph.cpython-310.pyc` |
| 用户以为的最终版（实测非论文版） | `trainers/MeDKCoOp/medkcoop.py` |
| base2new 产物 | `/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/{train_base,test_new}/<DS>/shots_16/GRAPH_DCPL_NO_TXT_VIS_GRAPH/vit_b16_0002_b50_randflip/seed{1,2,3}/` |
| few-shot 产物 | `/data/wangyijun/ckpt/Biomed_Graph_dcpl/few_shot/<DS>/shots_<K>/MeDKCoOp/nctx2_cscFalse_ctpend/seed<k>/` |
| rebuttal few-shot | `/data/wangyijun/ckpt/Biomed_Graph_dcpl_rebut/few-shot/<DS>/GRAPH_DCPL_NO_TXT_VIS_GRAPH1/vit_b16_base2new_<K>shots/MeDKCoOp/seed<k>/` |
| 数据集 | `/data/wangyijun/dataset/<DS>/`（`split_*.json` + `split_fewshot/`） |
| BiomedCLIP 权重 | `/data/wangyijun/model/BiomedCLIP/`、`/data/wangyijun/model/BiomedCLIP_local/` |
| MedSAM 权重 | `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` |
| 环境 | `/data/wangyijun/conda/dcpl` |
| 提示模板（72 类 × 50 条） | `trainers/prompt_templates.py`（主用）、`_incorrect.py`、`_outofdomain.py`、`prompt_templates1.py` |

## 附录 B：论文表格 ↔ 已确认 trainer 类

| 论文位置 | trainer 类 | 证据 |
|---|---|---|
| Table 1 平均 / Table 4 Ours | `GRAPH_DCPL_NO_TXT_VIS_GRAPH` | 8 数据集 16 格精确一致 |
| Table 5 (✗✗✗) | `GRAPH_DCPL_NO_TXT_VIS_FUSE` | 84.81 / 72.37 |
| Table 5 (✗✓✗) | `GRAPH_DCPL_NO_VIS_FUSE_GRAPH` | 83.77 / 74.70 |
| Table 5 (✓✗✓) | `GRAPH_DCPL_NO_VIS_GRAPH` | 87.53 / 71.04 |
| Table 5 (✗✗✓) | `GRAPH_DCPL_NO_TXT_VIS_GRAPH1` | 87.52 / 51.92 |
| Table 5 其余 3 行 | 待对齐（候选：`NO_TXT_GRAPH`、`NO_TXT_FUSE_GRAPH`、`NO_FUSE`、`NO_MEDSAM`、`NO_TXT_BIAS`、`NO_VIS_BIAS`、`hyper/b*`） | S6 阶段逐一对齐 |
| Table 2 / Table 3 | **无盘上产物** | 需重跑 |
