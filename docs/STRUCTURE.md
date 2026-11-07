# 目录结构说明

> 最后更新：2026-11-07
> 原则：只保留论文与实验实际用到的文件，每一层单一职责。
> 原项目 `/home/wangyijun/BiomedCoOp` 未做任何修改；完整快照见
> `/home/wangyijun/backup/BiomedCoOp_2026-11-05/`。

## 1. 当前结构

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
│   │   ├── reference/medkcoop.py        # 论文数值对应的原实现（对照组）
│   │   ├── reference/paper_bytecode/    # 丢失源码的 .pyc（重建依据）
│   │   ├── fixed/medkcoop.py            # 修复版 + 算法开关（实验组）
│   │   └── ablation/                    # 16 个论文消融变体
│   ├── baselines/               # BiomedCoOp CoOp CoCoOp KgCoOp MaPLe DCPL GraphAdapter Zeroshot
│   └── prompt_templates*.py     # LLM 知识模板
├── scripts/
│   ├── medkcoop/                # base2new / xd_train / xd_test / few_shot / 消融
│   ├── baselines/               # 8 个 baseline 的同名脚本
│   └── run_all_methods.sh       # 批量调度入口
├── third_party/                 # Dassl.pytorch open_clip clip MedSAM
├── tools/                       # 结果汇总 / 日志解析
├── tests/                       # 6 个冒烟测试
└── docs/                        # 见 docs/README.md
```

## 2. 语义边界

| 目录 | 只负责 | 不负责 |
|---|---|---|
| `configs/datasets/` | 数据集名与 split 入口（`DATASET.NAME`） | 不含数据本身 |
| `configs/trainers/` | 方法超参 | 不含数据/模型结构 |
| `datasets/` | 读 split、few-shot 采样、base/new 划分 | 不定义模型 |
| `trainers/medkcoop/reference/` | 论文原实现（不改动，作对照） | 不放消融 |
| `trainers/medkcoop/fixed/` | 修复与算法改进（带开关） | 不改变 reference 行为 |
| `trainers/medkcoop/ablation/` | Table 4/5、Fig 4 的 16 个变体 | 不放 baseline |
| `trainers/baselines/` | 论文对比方法 | 不放本文方法 |
| `scripts/` | 批量启动器 | 不含模型/数据逻辑 |
| `third_party/` | 外部代码 | 不放本文实验代码 |
| `tools/` | 结果汇总 | 不参与训练 |

## 3. trainer 注册类清单

| 类名 | 来源文件 | 用途 |
|---|---|---|
| `MeDKCoOp` | `reference/medkcoop.py` | 原实现（`GRAD_CTX=False`，论文数值） |
| `GRAPH_DCPL_NO_TXT_VIS_GRAPH` | `reference/medkcoop.py` | 同上，论文日志使用的类名 |
| `MeDKCoOpFix` | `fixed/medkcoop.py` | 修复版，默认 `GRAD_CTX=True` |
| `MeDKCoOpFixedImpl` | `fixed/medkcoop.py` | 修复版实现体 |
| `GRAPH_DCPL_NO_TXT_VIS_GRAPH_FIX` | `fixed/medkcoop.py` | 修复版的论文类名别名 |
| `GRAPH_DCPL_*`（14 个） | `ablation/*.py` | 论文 Table 4/5、Fig 4 |

## 4. 论文表格 ↔ 代码

| 论文位置 | trainer 类 | 文件 |
|---|---|---|
| Table 1 平均 / Table 4 Ours | `GRAPH_DCPL_NO_TXT_VIS_GRAPH` | `reference/medkcoop.py`（源码曾丢失，已按字节码重建为同名类） |
| Table 5 (✗✗✗) | `GRAPH_DCPL_NO_TXT_VIS_FUSE` | `ablation/..._no_text_vis_fuse.py` |
| Table 5 (✗✓✗) | `GRAPH_DCPL_NO_VIS_FUSE_GRAPH` | `ablation/..._no_vis_fuse_graph.py` |
| Table 5 (✓✗✓) | `GRAPH_DCPL_NO_VIS_GRAPH` | `ablation/..._no_vis_graph.py` |
| Table 5 (✗✗✓) | `GRAPH_DCPL_NO_TXT_VIS_GRAPH1` | `ablation/..._no_text_vis_graph1.py` |
| Table 5 其余 3 行 | 待对齐 | `..._no_text_graph.py`、`..._no_text_fuse_graph.py`、`..._no_fuse.py` |
| Table 4 w/o VDK / LDK / VK | `NO_VIS_BIAS` / `NO_TXT_BIAS` / `NO_MEDSAM` | `ablation/..._no_vis_bias.py` 等 |
| Table 4 文本干预 | `GENERAL` / `INCORRECT` / `OUTOFDOMAIN` | `ablation/..._general.py` 等 |
| Fig 4 backbone | `GRAPH_DCPL_CLIP` / `_PUBMEDCLIP` / `_PMCCLIP` | `ablation/..._clip.py` 等 |
| Table 1/2/3 对比方法 | 8 个 baseline | `trainers/baselines/<方法>/` |

## 5. 修复版开关（`fixed/medkcoop.py`）

| 配置项 | 默认 | 作用 |
|---|---|---|
| `TRAINER.GRAPHDCPL.GRAD_CTX` | **True** | 让 learnable context 参与梯度（原实现被 `no_grad` 切断） |
| `TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET` | False | 把 `visual_net` 纳入 optimizer（原实现有梯度但不更新） |
| `TRAINER.GRAPHDCPL.LEARN_BETA` | False | `beta` 改为可学习 sigmoid 标量（原为固定 0.3） |
| `TRAINER.GRAPHDCPL.GRAD_IMGPROTO` | False | 图像原型去掉 `.data`，参与梯度 |
| `TRAINER.GRAPHDCPL.TRANSFER_EVAL` | False | cross-data 时丢弃源域分类头（实测为 no-op） |

环境变量：`MEDSAM_CACHE`（特征缓存）、`MEDSAM_CHUNK`（分块大小，默认 2）、
`EVAL_BS`（评测 batch）、`DG_STRICT`（v2 数据集是否使用正常 train/test 划分，默认 1）。

细节见 [FIXES.md](FIXES.md)。

## 6. 已删除（未使用的历史内容）

| 类别 | 内容 | 可恢复位置 |
|---|---|---|
| 非论文 baseline | ClipAdapter、DPLQ、LP、LP2、ProGrad、TipAdapter | git 历史 / 备份 tar |
| 旧 few-shot 管线 | `main.py`、根 `utils.py`、`utils/`、`datasets/utils.py`、`trainers/{__init__,method,utils}.py` | 同上 |
| 非论文数据集 | BTMRI、BTMRIv2、BTMRI1v2、BUSI、CCBTM*、DermaMNIST*、KneeXray* | 同上 |
| 消融冗余 | 34→16 之外的 18 个 `.py`、`hyper/` | 同上 |
| 可视化/试跑 | `heatmap*.py`、`test_blip*.py`、`interpret_prompt.py`、`combine_acc_columns.py` | 同上 |
| 其他支线 | `qnn/`、`descriptors/`、上游 assets | 同上 |
| 重复/旧数据 | `trainers copy/`、`data/` 旧 split、`MedSAM/assets/`、`BLIP/`、`clip1/`（已并入 clip） | 备份 `repo_archived/` |

## 7. 当前状态

已完成：

1. `train.py` 只导入 15 个数据集模块与 68 个 trainer；
2. `bootstrap.py` 完成 `sys.path` 引导 + 离线 BiomedCLIP 加载（另提供
   `load_biomedclip()` / `load_biomedclip_tokenizer()` 显式 API）；
3. DASSL 已在 `conda medkcoop` 中重新 editable 安装；
4. 配置统一命名，脚本参数化（`DATA/CKPT/GPU/SHOTS/EPOCH/EXTRA_OPTS/RUN_TAG`）；
5. `--heatmap` 缺 hook 时给出明确错误；DASSL 内 `test1()`/`pdb` 已删除；
6. `datasets/*v2.py` 的训练/测试泄漏已修（`DG_STRICT`）；
7. MedSAM 特征缓存（原子写、容错读）；
8. 6 个冒烟测试通过。

仍需处理：

1. 论文 Table 5 剩余 3 行的消融对应关系待逐一对齐；
2. `scripts/baselines/*` 中部分脚本的配置指向仍是原项目风格（可运行但未逐一验证）。
