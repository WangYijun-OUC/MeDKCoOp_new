# 目录结构说明与迁移映射

> 结构原则：**只保留论文实际会用到的文件**；每一层都有单一职责，没有 `legacy/` 之类的兜底目录。
> 删除的内容都已在 git 基线提交 `5e7ecc2` 与 `/home/wangyijun/backup/BiomedCoOp_2026-11-05/` 中保留。
> 原项目 `/home/wangyijun/BiomedCoOp` 未做任何修改。

## 1. 最终结构

```
MeDKCoOp/
├── train.py
├── configs/
│   ├── datasets/       # 8 个 v1：BTMRI1 CHMNIST covid ctkidney kvasir lungcolon octmnist retina
│   │                   # 7 个 v2 ：CHMNIST covid ctkidney kvasir lungcolon octmnist retina
│   └── trainers/       # MeDKCoOp DCPL BiomedCoOp CoOp CoCoOp KgCoOp MaPLe GraphCLIP Zeroshot
├── datasets/           # 与 configs/datasets 一一对应（8 v1 + 7 v2）
├── trainers/
│   ├── medkcoop/
│   │   ├── reference/medkcoop.py            # 参考实现（原 GRAPH_DCPL/medkcoop.py）
│   │   ├── reference/paper_bytecode/        # 论文最终方法仅存的 .pyc
│   │   └── ablation/                        # 16 个论文消融变体
│   └── baselines/      # BiomedCoOp CoOp CoCoOp KgCoOp MaPLe DCPL GraphAdapter Zeroshot
├── scripts/
│   ├── medkcoop/       # base2new / cross_data / domain_genera / few_shot / 消融
│   └── baselines/      # 8 个 baseline 的脚本
├── third_party/        # Dassl.pytorch open_clip clip clip1 MedSAM
├── tools/parse_test_res.py
└── docs/
```

## 2. 语义边界（避免混用）

| 目录 | 只负责 | 不负责 |
|---|---|---|
| `configs/datasets/` | 数据集名 + split json 的入口（`DATASET.NAME`） | 不包含数据本身 |
| `datasets/` | 读取 split、few-shot 采样、base/new 划分 | 不定义模型 |
| `trainers/medkcoop/reference/` | 论文方法的主实现与重建依据 | 不放消融 |
| `trainers/medkcoop/ablation/` | 论文 Table 4/5、Fig 4 的 16 个变体 | 不放 baseline |
| `trainers/baselines/` | 论文对比方法 | 不放本文方法 |
| `scripts/` | 批量启动器（参数组合） | 不含模型/数据逻辑 |
| `third_party/` | 外部代码，不修改 | 不放本文实验代码 |
| `tools/` | 结果汇总等一次性工具 | 不参与训练 |

## 3. 论文表格 ↔ 代码文件

| 论文位置 | trainer 类 | 文件 |
|---|---|---|
| Table 1 平均 / Table 4 Ours | `GRAPH_DCPL_NO_TXT_VIS_GRAPH` | **源码丢失** → `reference/paper_bytecode/vip_graph_dcpl_biomedclip_adapter_no_text_vis_graph.cpython-310.pyc` |
| Table 5 (✗✗✗) | `GRAPH_DCPL_NO_TXT_VIS_FUSE` | `ablation/vip_graph_dcpl_biomedclip_adapter_no_text_vis_fuse.py` |
| Table 5 (✗✓✗) | `GRAPH_DCPL_NO_VIS_FUSE_GRAPH` | `ablation/..._no_vis_fuse_graph.py` |
| Table 5 (✓✗✓) | `GRAPH_DCPL_NO_VIS_GRAPH` | `ablation/..._no_vis_graph.py` |
| Table 5 (✗✗✓) | `GRAPH_DCPL_NO_TXT_VIS_GRAPH1` | `ablation/..._no_text_vis_graph1.py` |
| Table 5 其余 3 行 | 待对齐 | 候选：`..._no_text_graph.py`、`..._no_text_fuse_graph.py`、`..._no_fuse.py` |
| Table 4 w/o VDK / w/o LDK / w/o VK | `GRAPH_DCPL_NO_VIS_BIAS` / `NO_TXT_BIAS` / `NO_MEDSAM` | `ablation/..._no_vis_bias.py`、`..._no_txt_bias.py`、`..._no_medsam.py` |
| Table 4 文本干预 General / Incorrect / Out-of-domain | `GRAPH_DCPL_GENERAL` / `_INCORRECT` / `_OUTOFDOMAIN` | `ablation/..._general.py`、`..._incorrect.py`、`..._outofdomain.py` |
| Fig 4 backbone：CLIP / PubMedCLIP / PMC-CLIP | `GRAPH_DCPL_CLIP` / `_PUBMEDCLIP` / `_PMCCLIP` | `ablation/..._clip.py`、`..._pubmedclip.py`、`..._pmcclip.py` |
| Table 1/2/3 对比方法 | 8 个 baseline 类 | `trainers/baselines/<方法>/` |

## 4. 从旧路径到新路径

| 旧路径 | 新路径 |
|---|---|
| `trainers/GRAPH_DCPL/medkcoop.py` | `trainers/medkcoop/reference/medkcoop.py` |
| `trainers/GRAPH_DCPL/<消融>.py`（保留的 16 个） | `trainers/medkcoop/ablation/` |
| `trainers/GRAPH_DCPL/__pycache__/vip_..._no_text_vis_graph.pyc` | `trainers/medkcoop/reference/paper_bytecode/` |
| `trainers/{BiomedCoOp,CoOp,CoCoOp,KgCoOp,MaPLe,DCPL,GraphAdapter,Zeroshot}/` | `trainers/baselines/<同名>/` |
| `scripts/graph_dcpl/` | `scripts/medkcoop/` |
| `scripts/<baseline>/` | `scripts/baselines/<baseline>/` |
| `Dassl.pytorch/`、`open_clip/`、`clip/`、`clip1/`、`MedSAM/` | `third_party/{Dassl.pytorch,open_clip,clip,MedSAM}/`（clip 与 clip1 已合并） |
| `parse_test_res.py` | `tools/parse_test_res.py` |
| `REPRO_PLAN.md` | `docs/REPRO_PLAN.md` |
| 根 `README.md`（上游） | 已删除（见备份） |

## 5. 已删除（未使用）

| 类别 | 内容 | 可恢复位置 |
|---|---|---|
| 非论文 baseline | `ClipAdapter`、`DPLQ`、`LP`、`LP2`、`ProGrad`、`TipAdapter`（trainers + configs + scripts） | `git show 5e7ecc2:<path>` / 备份 tar |
| 旧 few-shot 管线 | `main.py`、根 `utils.py`、`utils/`、`datasets/utils.py`、`datasets/__init__.py`、`trainers/{__init__,method,utils}.py` | 同上 |
| 非论文数据集 | `BTMRI`、`BTMRIv2`、`BTMRI1v2`、`BUSI`、`CCBTM*`、`DermaMNIST*`、`KneeXray*`（datasets + configs） | 同上 |
| 消融冗余 | 34→16 之外的 18 个 `.py`、`hyper/`、`legacy/`（旧 MeDKCoOp 副本、rebuttal、BLIP 版） | 同上 |
| 可视化/试跑 | `heatmap*.py`、`test_blip*.py`、`interpret_prompt.py`、`combine_acc_columns.py`、`cal.sh` | `repo_archived/` 或 git 历史 |
| 其他支线 | `qnn/`、`descriptors/`、上游 `assets/`、上游 README | 同上 |
| 重复/旧数据 | `trainers copy/`、`data/` 旧 split、`MedSAM/assets/`、`BLIP/BLIP.gif` | `/home/wangyijun/backup/BiomedCoOp_2026-11-05/repo_archived/` |

> 删除理由：论文正文只用到 8 个数据集、8 个对比方法与 16 个消融变体；其余是探索过程，
> 保留会让"哪份代码产生论文哪张表"变得不可判定。

## 6. 代码适配状态

已在本次完成：

1. `train.py` 只导入保留下来的 15 个数据集模块与 51 个 trainer；
2. 新增 `bootstrap.py`：把 `third_party`、`third_party/open_clip` 加入 `sys.path`，
   并把 `hf-hub` 的 BiomedCLIP 加载重定向到 `/data/wangyijun/model/BiomedCLIP`
   （离线可用，`HF_HUB_OFFLINE=1` 已验证）；
3. `trainers/baselines/Zeroshot` 等跨目录 import 已改为 `trainers.baselines.<X>`；
4. DASSL 已在 `conda medkcoop` 环境中重新 `pip install -e third_party/Dassl.pytorch`；
5. 配置目录统一为 `configs/trainers/medkcoop/`，并删除已移除数据集的残留配置；
6. `scripts/medkcoop/base2new.sh` 已参数化（`DATA/CKPT/GPU/SHOTS/EPOCH`），训练+评测跑通；
7. `--heatmap` 在缺少 `heatmap_hook()` 时给出明确错误；
8. 恢复论文字节码中 3 行 RNG 消耗代码（`mean`/`noise` 未被使用），使随机序列与原实现一致。

仍需处理：

1. `scripts/medkcoop/` 其余脚本与 `scripts/baselines/*` 仍含旧 ckpt 路径与 `Device_id`，
   改写模板同 `base2new.sh`；
2. cross-dataset / domain generalization 的旧产物不在磁盘，需要按论文设置重跑；
3. 论文 Table 5 剩余 3 行消融与候选文件的对应关系待逐一对齐。

验证记录见 [ENVIRONMENT.md](ENVIRONMENT.md)。
