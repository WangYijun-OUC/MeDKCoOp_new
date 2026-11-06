# 实验脚本

每个方法目录都包含完整的独立运行脚本，脚本内直接调用 `train.py`，不依赖统一执行器。
额外的 `scripts/run_all_methods.sh` 只用于维护者批量调度。

## 方法目录

- 本文方法：`scripts/medkcoop/`
- Baseline：`scripts/baselines/{biomedcoop,coop,cocoop,kgcoop,MaPLe,dcpl,graph_adapter,zeroshot}/`

每个可训练 baseline 目录提供：

- `base2new.sh DATASET SEED`
- `few_shot.sh DATASET SEED SHOTS`
- `cross_data_eval.sh`：BTMRI1 训练，其他 7 个数据集评测，3 seeds
- `domain_genera.sh`：BTMRI1 训练，7 个 v2 目标数据集评测，3 seeds
- `xd_train.sh SOURCE SEED` 与 `xd_test.sh TARGET SEED SOURCE`：单次运行和排错

例如：

~~~bash
GPU=0 bash scripts/baselines/coop/base2new.sh BTMRI1 1
GPU=0 bash scripts/baselines/biomedcoop/few_shot.sh CHMNIST 2 4
GPU=0 bash scripts/baselines/MaPLe/xd_train.sh BTMRI1 1
GPU=0 bash scripts/baselines/MaPLe/xd_test.sh CHMNIST 1 BTMRI1
~~~

MeDKCoOp 对应入口：

~~~bash
GPU=0 bash scripts/medkcoop/base2new.sh BTMRI1 1
GPU=0 bash scripts/medkcoop/few_shot.sh BTMRI1 1 16
GPU=0 bash scripts/medkcoop/cross_data_eval.sh
GPU=0 bash scripts/medkcoop/domain_genera.sh
~~~

## 默认方法实现

| 目录 | Trainer | base2new epoch | few-shot / transfer epoch |
|---|---|---:|---:|
| biomedcoop | BiomedCoOp_BiomedCLIP | 50 | 100 |
| coop | CoOp_BiomedCLIP | 50 | 100 |
| cocoop | CoCoOp_BiomedCLIP | 50 | 100 |
| kgcoop | KgCoOp_BiomedCLIP | 50 | 100 |
| MaPLe | MaPLe_CLIP | 5 | 100 |
| dcpl | DCPL_biomedclip | 50 | 100 |
| graph_adapter | GraphCLIP | 50 | 100 |
| zeroshot | ZeroshotBiomedCLIP | 无训练 | 无训练 |

MaPLe 的 base2new 默认 5 epoch 来自现存实验配置；transfer 配置为 100 epoch。
`EPOCH` 可覆盖默认值。通用变量包括 `DATA`、`CKPT`、`GPU`、`SHOTS`、
`EVAL_BS`、`PYTHON_BIN`、`SOURCE` 和 `MODEL_DIR`。

默认优先使用 `/data/wangyijun/conda/medkcoop/bin/python`，不存在时使用当前环境的
`python`。输出默认写到 `/data/wangyijun/ckpt/MeDKCoOp/<method>/...`。
脚本只有在 checkpoint 与所需指标日志同时存在时才跳过；残缺产物会触发错误并停止。

## 统一调度

统一脚本调用各方法自己的公开入口：

~~~bash
GPU=0 bash scripts/run_all_methods.sh base2new BTMRI1 1
GPU=0 bash scripts/run_all_methods.sh few_shot BTMRI1 1 16
~~~

完整 cross-data 和 domain-generalization 会依次运行大量实验，建议按方法使用对应目录的脚本，
或自行分配不同的空闲 GPU。

## 消融

- `base2new_abla.sh`：Table 4 知识来源和文本干预消融
- `base2new_fuse.sh`：Table 5 组件组合消融
- `base2new_clip.sh`：Fig. 4 backbone 消融；PubMedCLIP/PMC-CLIP 需要额外本地权重

## 领域泛化解释限制

当前 `*v2` 数据集类把同一份 split 同时交给 `train_x` 和 `test`。MeDKCoOp 会用
`train_x` 构建视觉知识图，因此 Table 3 的实验语义仍需结合论文原设置核实，不能直接表述为
完全不使用目标域标签。脚本已可运行，这一限制不等于数值已完成复现。
