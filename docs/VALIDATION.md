# 运行与脚本验证记录

本文只记录已经实际执行或静态检查过的范围，不把单次 smoke test 表述为完整论文复现。

## MeDKCoOp cross-data 端到端验证

环境：conda `medkcoop`、单张 RTX 4090、代码提交 `d1ed23b`，数据根目录
`/data/wangyijun/dataset`。

执行命令：

```bash
GPU=5 bash scripts/medkcoop/xd_train.sh BTMRI1 1
GPU=5 bash scripts/medkcoop/xd_test.sh CHMNIST 1 BTMRI1
```

结果：

| 阶段 | 设置 | 结果 |
|---|---|---:|
| source | BTMRI1，16-shot，seed 1，100 epoch | accuracy 86.73%，macro-F1 86.29% |
| target | CHMNIST，加载 source 的 epoch-100 checkpoint | accuracy 23.20%，macro-F1 23.13% |

source checkpoint 位于：

```text
/data/wangyijun/ckpt/MeDKCoOp/cross_data/source/BTMRI1/shots_16/GRAPH_DCPL_NO_TXT_VIS_GRAPH/seed1/MultiModalPromptLearner/model.pth.tar-100
```

目标日志明确记录从上述路径加载 epoch 100，说明独立的 `xd_train.sh` → `xd_test.sh` 权重链路、
trainer 旧注册名和跨类别 checkpoint 加载均可工作。23.20% 只是本次单 seed、单目标的观测值，
不能替代论文 Table 2 的 3-seed 汇总，也不用于宣称数值已经复现。

## 其他实际运行

- CoOp 独立 `few_shot.sh`：BTMRI1、1-shot、seed 1、1 epoch，accuracy 56.12%。
- ZeroshotBiomedCLIP 独立评测入口：BTMRI1，accuracy 56.12%。
- PubMedCLIP 与 PMC-CLIP 本地 checkpoint 已分别完成反序列化检查。

## 脚本结构验收

- 每个可训练方法目录都包含独立的 `base2new.sh`、`few_shot.sh`、`xd_train.sh`、
  `xd_test.sh`、`cross_data_eval.sh` 和 `domain_genera.sh`。
- Zeroshot 不训练 source checkpoint，因此提供四类任务入口和独立 `xd_test.sh`，不伪造
  `xd_train.sh`。
- 各方法的 base2new、few-shot、`xd_train`、`xd_test` 直接调用 `train.py`，没有符号链接，
  也不 source 统一 runner；cross-data / DG 批量入口只调用同方法的 `xd_*`。
- `scripts/run_all_methods.sh` 只逐个调用各方法的公开入口。
- 全部 shell 脚本通过 `bash -n`；Python 文件通过 `compileall`（此前提交验证）。

## 尚未完成的验证

- cross-data 的全部 7 个目标、3 seeds 和全部 baseline；
- domain generalization 全表；
- few-shot 全数据集、全 shots、3 seeds；
- 修正 `visual_net` optimizer 归属后的独立对照实验。

另外，`*v2` 数据集当前把同一 split 同时交给 `train_x` 与 `test`，而 MeDKCoOp 会用
`train_x` 构建视觉知识图。完成 Table 3 数值重跑不等于消除了该实验语义风险。
