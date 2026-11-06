# 环境与运行

## 1. conda 环境

```bash
# 已在本机创建（从 dcpl 克隆，torch 1.12.0+cu113）
conda activate medkcoop

# 关键依赖版本
python 3.10.16 / torch 1.12.0+cu113 / torchvision 0.13.0+cu113
transformers 4.35.2 / open-clip-torch 2.23.0 / timm 1.0.15
dassl 0.5.0（editable，指向本仓库 third_party/Dassl.pytorch）
```

如果在新机器重建：

```bash
conda create -n medkcoop python=3.10.16 -y
conda activate medkcoop
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e third_party/Dassl.pytorch
```

> **必须重新 `pip install -e third_party/Dassl.pytorch`**：原项目环境里的 editable 安装指向
> `/home/wangyijun/BiomedCoOp`，不重装会 import 到旧代码。

## 2. 权重与数据（都不在仓库内）

| 资源 | 默认路径 | 覆盖变量 |
|---|---|---|
| 数据集 | `/data/wangyijun/dataset` | `DATA` |
| BiomedCLIP | `/data/wangyijun/model/BiomedCLIP`（含 `open_clip_pytorch_model.bin`、`BiomedBERT/`、tokenizer） | `BIOMEDCLIP_DIR` |
| MedSAM ViT-B | `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` | `MEDSAM_CKPT` |
| PubMedCLIP | `/data/wangyijun/model/PubMedCLIP/PubMedCLIP_ViT32.pth` | `PUBMEDCLIP_DIR` |
| PMC-CLIP | `/data/wangyijun/model/PubMed/` | `PMCCLIP_DIR` |
| PMC-CLIP 文本骨干 | `/data/wangyijun/model/BiomedCLIP/BiomedBERT` | `PMCCLIP_BERT_DIR` |
| 输出（日志/模型） | `/data/wangyijun/ckpt/MeDKCoOp` | `CKPT` |

`bootstrap.py` 会把 `hf-hub:microsoft/BiomedCLIP-...` 的加载重定向到本地权重，
因此无需联网；可用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制离线验证。
PubMedCLIP 与 PMC-CLIP trainer 同样默认读取上表的本地路径，不再依赖仓库相对路径。

需要更强的重复性时设置 `DETERMINISTIC=1`，或直接给 `train.py` 传
`--deterministic`。这会关闭 cuDNN benchmark 并启用 deterministic 模式；论文原始结果仍以
默认的 benchmark 模式为准。

## 3. 跑通一个实验（BTMRI1, 16-shot base2new, seed 1）

```bash
conda activate medkcoop
cd /home/wangyijun/MeDKCoOp

python train.py \
  --root /data/wangyijun/dataset --seed 1 --trainer MeDKCoOp \
  --dataset-config-file configs/datasets/BTMRI1.yaml \
  --config-file configs/trainers/medkcoop/base_to_novel/BTMRI1.yaml \
  --output-dir /data/wangyijun/ckpt/MeDKCoOp/base2new/BTMRI1/shots_16/MeDKCoOp/demo/train_base \
  --mode train \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base

python train.py \
  --root /data/wangyijun/dataset --seed 1 --trainer MeDKCoOp \
  --dataset-config-file configs/datasets/BTMRI1.yaml \
  --config-file configs/trainers/medkcoop/base_to_novel/BTMRI1.yaml \
  --output-dir /data/wangyijun/ckpt/MeDKCoOp/base2new/BTMRI1/shots_16/MeDKCoOp/demo/test_new \
  --model-dir /data/wangyijun/ckpt/MeDKCoOp/base2new/BTMRI1/shots_16/MeDKCoOp/demo/train_base \
  --load-epoch 50 --mode test --eval-only \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new
```

或者用封装脚本（训练+评测，支持断点跳过）：

```bash
GPU=0 SHOTS=16 EPOCH=50 bash scripts/medkcoop/base2new.sh BTMRI1 1
```

结果：

```bash
grep -E "\* accuracy:|\* total:" \
  /data/wangyijun/ckpt/MeDKCoOp/base2new/BTMRI1/shots_16/MeDKCoOp/vit_b16_0002_b50_randflip/seed1/*/log.txt
```

## 4. 用已有 checkpoint 复评（不训练）

```bash
python train.py \
  --root /data/wangyijun/dataset --seed 1 --trainer MeDKCoOp \
  --dataset-config-file configs/datasets/BTMRI1.yaml \
  --config-file configs/trainers/medkcoop/base_to_novel/BTMRI1.yaml \
  --output-dir /tmp/eval_btmri1 \
  --model-dir /data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/BTMRI1/shots_16/GRAPH_DCPL_NO_TXT_VIS_GRAPH/vit_b16_0002_b50_randflip/seed1 \
  --load-epoch 50 --mode test --eval-only \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new
```

## 5. 与论文对照

| 数据集 | 论文 base | 论文 new | 说明 |
|---|---|---|---|
| BTMRI1 | 85.57 | 92.10 | 3 seed 均值；seed1 原始日志 new=93.50 |
| CHMNIST | 93.67 | 60.17 | |
| Covid-19 | 78.97 | 91.07 | |
| CTKIDNEY | 90.70 | 84.83 | |
| LC25000 (lungcolon) | 94.27 | 96.27 | |
| RETINA | 88.60 | 58.50 | |
| OCTMNIST | 83.27 | 85.83 | |
| Kvasir | 88.00 | 70.83 | |

`trainers/medkcoop/reference/medkcoop.py` 与论文最终版（丢失源码，`GRAPH_DCPL_NO_TXT_VIS_GRAPH`）
在实现上等价（字节码比对仅有 3 行**未被使用**的 `noise` 计算、辅助函数改名、一句 print）。

### 本次实测（conda `medkcoop`，BTMRI1，16-shot，seed 1，50 epoch）

| 指标 | 本次重训 | 原始 seed1 日志 | 论文 3-seed 均值 |
|---|---|---|---|
| Base | **88.91%** | 88.90% | 85.57% |
| New | **92.20%** | 93.50% | 92.10% |

- Base 与原始 seed1 一致（1019 张里差 1 张），说明数据划分、模型与训练流程等价；
- New 与原始 seed1 差 1.3 pp、与论文报告的 3-seed 均值差 0.1 pp，差异来自
  `cudnn.benchmark=True` 导致的非确定性，以及 new 类原型对权重微扰更敏感；
- 另一条独立验证：用原始 checkpoint 做 eval-only，本实现得到 93.71%，原始日志 93.50%。

复现命令（`docs/ENVIRONMENT.md` 第 3 节）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/medkcoop/base2new.sh BTMRI1 1
# 输出: /data/wangyijun/ckpt/MeDKCoOp/base2new/BTMRI1/shots_16/MeDKCoOp/vit_b16_0002_b50_randflip/seed1/
```

## 6. 已知的暂时限制

- 只有 `configs/trainers/medkcoop/` + `scripts/medkcoop/base2new.sh` 完成了路径与新结构适配；
  其余脚本仍含旧的 ckpt 硬编码路径与 `Device_id`，改写方式同 `base2new.sh`。
- cross-dataset / domain generalization 的旧产物不在磁盘上（见 `docs/STRUCTURE.md`），需要重跑。
- 热图脚本已移出仓库，`--heatmap` 会给出明确提示。
