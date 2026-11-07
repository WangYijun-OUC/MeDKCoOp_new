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
| few-shot 采样缓存 | 默认写在各数据集的 `split_fewshot/` | `FEWSHOT_CACHE`（除 BTMRI1） |

`bootstrap.py` 会把 `hf-hub:microsoft/BiomedCLIP-...` 的加载重定向到本地权重，
因此无需联网；可用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制离线验证。
PubMedCLIP 与 PMC-CLIP trainer 同样默认读取上表的本地路径，不再依赖仓库相对路径。
对其余七个数据集，如需保持数据目录只读，可设置例如
`FEWSHOT_CACHE=/data/wangyijun/ckpt/fewshot_cache`；缓存会再按数据集名分目录，避免不同
数据集的 shot/seed 文件互相覆盖。BTMRI1 为保持现有 CRLF 源文件和历史缓存路径暂未迁移。

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

## 5. 复现结果

完整结果（base2new / cross-data / DG / 修复对照 / 算法改进）见 [RESULTS.md](RESULTS.md)。
这里只保留一条快速自检基线：

| 指标 | 3 seed 均值 | 论文 Table 1 |
|---|---|---|
| BTMRI1 Base | 85.87 | 85.57 |
| BTMRI1 New | 91.61 | 92.10 |
| 8 数据集平均 Base | 87.50 | 87.88 |
| 8 数据集平均 New | 79.75 | 79.95 |

`trainers/medkcoop/reference/medkcoop.py` 与论文最终版（源码丢失，类名
`GRAPH_DCPL_NO_TXT_VIS_GRAPH`）在实现上等价：字节码比对只差 3 行**未被使用**的
`noise` 计算、辅助函数改名、一句 print。

## 6. 已知的暂时限制

- MeDKCoOp 与 8 个 baseline 的 base2new、few-shot、cross-data、domain-generalization 入口均已
  完成新目录适配；各方法的单次训练/评测脚本直接调用 `train.py`，不依赖统一 runner。
  cross-data / DG 批量入口只调用同方法的单次脚本，统一跨方法脚本只做额外批量调度。
- cross-dataset 已完成一条 MeDKCoOp 的 BTMRI1 seed1 → CHMNIST 端到端验证，但完整的 3-seed、
  7-target 结果以及 domain generalization 仍需重跑，不能把单条验证当作论文 Table 2/3 复现。
- `*v2` 数据集把同一 split 同时用于 `train_x` 与 `test`；MeDKCoOp 构建目标视觉图时会读取
  `train_x`，因此 domain-generalization 的严格实验语义仍需核实。
- 最终 trainer 中 `visual_net` 虽被标为可训练，但未加入当前 optimizer；修正它会改变既有
  checkpoint 的训练语义，应作为 corrected implementation 另做对照，而不是静默改动。
- 热图脚本已移出仓库，`--heatmap` 会给出明确提示。

实际运行记录见 [RESULTS.md](RESULTS.md)；历史验证记录在 [archive/VALIDATION_2026-11-06.md](archive/VALIDATION_2026-11-06.md)。

## 7. 已知限制：eval-only 不能复现训练时的 base 指标

`--mode train|test` 同时决定"训练/评测"和 "fc / new_fc" 分支：
- train 模式构造 `fc`（源域类原型），test 模式构造 `new_fc`（目标域类原型）；
- 因此用 `--mode train --eval-only` 复评 base 时，模型里没有训练得到的 `fc`，
  实测只有 53.65%（论文训练日志是 80.5%）。

**结论：base 指标只能由完整训练流程产出，不能用 eval-only 复评。**
cross-data 的目标域评测走 test 分支、用目标集重建原型，属于设计行为，不是 bug。

## 8. 显式离线加载

`bootstrap.py` 提供：

```python
import bootstrap
model, preprocess = bootstrap.load_biomedclip()
tokenizer = bootstrap.load_biomedclip_tokenizer()
```

这两个函数不依赖全局 monkeypatch，可直接在任意脚本中使用；
`import bootstrap` 时仍会自动 patch `open_clip`，保证现有 trainer 无需修改。
