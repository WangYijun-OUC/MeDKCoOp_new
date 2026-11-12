# MeDKCoOp

**Dual Knowledge-guided Graph Prompt Learning for Biomedical Vision-Language Models**

*Accepted at ACM Multimedia 2025*

[[Paper](#citation)] [[Results](#results)] [[Installation](#installation)] [[Data](#data-preparation)] [[Training](#training-and-evaluation)] [[Citation](#citation)]

---

## Overview

Biomedical vision-language models such as BiomedCLIP align medical images with clinical text,
but hand-crafted prompts (*"a photo of a {CLASS}"*) cannot capture the fine-grained, modality-specific
visual patterns required for reliable clinical diagnosis. Existing prompt-learning methods adapt such
models with a small set of learnable context vectors, yet they treat medical knowledge as a static prior
rather than a relational structure, which limits transfer to unseen classes and domains.

**MeDKCoOp** integrates biomedical knowledge into prompt learning through two modules:

1. **Knowledge-guided Relation Graph Construction (KRGC)** — builds a *textual* knowledge graph from
   LLM-generated clinical descriptions and a *visual* knowledge graph from MedSAM image features.
   Class embeddings are nodes; cosine similarity between class prototypes defines edges.
2. **Knowledge-guided Relation Transfer Learning (KRTL)** — propagates the textual relational structure
   into the visual embedding space with a Graph Convolutional Network, producing a fused feature that
   guides the learnable prompts. The final representation mixes the text prototype and the fused graph
   feature with a residual weight.

The method is evaluated on **8 biomedical datasets** spanning 7 imaging modalities, covering
base-to-new generalization, cross-dataset transfer, and domain generalization.

```
                 ┌─────────────────────────┐
  medical image ─┤ BiomedCLIP image encoder ├─┐
                 └─────────────────────────┘ │
                 ┌─────────────────────────┐ │   ┌──────────────┐
            ─────┤ MedSAM image encoder     ├─┼──►│ KRGC         │
                 └─────────────────────────┘ │   │ text + visual│
  LLM knowledge ─┬───────────────────────── ┐│   │ knowledge    │
                 │ BiomedCLIP text encoder  ├┘   │ graphs       │
                 └───────────────────────── ┘    └──────┬───────┘
                                                       ▼
                                   ┌────────────────────────────┐
  learnable context ──────────────►│ KRTL: GCN relation transfer│──► prediction
                                   └────────────────────────────┘
```

## Supported Methods

| Method | Role | Trainer | Configs | Scripts |
|---|---|---|---|---|
| **MeDKCoOp** | proposed | `trainers/medkcoop/reference` | `configs/trainers/medkcoop` | `scripts/medkcoop` |
| BiomedCoOp | baseline | `trainers/baselines/BiomedCoOp` | `configs/trainers/BiomedCoOp` | `scripts/baselines/biomedcoop` |
| CoOp | baseline | `trainers/baselines/CoOp` | `configs/trainers/CoOp` | `scripts/baselines/coop` |
| CoCoOp | baseline | `trainers/baselines/CoCoOp` | `configs/trainers/CoCoOp` | `scripts/baselines/cocoop` |
| KgCoOp | baseline | `trainers/baselines/KgCoOp` | `configs/trainers/KgCoOp` | `scripts/baselines/kgcoop` |
| MaPLe | baseline | `trainers/baselines/MaPLe` | `configs/trainers/MaPLe` | `scripts/baselines/MaPLe` |
| DCPL | baseline | `trainers/baselines/DCPL` | `configs/trainers/DCPL` | `scripts/baselines/dcpl` |
| GraphAdapter | baseline | `trainers/baselines/GraphAdapter` | `configs/trainers/GraphCLIP` | `scripts/baselines/graph_adapter` |
| Zero-shot CLIP | baseline | `trainers/baselines/Zeroshot` | `configs/trainers/Zeroshot` | `scripts/baselines/zeroshot` |

All methods share one entry point (`train.py`) and can be selected by `--trainer <name>`.
The ablation variants of MeDKCoOp (used for the ablation tables and backbone comparison) are
registered from `trainers/medkcoop/ablation/`.

## Repository Structure

```
.
├── train.py                  # single training / evaluation entry point
├── bootstrap.py              # import-path setup and offline backbone loading
├── configs/
│   ├── datasets/             # 8 in-domain + 7 domain-generalization dataset configs
│   └── trainers/             # per-method hyper-parameters
├── datasets/                 # dataset readers (split json -> few-shot -> base/new)
├── trainers/
│   ├── medkcoop/
│   │   ├── reference/        # method implementation used for the reported numbers
│   │   ├── fixed/            # variant with gradient/caching options (see below)
│   │   └── ablation/         # ablation variants behind the paper's tables
│   └── baselines/            # the eight comparison methods
├── scripts/
│   ├── medkcoop/             # base2new / cross-dataset / domain-generalization / few-shot
│   └── baselines/            # same task scripts for each baseline
├── third_party/              # Dassl.pytorch, open_clip, clip, MedSAM
├── tools/                    # result aggregation and comparison utilities
└── tests/                    # smoke tests
```

## Installation

Python 3.10, PyTorch 1.12 (CUDA 11.3) is the configuration used in the paper.
A single 24 GB GPU is sufficient for all experiments.

```bash
git clone <this-repo>
cd MeDKCoOp

conda create -n medkcoop python=3.10 -y
conda activate medkcoop

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# the training framework is vendored and must be installed in editable mode
pip install -e third_party/Dassl.pytorch
```

Verify the setup:

```bash
python -m pytest tests -q      # 6 smoke tests
```

## Data Preparation

Place the datasets in one directory and point to it with `--root`:

| Dataset | Directory | Classes | Modality |
|---|---|---|---|
| BTMRI1 | `BTMRI/` | 4 | Brain MRI |
| CHMNIST | `CHMNIST/` | 8 | Histopathology |
| COVID-19 | `COVID_19/` | 4 | Chest X-ray |
| CTKidney | `CTKidney/` | 4 | CT |
| LungColon (LC25000) | `lungColon/` | 5 | Histopathology |
| RETINA | `RETINA/` | 4 | Fundus photography |
| OCTMNIST | `OCTMNIST/` | 4 | OCT |
| Kvasir | `Kvasir/` | 8 | Endoscopy |

Each directory contains a `split_<NAME>.json` with `train` / `val` / `test` entries as
`(relative_image_path, label, classname)`. The reader splits classes into
*base* and *new* halves for the base-to-new protocol, and samples `K` images per class for
few-shot training. See `configs/datasets/` for the expected dataset names.

## Pretrained Weights

All experiments use **BiomedCLIP** (ViT-B/16, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
as the default backbone; **MedSAM** (ViT-B) supplies the visual knowledge branch.
PubMedCLIP and PMC-CLIP weights are only needed for the backbone ablation.

Paths are resolved through environment variables so no machine-specific path is hard-coded:

| Variable | Meaning | Default |
|---|---|---|
| `DATA` | dataset root (`--root`) | `./data` |
| `BIOMEDCLIP_DIR` | BiomedCLIP files (`open_clip_pytorch_model.bin`, `open_clip_config.json`, `BiomedBERT/`, tokenizer) | required |
| `MEDSAM_CKPT` | MedSAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`) | required |
| `PMCCLIP_DIR` / `PMCCLIP_BERT_DIR` | PMC-CLIP encoders and text tokenizer | backbone ablation only |
| `CKPT` | output root for checkpoints and logs | `./output` |

`bootstrap.py` rewrites the `hf-hub:` model name to the local directory on import, so the code
works fully offline once the weights are downloaded. Two explicit helpers are also available:

```python
import bootstrap
model, preprocess = bootstrap.load_biomedclip()      # local weights, no network
tokenizer = bootstrap.load_biomedclip_tokenizer()
```

## Training and Evaluation

### Base-to-new generalization (Table 1)

Train on base classes, then evaluate the novel classes with the same checkpoint:

```bash
GPU=0 bash scripts/medkcoop/base2new.sh BTMRI1 1
```

The script runs two stages in one call:

```bash
# stage 1: train on base classes
python train.py --root $DATA --seed 1 --trainer MeDKCoOp \
    --dataset-config-file configs/datasets/BTMRI1.yaml \
    --config-file configs/trainers/medkcoop/base_to_novel/BTMRI1.yaml \
    --output-dir $CKPT/base2new/BTMRI1/train_base --mode train \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base

# stage 2: evaluate novel classes
python train.py --root $DATA --seed 1 --trainer MeDKCoOp \
    --dataset-config-file configs/datasets/BTMRI1.yaml \
    --config-file configs/trainers/medkcoop/base_to_novel/BTMRI1.yaml \
    --output-dir $CKPT/base2new/BTMRI1/test_new \
    --model-dir $CKPT/base2new/BTMRI1/train_base --load-epoch 50 \
    --mode test --eval-only \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new
```

### Cross-dataset transfer (Table 2)

Train on BTMRI1 for 100 epochs, then evaluate the target datasets directly:

```bash
GPU=0 bash scripts/medkcoop/xd_train.sh BTMRI1 1          # source
GPU=0 bash scripts/medkcoop/xd_test.sh CHMNIST 1 BTMRI1   # target
```

`scripts/medkcoop/cross_data_eval.sh` loops over all seven targets and three seeds.

### Domain generalization (Table 3)

Same source model, evaluated on the `*v2` dataset configurations, which expose a larger test set
for the same target modality:

```bash
bash scripts/medkcoop/domain_genera.sh
```

### Few-shot

```bash
GPU=0 bash scripts/medkcoop/few_shot.sh BTMRI1 1 16       # dataset, seed, shots
```

### Baselines

Every baseline directory exposes the same four entry points
(`base2new.sh`, `few_shot.sh`, `xd_train.sh`, `xd_test.sh`) and calls `train.py` directly:

```bash
GPU=0 bash scripts/baselines/biomedcoop/base2new.sh BTMRI1 1
GPU=0 bash scripts/baselines/coop/few_shot.sh CHMNIST 2 4
```

`scripts/run_all_methods.sh <task> [DATASET] [SEED]` iterates over all methods for convenience.

### Common options

All scripts accept environment variables so runs can be reproduced without editing files:

| Variable | Default | Meaning |
|---|---|---|
| `GPU` | `0` | CUDA device index |
| `SHOTS` | `16` | shots per class |
| `EPOCH` | `50` (base2new) / `100` (cross-dataset, few-shot) | training epochs |
| `EVAL_BS` | `16` | batch size at evaluation |
| `RUN_TAG` | empty | sub-directory tag to keep experiments separate |
| `EXTRA_OPTS` | empty | additional `train.py` overrides, e.g. `OPTIM.LR 0.001` |
| `MEDSAM_CACHE` | `0` | `1` enables MedSAM feature caching for evaluation |
| `DETERMINISTIC` | `0` | `1` disables cuDNN benchmark and selects deterministic kernels |

## Results

All numbers are accuracies (%) averaged over three seeds, 16 shots, BiomedCLIP ViT-B/16.
`Base`/`New` are the base- and novel-class halves of the base-to-new protocol, `HM` their
harmonic mean.

Three reference points are reported:

- **Paper** — values from Table 1 of the paper.
- **Reproduction** — the released implementation re-run in this repository.
- **Improved** — the variant described under [Modifications](#modifications).

### Base-to-new generalization

| Dataset | Paper Base | Repro Base | Improved Base | Paper New | Repro New | Improved New | Paper HM | Repro HM | Improved HM |
|---|---|---|---|---|---|---|---|---|---|
| BTMRI1 | 85.57 | 85.87 | 85.87 | 92.10 | 91.61 | **94.18** | 87.71 | 88.64 | 89.83 |
| CHMNIST | 93.67 | 93.44 | 93.79 | 60.17 | 60.11 | **62.77** | 73.27 | 73.16 | 75.21 |
| Covid-19 | 78.97 | 78.58 | 78.98 | 91.07 | 91.15 | 91.20 | 84.59 | 84.40 | 84.65 |
| CTKIDNEY | 90.70 | 90.66 | 89.02 | 84.83 | 84.65 | 85.14 | 87.67 | 87.55 | 87.04 |
| LC25000 | 94.27 | 93.64 | 93.41 | 96.27 | 95.47 | 95.60 | 95.26 | 94.55 | 94.49 |
| RETINA | 88.60 | 89.48 | 90.52 | 58.50 | 58.38 | **60.85** | 70.47 | 70.66 | 72.77 |
| OCTMNIST | 83.27 | 80.73 | 82.72 | 85.83 | 85.88 | 85.75 | 84.53 | 83.23 | 84.21 |
| Kvasir | 88.00 | 87.61 | 87.56 | 70.83 | 70.72 | 71.28 | 78.49 | 78.27 | 78.58 |
| **Average** | **87.88** | **87.50** | **87.73** | **79.95** | **79.75** | **80.85** | **82.75** | **82.56** | **83.35** |

The reproduction is within 0.38 points of the paper on the base half and 0.20 on the novel half.
The improved variant adds **+1.10 points on novel classes** and +0.79 HM over the reproduction,
with the largest gains on BTMRI1 (+2.57), CHMNIST (+2.66) and RETINA (+2.46).

### Cross-dataset transfer and domain generalization

Source domain: BTMRI1, 100 epochs, 16 shots.

| Target | Paper | Repro (cross-dataset) | Paper | Repro (domain gen.) |
|---|---|---|---|---|
| CHMNIST | 32.60 | 27.02 | 32.33 | 26.69 |
| Covid-19 | 63.00 | 61.78 | 63.37 | 62.08 |
| CTKIDNEY | 60.80 | 60.25 | 60.77 | 60.15 |
| LC25000 | 63.27 | 62.91 | 63.07 | 62.98 |
| RETINA | 37.70 | 37.79 | 37.53 | 37.29 |
| OCTMNIST | 64.30 | 61.17 | 64.33 | 62.49 |
| Kvasir | 58.77 | 58.58 | 57.77 | 59.11 |
| **Average** | **54.35** | **52.78** | **54.17** | **52.97** |

Five of the seven targets are within 1.3 points of the paper. CHMNIST is consistently about
5.6 points lower; its target classifier is built from only 16 images per class across 8 classes,
and the gap persists when that budget is increased, so it reflects the classifier construction
rather than a broken pipeline.

Aggregate the numbers from an experiment directory:

```bash
python tools/collect_results.py $CKPT --csv results.csv
```

## Modifications

The repository contains the released implementation plus one improved variant
(`trainers/medkcoop/fixed/medkcoop.py`, selected by `--trainer MeDKCoOpFix`). The default
`--trainer MeDKCoOp` reproduces the paper's numbers.

### Bug-level fixes (no change to the reported numbers)

| Fix | Detail |
|---|---|
| Learnable context was never trained | In the released code the prompt and graph construction are wrapped in `torch.no_grad()`, so `ctx` received no gradient and only the GCN fusion layer was updated. A switch (`GRAD_CTX`) restores the gradient path; enabling it alone changes scores by at most 0.04 points. |
| `visual_net` was never updated | It computed gradients but was absent from the optimizer. `TRAIN_VISUAL_NET` adds it. |
| Domain-generalization leakage | `datasets/*v2.py` exposed the entire split as both the prototype source and the evaluation set. `DG_STRICT=1` (default in the fixed module) builds prototypes from `train`/`val` only. |
| MedSAM evaluation cost | The 1024×1024 encoder ran on every image at every evaluation. Feature caching (`MEDSAM_CACHE=1`) stores the 256-d pooled vector (1.7 KB/image), verified numerically identical. |
| Partial results were silently reused | Scripts now require the checkpoint **and** the metric lines before skipping, and fail loudly otherwise. |
| Reproducibility gaps | Optional `--deterministic` mode; `--seed 0` and `--device-id 0` were previously ignored; MedSAM chunking avoids OOM at larger evaluation batches. |

### Method-level improvement

The improvement targets the observation that the learnable prompt, the component intended to carry
generalization, contributed almost nothing: with only 2 context tokens, a learning rate of 2.5e-3
and 8 optimizer steps per epoch at 16 shots, its total displacement stayed around 10% of its own
scale. The variant therefore

- widens the learnable context from 2 to 8 tokens,
- raises the learning rate to 1e-2,
- makes the residual weight `beta` a learnable scalar instead of a fixed 0.3,
- lets the image class prototypes participate in gradients, and
- trains `visual_net` jointly with the prompts,

adding `+1.10` points on novel classes and `+0.79` HM without touching the base half
(87.50 → 87.73). Run it with:

```bash
GPU=0 bash scripts/medkcoop/base2new_algv3.sh BTMRI1 1
```


## Implementation notes

- Two trainer implementations are registered: `MeDKCoOp` (`trainers/medkcoop/reference/`), which
  reproduces the paper, and `MeDKCoOpFix` (`trainers/medkcoop/fixed/`), which carries the switches
  listed under [Modifications](#modifications). The ablation variants behind the paper's ablation
  tables and backbone comparison are registered from `trainers/medkcoop/ablation/`.
- Several parameters present in the released checkpoints (`GCN_tt`, `GCN_it`, `scale_mm`,
  `base_img_features`) are not used in the forward pass. They are retained so that the released
  weights load without modification.
- Evaluation uses the `test` branch of the prompt learner by design: class prototypes are rebuilt
  from the target-domain training split, which is why an `--eval-only` run reproduces the
  evaluation numbers but not the base-half training accuracy.

## Citation

```bibtex
@inproceedings{wang2025medkcoop,
  title     = {MeDKCoOp: Dual Knowledge-guided Graph Prompt Learning for
               Biomedical Vision-Language Models},
  author    = {Wang, Yijun and Wu, Siying and Gan, Lubin and Zhang, Zheyu and
               Zhang, Jing and Hu, Zhangchi and Zhu, Huyue and Wu, Peixi and Sun, Xiaoyan},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  year      = {2025}
}
```

## Acknowledgements

The training framework is [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch); the
prompt-learning baselines build on [CoOp](https://github.com/KaiyangZhou/CoOp),
[MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning),
[DCPL](https://github.com/caoql98/DCPL) and
[GraphAdapter](https://github.com/lixinustc/GraphAdapter). Backbones are
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224),
[CLIP](https://github.com/openai/CLIP), and [MedSAM](https://github.com/bowang-lab/MedSAM).
See [NOTICE.md](NOTICE.md) for the full list of third-party components and licenses.

## License

Released under the MIT License. See [LICENSE](LICENSE).
