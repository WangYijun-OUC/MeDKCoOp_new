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
python train.py --root $DATA --seed 1 --trainer GRAPH_DCPL_NO_TXT_VIS_GRAPH \
    --dataset-config-file configs/datasets/BTMRI1.yaml \
    --config-file configs/trainers/medkcoop/base_to_novel/BTMRI1.yaml \
    --output-dir $CKPT/base2new/BTMRI1/train_base --mode train \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base

# stage 2: evaluate novel classes
python train.py --root $DATA --seed 1 --trainer GRAPH_DCPL_NO_TXT_VIS_GRAPH \
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

Base-to-new generalization, 16 shots, mean over three seeds (BiomedCLIP backbone):

| Dataset | Base | New | HM |
|---|---|---|---|
| BTMRI1 | 85.87 | 91.61 | 88.64 |
| CHMNIST | 93.44 | 60.11 | 73.16 |
| Covid-19 | 78.58 | 91.15 | 84.40 |
| CTKIDNEY | 90.66 | 84.65 | 87.55 |
| LC25000 | 93.64 | 95.47 | 94.55 |
| RETINA | 89.48 | 58.38 | 70.66 |
| OCTMNIST | 80.73 | 85.88 | 83.23 |
| Kvasir | 87.61 | 70.72 | 78.27 |
| **Average** | **87.50** | **79.75** | **82.56** |

Cross-dataset transfer (source: BTMRI1) and domain generalization:

| Target | Cross-dataset | Domain generalization |
|---|---|---|
| CHMNIST | 27.02 | 26.69 |
| Covid-19 | 61.78 | 62.08 |
| CTKIDNEY | 60.25 | 60.15 |
| LC25000 | 62.91 | 62.98 |
| RETINA | 37.79 | 37.29 |
| OCTMNIST | 61.17 | 62.49 |
| Kvasir | 58.58 | 59.11 |
| **Average** | **52.78** | **52.97** |

Aggregate the numbers from logs of an experiment directory:

```bash
python tools/collect_results.py $CKPT --csv results.csv
```

## Notes on the released implementation

- The trainer class used for the reported numbers is registered as
  `GRAPH_DCPL_NO_TXT_VIS_GRAPH` (an identifier inherited from the ablation naming scheme) and
  lives in `trainers/medkcoop/reference/medkcoop.py`. `MeDKCoOp` is a registered alias of the same
  implementation.
- In that implementation the learnable context is enclosed in `torch.no_grad()` while the graph
  features are built, so **only the GCN fusion layer receives gradients** during training.
  `trainers/medkcoop/fixed/medkcoop.py` adds explicit switches to change this
  (`GRAD_CTX`, `TRAIN_VISUAL_NET`, `LEARN_BETA`, `GRAD_IMGPROTO`). With `GRAD_CTX=True` the
  base-to-new scores stay within 0.04 points of the released numbers.
- `datasets/*v2.py` read a single split file and expose the full split as the evaluation set.
  Set `DG_STRICT=1` to build the visual prototypes from `train`/`val` only, which removes
  test-sample leakage in the domain-generalization protocol.

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
