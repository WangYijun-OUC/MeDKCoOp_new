# scripts

Task launchers. Each method directory is self-contained: its `base2new`, `few_shot`,
`xd_train` and `xd_test` scripts call `train.py` directly, and the batch entry points only call the
per-method scripts in the same directory.

```
scripts/
├── medkcoop/          # proposed method
├── baselines/         # biomedcoop, coop, cocoop, kgcoop, MaPLe, dcpl, graph_adapter, zeroshot
└── run_all_methods.sh # convenience loop over all methods
```

## Entry points

Every trainable method directory provides:

| Script | Usage | Purpose |
|---|---|---|
| `base2new.sh` | `DATASET SEED` | 16-shot base training, then novel-class evaluation |
| `few_shot.sh` | `DATASET SEED SHOTS` | few-shot training and evaluation |
| `xd_train.sh` | `SOURCE SEED` | cross-dataset source training |
| `xd_test.sh` | `TARGET SEED SOURCE` | evaluate a source checkpoint on a target dataset |
| `cross_data_eval.sh` | — | loop over all targets and seeds |
| `domain_genera.sh` | — | domain-generalization loop over the `*v2` targets |

```bash
GPU=0 bash scripts/medkcoop/base2new.sh BTMRI1 1
GPU=0 bash scripts/medkcoop/few_shot.sh BTMRI1 1 16
GPU=0 bash scripts/baselines/coop/base2new.sh BTMRI1 1
GPU=0 bash scripts/baselines/biomedcoop/few_shot.sh CHMNIST 2 4
```

Zeroshot has no training stage, so its directory provides evaluation entry points only.

## Default trainer per directory

| Directory | Trainer | base2new epochs | few-shot / transfer epochs |
|---|---|---:|---:|
| medkcoop | `GRAPH_DCPL_NO_TXT_VIS_GRAPH` | 50 | 100 |
| biomedcoop | `BiomedCoOp_BiomedCLIP` | 50 | 100 |
| coop | `CoOp_BiomedCLIP` | 50 | 100 |
| cocoop | `CoCoOp_BiomedCLIP` | 50 | 100 |
| kgcoop | `KgCoOp_BiomedCLIP` | 50 | 100 |
| MaPLe | `MaPLe_CLIP` | 5 | 100 |
| dcpl | `DCPL_biomedclip` | 50 | 100 |
| graph_adapter | `GraphCLIP` | 50 | 100 |
| zeroshot | `ZeroshotBiomedCLIP` | — | — |

`EPOCH` overrides the default. MaPLe keeps 5 epochs for base-to-new because that is the value in
the released configuration.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DATA` | `./data` | dataset root |
| `CKPT` | `./output` | output root |
| `GPU` | `0` | CUDA device index |
| `SHOTS` | `16` | shots per class |
| `EPOCH` | 50 / 100 | training epochs |
| `EVAL_BS` | `16` | evaluation batch size |
| `RUN_TAG` | empty | sub-directory tag for A/B runs |
| `EXTRA_OPTS` | empty | extra `train.py` overrides |
| `PYTHON_BIN` | `python` | interpreter to use |
| `SOURCE` / `MODEL_DIR` | `BTMRI1` / derived | cross-dataset source and checkpoint directory |

A run is skipped only when the checkpoint **and** the required metric lines are both present;
partial results raise an error instead of being silently reused.

## Batch scheduling

```bash
GPU=0 bash scripts/run_all_methods.sh base2new BTMRI1 1
GPU=0 bash scripts/run_all_methods.sh few_shot BTMRI1 1 16
```

Cross-dataset and domain-generalization loops launch many runs; run them per method and assign
different GPUs as needed.

## Ablation launchers

| Script | Covers |
|---|---|
| `medkcoop/base2new_abla.sh` | Table 4: knowledge sources and textual interventions |
| `medkcoop/base2new_fuse.sh` | Table 5: component combinations |
| `medkcoop/base2new_clip.sh` | Fig. 4 backbone comparison (PubMedCLIP / PMC-CLIP weights required) |
| `medkcoop/base2new_algv3.sh` | variant with learnable-context gradients and larger context |
