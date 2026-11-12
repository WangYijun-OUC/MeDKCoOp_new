# trainers/medkcoop

Implementation of **MeDKCoOp** (dual knowledge-guided graph prompt learning) and the ablation
variants reported in the paper.

```
medkcoop/
├── reference/
│   ├── medkcoop.py          # implementation used for the reported numbers
│   └── paper_bytecode/      # bytecode of the final variant (see below)
├── fixed/medkcoop.py        # same method with explicit gradient / caching options
└── ablation/                # variants behind the ablation tables and backbone figure
```

## Which class produces the reported numbers

The reported results are produced by the trainer registered as **`MeDKCoOp`**, implemented in
`reference/medkcoop.py`. The improved variant is registered as **`MeDKCoOpFix`** and lives in
`fixed/medkcoop.py`.

During the final stage of the project the source file of the reported variant was lost; only its
compiled bytecode remained (`reference/paper_bytecode/`). The implementation in
`reference/medkcoop.py` was recovered from that bytecode and differs only in:

- the class name;
- the helper name `edge_construct` (originally `cal_edge_emb`, identical logic);
- three lines of unused `mean`/`noise` computation in `_get_base_image_features` (they consume the
  RNG stream and therefore affect shuffling, so they are preserved);
- one debug `print` in `_get_base_text_features`.

The recovered module loads the original paper checkpoints without modification.

## Gradient flow in the released implementation

`MultiModalPromptLearner.build_prompt_graph` wraps the prompt/graph construction in
`torch.no_grad()`, and the image prototypes are read through `.data`. As a consequence only the
GCN fusion layer (`GCN_fuse`) — plus the alias-free parameters listed below — actually receives
gradients and is updated by the optimizer.

`fixed/medkcoop.py` exposes four switches to change this behaviour, all defaulting to the released
behaviour except `GRAD_CTX`:

| Config key | Effect when enabled |
|---|---|
| `TRAINER.GRAPHDCPL.GRAD_CTX` | learnable context participates in the forward graph |
| `TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET` | `visual_net` is added to the optimizer |
| `TRAINER.GRAPHDCPL.LEARN_BETA` | residual weight `beta` becomes a learnable scalar |
| `TRAINER.GRAPHDCPL.GRAD_IMGPROTO` | image class prototypes participate in gradients |

Enabling only `GRAD_CTX` keeps the base-to-new scores within 0.04 points of the released numbers;
the method-level effect of the remaining switches is documented in the paper's follow-up analysis.

Parameters present in the checkpoint but unused in the forward pass (`GCN_tt`, `GCN_it`,
`scale_mm`, `base_img_features`) are kept for checkpoint compatibility.

## Ablation variants

`ablation/` mirrors the naming used in the paper's experiments. Each file registers a distinct
trainer class, so a variant is selected with `--trainer <CLASS>`.

### Knowledge ablation (Table 4)

| Trainer class | File | Configuration |
|---|---|---|
| `GRAPH_DCPL_NO_VIS_BIAS` | `..._no_vis_bias.py` | w/o visual domain knowledge |
| `GRAPH_DCPL_NO_TXT_BIAS` | `..._no_txt_bias.py` | w/o prompt-branch domain knowledge |
| `GRAPH_DCPL_NO_MEDSAM` | `..._no_medsam.py` | w/o MedSAM visual knowledge |
| `GRAPH_DCPL_GENERAL` | `..._general.py` | generic textual knowledge |
| `GRAPH_DCPL_INCORRECT` | `..._incorrect.py` | incorrect textual knowledge |
| `GRAPH_DCPL_OUTOFDOMAIN` | `..._outofdomain.py` | out-of-domain textual knowledge |

### Component combination (Table 5)

| Trainer class | File |
|---|---|
| `GRAPH_DCPL_NO_TXT_VIS_FUSE` | `..._no_text_vis_fuse.py` |
| `GRAPH_DCPL_NO_VIS_FUSE_GRAPH` | `..._no_vis_fuse_graph.py` |
| `GRAPH_DCPL_NO_VIS_GRAPH` | `..._no_vis_graph.py` |
| `GRAPH_DCPL_NO_TXT_VIS_GRAPH1` | `..._no_text_vis_graph1.py` |
| `GRAPH_DCPL_NO_TXT_GRAPH` | `..._no_text_graph.py` |
| `GRAPH_DCPL_NO_TXT_FUSE_GRAPH` | `..._no_text_fuse_graph.py` |
| `GRAPH_DCPL_NO_FUSE` | `..._no_fuse.py` |

### Backbone comparison (Fig. 4)

`GRAPH_DCPL_CLIP`, `GRAPH_DCPL_PUBMEDCLIP`, `GRAPH_DCPL_PMCCLIP`
(from `..._clip.py`, `..._pubmedclip.py`, `..._pmcclip.py`).

Run any variant with the shared scripts, overriding the trainer name:

```bash
TRAINER=GRAPH_DCPL_NO_VIS_GRAPH GPU=0 bash scripts/medkcoop/base2new.sh BTMRI1 1
```
