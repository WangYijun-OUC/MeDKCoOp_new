# third_party

Vendored third-party code, restricted to what the trainers actually import. Contents are unmodified
upstream sources except where noted below.

| Directory | Used by | Notes |
|---|---|---|
| `Dassl.pytorch/` | all trainers | Training framework (registry, config, `TrainerX`). Install with `pip install -e third_party/Dassl.pytorch`. |
| `open_clip/` | `medkcoop/*`, BiomedCLIP variants of the baselines | Loads BiomedCLIP. Imported as `open_clip.src.open_clip`, so **do not rename the directory**. |
| `clip/` | all trainers | Merged CLIP package: standard CLIP + `pmcclip.py` (PMC-CLIP) + IVLP/MaPLe blocks. |
| `MedSAM/` | `medkcoop` visual knowledge branch | Requires the MedSAM ViT-B checkpoint (see below). |

## The `clip` package

`third_party/clip/` is the only CLIP implementation in the repository:

- standard CLIP (`clip.load()` and `clip.model.build_model(state_dict)` work as upstream);
- `pmcclip.py`, the PMC-CLIP ResNet-50 vision tower;
- the `ResidualAttentionBlock_IVLP` / `ResidualAttentionBlock_MaPLe` variants, enabled through
  `design_details`, used by MaPLe, DCPL and GraphAdapter.

The standard path was verified to match the original `clip` package bit-for-bit (maxdiff = 0) on the
same random weights.

## Offline backbone loading

`create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-...')` requires network access unless the
weights are available locally. Set `BIOMEDCLIP_DIR` to a directory containing
`open_clip_pytorch_model.bin`, `open_clip_config.json`, `BiomedBERT/` and the tokenizer files; the
loader in `bootstrap.py` then resolves the `hf-hub:` identifier to that directory.

```python
import bootstrap
model, preprocess = bootstrap.load_biomedclip()
tokenizer = bootstrap.load_biomedclip_tokenizer()
```

`MEDSAM_CKPT` points to the MedSAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`), and
`PMCCLIP_DIR` / `PMCCLIP_BERT_DIR` to the PMC-CLIP encoders and text tokenizer used by the
backbone ablation. All three are environment variables, so no machine-specific path is baked into
the code.
