# Third-party notices

This repository contains or adapts code from several research projects. The root `LICENSE` covers the
BiomedCoOp-derived repository code and must not be used to remove upstream notices.

Bundled third-party components:

| Component | Local path | License file |
|---|---|---|
| Dassl.pytorch | `third_party/Dassl.pytorch` | `third_party/Dassl.pytorch/LICENSE` (MIT) |
| open_clip | `third_party/open_clip` | `third_party/open_clip/LICENSE` |
| MedSAM / Segment Anything derived code | `third_party/MedSAM` | `third_party/MedSAM/LICENSE` (Apache-2.0) |
| OpenAI CLIP derived and prompt-learning extensions | `third_party/clip` | upstream notices must be retained when redistributed |

Trainer implementations under `trainers/baselines` are research-code adaptations of CoOp, CoCoOp,
KgCoOp, MaPLe, DCPL, GraphAdapter and BiomedCoOp. MeDKCoOp trainer files also derive from the same
Dassl/CLIP prompt-learning ecosystem. Publications or redistributed packages should cite the corresponding
papers and include the upstream license files required by the source projects.

Model weights and datasets are not covered by the repository license. Their original licenses and terms
apply separately. The repository does not redistribute those assets; local paths are documented only for
reproduction.
