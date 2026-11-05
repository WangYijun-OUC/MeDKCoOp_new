# third_party

外部代码，**只保留被现役 trainer 引用的部分**，内容未改动。

| 目录 | 谁在用 | 说明 |
|---|---|---|
| `Dassl.pytorch/` | 全部 trainer | 训练框架。当前环境是 editable 安装且指向**原项目**，换环境必须 `pip install -e third_party/Dassl.pytorch` |
| `open_clip/` | `medkcoop/*`、baseline 的 BiomedCLIP 版本 | BiomedCLIP 加载；代码以 `open_clip.src.open_clip` 绝对导入，**不要改目录名** |
| `clip/` | 所有 trainer | 合并后的 CLIP：标准 CLIP + `pmcclip.py`（PMC-CLIP）+ MaPLe/DCPL 的 IVLP 分支；标准路径与原实现位级一致 |

| `MedSAM/` | `medkcoop` 视觉知识分支 | 权重在 `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` |

## 为什么 clip 与 clip1 同时存在

两者都是 CLIP，但分支不同：`clip1` 增加了 `ResidualAttentionBlock_IVLP` 等 MaPLe 需要的模块，
`clip` 独有 `pmcclip.py`（PMC-CLIP 的 ResNet50 视觉塔）。现役 trainer 分别依赖其中之一，
合并会改动 import 与模型代码，属于"内容整改"阶段，不在结构整理范围内。

## 离线加载

`create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-...')` 在无外网、HF cache 为空时会失败。
本地权重：

- `/data/wangyijun/model/BiomedCLIP/`（完整 HF 仓库，含 `open_clip_config.json`、`open_clip_pytorch_model.bin`、`BiomedBERT/`、tokenizer）
- `/data/wangyijun/model/BiomedCLIP_local/`（单目录，缺 `tokenizer.json`）

整改时新增 `load_biomedclip()`：`add_model_config()` + 本地权重 + 本地 tokenizer，并与在线加载做 `allclose` 验证。
