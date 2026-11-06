"""运行时引导：路径、离线权重适配、可选环境变量。

在任何 trainer/dataset 导入之前调用（train.py 第一行）：
    import bootstrap  # noqa: F401

做三件事：
1. 把 third_party/ 及其子包加入 sys.path，使
   `from clip import ...`、`from MedSAM...`、
   `open_clip.src.open_clip` 都能直接解析；
2. 把 `open_clip` 里对 `hf-hub:microsoft/BiomedCLIP-...` 的加载替换为本地权重
   （默认 /data/wangyijun/model/BiomedCLIP，可用 BIOMEDCLIP_DIR 覆盖）；
3. 提供 MEDSAM_CKPT / DATA_ROOT 默认值，供 trainer 使用。
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THIRD_PARTY = ROOT / "third_party"

for _p in (str(ROOT), str(THIRD_PARTY), str(THIRD_PARTY / "open_clip")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BIOMEDCLIP_DIR = os.environ.get(
    "BIOMEDCLIP_DIR", "/data/wangyijun/model/BiomedCLIP"
)
BERT_DIR = os.path.join(BIOMEDCLIP_DIR, "BiomedBERT")
TOKENIZER_DIR = BIOMEDCLIP_DIR  # 根目录含 tokenizer.json / vocab.txt / tokenizer_config.json
WEIGHTS = os.path.join(BIOMEDCLIP_DIR, "open_clip_pytorch_model.bin")
CLIP_CONFIG = os.path.join(BIOMEDCLIP_DIR, "open_clip_config.json")

os.environ.setdefault("MEDSAM_CKPT", "/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth")
os.environ.setdefault("DATA_ROOT", "/data/wangyijun/dataset")


def _patch_open_clip():
    """让 hf-hub 的 BiomedCLIP 名称落到本地权重，避免联网。"""
    if not (os.path.exists(WEIGHTS) and os.path.exists(CLIP_CONFIG)):
        return  # 本地权重不存在时保持原行为（会尝试联网）

    import open_clip.src.open_clip as _oc
    from open_clip.src.open_clip.factory import (
        create_model_from_pretrained as _orig_create,
        get_tokenizer as _orig_get_tokenizer,
    )
    from open_clip.src.open_clip.tokenizer import HFTokenizer

    with open(CLIP_CONFIG, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)["model_cfg"]

    def create_model_from_pretrained(model_name, *args, **kwargs):
        if isinstance(model_name, str) and "BiomedCLIP" in model_name:
            text_cfg = dict(model_cfg["text_cfg"])
            text_cfg["hf_model_name"] = BERT_DIR
            text_cfg["hf_tokenizer_name"] = TOKENIZER_DIR
            text_cfg["hf_model_pretrained"] = False  # 权重从 checkpoint 加载，不联网
            kwargs.pop("cache_dir", None)
            kwargs.pop("force_custom_text", None)
            return _orig_create(
                "ViT-B-16",
                pretrained=WEIGHTS,
                embed_dim=model_cfg["embed_dim"],
                vision_cfg=model_cfg["vision_cfg"],
                text_cfg=text_cfg,
                force_custom_text=True,
                **kwargs,
            )
        return _orig_create(model_name, *args, **kwargs)

    def get_tokenizer(model_name="", context_length=None, **kwargs):
        if isinstance(model_name, str) and "BiomedCLIP" in model_name:
            return HFTokenizer(
                TOKENIZER_DIR, context_length=context_length or 256, **kwargs
            )
        return _orig_get_tokenizer(
            model_name, context_length=context_length, **kwargs
        )

    _oc.create_model_from_pretrained = create_model_from_pretrained
    _oc.get_tokenizer = get_tokenizer



def load_biomedclip(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    return_transform=True):
    """显式加载 BiomedCLIP（不依赖 monkeypatch，可在任意脚本中使用）。

    优先使用本地权重；本地不存在时回退到 open_clip 默认行为（可能联网）。
    """
    from open_clip.src.open_clip.factory import create_model_from_pretrained

    text_cfg = None
    if os.path.exists(WEIGHTS) and os.path.exists(CLIP_CONFIG):
        with open(CLIP_CONFIG, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)["model_cfg"]
        text_cfg = dict(model_cfg["text_cfg"])
        text_cfg["hf_model_name"] = BERT_DIR
        text_cfg["hf_tokenizer_name"] = TOKENIZER_DIR
        text_cfg["hf_model_pretrained"] = False
        model, preprocess = create_model_from_pretrained(
            "ViT-B-16",
            pretrained=WEIGHTS,
            embed_dim=model_cfg["embed_dim"],
            vision_cfg=model_cfg["vision_cfg"],
            text_cfg=text_cfg,
            force_custom_text=True,
            return_transform=return_transform,
        )
        return model, preprocess
    return create_model_from_pretrained(model_name, return_transform=return_transform)


def load_biomedclip_tokenizer(context_length=256, **kwargs):
    """本地 BiomedBERT tokenizer。"""
    from open_clip.src.open_clip.tokenizer import HFTokenizer

    return HFTokenizer(TOKENIZER_DIR, context_length=context_length, **kwargs)


_patch_open_clip()
