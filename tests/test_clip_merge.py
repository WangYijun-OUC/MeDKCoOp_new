"""clip 包完整性：标准路径与 MaPLe 分支都能构建。"""
import sys

import torch

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, ROOT + "/third_party")


def test_clip_standard_and_maple():
    import clip

    m = clip.model.CLIP(512, 224, 12, 768, 16, 77, 49408, 512, 8, 12)
    assert hasattr(m, "visual")
    dd = {"trainer": "MaPLe", "vision_depth": 2, "language_depth": 2,
          "vision_ctx": 2, "language_ctx": 2, "maple_length": 2}
    m2 = clip.model.CLIP(512, 224, 12, 768, 16, 77, 49408, 512, 8, 12, dd)
    assert hasattr(m2.visual, "transformer")
