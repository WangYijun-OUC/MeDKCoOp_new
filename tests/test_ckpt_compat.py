"""checkpoint 键兼容性：旧论文 checkpoint 的键必须能被当前模型加载。"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CKPT = ("/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/BTMRI1/"
        "shots_16/GRAPH_DCPL_NO_TXT_VIS_GRAPH/vit_b16_0002_b50_randflip/seed1/"
        "MultiModalPromptLearner/model.pth.tar-50")

EXPECTED = [
    "prompt_learner.ctx",
    "prompt_learner.fc.weight",
    "prompt_learner.GCN_fuse.gcn_weights",
    "visual_net.linear1.weight",
]


def test_old_checkpoint_keys():
    if not os.path.exists(CKPT):
        import pytest
        pytest.skip("论文 checkpoint 不在本机")
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    for key in EXPECTED:
        assert key in sd, f"缺少键 {key}"
    assert sd["prompt_learner.ctx"].shape == (2, 768)
    assert sd["prompt_learner.fc.weight"].shape[1] == 512
