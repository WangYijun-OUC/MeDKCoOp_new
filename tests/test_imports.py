"""导入冒烟测试：所有保留的 trainer 与数据集都必须能注册成功且无重名。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_all_trainers_import():
    import bootstrap  # noqa: F401
    import train  # noqa: F401  触发 train.py 中的 import 列表

    from dassl.engine import TRAINER_REGISTRY

    registry = TRAINER_REGISTRY._obj_map
    assert "MeDKCoOp" in registry
    assert "MeDKCoOpFix" in registry
    for name in ("CoOp_BiomedCLIP", "BiomedCoOp_BiomedCLIP", "DCPL_biomedclip", "MaPLe_CLIP"):
        assert name in registry, name


def test_all_datasets_registered():
    import bootstrap  # noqa: F401
    import train  # noqa: F401

    from dassl.data.datasets import DATASET_REGISTRY

    for name in ("BTMRI1", "CHMNIST", "COVID_19", "CTKidney", "Kvasir",
                 "LungColon", "OCTMNIST", "RETINA"):
        assert name in DATASET_REGISTRY._obj_map, name
