"""配置完整性：脚本引用的 trainer 与 config 都必须存在。"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_datasets_have_configs():
    for cfg in glob.glob(os.path.join(ROOT, "configs", "datasets", "*.yaml")):
        name = os.path.basename(cfg)[:-5]
        ds_py = os.path.join(ROOT, "datasets", name.lower() + ".py")
        assert os.path.exists(ds_py) or any(
            os.path.exists(os.path.join(ROOT, "datasets", f))
            for f in os.listdir(os.path.join(ROOT, "datasets"))
            if f.lower() == name.lower() + ".py"
        ), f"数据集配置 {name} 没有对应的 datasets/*.py"


def test_script_trainers_registered():
    import bootstrap  # noqa: F401
    import train  # noqa: F401

    from dassl.engine import TRAINER_REGISTRY

    reg = set(TRAINER_REGISTRY._obj_map)
    bad = []
    for sh in glob.glob(os.path.join(ROOT, "scripts", "**", "*.sh"), recursive=True):
        text = open(sh, errors="ignore").read()
        for m in re.finditer(r'TRAINER="\$\{TRAINER:-([A-Za-z0-9_]+)\}"', text):
            if m.group(1) not in reg:
                bad.append((os.path.basename(sh), m.group(1)))
        for m in re.finditer(r"^\s*TRAINER=([A-Za-z0-9_]+)\s*$", text, re.M):
            if m.group(1) not in reg:
                bad.append((os.path.basename(sh), m.group(1)))
    assert not bad, f"脚本引用了未注册的 trainer: {bad}"
