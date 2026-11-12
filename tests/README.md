# tests

Smoke tests covering the invariants most likely to regress.

| File | Covers |
|---|---|
| `test_imports.py` | All trainers and datasets register without name collisions |
| `test_configs.py` | Every dataset config has a matching reader class; every trainer name referenced by a script is registered |
| `test_ckpt_compat.py` | Checkpoint keys and shapes expected by the released weights |
| `test_clip_merge.py` | The merged `clip` package loads both the standard and MaPLe paths |

```bash
pip install pytest
python -m pytest tests -q      # 6 passed
```

`test_ckpt_compat.py` reads the checkpoint path from `MEKCOOP_TEST_CKPT` and skips itself when the
variable is unset or the file is missing:

```bash
MEKCOOP_TEST_CKPT=/path/to/model.pth.tar-50 python -m pytest tests -q
```
