# tests

四个冒烟测试，覆盖审查报告里最容易回归的点：

| 文件 | 覆盖 |
|---|---|
| `test_imports.py` | 所有 trainer / dataset 注册成功且无重名 |
| `test_configs.py` | 每个数据集配置有对应类；脚本引用的 trainer 都已注册 |
| `test_ckpt_compat.py` | 旧论文 checkpoint 的键与形状 |
| `test_clip_merge.py` | 合并后的 clip 包：标准路径 + MaPLe 分支 |

运行：

```bash
conda activate medkcoop
pip install pytest  # 若未安装
cd /home/wangyijun/MeDKCoOp
python -m pytest tests -q
```
