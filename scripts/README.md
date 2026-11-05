# scripts

批量启动器，按方法分组。**脚本内容未改动**，仍保留旧路径与硬编码，修复安排在内容整改阶段。

```
scripts/
├── medkcoop/          # 论文方法：11 个
│   ├── base2new.sh            # 16-shot 训练 base → eval-only 评测 new
│   ├── base2new_abla.sh       # 消融批量跑（Table 4/5）
│   ├── base2new_fuse.sh       # 融合相关消融
│   ├── base2new_clip.sh       # backbone 消融（CLIP）
│   ├── cross_data_eval.sh     # 跨数据集批量评测入口
│   ├── xd_train.sh / xd_test.sh  # 源域 BTMRI1 训练 / 目标域评测
│   ├── domain_genera.sh       # 用 *v2 数据集做 domain generalization
│   └── few_shot.sh / few_shot1.sh / few_shot2.sh
└── baselines/         # 8 个 baseline 的同类脚本
    ├── biomedcoop/ coop/ cocoop/ kgcoop/ MaPLe/ dcpl/
    ├── graph_adapter/ zeroshot/
```

## 使用前必读（全部属于内容整改范围）

1. 脚本内部仍调用 `bash scripts/graph_dcpl/xd_test.sh ...`，目录已改名为 `scripts/medkcoop/`。
2. `--config-file` 多写 `configs/trainers/GRAPH_DCPL/...`，该目录不存在；实际用
   `configs/trainers/medkcoop/{base_to_novel,few_shot}/<DATASET>.yaml`。
3. 输出目录写死 `/home/wangyijun/ckpt/...` 或 `/data/wangyijun/ckpt/...`，GPU 写死 `Device_id=5..9`。
4. `CFG=vit_b16_0002_b50_randflip` 只是目录名，不是配置路径。
5. few-shot 脚本传 `TRAINER.BIOMEDCOOP.*`，但 GRAPHDCPL trainer 读 `TRAINER.GRAPHDCPL.*`，覆盖无效。
6. 已删除的 baseline 目录（clip_adapter/dplq/linear_probe*/prograd/tip_adapter）对应代码也已移除，
   需要时从 git 基线提交 `5e7ecc2` 或备份恢复。
