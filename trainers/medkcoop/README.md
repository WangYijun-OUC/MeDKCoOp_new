# trainers/medkcoop

论文方法与消融。**代码内容未改动，只调整了位置与文件集合。**

```
medkcoop/
├── reference/
│   ├── medkcoop.py                # 原 trainers/GRAPH_DCPL/medkcoop.py（类 MeDKCoOp）
│   └── paper_bytecode/            # 论文最终方法丢失源码后仅存的 .pyc
└── ablation/                      # 16 个变体，对应论文 Table 4/5 与 Fig 4
```

## 两个关键事实

1. **论文 Table 1 的 "Ours" 不是 `reference/medkcoop.py`**。
   8 个数据集 16 个 base/new 数值与日志中 **`GRAPH_DCPL_NO_TXT_VIS_GRAPH`** 完全一致；
   该类源码已丢失，只剩 `reference/paper_bytecode/vip_graph_dcpl_biomedclip_adapter_no_text_vis_graph.cpython-310.pyc`
   （Python 3.10 可 `marshal` 载入并反汇编）。
2. `reference/medkcoop.py` 与字节码的差异只有：
   - 类名不同；
   - `build_prompt_graph` 中 `edge_construct` vs `cal_edge_emb`（实现一致）；
   - `_get_base_image_features` 少 3 行图像特征加噪；
   - `_get_base_text_features` 少一句 `print("构建文本子图......")`。

   → 重建时以 `reference/medkcoop.py` 为骨架，恢复上述差异即可；checkpoint 键完全兼容。

## 消融文件与论文位置

| 文件 | trainer 类 | 论文位置 |
|---|---|---|
| `..._no_text_vis_fuse.py` | `GRAPH_DCPL_NO_TXT_VIS_FUSE` | Table 5 (✗✗✗) |
| `..._no_vis_fuse_graph.py` | `GRAPH_DCPL_NO_VIS_FUSE_GRAPH` | Table 5 (✗✓✗) |
| `..._no_vis_graph.py` | `GRAPH_DCPL_NO_VIS_GRAPH` | Table 5 (✓✗✓) |
| `..._no_text_vis_graph1.py` | `GRAPH_DCPL_NO_TXT_VIS_GRAPH1` | Table 5 (✗✗✓) |
| `..._no_text_graph.py` | `GRAPH_DCPL_NO_TXT_GRAPH` | Table 5 待对齐行候选 |
| `..._no_text_fuse_graph.py` | `GRAPH_DCPL_NO_TXT_FUSE_GRAPH` | 同上 |
| `..._no_fuse.py` | `GRAPH_DCPL_NO_FUSE` | 同上 |
| `..._no_vis_bias.py` | `GRAPH_DCPL_NO_VIS_BIAS` | Table 4 w/o VDK |
| `..._no_txt_bias.py` | `GRAPH_DCPL_NO_TXT_BIAS` | Table 4 w/o LDK |
| `..._no_medsam.py` | `GRAPH_DCPL_NO_MEDSAM` | Table 4 w/o VK |
| `..._general.py` | `GRAPH_DCPL_GENERAL` | Table 4 General |
| `..._incorrect.py` | `GRAPH_DCPL_INCORRECT` | Table 4 Incorrect |
| `..._outofdomain.py` | `GRAPH_DCPL_OUTOFDOMAIN` | Table 4 Out-of-domain |
| `..._clip.py` / `_pubmedclip.py` / `_pmcclip.py` | `GRAPH_DCPL_CLIP` 等 | Fig 4 backbone |

已确认数值的消融（8 数据集 base/new 均值）：

| 类 | base | new |
|---|---|---|
| `NO_TXT_VIS_GRAPH`（论文 Ours） | 87.88 | 79.95 |
| `NO_TXT_VIS_FUSE` | 84.81 | 72.37 |
| `NO_VIS_FUSE_GRAPH` | 83.77 | 74.70 |
| `NO_VIS_GRAPH` | 87.53 | 71.04 |
| `NO_TXT_VIS_GRAPH1` | 87.52 | 51.92 |

## 阅读提示

`reference/medkcoop.py` 前向中 `GCN_tt`、`GCN_it`、`scale_mm`、`base_img_features` 是**死参数/仅初始化用**，
但仍存在于 checkpoint，容易误读。MedSAM 路径在 `reference/` 中已指向
`/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth`。
