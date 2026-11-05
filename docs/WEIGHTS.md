# /data 下的权重与实验产物清理清单

> 本文只做盘点与建议，**尚未删除任何文件**。删除是不可逆操作，确认后再执行第 5 节的命令。

## 1. 结论

| 类别 | 容量 | 建议 |
|---|---|---|
| 本项目运行必需 | 约 54 GB | **保留** |
| 本项目可清理的模型权重 | 约 **1.39 TB** | 归档日志后删除 |
| 与本项目无关的 ckpt / 权重 | 约 3.29 TB | 需你确认（可能是其他项目） |
| 与本项目无关的数据集 | 约 740 GB | 可选清理，需你确认 |

清理后，本项目保留约 **54 GB**：8 个数据集 + BiomedCLIP + MedSAM + PubMedCLIP/PMC-CLIP + 论文最终方法的 24 个模型 + 全部日志。

## 2. 必留（约 54 GB）

| 路径 | 大小 | 理由 |
|---|---|---|
| `/data/wangyijun/dataset/{BTMRI,CHMNIST,COVID_19,CTKidney,Kvasir,lungColon,OCTMNIST,RETINA}` | 22.6 GB | 论文 8 个数据集（含 split 与 few-shot 缓存） |
| `/data/wangyijun/model/BiomedCLIP` | 2.3 GB → 清理后约 1.1 GB | 主 backbone；`open_clip_pytorch_model.bin` + `BiomedBERT/` + tokenizer |
| `/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth` | 375 MB | 视觉知识分支 |
| `/data/wangyijun/model/PubMedCLIP/PubMedCLIP_ViT32.pth` | 1.5 GB | 论文 Fig 4 backbone 消融 |
| `/data/wangyijun/model/PubMed/*` | 565 MB | PMC-CLIP 三件套，Fig 4 backbone 消融 |
| `/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/*/shots_16/GRAPH_DCPL_NO_TXT_VIS_GRAPH/` | 26.0 GB | 论文 Table 1/4 的 **Ours**：8 数据集 × 3 seed 的最终方法模型 |
| `/data/wangyijun/ckpt/Biomed_Graph_dcpl` 全部 `log.txt` | 69 MB | 2060 份日志，可复算 Table 1/4/5 |
| `/data/wangyijun/ckpt/Biomed_Graph_dcpl/heatmap` | 24 MB | 可视化产物 |
| `/data/wangyijun/ckpt/Biomed_Graph_dcpl_time` | 220 KB | 计时实验 |
| `/data/wangyijun/ckpt/MeDKCoOp` | 1.1 GB | 本次整理后重训验证的 BTMRI1 seed1 |

### BiomedCLIP 内部可顺手去重（约 1.9 GB）

| 子项 | 大小 | 说明 |
|---|---|---|
| `BiomedCLIP/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/` | 748 MB | HF cache 形式的重复权重；`bootstrap.py` 只读根目录的 `open_clip_pytorch_model.bin` |
| `BiomedCLIP/BiomedBERT/flax_model.msgpack` | 438 MB | Flax 权重，PyTorch 不用 |
| `/data/wangyijun/model/BiomedCLIP_local/` | 748 MB | 与 `BiomedCLIP` 重复的旧副本，且缺 tokenizer.json |

删除前建议先做一次离线加载自检（第 5 节最后一条），确认 `bootstrap.py` 仍可用。

## 3. 本项目可清理（约 1.39 TB）

| 路径 | 大小 | 删除后怎么复算指标 |
|---|---|---|
| `ckpt/Biomed_Graph_dcpl/ablation/train_base/*/shots_16/<非最终方法变体>/.../model*.pth*` | **783.9 GB** | 日志保留，Table 4/5 数值仍可复算；只是不能再 eval-only |
| `ckpt/Biomed_Graph_dcpl/few_shot/*/.../model*.pth*` | **129.0 GB** | few-shot 日志保留 |
| `ckpt/Biomed_Graph_dcpl_rebut/.../model*.pth*` | **130.4 GB** | rebuttal 日志保留 |
| `ckpt/BiomedCoOp/.../model*.pth*` | **145.4 GB** | baseline 日志保留 |
| `ckpt/DCPL_BiomedCLIP/.../model*.pth*` | **132.2 GB** | baseline 日志保留 |
| `ckpt/MaPLe/.../model*.pth*` | **59.5 GB** | baseline 日志保留 |
| `ckpt/GraphAdapter/.../model*.pth*` | **2.8 GB** | baseline 日志保留 |
| `ckpt/coop/.../model*.pth*` | **0.9 GB** | baseline 日志保留 |
| `/data/wangyijun/model/BLIP/` | 2.0 GB | BLIP trainer 已从仓库移除，论文未用 |
| BiomedCLIP 内部重复（见 2.1） | 1.9 GB | 去重 |

合计约 **1.39 TB**。

> 如果以后还想对某个变体做 eval-only，可以只保留每个变体 seed1 的模型（约 1.1 GB/数据集），
> 这样大约多留 30–40 GB，但不再需要 784 GB。

## 4. 与本项目无关，需你确认（约 3.29 TB）

这些是其他项目的产物，我没有动，也不建议在未确认前删除：

| 路径 | 大小 |
|---|---|
| `ckpt/MSCPT` | 2.6 TB |
| `ckpt/mgpath` | 473 GB |
| `ckpt/DiffCoOp`、`DiffCoOp_05NCE05diff`、`DiffCoOp_lamda05`、`DiffCoOp_noNCE` | 各 30 GB，共 120 GB |
| `ckpt/DiffMICv2` | 23 GB |
| `ckpt/DiffMIC` | 9.3 GB |
| `ckpt/cs336` | 3.0 GB |
| `ckpt/DiffusionClassifier` | 391 MB |
| `ckpt/{BiomedCLIP,clipadapter,lp,lp2}` | 各 520 KB（只有日志） |
| `ckpt/tipadapter` | 111 MB |

`/data/wangyijun/model` 下与本文无关的权重（合计约 63 GB）：
`KAD 22G`、`MVFA 17G`、`MAVL 8.7G`、`Gigapath 5.0G`、`Qwen 2.9G`、`hovernet 1.8G`、
`conch 1.5G`、`medsam2 1.1G`、`KG_RAG 932M`、`MGPATH 793M`、`sd-webui-forge-classic-neo 728M + zip 29M`、
`LRT 402M`、`EfficientSAM 195M`。

## 5. 建议的执行方式（确认后执行）

### 5.1 先归档日志（很小，约 100 MB）

```bash
BK=/home/wangyijun/backup/ckpt_logs_2026-11-05
mkdir -p "$BK"
for d in Biomed_Graph_dcpl Biomed_Graph_dcpl_rebut BiomedCoOp DCPL_BiomedCLIP MaPLe GraphAdapter coop; do
  find "/data/wangyijun/ckpt/$d" -name log.txt -print0 \
    | tar --null -czf "$BK/$d.logs.tar.gz" --files-from=-
done
```

### 5.2 删除本项目不需要的模型（先 dry-run）

```bash
# 1) 本项目方法：只保留最终方法 GRAPH_DCPL_NO_TXT_VIS_GRAPH
find /data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base \
  -name 'model*.pth*' \
  -not -path '*/GRAPH_DCPL_NO_TXT_VIS_GRAPH/*' -print   # 把 -print 换成 -delete 才真正删除

# 2) few-shot 模型
find /data/wangyijun/ckpt/Biomed_Graph_dcpl/few_shot -name 'model*.pth*' -print

# 3) rebuttal 模型
find /data/wangyijun/ckpt/Biomed_Graph_dcpl_rebut -name 'model*.pth*' -print

# 4) baseline 模型（日志已在 5.1 归档）
for d in BiomedCoOp DCPL_BiomedCLIP MaPLe GraphAdapter coop; do
  find "/data/wangyijun/ckpt/$d" -name 'model*.pth*' -print
done

# 5) 未使用的权重
rm -rf /data/wangyijun/model/BLIP
rm -rf /data/wangyijun/model/BiomedCLIP_local
rm -rf /data/wangyijun/model/BiomedCLIP/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
rm -f  /data/wangyijun/model/BiomedCLIP/BiomedBERT/flax_model.msgpack
```

### 5.3 删除后自检

```bash
conda activate medkcoop
cd /home/wangyijun/MeDKCoOp
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'EOF'
import bootstrap
from open_clip.src.open_clip import create_model_from_pretrained
m, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
print("BiomedCLIP 离线加载 OK")
EOF
# 再用最终方法的 checkpoint 跑一次 eval-only（docs/ENVIRONMENT.md 第 4 节）
```

## 6. 可选：与本项目无关的数据集（约 740 GB）

这些不是模型权重，只是磁盘占用较大，列出来供你决定：

| 路径 | 大小 |
|---|---|
| `dataset/MIMIC-CXR` | 574 GB |
| `dataset/NIH_CheXray` | 43 GB |
| `dataset/ImageNet100` | 33 GB |
| `dataset/ImageNet` | 30 GB |
| `dataset/Preumonia_ViP` | 24 GB |
| `dataset/Tumor` | 15 GB |
| `dataset/cs336` | 14 GB |
| `dataset/ISIC2018` | 5.2 GB |
| `dataset/Pneumonia_XCOOP` | 4.7 GB |
| `dataset/SIIM`、`dataset/Derm7pt`、`dataset/miniBTMRI*`、`dataset/CCBTM` 等 | 合计约 1 GB |

## 7. 建议的决策

1. **保留**：第 2 节全部（约 54 GB）。
2. **删除**：第 3 节（约 1.39 TB），先做 5.1 日志归档。
3. **待确认**：第 4、6 节属于其他项目/数据集，你确认后我再处理。

回复「按第 3 节删除」即可，我会先 dry-run 给你看命中列表和总量，再执行删除与自检。
