# RTFormer · ImageNet-1K 预训练笔记

> 日期：2026-06-09 · 范围：给 RTFormer backbone 补 ImageNet-1K 分类预训练权重，供下游 seg/多任务以 `backbone.args.pretrained` 加载
> 环境：`sidney-dev` 容器（NGC pytorch 25.02，torch 2.4.0a0，py3.10），4×A100-80G（卡 4/5/6/7）
> 脚本：`tools/pretrain_imagenet.py`（已有，本轮小改：加 `--snapshot-every` 周期快照 + 减轻 val loader 的 shm 占用）

---

## 0. 一句话结论

RTFormer trunk（stem→stage5，经 `forward_trunk` 取 s32）+ linear 分类头，ImageNet-1K 训 **100 epoch**，
**val top1 = 71.82%**，7h40m 跑完。这是 Demo1 里 RTFormer 缺的那块 ImageNet 预训练权重（MiT-B0 早有，RTFormer 之前从零 → 是 seg mIoU 偏低归因之一）。**可直接放行下游 seg 微调。**

为什么不追更高（paper ~74-75%）：目标是"够格当下游 seg 的 teacher backbone"，不是刷分类 SOTA。达此目的即停，符合 CLAUDE.md"达 bar 即停"。

---

## 1. 最终结果与收敛曲线

| epoch | 5 | 10 | 15 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top1 | .418 | .533 | .573 | .599 | .634 | .651 | .660 | .678 | .696 | .707 | .716 | **.718** |

平滑单调上升，无塌陷/过拟合迹象；后段 cosine 退火（lr 2.8e-3→1e-5）贡献稳定收尾增益。

---

## 2. 关键配置（修正后）

| 项 | 值 | 说明 |
|---|---|---|
| GPU | 4,5,6,7 | 避开 GPU0（团队常占）、**GPU2（坏卡 730 ECC）** |
| per-GPU batch | **1024** | global 4096 |
| lr / 调度 | 2.8e-3 / cosine + 5ep warmup | √ 缩放（基线 2e-3@2048 → 4096）|
| optimizer | AdamW，wd 0.05 | label smoothing 0.1 |
| workers | train 10/卡，val 2/卡非常驻 | 见 §3 shm |
| AMP | fp16 autocast + GradScaler | |
| 吞吐 | 0.84 s/step，~4900 img/s，4 卡 100% util | GPU-bound（非 dataloader-bound）|

启动命令：
```bash
HF_HOME=/data/sidney/hf-cache CUDA_VISIBLE_DEVICES=4,5,6,7 \
  torchrun --nproc_per_node=4 --master_port=29712 tools/pretrain_imagenet.py \
  --epochs 100 --batch-size 1024 --lr 2.8e-3 --warmup-epochs 5 \
  --workers 10 --eval-every 5 --snapshot-every 20 --resume \
  --out /data/sidney/checkpoints/pretrained/rtformer_in1k.pth
```
中断可加 `--resume` 从 `.last.pth` 续跑（含 opt/scaler/epoch）。

---

## 3. 踩坑：64G /dev/shm（重要，重训必看）

**首次 batch 2048 + 16workers 启动即 SIGBUS**：`Unexpected bus error ... insufficient shared memory (shm)`，torchrun 退出码却是 0（假成功）。

- 根因：DataLoader 经 shm 传 collated batch，占用 ≈ `batch × 3·224·224·4B × workers × prefetch × procs`。
  2048×16×2×4 ≈ 156G ≫ 容器 **64G** `/dev/shm`。`persistent_workers=True` 的 val loader 与 train 同时常驻 → shm 翻倍。
- 容器内 **无法扩 shm**（`mount -o remount,size=` 报 permission denied，没 CAP_SYS_ADMIN）；扩只能 host 侧 `--shm-size` 重建容器。
- 本轮修法（不重建容器）：global batch 降到 4096 + train workers 10 + **val loader 改 `persistent_workers=False` & 2 workers**。shm 峰值 ~47G（eval 时 ~56G），稳在 64G 内。
- 认知点：64G shm 逼着"大 batch 吃显存"与"多 worker 高吞吐"二选一。本任务 batch 1024 时 4 卡已 100% util（算力满载），**显存只用 35% 不是浪费——显存空 ≠ 卡闲**。

---

## 4. 产物

落点 `/data/sidney/checkpoints/pretrained/`：

| 文件 | 内容 | 用途 |
|---|---|---|
| `rtformer_in1k.pth` | backbone-only（best/最终，67M） | **下游 seg 加载这个** |
| `rtformer_in1k.last.pth` | model+backbone+opt+scaler（190M）| `--resume` 续训 |
| `rtformer_in1k.ep{20,40,60,80,100}.pth` | 周期 backbone 快照（各 67M）| 回溯/对比 |

加载校验（已通过）：
```python
from src.models.backbones import RTFormer
m = RTFormer(); m.load_pretrained('/data/sidney/checkpoints/pretrained/rtformer_in1k.pth')
# missing=0 unexpected=0（含未训练的 DAPPM 键，seg 阶段会微调）
```
即 YAML 里 `backbone: {name: rtformer, args: {pretrained: /data/sidney/checkpoints/pretrained/rtformer_in1k.pth}}`。

---

## 5. 下一步

- 用此权重重跑 `cityscapes_seg` / `cityscapes_multitask`，验证 seg mIoU 是否较 Demo1（RTFormer 从零）回升——这是当初 60.98 的归因项之一（另两项：无 seg 数据增强、decoder/head 从零，见 `demo1_notes.md`）。
- DAPPM 在预训练里未参与（`forward_trunk` 不走），seg 阶段从随机初始化微调，符合设计。
