# RTFormer 单任务 seg · 笔记

> 🔔 **2026-06-30 方向更新**：项目已收敛为「主用 RTFormer + 先只做单任务 seg」，本笔记的单任务线
> 从"补对照"升为**当前主线**。§3 里"被暂停、先统一 teacher 再回头"的待办随之取消——直接用对齐
> 原文配方（§2）的 `cityscapes_seg.yaml` 把单任务天花板跑出来。详见
> `notes/direction_2026-06-30_rtformer_singletask.md`。

> 日期：2026-06-14（2026-06-15 重大更正）· 范围：补 RTFormer 纯单任务分割 run。
> ⚠️ **本次结果作废为"不可比的半残基线",见 §0。**
> ckpt：`/data/sidney/checkpoints/demo1/cs_seg_rtformer.pth`｜日志：`logs/cs_seg_rtformer.log`
> 硬件：4×A100（GPU 4-7）DDP，eff batch 16，~2h37m。

## 0. 结论（已更正）

跑出 **64.1 mIoU**，但这是用**遗留半残配方** `cityscapes_seg.yaml` 得到的，**与多任务 73.3 / SegFormer teacher 74.54 完全不可比**。两者差了约 6 个变量：

| 维度 | 本次单任务(`cityscapes_seg.yaml`) | 多任务 73.3(`cityscapes_mt_rtformer.yaml`) |
|---|---|---|
| 数据增强 | ❌ 无（仅 resize） | ✅ scale0.5-2.0 + crop + hflip |
| 训练输入 | 半分辨率 resize 512×1024 | 全分辨率 native + 随机缩放裁剪 |
| **eval 分辨率** | **半分辨率 512×1024** | **全分辨率 1024×2048** |
| neck | mlp_fusion | passthrough (s8) |
| decoder | conv_decoder (3层) | identity |
| seg head | upsample 4 | upsample 8 + mid(3×3) |
| depth 头 | 无 | 有 |
| 权重 | fixed | uncertainty |

## 1. 因此

- **不能**从 64.1 得出"卸掉 depth 让 seg 掉点 / depth 是正则项"——之前那个结论**已收回**，因为根本没控变量。
- **RTFormer 单任务 seg 的真实天花板仍未测出。**
- LR 不是问题：原文(arXiv:2210.07124)Cityscapes 用 **AdamW + lr 4e-4 + wd 0.0125 + poly0.9 + 484ep**，我们完全一致；唯一实质偏离是**数据增强**(原文有，遗留 seg config 无)。loss 轨迹平滑收敛到 0.05、无震荡，也排除 LR 过大。

## 2. RTFormer 原文 Cityscapes 配方（arXiv:2210.07124，已联网核实）

AdamW / lr 4e-4 / wd 0.0125 / poly power 0.9 / 484 epochs(~120K iters) / batch 12(4×V100 syncBN) /
augment: random scale 0.5–2.0 + random crop 512×1024 + hflip / **OHEM 不用** / ImageNet 预训练。

## 3. 待办（被 Sidney 暂停）

Sidney 决定先**统一制定完善训练策略，再统一重训最完美的 teacher**，然后才回头补单任务对照。
真要测单任务 seg 天花板时：用对齐 `cityscapes_mt_rtformer.yaml` 的配方，仅去掉 depth 头（增强已在 dataset 实现，开 `augment:true` 即可）。
