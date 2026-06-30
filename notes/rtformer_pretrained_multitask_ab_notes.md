# RTFormer 多任务 · ImageNet 预训练 A/B 验证笔记

> 日期：2026-06-09 · 范围：用 ImageNet-1K 预训练的 RTFormer backbone 重跑 paper-faithful 多任务(seg+depth),与从零的 `faithful` 做受控对比
> config：`configs/cityscapes_mt_rtformer_pretrained.yaml`(_base_ = `cityscapes_mt_rtformer.yaml`,只改 backbone 初始化 + per-GPU batch)
> 权重：`/data/sidney/checkpoints/demo1/cs_mt_rtformer_pretrained.pth`(68M)｜日志：`logs/cs_mt_rtformer_pretrained.log`

## 0. 一句话结论(与预期相反,如实记)

ImageNet 预训练**大幅提升 seg(+3.3 mIoU → 73.3%,逼近 SegFormer 74.5),但 depth 反而略降**(RMSE 7.77→7.98,δ1 0.749→0.696)。
→ **depth 的弱不是 backbone 初始化造成的**(预训练没救回来),主因是**多任务容量竞争 + uncertainty weighting 偏袒 seg**。预训练只是把这个 seg-depth 权衡推到了更极端的一端。

## 1. 受控对比(唯一变量 = backbone 初始化)

| 指标 | faithful(从零) | **本轮(预训练)** | Δ |
|---|---|---|---|
| seg mIoU | 0.7002 | **0.7328** | **+3.3** ✅ |
| seg pixel_acc | 0.9506 | 0.9534 | +0.3 |
| depth RMSE (m) | 7.774 | **7.981** | +0.21 ❌ |
| depth AbsRel | 0.1549 | **0.1787** | ❌ |
| depth δ1 | 0.749 | **0.696** | −0.053 ❌ |
| depth δ2 / δ3 | .931/.978 | .932/.982 | ~持平 |

三个 depth 指标同向变差 → 真实方向性效应,非噪声。

## 2. RTFormer 多任务里的 seg–depth 权衡(全量对照)

| run | backbone | seg mIoU | depth RMSE | depth δ1 |
|---|---|---|---|---|
| `cs_mt_rtformer`(200ep, 从零) | RTFormer | 64.1 | 7.06 | 0.817 |
| `faithful`(484ep, 从零) | RTFormer | 70.0 | 7.77 | 0.749 |
| **本轮(484ep, 预训练)** | RTFormer | **73.3** | **7.98** | **0.696** |
| 对照:`segformer`(200ep, 预训练) | MiT-B0 | 74.5 | **5.70** | 0.889 |

**RTFormer 内部:seg 越好(64→70→73),depth 单调越差(7.06→7.77→7.98,δ1 0.817→0.749→0.696)。** 强证据 = 共享 backbone 的容量竞争里 **seg 持续赢、depth 持续被饿**。预训练给的是**语义**特征(分类学来的),对 seg(语义任务)极有利,但对 depth(几何/度量任务)无益甚至有害,并通过 uncertainty weighting 进一步把容量拉向 seg。

**架构层面**:MiT-B0(SegFormer)depth 一直明显更好(5.70 vs ~7-8),说明 RTFormer 的双分辨率 + DAPPM-s8 融合结构对 dense 度量深度本就不如 MiT 的层级特征友好——这与是否预训练无关。

## 3. 诊断更新

- ❌ 旧假设"depth 弱 = backbone 从零、表征不足" → **被本实验部分证伪**:换上 71.8% 的预训练 backbone,depth 没改善。
- ✅ 主因锁定:**多任务容量竞争 + uncertainty weighting 偏袒 seg**(次因:RTFormer 结构对 depth 不友好;Cityscapes disparity 深度 GT 在远处/天空噪声大)。
- 注:本配置共享到 s8、两个 head 只各一个 3×3+1×1(任务专属容量极少),放大了竞争。

## 4. 配置/资源

- 4×A100(卡4-7),per-GPU batch 4 → effective 16(对齐 faithful),lr 4e-4,wd 0.0125,poly,uncertainty weighting,484ep,89540 iters。
- 显存仅 ~4.7GB/卡(6%);耗时 ~5h(11:42→16:42),比 2 卡 faithful 快约 2×。
- 训练中一次单步 loss 尖峰(ep100,4.56)即时自愈,无发散无 NaN。

## 5. 想救 depth 的下一步(按性价比,未做)

1. **改任务权重**:打印 uncertainty 学到的 σ 看实际偏袒;或试 DWA / 固定权重抬高 depth;或 depth 单任务上限对照(看多任务到底拖了多少)。
2. **给 depth 专属容量**:两 head 提前分叉 / depth 侧加 decoder 层,减少与 seg 抢 backbone。
3. **depth 配方**:抬 grad 权重、或屏蔽远处/天空无效区重算指标。
4. 若只在意 seg:本轮已基本达 SegFormer 水平,可直接用。
