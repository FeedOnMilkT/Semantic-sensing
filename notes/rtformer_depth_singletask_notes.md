# RTFormer 单任务 depth · 控制变量诊断笔记

> 日期：2026-06-10 · 范围：单训 depth(去掉 seg),与多任务做受控对比,判定 depth 弱 = 多任务竞争 还是 结构/数据天花板
> config：`configs/cityscapes_depth_rtformer_pretrained.yaml`(由多任务配置程序化派生,只删 seg)
> 权重：`/data/sidney/checkpoints/demo1/cs_depth_rtformer_pretrained.pth`｜日志：`logs/cs_depth_rtformer_pretrained.log`
> 代码改动：`src/datasets/cityscapes.py` 给 seg 加 5 处 None 守卫(支持 depth-only;seg 在场时行为不变)

## 0. 一句话结论

去掉 seg 竞争、depth 独占 backbone 484ep,**depth 没救回来**(δ1 +0.02,RMSE 反更差)。
→ **depth 弱不是多任务竞争**,是 **RTFormer 单尺度 s8 head 丢分辨率 + Cityscapes stereo-GT 天花板**。调权重/DWA 不用做。

## 1. 三方对照(控制变量:同 backbone/预训练/recipe/eff-batch16,唯一变量=有无 seg)

| 指标 | 多任务(预训练) | 单任务 depth | MiT-B0(同 setup) | bar |
|---|---|---|---|---|
| RMSE (m) | 7.98 | **8.39** ⬆ | 5.70 | <8 |
| AbsRel | 0.179 | 0.171 | 0.102 | — |
| δ1 | 0.696 | 0.717 | 0.889 | — |
| δ2 / δ3 | .932/.982 | .914/.976 | .977/.993 | — |

## 2. 推理链

- 竞争假说**证伪**:单任务该大涨却没有(δ1 仅 +0.02,RMSE 更差)。
- 数据**支持**更好深度:同数据同 setup,MiT-B0 δ1 0.889 / RMSE 5.70。
- RTFormer 单/多任务都卡 ~0.7 → **落在数据天花板之下 = 结构问题**。
- **结构根因定位**:本 paper-faithful 配置 depth head 只吃单个 **stride-8 粗特征**(neck=passthrough s8 → identity decoder → ×8 上采样)。MiT-B0 用 `mlp_fusion` 融合 s4/s8/s16/s32,depth head 能看 s4 精细结构。**深度需要精细分辨率,RTFormer 在入 head 前已降采样丢失。** 这是它快/可上 Hailo 的同一个取舍。

## 3. 副发现:seg 是辅助正则,不只是竞争者

单任务训练 loss 更低(ep400 0.73 vs 多任务 ~1.2)、δ1/AbsRel 略好,但 **RMSE 反更差**。两因:
1. SiLog **尺度不变** → 练狠了改善相对指标(δ1/AbsRel),不直接压绝对 RMSE;
2. 单任务仅 2975 张、失去 seg 辅助信号 → 略过拟合。
→ 多任务里 seg 之前还帮 depth 稳住了 RMSE。

## 4. 修正诊断 & 下一步(按性价比)

- ❌ 调任务权重 / DWA：竞争已证伪,**别做**。
- ✅ 唯一有效杠杆 = **改结构**：给 depth 路径加多尺度融合 / 接 s4 精细特征(代价:偏离 paper-faithful + 加算力)。建议先小改 depth head 接 s4，做 A/B。
- ⏸ NYU(干净 GT)对照：优先级降低——MiT-B0 对照已基本证明"数据支持、瓶颈在结构"。仅当要确认"RTFormer 在干净 GT 上的深度上限"时才做(需先写 NYU loader)。
- ✅ 交付层面:多任务版 depth 7.98 < 8m **达标**;单任务 8.39 仅诊断用,非交付。
