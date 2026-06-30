# RTFormer depth · fusion-neck A/B + 总定论

> 日期：2026-06-10 · 范围：给 RTFormer depth 换多尺度融合 neck,验证"分辨率瓶颈"假说;汇总三个受控实验的总结论
> config：`configs/cityscapes_depth_rtformer_fusion.yaml`(由单任务 passthrough 版派生,只换 neck + head upsample)
> 权重：`/data/sidney/checkpoints/demo1/cs_depth_rtformer_fusion.pth`｜日志：`logs/cs_depth_rtformer_fusion.log`

## 0. 一句话结论

把 MiT-B0 同款"四尺度融合到 1/4"接给 RTFormer depth,**指标纹丝不动**(δ1 0.717→0.705,RMSE 8.39→8.38)。**分辨率假说证伪。** 叠加前两个实验 → depth 弱是 **RTFormer backbone 特征本身**的问题,非初始化/竞争/neck。→ depth 要紧用 SegFormer/MiT-B0;RTFormer 接受 ~8m 天花板(实时/Hailo 的代价)。

## 1. fusion vs passthrough vs MiT-B0

| 指标 | passthrough(s8,1/8) | fusion(s4-s32,1/4) | MiT-B0 |
|---|---|---|---|
| RMSE (m) | 8.389 | 8.384 | 5.70 |
| AbsRel | 0.171 | 0.178 | 0.102 |
| δ1 | 0.717 | 0.705 | 0.889 |
| δ2/δ3 | .914/.976 | .921/.978 | .977/.993 |

唯一变量=neck(passthrough→mlp_fusion,head upsample 8→4)。fusion 比 passthrough 慢约 +35%(算力 ~4× neck/head 像素),换来 0 收益。

## 2. 三个受控实验汇总(全部排除廉价修法)

| # | 改的变量 | δ1 轨迹 | 排除 |
|---|---|---|---|
| 1 | backbone 从零→ImageNet预训练 | 0.749→0.696 | 初始化 |
| 2 | 多任务→单任务(去 seg 竞争) | →0.717 | 多任务竞争 |
| 3 | passthrough→fusion neck | →0.705 | neck/分辨率 |
| 对照 | MiT-B0 同 setup 同 fusion | **0.889** | (数据/neck 都支持) |

所有 RTFormer 变体钉在 δ1 ~0.70-0.75 / RMSE ~7.8-8.4;MiT-B0 在同数据是 0.889。

## 3. 定论:RTFormer backbone 本身

实验 3 后 RTFormer-fusion 与 MiT-B0 **neck 已一致**(都 mlp_fusion→1/4),剩 0.18 δ1 差距**只能归 backbone**。RTFormer 为实时分割(语义)设计——DDRNet 式双分辨率 + DAPPM + GPU 友好注意力,特征强语义弱几何细节;s4 仅来自浅 stem+2 block,低容量早期特征。给再好的 neck 也喂不出 MiT 那种适合稠密度量回归的特征。这是架构取舍 = 它快/可上 Hailo 的代价。
- caveat:本轮只换 neck,decoder 仍 identity(未加 MiT-B0 的 conv_decoder)。但 neck 对齐后仍差 0.18,小处理补不回,结论不变。

## 4. 决策

- **depth 质量要紧 → SegFormer/MiT-B0**(5.70m / δ1 0.889)。
- **要实时 + Hailo + 可接受 depth ~8m → RTFormer**(多任务版 7.98<8m 达标,够当 Phase 2 teacher)。
- 不再投:权重/DWA、backbone 预训练、neck 改造——三条路都验过无效。
