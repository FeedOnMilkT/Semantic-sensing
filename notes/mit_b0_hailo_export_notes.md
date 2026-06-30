# MiT-B0 (SegFormer) · Hailo 分模块导出 + 可编译性裁决

> 日期：2026-06-10 · 范围：MiT-B0 逐模块导出静态 ONNX + onnxsim + ORT 对齐,对照 Hailo §7 算子表判可编译性
> 动机：depth 主线选了 SegFormer(MiT-B0,§见 rtformer_depth_fusion_ab_notes.md),但 MiT 是 transformer——
>   其算子(LayerNorm/动态 MatMul/Softmax/5D permute/GELU)正是 RTFormer 当初为规避而被选中的那些。换线前必须验它能否上 Hailo。
> 工具：`tools/export_modules_mit.py`(新建,仿 `export_modules.py`)｜产物：`deploy/onnx_mit/*.onnx`
> 权威依据：`deplyment-demo/hailo_deployment_guide.md` §7(DFC v3.33.1 Supported Layers)、§6.3(transformer 命令)

## 0. 一句话裁决

MiT-B0 的算子**对照 §7 基本在支持集内,不是明显拦路虎**:标准 softmax MHA 是 Hailo **一等支持**、LayerNorm 可拼(有专用 `layer_norm_decomposition`)、GELU/conv/reshape 皆可。**反直觉:MiT 的标准 MHA 可能比 RTFormer 的 GFA grouped double-normalization(demo3 真正 #1 风险)更干净。** 但最终 yes/no 必须进 suite 容器 parse(本容器无 `hailo_sdk_client`)。

## 1. 导出结果(本容器,host 侧)

6 模块全部导出干净、ORT 对齐 ~1e-6,无导出 bug:

| 模块 | nodes | max\|torch−ort\| | 风险算子 |
|---|---|---|---|
| patch_embed | 10 | 2.4e-6 | ReduceMean(LN) |
| attn_sr(stage1,2头,SR4) | 31 | 1.3e-7 | MatMul, Softmax, ReduceMean |
| attn_full(stage4,8头,SR1) | 19 | 8.6e-8 | MatMul, Softmax |
| mix_ffn | 14 | 3.6e-7 | Erf, MatMul |
| block | 61 | 6.0e-7 | 全部 |
| full_backbone | 540 | 6.9e-6 | 全部 |

risk ops 汇总:`Erf, MatMul, ReduceMean, Softmax`(LayerNorm 在 opset13 分解为 ReduceMean/Sub/Pow/Sqrt/Div)。

## 2. 对照 §7 裁决

| MiT 算子 | 支持 | guide 依据 |
|---|---|---|
| EfficientAttention = softmax MHA | ✅ 一等支持 | §7 L237(官方 PyTorch 示例,K/Q/V 数据驱动) |
| 动态 q@k^T / attn@v | ✅ 仅 MHA 内 | §7 L238(自由 data×data matmul 不支持) |
| LayerNorm | ✅ 可拼+专用命令 | §7 L240 + §6.3 layer_norm_decomposition |
| GELU/Erf、Linear、dwconv、Conv | ✅ | LUT / conv 类 |
| reshape flatten(2)+permute | ✅ 最稳 | §7 L251 |

## 3. 真正待验项(suite 容器,risk 序)

1. **attn_sr.onnx** 先验:① SR 使 Q(N) 与 K/V(N/sr²)**非对称长度**,DFC 的 MHA 识别能否吃;② kv 拆分是 **5D permute**(reshape(B,-1,2,heads,d).permute(2,0,3,1,4)),§7 只祝福 4D flatten+permute,5D 可能要重写。
2. **block.onnx**:LayerNorm 分解过 parse。
3. **full_backbone.onnx**:整网。
4. 量化精度:softmax chain 不能简单 16-bit(§7 L183),开 §6.3 transformer 组合拳(smart_softmax_stats / matmul_equalization / matmul_correction / matmul_decomposition)+ 敏感层 16-bit。

parse 命令(每模块):`ClientRunner(hw_arch='hailo8').translate_onnx_model('<m>.onnx','<m>')` → 不支持算子在此暴露。

## 4. 成本提醒

MiT 是端到端 transformer(4 stage × 2 block = 8 attention + 满网 LayerNorm),vs RTFormer 仅少数 attention 块。即便可编译,**16-bit 敏感层更多 → HEF 更大/更慢、量化调参更费**。这是"换 SegFormer 救 depth"在部署侧的隐性代价,需与 depth 收益(δ1 0.717→0.889)权衡。

## 5. 边界(没做的)

本容器只到 PyTorch→ONNX→onnxsim→ORT 对齐;**DFC parse/optimize/compile、量化掉点、上板**都在 suite 容器/Phase 3,未做。

---

## 6. suite 容器 parse 实测结果(2026-06-10,DFC 3.33.0)

> 用 `deplyment-demo/` 同一套 pipeline 跑 parse + NATIVE。完整记录见
> `deplyment-demo/artifacts/reports/findings.md §6` 与 `results/mit_b0.json`。

**parse:6/6 全过 ✅**。§3 的待验项裁决:

| §3 待验项 | 实测落地 | 裁决 |
|---|---|---|
| ①SR 非对称 Q/KV 长度 DFC 能否吃 | parse 未报错 | ✅ 接受 |
| ②kv 拆分 5D permute(§7 只祝福 4D) | 映射为 **`feature_splitter`** | ✅ **直接吃掉,5D 顾虑解除** |
| LayerNorm 分解过 parse | 映射为专用 **`layer_normalization`** | ✅ 比 reduce+EW 更干净 |
| GELU/Erf | `activation` | ✅ |

**反直觉印证**(§0):MiT 标准 MHA 的 parse 比 RTFormer GFA double-norm 还干净。

**NATIVE:暂不可信** —— `deplyment-demo` 的 emulate harness 按 4D 图像写,MiT 子模块是
3D token `[1,N,C]`,转换/配对逻辑崩(非 MiT 问题)。唯一 4D 进出的 full_backbone 跑通,
per-output 深度递增 5.5e-5→2.8e-3(深 transformer FP 累积,略超 1e-3 阈)。待 harness 升级。

**仍未做**:量化掉点、compile(HEF + # of Contexts)—— 需校准集。# of Contexts 是 §4
"MiT 是否对 26 TOPS 偏大"的关键信号,留待有校准集时。
