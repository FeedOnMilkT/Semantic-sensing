# Demo 3 · 整网(backbone + seg/depth 头)结构 de-risk 笔记

> 日期：2026-06-09 · 范围：完整多任务网络过 DFC **parse + NATIVE 对齐**，预训练前最后一道结构验证
> 关联：`deplyment-demo/`（代码骨架 + 报告）、`hailo_deployment_guide.md`（§7 算子表）、
> 上一轮分模块笔记 `demo3_deploy_modular_notes.md`（只验了骨干主干）
> 代码：复用 `deplyment-demo/hailo_deploy/` 同一套 pipeline，只换输入 ONNX + 追加报告

---

## 0. 一句话结论

完整网络（RTFormer backbone + seg + depth 两个头）**过 DFC parse、NATIVE 与 ONNXRuntime 数值对齐**
（out_seg 4.5e-5 / out_depth 2.2e-4，均 < 1e-3）。**结构层面无阻断项 → 放行预训练。**

这次相比骨干轮新增验的是**两个头的尾部算子在【完整图】里**的落地：Resize 上采样、depth 尾
sigmoid×max_depth、头部 depthwise —— 全部正常。

---

## 1. 怎么跑（复用上一轮 pipeline）

```bash
cd /workspace/demo/deplyment-demo    # 在 Hailo suite 容器内（DFC 3.33.0, hw_arch=hailo8）
python scripts/run_pipeline.py --only full_net_seg_depth --section "整网 backbone+heads（seg+depth）"
```
- ONNX：`backbone-demo/deploy/onnx/full_net_seg_depth.onnx`（image[1,3,512,1024] → out_seg[1,19,512,1024], out_depth[1,1,512,1024]，静态）
- 复用：`hailo_deploy/{config,parse,emulate,onnx_io,reporting}.py`；本轮仅
  ① config 登记新模块；② emulate 输入/输出改按 `original_names` 配（findings §1.3）；
  ③ reporting/run_pipeline 加 `--section` 追加模式。**没重写 pipeline。**
- 产物：`artifacts/har/full_net_seg_depth_hailo_model.har`、
  `artifacts/results/full_net_seg_depth.json`、报告追加进 `artifacts/reports/{pipeline,findings}.md`。

---

## 2. 结果

| 项 | 结果 |
|---|---|
| parse | ✅ PASS |
| NATIVE out_seg max\|diff\| | 4.51e-05 |
| NATIVE out_depth max\|diff\| | 2.21e-04 |
| 判定 | ✅ PASS（< 1e-3） |

HN 算子集（无一未支持/未映射）：
`activation, avgpool, concat, conv, dense, dw, ew_add, ew_mult, external_pad,
format_conversion, input_layer, matmul, normalization, output_layer, reduce_sum,
resize, softmax`

---

## 3. 头尾部算子落地裁决（本轮重点）

| 部件 | ONNX | HN | 验证 |
|---|---|---|---|
| 各头 bilinear 上采样（align_corners=False） | Resize×10 | `resize` | ✅ 该 align_corners 组合过 parse；out_seg 经其到全分辨率，对齐 4.5e-5 |
| depth 尾 sigmoid × max_depth | Sigmoid + Mul | `activation` + `ew_mult` | ✅ out_depth 对齐 2.2e-4 |
| 头部 depthwise 3×3 | Conv(group) | `dw` | ✅ 受支持 |
| GFA double-norm（带头整网复核） | Softmax+ReduceSum/Div | `softmax`+`reduce_sum`+`normalization` | ✅ 加头不破坏 |
| attention（带头整网复核） | MatMul+Softmax | `matmul`+`softmax` | ✅ |

---

## 4. caveat / 待改（非阻断）

1. **输出按 shape 兜底配，未走按名配**：本 SDK 版本 NATIVE `infer` 返回 list 不是 dict，
   findings §1.3 基于 `original_names` 的按名匹配（dict 路径）未触发。但 out_seg(19ch)/
   out_depth(1ch) shape 互异 → shape 配唯一且非靠顺序，极小 diff 反证没配错。
   待改：list 路径下按「HN 输出层顺序 ↔ 返回顺序」归名（需确认该版本返回顺序约定）。
2. **量化未做**（本轮按要求不做校准）：Resize / sigmoid / normalization 的**量化**精度
   未验，与掉点 ③ 一起留到有校准集时。

---

## 5. 对 Demo 3 验收线的进度（结构验证收尾）

| checkpoint | 状态 |
|---|---|
| ① 关键 block + 整网(含头) 过 DFC parse | ✅ 达成 |
| ② NATIVE 数值对齐 | ✅ 达成（骨干 + 整网带头均过） |
| ③ 量化掉点 < 2 mIoU | ⏳ 待校准集 |
| ④ compile 出 HEF + # of Contexts | ⏳ 待 ③（compile 需先量化） |

**结论：结构 de-risk 完成，放行预训练。** 量化/编译是预训练后、有校准集时的下一道关。
