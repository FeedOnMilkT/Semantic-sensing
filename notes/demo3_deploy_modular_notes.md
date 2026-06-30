# Demo 3 · 分模块部署 de-risk 笔记（DFC 前半段 / host 侧）

> 日期：2026-06-08 · 范围：RTFormer → Hailo-8 部署可行性，**分模块、不做完整校验、先跑通导出**
> 关联：`deplyment-demo/CLAUDE.md`（总览）、`hailo_deployment_guide.md`（权威，§7 算子表）、`demo3_prompt.md`（落地顺序）
> 脚本：`tools/export_modules.py`（新建，不动训练主代码）

---

## 0. 一句话结论

RTFormer 的关键算子**全部能干净导出为静态 ONNX 并通过 onnxsim + onnxruntime 数值对齐**（max|torch−ort| ~1e-6）。
这是 DFC 链路的**前半段（host 侧、不上板、不进 Hailo 容器）**已跑通。真正的 `ClientRunner` parse/optimize/compile 是下一步，需进 Hailo suite 容器（本训练容器**没装 `hailo_sdk_client`**，按 guide §2 训练/部署环境本就隔离）。

**做了什么**：把 guide §7 的每个风险算子拎成独立小图，按 demo3_prompt 的风险顺序逐个导出 + 化简 + 数值校验。
**没做什么**：DFC parse/optimize/compile、量化掉点、上板——都在 suite 容器/Phase 3。

---

## 1. 环境关键事实（先记死）

- 本容器（训练环境，PyTorch 2.4）：`torch / onnx 1.16 / onnxsim 0.6.4 / onnxruntime 1.23` ✅；**`hailo_sdk_client` ❌ 未装**。
- 含义：这里只能跑到 **PyTorch → 静态 ONNX → onnxsim → ORT 数值对齐**（= DFC 的 NATIVE-emulator 等价校验的 host 预演）。
- DFC（`hw_arch="hailo8"`, 3.33.x）必须进 `./hailo_ai_sw_suite_docker_run.sh` 容器跑，ONNX 经共享文件夹 `./shared_with_docker` 进出。**别把 hailo 装进训练容器**（guide §2.1）。

---

## 2. 怎么跑

```bash
cd /workspace/demo/backbone-demo
python tools/export_modules.py                          # 5 个模块全导，512x1024
python tools/export_modules.py --only gfa cross_res_attn
python tools/export_modules.py --shape 1 3 512 1024 --out-dir deploy/onnx
```

产物：`deploy/onnx/*.onnx`（共 ~105MB，<100MB/个，可随时由脚本重生 → 可不入库）。
> 注意 `full_backbone.onnx` 67MB；都在 /workspace（小盘）但总量 105MB 可接受。若以后导更大/批量，按全局 CLAUDE.md 挪到 `/data/sidney/`。

---

## 3. 分模块结果（按风险顺序）

| 模块 | 角色 | 输入 shape | nodes | onnxsim | max\|torch−ort\| | 判定 |
|---|---|---|---|---|---|---|
| `conv_sanity` | stem+stage1~3 纯卷积，隔离环境 | (1,3,512,1024) | 72 | Y | 4.2e-07 | ✅ 零风险，链路通 |
| `gfa` | **#1 风险** GFA double-norm | (1,512,16,32) | 29 | Y | 2.4e-07 | ✅ 导出干净，待 DFC parse 裁决 |
| `cross_res_attn` | cross-res 注意力（softmax） | (1,128,64,128)+(1,512,16,32) | 51 | Y | 4.8e-07 | ⚠️ 导成裸 matmul，见 §4.2 |
| `rtformer_stage` | 一个完整 stepped block | 同上 | 94 | Y | 1.6e-06 | ✅ GFA+CA+2×FFN 合体可导 |
| `full_backbone` | 整网 | (1,3,512,1024) | 400 | Y | 8.9e-07 | ✅ 整网可导，feeds ClientRunner |

全部 max diff < 1e-3，导出与 ORT 推理**数值一致**——ONNX 层面无算子语义破坏。

---

## 4. op 覆盖分析（对照 guide §7，真正的风险裁决在这里）

### 4.1 GFA double-normalization（#1 风险）→ 乐观
ONNX 算子：`MatMul, Softmax, ReduceSum, Div, Reshape, Transpose, Add, Conv`。
- double-norm 落地为：`Softmax`(第一归一，over N tokens) + `ReduceSum`→`Div`(第二归一，grouped L1)——**都是标准 reduce + elementwise**，正是 guide §7 说"可用 Reduce+EW 拼出"的形态。
- **关键好消息**：两个 matmul（`n@kg.t()`、`attn@vg`）里 `kg/vg` 是 `nn.Parameter` → 在 ONNX 里是**权重 matmul（常量算子作初始化器）**，**不是 data×data**。即 guide §7 说的"冻结记忆 matmul 可重写为 1×1 conv"——天然成立，不踩"自由 matmul 不支持"的坑。
- **仍需亲手验**：double-norm 是 guide §7 唯一"未明确祝福"项。ONNX 能拼 ≠ DFC parser 一定接。**下一步：单独 `translate_onnx_model('gfa.onnx')` 看能否过 parse**。

### 4.2 cross_res_attn → ⚠️ 唯一要盯的点
ONNX 算子：`MatMul(×2), Softmax, Transpose, Reshape, Mul, Resize, Conv`。
- 这里 `q@k.T` 和 `attn@v` **两个操作数都是 data**（K/V 来自 low 分支）→ 这是 guide §7 说的 **data×data matmul，仅 MHA block 内支持**。
- **风险**：导出的是**裸 `MatMul+Softmax+MatMul` 链，不是 Hailo 的融合 MHA 算子**。DFC parser 能否把这个 pattern 识别/映射成它一等支持的 Multi Head Attention，**是 Demo 3 真正要验的事**。
- **缓解预案**（按 guide §7 + §6.3）：① 若 parser 不认，照官方 MHA 示例的 `flatten(2)+permute` 固定 reshape 模式对齐写法；② 量化掉点靠 §6.3 transformer 组合拳（`layer_norm_decomposition`+`smart_softmax_stats`+`matmul_equalization/correction/decomposition`+`quarot`），别只靠 16-bit（softmax chain 不让提 16-bit）。
- 注意 `Resize`（cross-feature 固定 12×12，用 `interpolate` 非 `adaptive_pool`，已是 Hailo 友好的静态 Resize）；`align_corners=False` 对应 guide §7 的 align_corners/half_pixel 组合限制，需在 parse 后确认。

### 4.3 full_backbone → 多了 DAPPM/动态轴残留
整网额外算子：`AveragePool, GlobalAveragePool`(DAPPM 的 AvgPool2d k=5/9/17 + AdaptiveAvgPool2d(1)) ✅ 支持；
`Gather, Cast, Unsqueeze, Shape, Slice`——来自 `F.interpolate` 的动态尺寸推断残留。
- 静态 shape 导出 + onnxsim 已把大部分折掉，但仍残留少量 Shape 系算子。**DFC parse 时若卡在这些动态算子上**，按 guide §4 用 `hailo visualizer` 看图定位，必要时固化/`end_node_names` 切分。

---

## 5. 与切分架构的衔接（guide §11）

bottleneck 在 **stage3 后（conv→attention 边界）**：发送端纯卷积、attention 在接收端 decoder，两段各自 parse/optimize/compile（对应两块对称 Hailo-8）。
- `conv_sanity` 子图 ≈ **发送端 encoder** 的算子画像（纯卷积，零风险）→ 上板侧最稳。
- `gfa / cross_res_attn / rtformer_stage` ≈ **接收端 decoder** 的算子画像 → 风险集中在这半边。
- 整网切分用 `translate_onnx_model(..., end_node_names=[stage3 输出节点])`，两段独立成 HAR。本脚本已把两半的算子风险分别隔离验过。

---

## 6. 下一步（进 Hailo suite 容器，本环境做不了）

逐模块按 guide §3 状态机走（先小后大，隔离风险）：
```python
runner = ClientRunner(hw_arch="hailo8")
hn, npz = runner.translate_onnx_model("gfa.onnx", "gfa")     # ← 先验 #1 风险
runner.save_har("gfa_hailo_model.har")
# NATIVE emulator 验数值对齐 → optimize(calib≥1024, NHWC) → QUANTIZED 验掉点 → compile()
```
顺序建议：`conv_sanity`(环境) → `gfa`(#1) → `cross_res_attn`(MHA 映射) → `rtformer_stage` → `full_backbone`（必要时 `end_node_names` 切分）。
- 整网 compile 后**先看 `# of Contexts`**——RTFormer 对 26 TOPS 是否偏大的早期信号（guide §8/§9）。
- 掉点 >2 mIoU → guide §5.4 十一步 + §6.3 transformer 组合拳。

---

## 7. 对 Demo 3 验收线的进度

| Demo 3 checkpoint | 状态 |
|---|---|
| ① 关键 block（尤其 GFA double-norm）能过 DFC parse | ⏳ ONNX 就绪，parse 待 suite 容器 |
| ② NATIVE emulator 数值与 PyTorch 对齐 | 🟢 host 预演已过（ORT vs torch ~1e-6） |
| ③ 量化掉点 < 2 mIoU | ⏳ 未做（需 optimize/QUANTIZED） |
| ④ compile 成功出 HEF + 记 # of Contexts | ⏳ 未做 |
| ⑤（可选）单 context 时 Profiler FPS/latency | ⏳ 未做 |

**本轮交付**：① 分模块静态 ONNX（DFC-ready）；② 算子覆盖逐项对照 guide §7，定位出唯一真正要盯的点（cross_res_attn 的裸 matmul→MHA 映射）；③ 可复现脚本 `tools/export_modules.py`。
**达标即停原则**：不追完整量化校验，把 GFA/MHA 的 parse 可行性留给 suite 容器一次性验。
