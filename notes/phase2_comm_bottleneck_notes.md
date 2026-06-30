# Phase 2 · 通信 bottleneck 实现笔记（2026-06-30）

> 把 demo2（`demo/bottleneck-demo/demo2/`）验过的监督压缩部件**完整**移植进正式工程的
> `bottleneck` 槽。非简化版：用真正的 CompressAI `EntropyBottleneck`（含 rANS range coder、
> 学习熵模型、aux 量化损失、真 bitstream）。

## 落点

```
src/_compat.py                                  # add_safe_globals shim + 线程封顶（import compressai 前）
src/models/bottlenecks/entropy_bottleneck.py    # EntropyBottleneckModule：rate 机器件（CompressAI 薄封装）
src/models/bottlenecks/compression.py           # CommBottleneck = g_a + EntropyBottleneck + g_s
src/models/bottlenecks/__init__.py              # 工厂：identity | compression（compressai 惰性导入）
src/losses/compression.py                       # feature_mse + beta*bpp
tools/smoke_student.py                           # student 冒烟（4/4）
```

## 架构（split-inference 的压缩变换）

```
f --g_a(analysis)--> y --EntropyBottleneck(量化+学习rate)--> y_hat --g_s(synthesis)--> f_hat
   送端                  过信道的就是 y_hat                       收端
```

- **g_a/g_s 用 conv+BN+ReLU**（无 GDN/LN）——满足项目 Hailo 约束（收端 decoder 必须 DFC 可编译、可折叠）。
- g_s 重建回**与 f 同通道数**，下游 decoder/heads 无需改动即可消费。
- 默认 `latent_channels=64, hidden=128, num_down=1`（通道 + 一次空间下采样，真正压缩）。

## 关键工程决策

1. **统一契约** `forward(f) -> (f_hat, info)`：identity 返回 `(f, None)`，compression 返回
   `(f_hat, {"likelihoods","latent"})`。`MultiTaskNet` 每次 forward 暂存 `neck_feature`（切分点/蒸馏目标）
   与 `bottleneck_info`（rate）。改了 identity 契约但 seg 主线 smoke 仍 5/5。
2. **CompressAI 惰性导入**：只有真正 build `compression` 时才 import compressai → seg 训练**零依赖**
   （已验证 `import bottlenecks` 不加载 compressai）。
3. **环境兼容**：本机 NGC torch 2.4 没有 `torch.serialization.add_safe_globals`，而 compressai 的
   point-cloud registry 急切 import torch_geometric 会炸。`src/_compat.py` 给该符号打 no-op shim
   （只用 EntropyBottleneck，不碰 point-cloud，安全）。
4. **优化器拆分**（CompressAI 惯例）：`codec_parameters()`（g_a/g_s+熵模型）走主优化器训
   distortion+β·rate；`aux_parameters()`（CDF quantiles）走单独 aux 优化器训 `aux_loss()`；
   `compress` 前必须 `update()` 建 CDF 表。

## student 冒烟结果（`tools/smoke_student.py`，4/4）

冻结 seg teacher，在切分点（neck→decoder 槽）插入可训练 CommBottleneck：

| 检查 | 结果 |
|---|---|
| #1 训练：RD 损失下降 + codec/aux 双优化器步进 | dist 0.203→0.028，bpp 有限 ✅ |
| #2 完整 split-inference 前向（冻结 teacher tail 消费 f_hat） | seg preds shape 正确 ✅ |
| #3 框架集成：`build_model(compression)` 前向 + 暴露 `bottleneck_info` | ✅ |
| #4 真 bitstream roundtrip + 估计 bpp ≈ 实测 | **max\|diff\|=0**，est 3.585 vs 实测 3.776（5.1%）✅ |

## 下一步（未做，Phase 2 训练）

- 把 rate 项 + aux 优化器接进 seg `Trainer`（目前 `Trainer` 只跑任务 loss，不加 β·bpp）。
- 定 teacher：用 `ade20k_rtformer_bs32_lr4e4`（38.2）作冻结单任务 seg teacher（选项 A）。
- β 扫描出 rate-distortion / bpp-mIoU 曲线。
- 真实切分点对齐 RTFormer stage3（当前 student 冒烟用的是 neck 融合特征作通用验证）。
