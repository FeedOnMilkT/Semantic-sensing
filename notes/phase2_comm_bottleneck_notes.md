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

---

# 更新（2026-06-30 续）：两个 split student + 完全版 hyperprior bottleneck

为 split-point ablation 构建了**早切/晚切两版 student** + **完整 mean-scale hyperprior**，
全部走模块化 + 配置文件，未破坏源网络拓扑。

## 新增/改动
- `src/models/bottlenecks/hyperprior.py`：**完全版** bottleneck。mean-scale 超先验
  `g_a→h_a→EB(z)→h_s→(μ,σ)→GaussianConditional(y)→g_s`，两路 bitstream。全 conv+BN+ReLU
  （Hailo 可折叠），**不带 autoregressive context**（不可部署）→ "可部署范围内最优"。工厂注册
  `hyperprior`（compressai 惰性导入）。
- `src/models/backbones/rtformer.py`：**无损重构** `forward()` 为 `forward_sender`
  (stem+s1+s2 纯卷积) + `forward_receiver` (s3~5+DAPPM)。`forward()` 行为不变
  （断言 `forward == forward_receiver(forward_sender)`，**max|diff|=0**）。
- `src/models/student.py`：配置驱动 `SplitStudent`，冻结 teacher + 可训 bottleneck，
  `split=early|late`。
- `configs/student_rtformer_{early,late}.yaml`：同 teacher、同 bottleneck，**只切点不同**。
- `tools/smoke_student_split.py`：**9/9 通过**（重构等价 + identity-student==teacher 接线无损
  + RD 训练下降 + hyperprior 两路 bitstream roundtrip **精确 max|diff|=0** + bpp 误差 <2.5%）。

## 两版定义（受控：唯一变量=切点）
- **late（语义腰）**：切 neck 输出（单张融合特征）。送端=backbone+neck，收端=decoder+head。
- **early（纯卷积送端）**：切 stage2 特征（双分辨率分叉前）。送端=stem+s1+s2（纯卷积，注意力全在收端）。
  收端=s3~5+DAPPM+neck+decoder+head。**s4 高分旁路**（开放问题#1），只压语义的 stage2 特征。

## ⚠️ 两个必须记住的 caveat
1. **early 的 bpp 被低估**：s4 旁路目前**不计 rate**。smoke 里 early bpp 看着比 late 低，是因为
   ① 压的特征更小 + ② s4 没算钱。**这不是结论**（且权重是随机的）。要公平比，得把 s4 也压（多尺度
   bottleneck）或显式把旁路码率计入。
2. **GPU 段错误**：compressai 的 rANS 原生编码器和 CUDA 上下文同进程混用会 segfault（range coder
   本就 CPU-only）。**训练 forward/backward 在 GPU 正常**（已单独验证）；**compress/decompress 走 CPU**。
   smoke 整体固定 CPU。

## 下一步（Phase 2 训练）
- 把 rate 项 + aux 优化器接进 `Trainer`（目前只跑 task loss）。
- 载入真 teacher `ade20k_rtformer_bs32_lr4e4`(38.2) 冻结（选项 A）；当前 smoke 用随机权重只验接线。
- 解决 early 的 s4 旁路码率计入，使 early/late bpp 可公平比。
- 各跑 β 扫描 → 两条 bpp-mIoU 曲线 + 部署对比表（送端是否含注意力）。
