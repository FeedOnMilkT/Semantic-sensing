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
- ✅ **① rate 接进训练（done）**：`src/engine/student_trainer.py` 的 `StudentTrainer`
  （L=feature_mse+β·bpp，codec/aux 双优化器，poly LR）+ `evaluate_student`（一遍出 bpp-mIoU 点）；
  CLI `tools/train_student.py`（`--smoke` 合成验证 / `--teacher-ckpt` 载真 teacher / `--beta` 扫描）。
  合成数据已验证两版都能训 + 出 `[POINT] bpp/mIoU`。seg 路径仍零 compressai 依赖。
- ✅ **② 载真 teacher + 首个真实点（done, 2026-06-30）**：
  - Pre-flight：HF 缓存在；teacher 载入 missing=0 unexpected=0；冻结 teacher 在 128 val 子集
    mIoU **0.334** / pixel_acc 0.775（全集 38.2，子集欠采样所以偏低，pixel_acc 对得上）。
  - 首个真实点：late、**β=0.001**、1 epoch(~2526 步)、val-limit 128 →
    **bpp=1.467, mIoU=0.306, pixel_acc=0.757**。即低 β 下 student 几乎追回 teacher（0.306 vs 0.334，
    **−0.028 / ~9% 相对，仅训 1 epoch**）。**整条真实管道端到端验证通过。**
    ckpt: `students/late_b0.001_sanity.pth`。
  - ⚠️ 已废弃的坑（2026-06-30 修复）：`--epochs` 曾默认 **1**，导致 sanity 跑只训 1 epoch（严重欠训、
    mIoU 塌）。现 `train_student.py` `--epochs` 默认 **None → 走 per-stage 真实排期**（Stage1
    `student.stage1_epochs`=100 / Stage2 `stage2.epochs`=20），`--smoke` 才用 1。**不再有静默 1-epoch。**
- ✅ **③ early 的 s4 旁路码率计入（done, 2026-06-30）**：`student.compress_s4: true` 给 early 加第二个
  hyperprior 压 s4，bpp 计入两路。三方对照（β=0.001，1 epoch，val-128，teacher 上界 mIoU 0.334）：

  | split | bpp | mIoU | 备注 |
  |---|---|---|---|
  | late | 1.467 | 0.306 | 语义腰 |
  | early（s4 免费，低估） | 0.414 | 0.283 | 旧 |
  | early（s4 计入，**公平**） | **1.272** | 0.302 | ckpt `early_b0.001_fair.pth` |

  **发现**：s4 一计入，early bpp 0.41→1.27（×3.1）——"early 便宜 3.5×"是 s4 免费的假象。公平比下
  **early ≈ late**（1.27/0.302 vs 1.47/0.306），bpp-mIoU 基本打平 → **early vs late 的取舍落在部署轴
  （纯卷积送端）而非码率**。Caveat：单点/1 epoch/子集，且 s4 bottleneck 未调（1.27 还能再压）。

## 真正的下一步（出可发表的结论）
- **β 扫描**：late 与 early 各扫 β∈{0.002,0.005,0.01,0.02,0.05}，训到收敛、全 val，画两条 bpp-mIoU 曲线。
- s4 bottleneck 调参（latent/num_down），让 early 曲线公平且最优。
- 部署对比表：送端 FLOPs/params + 注意力在哪端（early 纯卷积 vs late 含注意力）。

---

# 更新（2026-06-30 续 II）：中切主线 dual-latent bottleneck + Stage2（按 phase2_rtformer_distill_prompt.md）

> 实现 spec 的**中切主线**（stage3 后 conv→attention 边界，双支紧致 payload）+ 双 latent/
> 单容器 bitstream + 两阶段蒸馏。**只写代码不训练**（Sidney 本轮明确）；全部 smoke/pytest 绿。

## 关键决策（与 Sidney 对齐后锁定）
- **teacher = `ade20k_rtformer_paperhead`（passthrough(s8)+DAPPM+3x3 头, 36.55）**。原因：generic
  head 的 mlp_fusion 会重融 4 尺度、逼着 s4（1/4 大图）过信道、码率被它主导、破坏"干净中切"故事；
  paperhead 的 tail 只吃 s8 → 中切 payload **正好 = (x_h 128@1/8, x_l 256@1/16) 两支，无 s4**，合 spec。
  天花板低 1.6 点但论文报相对 RD 曲线，不影响对比。
- **激活两档都建**：GDN/IGDN = 主 RD 曲线；convbn = Hailo 可折叠档（报掉点）。
- **三切分按 spec 重定义**：early(stage2,单)/ **mid(stage3,双,主线)** / late(stage5,双)。旧 "late"=neck
  语义迁到 split 名 `neck`（`student_rtformer_late.yaml` + smoke 已改），保旧 9/9 不破。

## 落点（全在 src/ 模块框架内，无新 phase2/ 树）
- `models/backbones/rtformer.py`：加 `forward_sender_mid/receiver_mid`（stage3 后切，等价）+
  `forward_sender_late/receiver_late`（stage5 后切）。抽出 `_stage3`/`_stage45`。forward() 不变。
- `models/bottlenecks/hyperprior.py`：加 `act: gdn|convbn`（主 g_a/g_s；hyper 仍 conv+ReLU）。
  `_hyper_params` 把 means/scales 对齐 y 尺寸（奇数/小尺寸鲁棒），payload 存 `y_shape`，
  compress/decompress 同步对齐。
- `models/bottlenecks/rtformer_distill.py`：**新** `DualLatentBottleneck` = 两个独立
  HyperpriorBottleneck（high/low）+ 单容器序列化 `_pack/_unpack`（magic RTFD | n | 每支 z/y 串）。
  工厂注册 `rtformer_distill`（in_channels=(c_high,c_low)）。
- `models/student.py`：`SplitStudent` 支持 early/mid/late/neck；mid/late 走双支、元组 target/recon；
  暴露 `receiver_modules/receiver_parameters/set_stage2_trainable`（Stage2 微调接收端 tail）。
- `engine/student_trainer.py`：`_distortion` 支持元组（双支 feature-MSE 求和）；**新** `Stage2Trainer`
  （冻 encoder+bottleneck，eval 模式量化态，微调 tail，L=CE+Dice，Lovász 开关）。
- `tools/train_student.py`：加 `--stage 1|2` + `--student-ckpt`（Stage2 载 Stage1 student）。
- configs：`phase2_mid_stage1.yaml`(主线,GDN)/`_convbn`/`phase2_mid_stage2.yaml` /
  `ablation_early.yaml`(单latent,compress_s4=false 因 passthrough 不用 s4)/`ablation_late.yaml` /
  `phase2_betasweep.yaml` / `baseline_jpeg.yaml`。均 `_base_: ade20k_rtformer_paperhead.yaml`。

## 验证（全绿，CPU）
- `tests/`（pytest, 8/8 稳定，单线程避 CompressAI erfc 段错误）：
  - test_split_equivalence：identity bottleneck 下 encoder+tail==teacher（early/mid/late/neck）+
    backbone refactor 无损（mid/late s8/s16/s32 == forward）。
  - test_bottleneck_contract：dual 契约 + GDN/convbn 两档 + codec 有梯度。
  - test_bitstream_roundtrip：单容器 pack/unpack 逆 + roundtrip==eval forward。
  - test_rate_vs_filesize：估计 bpp ≈ 实测容器 bit（<20%）。
  - 跑法：`CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q`
- smoke（合成数据，CPU）：MID Stage1(GDN/convbn)、MID Stage2、EARLY 均出 [POINT]；
  既有 `smoke_student_split.py` 9/9（neck+early）、`smoke_test.py` 5/5、`smoke_student.py` 4/4 不破。

## 真正的下一步（训练态，需 Sidney green-light GPU）
- 用真 paperhead teacher 跑 mid Stage1 单 β sanity（验证重建特征过冻结 tail 掉点 <2）→ Stage2 → β 扫描。
- 待补**代码**（Step 5/6，本轮未写）：`tools/eval_rd_curve.py`（汇总 [POINT]→csv/png）、
  `tools/jpeg_baseline.py`（扫 Q）、`tools/export_onnx.py` 发送端/接收端预检、`viz_latent.py`。
