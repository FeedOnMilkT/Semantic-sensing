# Phase 2 实施任务：RTFormer split-computing 蒸馏式语义压缩（单分割任务）

> 给 Claude Code 的实施 spec。Sidney 已用 PyTorch 训好无压缩 RTFormer-Base teacher（ADE20K，150 类）。本阶段在 RTFormer 的**卷积→注意力分界（stage3 之后）**对半切，插入 CompressAI bottleneck，按 split computing（Entropic Student / Ladon）范式做**特征蒸馏 + latent 率约束**，产出 bitstream，扫 β 画 **bpp vs mIoU** 曲线。
> 完整项目背景见 `CLAUDE.md`。开始前请把本文件 + `CLAUDE.md` + `notes/RTFormer_reading_notes.md` + `notes/Matsubara2022_supervised_compression_reading_notes.md` + `notes/Matsubara2025_Ladon_multitask_SC_reading_notes.md` 读一遍。架构层面有不清楚的先问,不要拍脑袋。

---

## 0. 一句话总结

把已训好的 RTFormer-Base 在 stage3 后切成**发送端(纯卷积)+ 接收端(注意力+头)**,切口插入一个 **双 latent / 单容器 bitstream** 的 CompressAI bottleneck;用冻结 teacher 的 stage3 特征做 **feature-MSE 蒸馏 + GaussianConditional+hyperprior 估率**先训出可压缩表示,再(可选冻结)发送端、用 **CE+Dice** 微调接收端 tail;扫 β 出 bpp-mIoU 曲线,对照**早切/晚切**两个切分位置,外部只比 **JPEG**。**单任务(seg),不仿真信道、不注入噪声(pure semantics)。**

---

## 1. 背景（精简,必读）

- 方法论 = **Learned Source Coding for Machines / Supervised Compression for Split Computing**(Matsubara Entropic Student WACV'22、Ladon WACV'25)。**不是 JSCC**:训练时不仿真信道、不注入噪声、不出 analog symbols。引 prior art 时不要把 Shao/Xie 当正面参考。
- 拓扑:发送端(Pi5+Hailo-8)跑卷积前半 → bottleneck 出 bitstream → [队友信道 LDPC/Turbo,本项目不涉及] → 接收端(Pi5+Hailo-8,**同等算力**)跑注意力后半 + 分割头。
- **两端对称**:接收端 tail 也受 Hailo 约束,没有"更强接收端"。
- **核心创新点是 bottleneck**,不是 backbone、不是 decoder。

---

## 2. 已锁定决策（single source of truth,代码与本表冲突时以本表为准）

| 项 | 决策 |
|---|---|
| teacher | PyTorch, **RTFormer-Base**(`base_channels=64`), **ADE20K(150 类)**, 已训好、冻结 |
| 主线切分点 | **中切 = stage3 之后**(conv→attention 分界) |
| 切口 payload | 双分支 `(x3_ 128ch@1/8, x3 256ch@1/16)` |
| 任务 | **单语义分割**;loss = CE + Dice(Lovász 作 config 开关) |
| bottleneck | **双 latent / 单容器 bitstream**(每分支独立 analysis/synthesis/熵模型,序列化拼一条流) |
| 熵模型 | 直接上最优:**MeanScale Hyperprior**(Minnen 2018, `GaussianConditional` + hyper),CompressAI |
| 训练 | 两阶段蒸馏:Stage1(feature-MSE + β·rate)→ Stage2(微调 tail, CE+Dice) |
| 发送端卷积 | **允许微调**(默认 Stage1 冻结、可解冻;解冻则熵模型必须联动训,见 §5.3) |
| 蒸馏监督面 | v1 **只对齐切口特征**;FitNets 多层 / logit-KD / task-consistency 作 ablation 开关 |
| 切分位置对照 | **早切(stage3 前)** + **晚切(backbone 后)**,各一组;主线 = 中切 |
| 外部 baseline | 只比 **JPEG** → 解码图像 → 过冻结 teacher |
| 框架 | 纯 PyTorch + **CompressAI** + OmegaConf + TensorBoard |

---

## 3. ★ 切分点（已对照 PaddleSeg 官方 `rtformer.py` 精确定稿）

RTFormer-Base(`base=64`)前向逐层(通道/stride 已核对):

```python
x1  = layer1(conv1(x))              # 64ch,  1/4    纯卷积(stem+BasicBlock)
x2  = layer2(relu(x1))              # 128ch, 1/8    纯卷积
x3  = layer3(relu(x2))             # 256ch, 1/16   纯卷积           ← 低分支 x_l
x3_ = x2 + interp(compression3(x3))# 128ch, 1/8    (compression3: 256→128)
x3_ = layer3_(relu(x3_))           # 128ch, 1/8                     ← 高分支 x_h
# ===================== ★ 中切 = CONV→ATTENTION 分界(主线切分点)★ =====================
x4_, x4 = layer4([relu(x3_), relu(x3)])  # EABlock: 128@1/8 ; 512@1/16
x5_, x5 = layer5([relu(x4_), relu(x4)])  # EABlock: 128@1/8 ; 512@1/32
# ----------------------- 晚切 = backbone 之后(对照臂)-----------------------
x6     = spp(x5)                    # DAPPM: 512→128, 1/32 → interp 1/8
x_out  = seghead(concat([x5_, x6]))# in=256 → 150 logits, 1/8 → 上采到输入
```

### 3.1 三个切分位置(实验矩阵的主轴)

| 切分 | 位置 | 发送端 | 切口 payload | 接收端 | 角色 |
|---|---|---|---|---|---|
| **早切** | layer2 后 | `conv1+layer1+layer2` | **单**张 `x2 128@1/8` | `layer3+layer3_+compression3 + layer4/5 + spp + seghead` | 对照(可压性差) |
| **★中切** | stage3 后 | `conv1+layer1+layer2+layer3+layer3_+compression3` | **双** `(x3_ 128@1/8, x3 256@1/16)` | `layer4 + layer5 + spp + seghead` | **部署主线** |
| **晚切** | layer5 后 | 中切发送端 `+ layer4 + layer5` | **双** `(x5_ 128@1/8, x5 512@1/32)` | `spp + seghead` | 对照(可压性好但 attention 落发送端 Hailo) |

> 早切是**单分支**(双分支结构在 compression3/layer3_ 才形成)→ 单 latent。中切/晚切是**双分支**→ 双 latent。
> 晚切**不是可部署主线**:它把 EABlock(external/cross attention)压到发送端 Hailo,违反"发送端纯卷积",且 bring-up 风险高。仅用于画 RD-vs-切深曲线。单任务下"task-collapse"的反对意见消失,故晚切作对照合理。

### 3.2 中切契约(本阶段主战场)

- **发送端 Encoder**(纯 Conv+BN+ReLU,Hailo 极友好):`conv1, layer1, layer2, layer3, layer3_, compression3`,输出双分支 `(x3_, x3)`。
- **★ Bottleneck**:见 §4。输入 `(x3_, x3)`,输出 `((x̂3_, x̂3), aux)`。
- **接收端 Tail**(external/cross attention + conv,受 Hailo 约束、需 DFC):`layer4, layer5, spp(DAPPM), seghead`,输出 150 类 logits。
- `seghead_extra`(aux head)只在 teacher 训练用,本阶段部署链路丢弃。

---

## 4. Bottleneck 设计（双 latent / 单容器 bitstream）

### 4.1 数据流（中切,双分支各一套）

```
发送端:
  x3_ (128@1/8)  → g_a^h → y_h → [h_a^h→z_h→EB→ẑ_h→h_s^h→(μ_h,σ_h)] → GC.compress → str_y_h, str_z_h
  x3  (256@1/16) → g_a^l → y_l → [h_a^l→z_l→EB→ẑ_l→h_s^l→(μ_l,σ_l)] → GC.compress → str_y_l, str_z_l
  容器: pack([shapes | str_z_h | str_y_h | str_z_l | str_y_l]) → ONE bitstream  ← 信道接口

接收端:
  unpack → str_*  → GC/EB.decompress → ŷ_h, ŷ_l
  ŷ_h → g_s^h → x̂3_ (128@1/8)
  ŷ_l → g_s^l → x̂3  (256@1/16)
  [x̂3_, x̂3] → layer4(EABlock)
```

- 每分支 = 一个**特征域 MeanScale Hyperprior 压缩器**(Minnen 2018):`g_a`(analysis)、`h_a/h_s`(hyper)、`EntropyBottleneck`(压 z)、`GaussianConditional`(压 y,均值/尺度来自 hyper)、`g_s`(synthesis)。**输入是 teacher 特征张量,不是图像**(channel = 128 或 256)。
- 两分支**互不耦合**:还原是确定性的"拆容器 + 各自熵解码 + 各自 synthesis",不存在融合后劈不准的问题。
- **g_s 不是 g_a 的代数逆,是学出来的**:feature-MSE 蒸馏(§5)把每个 `g_s` 的输出训练成逼近 teacher 真实分支。

### 4.2 模块结构（每分支,CompressAI 风格）

```python
# g_a:  Conv(c_in→N) → GDN → Conv(N→N,s2) → GDN → Conv(N→M,s2)         # 特征→latent y (M ch)
# g_s:  对称转置(ConvTranspose/Upsample+Conv) + IGDN,M→...→c_in,空间还原
# h_a/h_s: 标准 hyper analysis/synthesis(Conv + ReLU),出 (μ, σ)
# 熵: EntropyBottleneck(z) + GaussianConditional(y | μ, σ)
```

- **激活两档(Hailo 训练≠部署)**:RD 主曲线用 **GDN/IGDN**(CompressAI 最优档);另提供 **Conv+BN+ReLU** 变体(config `bottleneck.act: gdn|convbn`),因 GDN 在 Hailo DFC 未必可编译。报告两档 RD 差距。
- latent 维度 `M`(主)、`N`(hyper)走 config;`M` 是核心压缩超参,和 β 一起决定 bpp。
- 下采步数让 y 的空间比输入分支再缩 2~4 倍(经验起点,config 可调)。

### 4.3 接口契约（沿用 Demo 1,所有 bottleneck 实现统一）

```python
# bottleneck.forward(feats) -> (feats_hat, aux)
#   中切/晚切: feats = (x_h, x_l) 双张量; 早切: feats = x 单张量
#   feats_hat: 同结构重建
#   aux = {"rate_loss": <标量, -log2 likelihood 求和 / num_pixels_image>,
#          "bpp": <同上, 估计值>, "likelihoods": {...}}
# 另需 compress(feats)->bytes / decompress(bytes)->feats_hat 两个部署态方法
```

- **bpp 归一化用原始输入图像像素数 H×W**(不是特征像素),保证早/中/晚切 RD 可比。
- registry 注册 `rtformer_distill`,与 `identity`/`vib` 可 config 互换。

### 4.4 序列化容器

- `compress` 返回单 `bytes`:`[magic | n_streams | (len, payload)*]`,内含各分支 y/z 的 CompressAI strings + z 的 shape 头。
- `decompress` 逆操作。**必须有 roundtrip 测试**(§9)。
- 这条 bytes 就是给队友信道编码器的接口,本项目不对它做任何信道处理。

---

## 5. 训练范式（两阶段蒸馏 + β 扫描）

teacher = 冻结的无压缩 RTFormer-Base。其在切口的特征即蒸馏目标 `h`(中切 = `(x3_, x3)`)。

### 5.1 Stage 1 —— 学可压缩表示

- 发送端 Encoder = teacher 卷积前半,**权重拷贝**;默认冻结(`encoder.freeze: true`)。
- 训 **Bottleneck**,让重建逼近 teacher 切口特征:

```
L_stage1 = Σ_branch ‖ x̂_branch − x_branch ‖²        # distortion = 特征空间 MSE(冻结 teacher)
           + β · L_rate                              # rate = Σ −log2 p(ŷ)/HW (主 + hyper)
训练态量化代理:  ŷ = y + U(−0.5, 0.5)               # CompressAI 标准,非高斯采样
```

- distortion 是**特征 MSE,不是像素 MSE、不是任务 loss**(这就是蒸馏式与纯 ICM/纯 LIC 的分界)。
- 接收端 tail 在 Stage 1 = 冻结 teacher,仅前向、不进 distortion。
- **ablation 开关**(v1 默认全关):
  - `distill.fitnets: [layer4, layer5]` —— 额外对齐更深层特征(信号过 bottleneck+tail 后);
  - `distill.logit_kd: true` —— 加 teacher 软标签 KL(温度 T);
  - `distill.task_consistency: w` —— Stage1 就把重建特征过冻结 tail 加小权重 CE。

### 5.2 Stage 2 —— 任务微调

- **冻结发送端 Encoder + Bottleneck**(部署侧固定,熵模型随之锁定)。
- 解冻并微调接收端 tail(`layer4/5 + spp + seghead`):

```
L_stage2 = CE + Dice        # Lovász-Softmax 作 config 开关
```

- ignore_index 与类映射**严格对齐 teacher 的 ADE20K 约定**(确认 reduce_zero_label / ignore=255 与 teacher 一致,否则 mIoU 全错)。

### 5.3 发送端微调的联动约束（重要）

若开 `encoder.freeze: false`(允许发送端卷积微调):latent 分布会漂,**熵模型(GaussianConditional/EntropyBottleneck)必须与 encoder 同阶段联合训练,不能冻**;否则估计 bpp 与真实 bit 数对不上。提供 `encoder.freeze: true|false` 两条路径,默认 true(纯净 split)。

### 5.4 率失真扫描

- 扫 `β ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`(config 列表,每个 β 一次 Stage1+Stage2)→ **bpp vs mIoU** 曲线;另出 **β vs mIoU**。
- 多 β 共享同一冻结 teacher,可并行。

---

## 6. 实验矩阵

| 组 | 切分 | 方法 | 产出 | 优先级 |
|---|---|---|---|---|
| **主线** | 中切 | 蒸馏式(§5),β 扫描 | bpp-mIoU 核心曲线 | P0 |
| 对照A | 早切 | 同蒸馏配方,单 latent | RD-vs-切深(浅) | P1 |
| 对照B | 晚切 | 同蒸馏配方,双 latent | RD-vs-切深(深) | P1 |
| baseline | — | **JPEG** 压原图(扫质量因子 Q)→ 解码 → 冻结 teacher | bpp-mIoU 外部对照 | P0 |

- 三切分共用同一套 bottleneck/trainer 代码,只换"切在哪"(config)。
- JPEG baseline:用 `Pillow`/`torchvision` 编解码,bpp = JPEG 文件 bit / (H×W),mIoU = 解码图过 teacher。
- **不做** 任务驱动 ICM 臂、纯神经 LIC 臂、纯 VIB 臂(Sidney 已决定砍掉以省训练时间);如需可后补为 ablation。

---

## 7. 代码结构与改动（基于已有 PyTorch teacher）

> ⚠️ 先定位 teacher 代码:RTFormer 的 PyTorch 实现文件、forward、checkpoint 路径。把 teacher 当**只读依赖**,不改它,只 import + 拷权重 + 拆 forward。

```
phase2/
├── models/
│   ├── rtformer_split.py        # 把 teacher RTFormer 拆 RTFormerEncoder / RTFormerTail
│   │                            #   按 split_point(early|mid|late)切;load_from_teacher(ckpt)
│   ├── bottlenecks/
│   │   ├── __init__.py
│   │   ├── base.py              # 接口契约 forward/compress/decompress
│   │   ├── feature_hyperprior.py# 单分支 MeanScale Hyperprior 压缩器(§4.2)
│   │   └── rtformer_distill.py  # 组合双分支 + 单容器序列化(§4.1/4.4)
│   └── registry.py              # 复用 Demo 1 风格(若已有则 import)
├── engine/
│   ├── distill_trainer.py       # Stage1 + Stage2 两套循环, freeze 规则, β 扫描
│   └── evaluator.py             # mIoU(150 类) + bpp(估计/实测)
├── losses/
│   ├── feature_mse.py           # 双分支特征 MSE(+ 可选 FitNets/KD/task-consistency)
│   ├── rate.py                  # -log2 likelihood 聚合 / num_pixels=HW
│   └── seg.py                   # CE + Dice (+ Lovász 开关)
├── data/
│   └── ade20k.py                # ADE20K 150 类 dataset(对齐 teacher 预处理/类映射)
├── configs/
│   ├── _base_/ade20k.yaml
│   ├── phase2_mid_stage1.yaml   ├── phase2_mid_stage2.yaml
│   ├── phase2_betasweep.yaml    # β 列表驱动
│   ├── ablation_early.yaml      ├── ablation_late.yaml
│   └── baseline_jpeg.yaml
├── tools/
│   ├── train_distill.py         # --stage 1|2 --config ...
│   ├── eval_rd_curve.py         # 汇总多 β/多切分 → bpp-mIoU 曲线 + csv/png
│   ├── bitstream_roundtrip.py   # compress→写盘→读→decompress→mIoU 一致性 + bpp≈文件
│   ├── jpeg_baseline.py         # 扫 Q 出 JPEG bpp-mIoU
│   ├── viz_latent.py            # latent/重建特征/分割预测可视化
│   └── export_onnx.py           # 发送端 + 接收端分别静态导出(Hailo 兼容预检)
└── tests/
    ├── test_split_equivalence.py# 未插 bottleneck 时 encoder+tail 前向 == teacher 前向
    ├── test_bottleneck_contract.py
    ├── test_bitstream_roundtrip.py
    └── test_rate_vs_filesize.py # 估计 bpp ≈ 实际文件 bit(偏差 <10–20%)
```

---

## 8. 关键实现要点

1. **拆 teacher 等价性**:`RTFormerEncoder` + `RTFormerTail` 拼起来、bottleneck 设 identity 时,前向数值必须 == 原 teacher(`test_split_equivalence.py`)。这是后续一切的地基。
2. **特征 hook**:Stage1 需要 teacher 在切口的特征作 target。用 forward hook 或直接调 `encoder`(=teacher 卷积前半)得到 `(x3_, x3)`,detach 当 target。
3. **bpp 归一化**:一律 `/(H×W)`,H、W = 原始输入图像尺寸,不是特征图。早/中/晚切才可比。
4. **量化代理**:训练 `y + U(-0.5,0.5)`;部署 `torch.round` + range coder。CompressAI 的 `EntropyBottleneck`/`GaussianConditional` 自带 `compress/decompress`。
5. **roundtrip 校验**:`compress→bytes→写盘→读→decompress→过 tail` 的 mIoU,与训练态(加噪)前向的 mIoU 一致(±小量);实际文件 bit ≈ 估计 bpp×HW。
6. **GDN/Hailo 双档**:主 RD 用 GDN;另跑一遍 `act: convbn` 报掉点,为 Phase 3 Hailo bring-up 留数据。
7. **ONNX 预检**:发送端(纯卷积+g_a/h_a)应干净导出;接收端含 EABlock,重点看**非 last-dim softmax(GDN double-norm)、动态 reshape、`grid_sample`**——尽早暴露(`CLAUDE.md` 第三节算子黑名单)。
8. **类映射**:ADE20K 与 teacher 训练时的 `reduce_zero_label`/`ignore_index` 必须逐字一致,否则 mIoU 失真。先用 teacher 复算一遍 val mIoU 核对,再开始压缩实验。

---

## 9. 验收标准

- [ ] `test_split_equivalence`:identity bottleneck 下 encoder+tail 前向 == teacher(三种切分都过)。
- [ ] teacher 在 ADE20K val 复算 mIoU == 已知 teacher 指标(类映射核对通过)。
- [ ] Stage 1:feature-MSE 收敛;某 β 下重建特征过冻结 tail,mIoU 掉点 < 2。
- [ ] Stage 2:任务微调后目标 bpp 档 mIoU 接近无压缩 teacher(差距按曲线报,不设绝对值)。
- [ ] bitstream roundtrip 正确;实测 bit ≈ 估计 bpp(偏差 <10–20%)。
- [ ] β 扫描出**单调** bpp-mIoU 曲线;主线(中切)曲线在同 bpp 下 ≥ JPEG baseline。
- [ ] 早/中/晚切三条 RD 曲线齐全,能讲"切深 vs 可压性"的故事。
- [ ] 发送端/接收端 ONNX 各自导出不报错(或明确列出 Hailo 不友好算子清单)。

---

## 10. 严格禁止 / 不要做

1. **不要训练时注入任何信道噪声** —— 这不是 JSCC,信道训练里不存在。
2. **不要改 teacher 代码/权重** —— 只 import、拷权重、拆 forward。
3. **不要把 distortion 写成像素 MSE 或任务 loss** —— Stage1 是**特征 MSE**;任务 loss 只在 Stage2。
4. **不要把 bpp 用特征像素归一化** —— 必须 `/(H×W)` 图像像素,否则切分位置不可比。
5. **不要在融合后劈分支** —— 用**双 latent**,各分支独立 analysis/synthesis/熵模型。
6. **不要在解冻发送端时还冻熵模型** —— 见 §5.3,必须联动训。
7. **不要做 PyTorch QAT** —— Hailo 期望干净 fp32 ONNX,量化由 Phase 3 处理。
8. **不要硬编码 base_channels/类数/β/lr/M/N 在 .py** —— 全走 config(150 类、base=64 从 config 读)。
9. **不要引入 MMSeg/MMCV 作依赖** —— 摘代码可以,加依赖不行。
10. **不要假设三分支或多任务** —— 单任务 seg;早切是单分支,中/晚切是双分支。

---

## 11. 实施顺序（给 Claude Code 的工作计划）

```
Step 0 (定位与对齐, ~0.5 天):
  - 找到 teacher RTFormer PyTorch 代码 + checkpoint,确认 base=64/150 类/ADE20K 类映射
  - 复算 teacher val mIoU,核对类映射;写 ade20k.py dataset

Step 1 (拆分等价, ~0.5 天):
  - rtformer_split.py:按 split_point 拆 encoder/tail,load_from_teacher
  - test_split_equivalence 三切分全过(bottleneck=identity)

Step 2 (单分支压缩器, ~1 天):
  - feature_hyperprior.py:MeanScale Hyperprior 特征域压缩器(GDN 档)
  - 先在随机特征张量上 sanity:forward 出 likelihood、compress/decompress roundtrip、bpp≈文件
  - test_bitstream_roundtrip + test_rate_vs_filesize

★ Step 3 (中切主线, ~1.5 天):
  - rtformer_distill.py:双分支 + 单容器序列化
  - distill_trainer.py Stage1(feature-MSE + rate, 冻结 encoder/tail)
  - 单 β 跑通 Stage1,验证重建特征过冻结 tail 掉点 <2

Step 4 (任务微调 + β 扫描, ~1.5 天):
  - Stage2(冻 encoder+bn,微调 tail,CE+Dice)
  - betasweep,出中切 bpp-mIoU 曲线;eval_rd_curve 汇总

Step 5 (对照, ~1.5 天):
  - ablation_early(单 latent)+ ablation_late(双 latent)各跑 β 扫描
  - jpeg_baseline 扫 Q;三切分 + JPEG 四条曲线同图

Step 6 (工程收尾, ~0.5 天):
  - convbn 档跑一遍报 Hailo 掉点;export_onnx 发送端/接收端预检
  - viz_latent;README 写清复现命令 + 决策表追溯
```

每步跑对应 pytest,绿了再进下一步。**Step 0-4 是 critical path**;Step 5 对照可在 Step 4 后并行。

---

## 12. 交接 / 追溯

- 所有设计决策在代码 docstring 追溯到:`CLAUDE.md` 章节 / 本文件 §号 / Entropic Student / Ladon 论文节号 / 官方 `rtformer.py`。
- 完成后 Phase 3 = 发送端 + 接收端两块 HEF 分别 Hailo bring-up(见 `hailo_deployment_guide.md`)。

---

**任务结束。开干前先把本文件 + CLAUDE.md + 三篇 notes 读完,确认 teacher 代码与 ADE20K 类映射后再动手。架构有疑问先问。**
