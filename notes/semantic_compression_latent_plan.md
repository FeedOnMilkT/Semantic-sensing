# 语义压缩 + 收紧 Latent · 设计与策略（2026-06-30）

> 背景：late student（β=0.001 收敛）全 val = **bpp 1.05 / mIoU 37.51**，teacher 38.23。
> 但 compress-then-analyze 的 **JPEG q50 在子集上用更低 bpp(0.79) 达到同样 mIoU** → 我们当前
> **没赢图像编码器**。本文给出"为什么"的诊断 + 两条改进线（任务驱动语义压缩 / 收紧 latent）的
> 原理、技术、文献与落地计划。引用见 §6（均已核实）。

---

## 0. 诊断：为什么现在赢不了 JPEG

两条根因，都不是"方法上限低"，而是"没用对方法"：

1. **distortion 用的是 feature-MSE（像素保真档）**。`L = ‖f − f̂‖² + β·R` 保全整张特征——
   连**任务无关的纹理/细节**也照传。这恰是 JPEG 几十年最擅长的"保真压缩"赛道。我们在拼像素，
   当然拼不过专做像素的 JPEG。**JPEG 唯一做不到的是"任务感知"，而我们偏偏没用。**
2. **latent 结构上几乎没收紧**：`latent_channels=128`（=输入通道）、`num_down=1`。瓶颈维度没卡，
   压缩全靠熵模型 + β 硬扛 → RD 工作点先天不利。

> 结论：要压过图像编码器，得（A）把 distortion 变**任务驱动**，（B）把 latent **结构收紧**。

---

## 1. 理论原理：面向机器的率失真（地基）

- **率失真 (Rate–Distortion)**：在失真 D 约束下最小化码率 R，拉格朗日式 `L = D + β·R`（我们的 β
  就是这个 λ）。β↑ → R↓、D↑，沿曲线走。[Shannon RD theory]
- **信息瓶颈 (Information Bottleneck, IB)**：对一个任务而言，"该保留的"不是像素/特征保真，而是
  **任务相关信息**——`min I(X;Z) s.t. 保住 I(Z;Y)`。把 Z 压到只含预测 Y 所需的信息。[6 Tishby'99]
  深度化 = Deep VIB（变分上界 + 重参数化）。[7 Alemi'17]
- **对我们的含义**：把 IB 里的 Y 设成**分割标签** → distortion 应是**任务 loss**而非 feature-MSE。
  这样码只保 `I(Z; seg)`，把任务无关信息丢掉，同 mIoU 下 bit 更少。这就是 §2 的全部动机。

---

## 2. 语义压缩（任务驱动 / Image Coding for Machines）

### 2.1 原理
**ICM/VCM**：为**机器视觉任务**而非人眼感知优化码流。理论与实证都表明：丢掉任务无关信息后，
同任务精度下码率显著低于通用图像编码器。[15 Duan'20 / MPEG-VCM][10 Dubois'21 给出"为保任务不变性
可无损丢弃"的码率理论]

### 2.2 技术（怎么改我们的代码）
当前 `src/losses/compression.py`：`L = feature_mse(f, f̂) + β·bpp`。改为**任务驱动**：

```
L = L_task( frozen_tail(f̂), seg_GT ) + γ·feature_mse(f, f̂) + β·R
        └ 主项：分割 CE/Lovász，穿过冻结 teacher tail，梯度回流到 bottleneck
                                  └ 小权重稳定项（可选，γ 小）
```
- **机制**：student 的重建特征 f̂ → 冻结 decoder/heads → 分割预测 → 任务 loss。梯度只逼 bottleneck
  保留"能让冻结 tail 正确分割"的信息 → 自动丢弃纹理细节。
- 我们的 `StudentTrainer` 已有 `--task-weight`；只需把它扶为主项（并加 `distortion: feature_mse |
  task | hybrid` 开关）。
- **稳定性**：纯 task loss 可能不稳（冻结 tail 约束 + 量化噪声）→ 推荐 **hybrid 起步**（task 主 +
  小 feature-MSE），或 **两阶段**：先 feature 蒸馏预热、再 task 微调。

### 2.3 训练配方（文献成熟范式）
- **Entropic Student**[8 Matsubara WACV'22]：teacher–student 监督压缩，stochastic bottleneck +
  可学熵先验。**我们这套就是它的工程化**——但它的主线含任务信号，我们目前只做了 feature 蒸馏那半。
- **Ladon**[13 Matsubara WACV'25]：多任务监督压缩，**三步**：feature 蒸馏预训 encoder-decoder →
  fine-tune decoder → 冻结后挂任务头。和 Sidney 的多任务设定同构，是 Phase-2 多任务时的直接蓝本。
- **SC2 Benchmark**[9 Matsubara TMLR'23]：split computing 监督压缩的评测范式（分类/检测/**分割**），
  我们出 bpp-mIoU 曲线时对齐它的口径。
- **可压缩特征端到端学习**[11 Singh ICIP'20]：把特征"可压缩性"和任务目标联合优化的早期工作。
- **设备-边缘协同的特征压缩**[14 Shao BottleNet++'20]：split inference 里压中间特征的工程参照。

### 2.4 预期与诚实声明
ICM 文献普遍能**压过图像编码器**（同精度更低 bpp）。所以"任务驱动"大概率把曲线压到 JPEG 之下。
**但不打包票**：若上了任务驱动 + 窄 latent **仍赢不了 LIC**，那是真结论（ADE/RTFormer 设定下语义
优势有限），如实报告、回头反思 framing。

---

## 3. 收紧 Latent（率失真结构杠杆）

### 3.1 原理
码率下限由**潜变量元素数（通道 × 空间）及其熵**决定。latent 越宽、空间越大 → 熵模型负担越重、
可达码率越高。**结构性收紧**（少通道 + 多下采样）直接降低 RD 工作点，再让熵模型在更紧的码上分配比特。

### 3.2 技术（对应 `hyperprior.py`）
我们的 bottleneck 是 **mean-scale 超先验**[2 Ballé'18 + 3 Minnen'18]，构件与可调点：

| 构件 | 现状 | 收紧手段 | 文献 |
|---|---|---|---|
| 分析变换 g_a（C→D） | D=128(=输入) | **D↓ 到 48/32**：卡瓶颈维度 | [1 Ballé'17] |
| 空间下采样 num_down | 1（stride-8 latent） | **→2（stride-16）**：空间元素再 ÷4 | [1 Ballé'17] |
| 量化 | 训练态加 U(−.5,.5) 噪声 | （不变，可学的标准做法） | [5 Theis'17][1 Ballé'17] |
| 熵模型 | 超先验逐元素预测 (μ,σ) | 已是可部署最优；比特按需分配 | [2][3] |
| —（不上）AR context | 无 | 不可 Hailo 部署，**排除** | [3 Minnen'18] |
| —（不上）GDN | 用 conv+BN+ReLU | GDN 不可折叠，**排除** | [1 Ballé'17] |

> 注：我们刻意用 conv+BN+ReLU 代替 GDN、不上 autoregressive context——这是**部署约束下的取舍**
> （§5），所以是"**可部署范围内的最优熵模型**"，非纯 RD SOTA[4 Cheng'20 那类]。

### 3.3 量化与可微
量化不可导 → 训练态用**加性均匀噪声**近似（`y + U(−.5,.5)`），部署态 round + 算术编码。[5 Theis'17]
[1 Ballé'17]。我们用 CompressAI 的 `EntropyBottleneck`/`GaussianConditional` 实现[12 CompressAI]。

---

## 4. 本项目落地计划（实验矩阵）

**Config A —「任务驱动 + 窄 latent」**（方法的"真正一枪"）
- `student_rtformer_late_tuned.yaml`：`latent_channels=48, num_down=2`；
- `distortion: hybrid`（task 主 + γ·feature_mse 稳定），β 扫描 {0.003,0.01,0.03,0.1}。
- 代码改动：`StudentTrainer` 加 distortion 开关（task/hybrid/feature_mse）；loss 加 task 项（穿冻结 tail）。

**对照矩阵（同全 val、同 bpp 轴）**
| 曲线 | 说明 | 状态 |
|---|---|---|
| naive（feature-MSE，宽 latent） | 当前 β 扫描 | 跑中 |
| **tuned（task + 窄 latent）** | 方法真实曲线 | 待做 |
| JPEG / WebP | compress-then-analyze 弱 baseline | JPEG 跑中 |
| **LIC**（CompressAI 预训练 bmshj2018/mbt2018） | 真正对手 | 待做 |
| 参照线 | 原图 24bpp / 裸 float 特征 bpp | — |

**判读**：tuned 曲线在 JPEG（理应）与 **LIC** 左下方 → 语义传输价值成立；否则如实记录。

---

## 5. 部署约束（Hailo，贯穿所有选择）
- **送/收两端皆 Hailo-8**，decoder 与 bottleneck 都要 DFC 可编译。
- ⇒ 变换用 **conv+BN+ReLU（可折叠）**，不用 GDN[1]；熵模型用**超先验**[2][3]，**不上 AR context**[3]
  （动态、不可静态编译）。这把我们钉在"可部署最优"而非"纸面 RD 最优"。
- 任务驱动不改变这一点（只换 loss，不换算子）。

---

## 6. 参考文献（均已核实）

**学习式图像压缩 / 熵模型（latent 这条线）**
- [1] J. Ballé, V. Laparra, E. P. Simoncelli. *End-to-end Optimized Image Compression.* ICLR 2017. arXiv:1611.01704.
- [2] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston. *Variational Image Compression with a Scale Hyperprior.* ICLR 2018. arXiv:1802.01436.
- [3] D. Minnen, J. Ballé, G. Toderici. *Joint Autoregressive and Hierarchical Priors for Learned Image Compression.* NeurIPS 2018.
- [4] Z. Cheng, H. Sun, M. Takeuchi, J. Katto. *Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules.* CVPR 2020. arXiv:2001.01568.
- [5] L. Theis, W. Shi, A. Cunningham, F. Huszár. *Lossy Image Compression with Compressive Autoencoders.* ICLR 2017. arXiv:1703.00395.
- [12] J. Bégaint, F. Racapé, S. Feltman, A. Pushparaja. *CompressAI: a PyTorch library and evaluation platform for end-to-end compression research.* arXiv:2011.03029, 2020（技术报告，无正式 venue）。

**信息瓶颈 / 率失真理论（语义压缩地基）**
- [6] N. Tishby, F. C. Pereira, W. Bialek. *The Information Bottleneck Method.* Allerton 1999. arXiv:physics/0004057.
- [7] A. A. Alemi, I. Fischer, J. V. Dillon, K. Murphy. *Deep Variational Information Bottleneck.* ICLR 2017. arXiv:1612.00410.
- [10] Y. Dubois, B. Bloem-Reddy, K. Ullrich, C. J. Maddison. *Lossy Compression for Lossless Prediction.* NeurIPS 2021. arXiv:2106.10800.

**监督压缩 / 面向机器编码（语义压缩主线）**
- [8] Y. Matsubara, R. Yang, M. Levorato, S. Mandt. *Supervised Compression for Resource-Constrained Edge Computing Systems*（"Entropic Student"）. WACV 2022. arXiv:2108.11898.
- [9] Y. Matsubara, R. Yang, M. Levorato, S. Mandt. *SC2 Benchmark: Supervised Compression for Split Computing.* TMLR 2023. arXiv:2203.08875.
- [11] S. Singh, S. Abu-El-Haija, N. Johnston, J. Ballé, A. Shrivastava, G. Toderici. *End-to-end Learning of Compressible Features.* IEEE ICIP 2020. arXiv:2007.11797.
- [13] Y. Matsubara, M. Mendula, M. Levorato. *A Multi-Task Supervised Compression Model for Split Computing*（模型名 "Ladon"）. WACV 2025, pp. 4913–4922. arXiv:2501.01420.
- [14] J. Shao, J. Zhang. *BottleNet++: An End-to-End Approach for Feature Compression in Device-Edge Co-Inference Systems.* IEEE ICC Workshops 2020.
- [15] L.-Y. Duan, J. Liu, W. Yang, T. Huang, W. Gao. *Video Coding for Machines: A Paradigm of Collaborative Compression and Intelligent Analytics.* IEEE TIP 2020. arXiv:2001.03569.（配 MPEG-VCM/AI 官方探索文档一并引用；"ICM" 宜引标准化文档/综述而非单一论文）

> 引用已核实，但写进论文前仍建议二次确认页码/卷期。CompressAI 注明为 arXiv 报告；VCM/ICM 用
> Duan'20 综述 + MPEG 官方探索配对引用最稳妥。
