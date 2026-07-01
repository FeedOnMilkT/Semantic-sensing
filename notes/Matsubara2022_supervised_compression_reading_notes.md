# 论文笔记：Supervised Compression for Resource-Constrained Edge Computing Systems
> Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt | UC Irvine | WACV 2022  
> 关键词：Split Computing, Feature Compression, Knowledge Distillation, Entropic Student, Rate-Distortion

---

## 这篇论文是干什么的

Split computing 场景：移动端（Raspberry Pi 级设备）跑编码器，边缘服务器跑解码器 + 任务头。
核心问题——怎么压缩中间特征，使得传输数据量尽量小，同时任务性能尽量高。

提出 **Entropic Student**：knowledge distillation + 神经图像压缩的结合体。在率失真性能和端到端延迟上超过七个对照方法。

这是 Sidney 项目的**系统拓扑参考**和**主要 baseline 对照**。

---

## 与 Sidney 项目的关键关系

| 维度 | Matsubara WACV 2022 | Sidney 项目 |
|---|---|---|
| distortion 项 | 特征空间 MSE（模仿 teacher 中间特征 h） | 任务 loss（mIoU + SiLog + cosine） |
| 训练范式 | 两阶段：先特征重建，再 fine-tune 任务头 | 一阶段端到端任务驱动（ICM 范式） |
| 多任务方式 | 顺序 fine-tune，encoder 冻结后各任务独立 | 多任务同时端到端训练，共同约束 bottleneck |
| 边缘设备假设 | CPU-only（RPi4），encoder 必须极轻 | Hailo-8 NPU（26 TOPS），完整轻量 backbone 可跑 |
| 任务覆盖 | 分类 / 检测 / 分割（单任务分别做） | seg + depth + normal 三任务同时 |
| bottleneck 类型 | Factorized prior（Ballé 2018 风格） | ICM 主方案：hyperprior；简化版：VIB KL |

**论文叙事定位**：Matsubara 建立了 split computing + learned feature compression 的系统框架；Sidney 在这个框架上把训练范式从"特征蒸馏两阶段"升级为"端到端多任务驱动"，把任务从单一感知扩展到语义 + 粗几何 + 细几何三类 dense prediction。

---

## 系统结构

```
移动端
  Image x
    ↓
  Encoder f_θ（轻量，conv + GDN）
    ↓ z（量化 + 熵编码 → bitstream）

边缘服务器
  z（熵解码）
    ↓
  Decoder g_φ（conv + IGDN）
    ↓ decoded feature ≈ h
    ↓
  Task tail（复用 teacher 后半段）
    ↓ 预测结果
```

核心数据流：x → z → h → y（student），对比 teacher：x → h → y

---

## 方法：Entropic Student（两阶段训练）

### 第一阶段：学习可压缩的特征表示（Section 3.2）

目标：用尽量少的 bit 重建 teacher 的中间特征 h。

**量化处理**：训练时往 encoder 输出加均匀噪声 ε ~ U(-1/2, 1/2) 模拟 round，保证梯度可流。

**损失函数**（Eq. 2）：
```
L(x, h) = ||h - g_φ(f_θ(x) + ε)||²   ← distortion（特征空间 MSE）
         - β · log p_φ(f_θ(x) + ε)    ← rate（learned prior 负对数似然）
```

- distortion = teacher 特征 h 和 student 解码特征的 MSE，**不是任务 loss**
- rate = Ballé factorized prior，可学习，不是固定 N(0, I)
- β 是拉格朗日乘子，控制率失真权衡；一个模型调 β 就能在 RD 曲线上移动

部署时：z = round(f_θ(x))，用 p_φ 做熵编码。先验各维度独立分解，可并行编码。

训练辅助：除了 h 之外，还用 teacher 冻结中间层的多层特征作为辅助监督（类似 FitNets），帮助 bottleneck 在早层时收敛。

### 第二阶段：Fine-tune 到下游任务（Section 3.3）

**冻结 encoder**，只训 decoder + 任务头。

对分类：hard-target（cross-entropy）+ soft-target（student/teacher softened 分布的 KL 散度）加权。

**多任务复用**：同一冻结 encoder → 分别 fine-tune 出分类头、RetinaNet 检测头、DeepLabv3 分割头。设备端只存一个 encoder，部署成本约为"每任务独立 encoder"方案的 1/3。

**注意词选择**：结论里用的是 "tuned to serve multiple tasks"，不是 "jointly trained"——多任务是顺序 fine-tune，bottleneck 没有被多任务同时约束。

---

## Bottleneck 位置的设计决策（Section 4.2）

**为什么放在早层**：现代视觉 backbone（ResNet-50）的特征被 FPN 等多尺度结构引用，若 bottleneck 放晚层，服务器端需接收多个层的特征图，通信开销倍增。放在第一个 residual block 之前，只传一个压缩表示。

**Encoder 结构**：conv + 简化版 GDN 层。GDN（Generalized Divisive Normalization）是 Ballé 为图像压缩设计的非线性激活，能更好地建模自然图像统计。

> **Sidney 的对应**：PyTorch 标准 Conv + BN + ReLU 是合理简化。GDN 在 Hailo-8 上未必能编译，且 Sidney 的 distortion 是任务 loss，GDN 对特征分布建模的优势在任务驱动设定下打折扣。

---

## 实验结果（Section 4）

### Baseline 体系（Section 4.1）

| 类型 | 方法 | 说明 |
|---|---|---|
| IC | JPEG / WebP / BPG | 传统 codec 还原图像 |
| IC | Factorized Prior / Mean-Scale Hyperprior | 神经图像压缩（CompressAI） |
| FC | CR + BQ | channel reduction + INT8 量化 |
| SC | Singh 2020 style（无 teacher） | 端到端压缩但无蒸馏，bottleneck 放太晚 |

> 这套分类体系就是 SC2 Benchmark TMLR 2023 Table 1 的前身。IC 类对应 Sidney 的"纯 LIC baseline"。

### 主要结论

- **率失真**：Entropic Student 超过所有七个 baseline（三个任务）；无 teacher 版本（= Singh 2020）显著更差，验证了知识蒸馏的必要性
- **Bitrate Allocation 可视化**（Section 4.5 / Figure 7）：Entropic Student 把 bit 集中在任务相关区域（人、球），通用图像压缩均匀分配甚至浪费在背景水印上——直观证明"任务感知压缩"
- **模型大小**（Section 4.6）：encoder 比 mean-scale hyperprior 小约 40 倍，比 MobileNetV2/V3 也小；三任务共享同一 encoder
- **端到端延迟**（Section 4.7）：RPi4 + LoRa（37.5 Kbps）场景下，比最强 baseline 延迟减少 47–62%；编码时间 + 通信延迟是主要瓶颈，服务器端推理时间可忽略

---

## 关于"边缘侧性能估计过于保守"的批判性观察

论文的隐含假设是移动端 = CPU-only（RPi4），所以 encoder 稍重就担心功耗和延迟。但这个假设在 2022 年已有些过时：

- Hailo-8（26 TOPS）可跑完整 SegFormer-B0、DDRNet-23-slim
- Jetson Orin、高通 NPU、Apple ANE 同期都打破了这一假设

**对 Sidney 的意义**：Sidney 的 encoder 是完整的轻量 backbone（MiT-B0 / DDRNet），在 Matsubara 的框架里算"不能部署"的级别，但 Hailo-8 跑起来没问题。Sidney 的 bottleneck 压缩的是经过完整 backbone 提炼的语义特征，信息密度更高——这是比 Matsubara 更难也更有意义的设定。

论文中可以点一句：随着边缘 AI 专用硬件普及，"encoder 必须极轻"的前提已松动，瓶颈转移到了语义特征的压缩传输本身。

---

## 与 SC2 Benchmark TMLR 2023 的关系

WACV 2022 = 方法论文（提出 Entropic Student）  
SC2 TMLR 2023 = 评估框架论文（标准化 benchmark，180+ 模型，新增 ExR-D 三维权衡指标）

SC2 的新贡献：提出 **三维权衡**（rate × encoder_size × accuracy），揭示"bottleneck 放晚层 RD 好但 encoder 太重"的陷阱，推荐早层 bottleneck（即 Entropic Student 的设计）。

---

## Sidney 论文写作用法总结

**引用 Matsubara 时的 framing**：
> "Following the supervised compression for split computing framework (Matsubara et al., 2022c), and different from two-stage feature distillation approaches, we propose an end-to-end task-driven training paradigm where the bottleneck is jointly optimized with multiple dense prediction tasks."

**可以借鉴的实验设计**：
1. Section 4.5 的 bitrate allocation 可视化 → 在 Cityscapes 上做类似分析，展示 bottleneck 关注语义区域（车、行人）而非天空背景
2. Section 4.7 的端到端延迟测试框架 → 对应 Hailo-8 实测延迟实验
3. IC baseline 对比体系（JPEG/BPG/hyperprior + SegFormer-B0）→ 对应 Sidney 的"纯 LIC baseline"

**Sidney 相对 Matsubara 的三点差异（可写进 contribution）**：
1. distortion = 任务 loss，而非特征重建 MSE → 更端到端，更直接优化任务指标
2. 多任务同时训练，bottleneck 被 seg + depth + normal 同时约束 → 表示更通用
3. 边缘设备用 Hailo-8 NPU，encoder 是完整轻量 backbone → 更接近 V2X 实际部署场景
