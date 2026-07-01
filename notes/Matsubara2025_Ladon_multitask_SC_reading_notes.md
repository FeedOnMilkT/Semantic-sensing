# 论文笔记：Ladon — A Multi-task Supervised Compression Model for Split Computing

**作者**：Yoshitomo Matsubara, Matteo Mendula, Marco Levorato  
**发表**：WACV 2025  
**代码**：https://github.com/yoshitomo-matsubara/ladon-multi-task-sc2  
**关键词**：Split Computing, Multi-task, Supervised Compression, Knowledge Distillation, Rate-Distortion  
**与项目关系**：SC2 Benchmark（Matsubara 2023）的直接续集，多任务 SC 先驱，作为 related work 引用；训练范式是 distillation-based，与 Sidney 的 ICM task-driven 主方案形成对比

---

## 一、核心问题

已有 SC 方法（含 Entropic Student）在多任务场景下要对同一张图跑多次 inference（每个任务一次），边缘侧延迟和能耗不可接受。如何让一次 inference 同时服务多个任务？

---

## 二、系统设计

**拓扑**（与 Sidney 项目同构）：
```
边缘设备（弱）→ Encoder → 压缩表示 ẑ → [无线信道] → 服务器 → Decoder + 多任务头
```

**与 Entropic Student 的对比**：

| | Entropic Student | Ladon（本文） |
|---|---|---|
| Encoder | 共享，task-agnostic | 共享，task-agnostic |
| 服务器端 backbone | 每个任务独立参数 | 所有任务共享参数 |
| 运行时 inference | 每张图跑 N 次 | 每张图跑 1 次 |
| 预处理 pipeline | 每任务各自（分类用 crop）| 统一 pipeline |

**任务组合**：分类（ILSVRC 2012）+ 检测（COCO 2017）+ 分割（PASCAL VOC 2012）

**Backbone**：ResNet-50（25.6M）和 ResNeSt-269e（110M），均在 ImageNet 预训练。把前几层（含第一个 residual block）替换为 supervised compression encoder-decoder，后接 Faster R-CNN + FPN（检测）和 DeepLabv3（分割）。

---

## 三、训练流程（三步分阶段）

### Step 1：预训练 encoder-decoder

**Loss**：
```
L_pre(x) = Σ_{i∈I} ||h_teacher_i(x) - h_student_i(x)||²   ← distortion（feature MSE）
           − β · log p_ψ(f_θ(x) + ε)                       ← rate（Ballé hyperprior）
ε ~ Unif(-1/2, 1/2)   ← 均匀噪声模拟量化误差
```

- Teacher：原始预训练 backbone（不修改）
- Student：插入 encoder-decoder 后的改造版
- 目标：压缩再重建后，backbone 看到的特征≈没压缩时的特征
- **这一步不知道最终任务是什么，只保特征保真度 + 率约束**
- ε 的均匀噪声是 LIC 标准做法（精确模拟 round 量化误差，比 VIB 的高斯采样更贴近部署）

### Step 2：固定 encoder，fine-tune decoder + backbone 后续层

- 在 ImageNet 上用标准知识蒸馏继续训
- 目的：让 backbone 参数对多个下游任务可复用

### Step 3：冻结全部参数，接入并 fine-tune task-specific heads

- 检测头（Faster R-CNN + FPN）和分割头（DeepLabv3）独立 fine-tune
- encoder 参数在整个 Step 2/3 中完全冻结

**训练范式定性**：分阶段冻结，非端到端联合优化。distortion = feature MSE（蒸馏），不是任务损失。

---

## 四、核心创新点

**唯一核心创新**：把 split computing 从单任务推进到多任务运行时效率问题。统一预处理 pipeline + 共享 backbone 参数 + 末端分叉 heads = 多任务单次 inference。

方法层面没有新东西：encoder-decoder 来自 Entropic Student，率失真框架来自 Ballé，backbone 用现成预训练模型。论文价值更多在**实验评估**：真实 Jetson 设备上的延迟/能耗测量（30 种 baseline 配置）。

---

## 五、实验结果

**硬件**：
- 移动端（弱）：Jetson Nano（~0.5 TOPS）、Jetson Xavier NX（21 TOPS）
- 服务器端（强）：RTX A500 Mobile 笔记本
- 信道：100 Kbps / LoRa 37.5 Kbps

**性能**（rate-distortion tradeoff）：
- 分类、检测：超过轻量 baseline（MobileNet 系列）
- 分割：与 baseline 持平，未超越
  - 原因：pixel-level 预测对压缩损失更敏感（论文自认的局限）

**延迟**：相比 LC baseline 降低最多 95.4%，相比 SC（Entropic Student）baseline 降低最多 95.3%

**能耗**：移动端能耗降低 65.0–88.2%

---

## 六、与 Sidney 项目的对比定位

### 相同点
- 系统拓扑完全一致：edge encoder → bottleneck → server decoder + task heads
- 率约束机制相同：Ballé hyperprior 风格，均匀噪声模拟量化
- 多任务共享 backbone + 末端分叉 heads 的架构哲学

### 关键差异

| 维度 | Ladon | Sidney 主方案（ICM） |
|---|---|---|
| Distortion 项 | Feature MSE（蒸馏） | 任务损失（CE+Dice / SiLog / Cosine） |
| 训练方式 | 分阶段冻结，非端到端 | 端到端联合优化 |
| 任务性质 | 混合（分类+检测+分割） | 全 dense prediction（seg+depth+normal） |
| 任务粒度 | 图像级/框级/像素级混合 | 像素级统一 |
| 预处理统一难度 | 需额外设计（分类 crop 问题）| 天然统一 |
| 核心问题 | 多任务运行时效率 | bottleneck 对不同性质 dense 任务的泛化性 |

### 在论文中的定位

Ladon 作为"多任务 SC 先驱"在 related work 引用。表述方向：

> "Ladon [WACV 2025] is the first multi-task supervised compression model for split computing but adopts distillation-based distortion and mixes sparse and dense prediction tasks. Unlike Ladon, our approach adopts end-to-end task-driven training following the ICM paradigm and focuses exclusively on dense prediction tasks (segmentation, depth, surface normals) to investigate bottleneck generality across semantic and geometric understanding."

---

## 七、对 Sidney 项目的可借鉴点

**可直接借鉴**：
- Figure 4（PASCAL VOC 分割曲线）的格式即是 Sidney 核心实验图的模板：x 轴 compressed data size / bpp，y 轴 mIoU，多条 β 扫描曲线
- 分割比分类/检测更难压缩的实验结论，可引用支撑 dense prediction 对 bottleneck 要求更高的论点
- SC2 三角权衡（encoder 大小 / 数据大小 / 精度）作为评估框架

**可作为 ablation baseline**：
- Feature MSE distortion（Ladon 风格）vs 任务驱动 distortion（Sidney 主方案）
- 两者使用相同熵模型和率约束，唯一差异是 distortion 定义，对比有说服力

**Feature MSE 作为 Phase 2 预热的可行性**：
- 先用 feature MSE 预训练 bottleneck，再切换 task loss fine-tune
- Teacher 必须是多任务模型（Phase 1b 的无压缩 baseline），否则 teacher 特征不含深度/法线信息
- 可降低 Phase 2 多任务训练不稳定风险

**不适合借鉴**：
- 分阶段冻结训练策略（Sidney 端到端更激进，bottleneck 随任务 loss 持续更新是优势）
- Baseline 模型选择（Ladon 用 MobileNet 系列是因为 SC2 benchmark 框架，Sidney 用 SegFormer-B0 量级更合适）

---

## 八、关键引用

- Entropic Student [22]：Matsubara et al., WACV 2022（Step 1 的直接来源）
- SC2 Benchmark [23]：Matsubara et al., TMLR 2023（评估框架来源）
- Ballé Hyperprior [3]：Ballé et al., ICLR 2018（率约束机制）
- Chimera [26]：Nimi et al., SMARTCOMP 2022（另一个多任务 SC 工作，需至少跑 34 层才能压缩，Ladon 对其的批评值得注意）
