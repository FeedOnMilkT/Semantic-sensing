# RTFormer 精读笔记
> NeurIPS 2022 · Baidu VIS
> 论文全称：RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer

---

## 一、论文定位

**一句话**：CNN 实时分割的 Transformer 替代方案，核心贡献是解释清楚了为什么原版 Transformer 在 GPU 上做实时分割慢，并给出了有针对性的工程修复。

**与 Sidney 项目的关系**：
- RTFormer 是 backbone 级别的文章，不涉及压缩表示学习，对 bottleneck 设计无直接影响
- 提供了 MiT-B0 和 DDRNet-23-Slim 之外的第三条对照线（hybrid CNN+Transformer）
- 对 MiT-B0 做 Hailo 适配改造时，有几个设计思路可以借鉴

---

## 二、核心问题：为什么 Transformer 在 GPU 上做实时分割慢

论文明确指出两个障碍：

**障碍一：Multi-Head Attention 在 GPU 上不友好**

标准 MHA 把矩阵乘法沿 channel 维切成 H 份，每份做一次小矩阵乘，类比于 Group Convolution。GPU Tensor Core 专为大块连续 GEMM 优化，切成 H 组之后每组的收缩维只有 $d/H$，计算单元大量空转。

**障碍二：高分辨率分支做自注意力效率低**

高分辨率特征（stride 1/8）每个 token 感受野有限，让它跟自己做 attention 学不到真正的全局语义。场景级语义（"这是路口"、"这是天空"）在低分辨率特征里才有。

---

## 三、RTFormer Block 设计

双分辨率结构，**Stepped Layout**（低分辨率先走，高分辨率再查询低分辨率的输出，不是并行）。

### 3.1 GPU-Friendly Attention（GFA）— 用在低分辨率支

**出发点：External Attention（EA）**

$$\text{EA}(X, K, V) = \text{DN}(X K^\top) \cdot V$$

- $K, V \in \mathbb{R}^{M \times d}$ 是可学习的固定参数（训练出来的，类似 Conv 权重），与输入无关
- 复杂度 $O(NM)$，$M$ 是固定超参数，不随分辨率增长
- DN（Double Normalization）：先行 softmax，再列归一化

**Multi-Head EA 的问题**：把 $X$ 切成 H 份，每份做 $(N, d/H) \times (d/H, M)$ 的小矩阵乘，收缩维只有 $d/H$，GPU 利用率低。

**GFA 的解法**：不切 $X$，把参数扩大 H 倍，$K_g, V_g \in \mathbb{R}^{M_g \times d}$，$M_g = M \times H$：

$$\text{GFA}(X, K_g, V_g) = \text{GDN}(X K_g^\top) \cdot V_g$$

- $XK_g^\top$：$(N, d) \times (d, M_g)$，一次完整大矩阵乘，收缩维是完整的 $d$，GPU 满载
- GDN（Grouped Double Normalization）：行 softmax 后，把 $M_g$ 列切成 H 组，每组独立做列归一化 → 每组是一个"虚拟头"，多样性靠归一化而非矩阵乘法来实现
- 参数量是 MHEA 的 H 倍，但全部在一次大 GEMM 里消化

**直觉**：把多头的多样性从"矩阵乘法的 split"转移到"归一化的 grouping"，前者 GPU 不喜欢，后者 GPU 无所谓。

> ⚠️ **Hailo 兼容性**：GDN 的第二步归一化作用在非最后维（列方向 = N 维），Hailo 只支持最后维 Softmax，GFA 可能在 DFC 编译阶段卡住。

### 3.2 Cross-Resolution Attention（CA）— 用在高分辨率支

高分辨率像素（Q）不跟自己做 attention，而是去查询低分辨率特征压缩而来的固定小尺寸 token pool：

$$\text{CA}(X_h, K_c, V_c) = \text{Softmax}\!\left(\frac{X_h K_c^\top}{\sqrt{d_h}}\right) V_c, \quad K_c, V_c = \phi(\theta(X_l))$$

- $\theta$：pooling + conv，把低分辨率特征压成固定 $8\times8$（Slim）或 $12\times12$（Base）的 token
- $\phi$：reshape 操作
- attention map 形状 $N_h \times 64$（固定），Softmax 在最后维 ✅ Hailo 支持
- 不用 multi-head，不用 Double Normalization（K/V 不是外部参数时单 softmax 表现更好）

**Stepped Layout 的意义**：高分辨率查询时，K/V 来自已经经过 GFA 处理过的低分辨率特征，不是原始输入。低分辨率先"想清楚"全局语义，高分辨率再来问，信息质量更高。

**Cross-feature 尺寸选择**：最优尺寸约等于高分辨率分支通道数的开方（Slim 通道 64 → 8×8，Base 通道 128 → 12×12）。

### 3.3 FFN 设计

用两个 3×3 conv 替代标准 FFN（MLP 升维 4× + depthwise 3×3 + MLP 降维）。更简单、无 channel expand，实验结果 mIoU 和 FPS 同时更好（33.0 vs 32.15 mIoU，187.9 vs 178.5 FPS）。

---

## 四、整体网络结构

```
stem（2×3×3 conv）
Stage 1-2：纯 Conv Block（学局部信息）
Stage 3：双分辨率 Conv Block（开始高/低分辨率分支）
Stage 4-5：RTFormer Block（GFA 在低分，CA 在高分）
Head：DAPPM（多尺度池化，借自 DDRNet）→ 融合高分辨率特征 → 3×3 + 1×1 分类头
```

- 高分辨率始终 stride 1/8
- 低分辨率 stride 逐步 1/16 → 1/32 → 1/32

**DAPPM**（Deep Aggregation Pyramid Pooling Module）：多个不同尺度 avg pooling 逐级累加（瀑布式传递），比 PSPNet 的直接 concat 信息融合更充分。在边缘侧可用轻量版本（ASPP 或 global avg pool）替代。

**两个变体**：

| | 参数量 | 通道数 | Cross-feature |
|---|---|---|---|
| RTFormer-Slim | 4.8M | 32/64 起步 | 8×8 |
| RTFormer-Base | 16.8M | 64/128 起步 | 12×12 |

---

## 五、实验结果

**Cityscapes**：比 DDRNet 只多 0.2~0.4 mIoU，差距很小。场景统一、数据量大，全局语义的边际收益低。

**CamVid**：比 DDRNet-23-Slim 多 6.7 mIoU（81.4 vs 74.7），差距悬殊。数据量小、场景多样，全局 context 价值被放大。

**ADE20K / COCOStuff**：差距 3~4 个点。类别越多、场景越复杂，全局 context 收益越明显。

**规律**：RTFormer 的 Transformer 全局语义优势在小数据集、多样场景下才充分体现，Cityscapes 这类大数据集局部纹理信息已经足够。

---

## 六、消融实验

| 实验 | 结论 |
|---|---|
| Attention 类型 | GFA+CA 最优（33.0 mIoU），SA+SA 速度慢一倍但精度持平，加 CA 后精度涨 0.2 FPS 几乎不变 |
| FFN 类型 | 两个 3×3 conv 比标准 MLP+depthwise FFN 精度和速度同时更好 |
| GDN 分组数 | 低分辨率 8 组 + 高分辨率 2 组最优，改分组数不影响 FPS |
| Cross-feature 尺寸 | 8×8 是 Slim 的最优点，尺寸与高分辨率通道数对应 |

---

## 七、对 Sidney 项目的借鉴

### 7.1 Hailo 部署兼容性评估

| 组件 | Hailo 兼容性 | 说明 |
|---|---|---|
| GFA 矩阵乘 | ✅ | 标准线性层 |
| GFA GDN 第一步（对 $M_g$ 维 softmax） | ✅ | 最后维 |
| GFA GDN 第二步（对 N 维列归一化） | ⚠️ | 非最后维，可能不支持 |
| CA Softmax（对 64 token 维） | ✅ | 最后维 |
| CA Cross-feature 生成（pooling + conv） | ✅ | 标准算子 |
| DAPPM | ✅ | 纯 conv |

**结论**：Hailo-8 上 SegFormer MiT-B0 更合适（Model Zoo 有官方支持），RTFormer 的 GPU-Friendly 优化是针对 GPU 的，不适用于 NPU。RTFormer-Slim FLOPs 是 MiT-B0 的 2 倍以上，也不利于边缘部署。

### 7.2 可借鉴的设计哲学

**FFN 替换**（优先级高）：把 MiT-B0 的 Mix-FFN 换成两个 3×3 conv，有文献证明精度和速度同时更好，Hailo 更友好。

**分层 attention 策略**（Hailo 适配应急方案）：如果 MiT-B0 早期 stage 的 attention 编译有问题，把 Stage 1（stride 1/4）的 attention 换成 Conv Block，只保留 Stage 3/4 的 attention。早期 stage 感受野小，换 conv 精度损失最小。

**Cross-resolution Attention 替代 FPN fusion**：固定 8×8 token pool 作为语义查询节点，高分辨率特征来查询，比 bilinear resize + add 信息密度更高，且 Hailo 完全兼容（最后维 softmax + 固定形状）。可作为 bottleneck 前 multi-scale fusion 模块的备选方案。

**Stepped Layout 的启示**：做 multi-scale fusion 时，让最低分辨率特征先多跑几层聚合语义，再让浅层特征来查询，送进 bottleneck 的语义会更纯粹。

### 7.3 RTFormer 在 backbone 选型图谱中的位置

```
Sidney 的 backbone 选型：
├── MiT-B0（纯 Transformer）          ← Transformer 支，Hailo Model Zoo 支持
├── DDRNet-23-slim（纯 CNN）           ← CNN 支，Hailo 量化友好
└── RTFormer-Slim（CNN+Transformer）  ← Hybrid 第三条线（如时间允许）
```

RTFormer 是 CLAUDE.md 里提到的 "TopFormer-T 类 hybrid 架构"的更好候选——比 TopFormer-T 在文献里更知名，且跟 DDRNet 有直接架构继承关系（都用 DAPPM、双分辨率）。作为 stretch goal，不影响主线。

### 7.4 一个值得探索的实验叙事

RTFormer 在 Cityscapes 上比 DDRNet 只好 0.2，在 CamVid 好 6.7——说明全局语义在小数据集/场景多样性高的情况下价值更大。

对应 Sidney 的 bottleneck 实验：低 bpp 压缩时，哪些类的 mIoU 先崩？如果细长/边界密集类（Pole、Rider）先崩而大面积语义类（Sky、Road）保持好，说明 bottleneck 优先保语义、舍几何边界，这是一个有价值的实验叙事。

---

## 八、关键数字速查

| 模型 | 参数量 | FLOPs (512×2048) | Cityscapes mIoU | CamVid mIoU | GPU FPS |
|---|---|---|---|---|---|
| SegFormer-B0 | 3.8M | 8.4G | — | — | 50.5 |
| DDRNet-23-Slim | 5.7M | 18.2G | 76.1 | 74.7 | 101.0 |
| DDRNet-23 | 20.1M | 71.6G | 78.9 | 76.3 | 38.3 |
| RTFormer-Slim | 4.8M | 17.5G | 76.3 | 81.4 | 110.0 |
| RTFormer-Base | 16.8M | 67.4G | 79.3 | 82.5 | 39.1 |

---

*笔记整理日期：2026-05-30*
