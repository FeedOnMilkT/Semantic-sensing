# ELEC0054 项目上下文 · 语义感知通信系统
> 本文件同时作为：① Sidney 的项目知识笔记 ② Claude 在本项目下的上下文提示词
> 每次新对话开始前请先阅读此文件，以恢复完整的项目认知。
> 最后更新：2026-06-30（**方向收敛**：① **主用 RTFormer** 作 backbone——论文部署贡献主体集中在它身上，其余骨干降级为对照/基线；② **先只做单任务分割（seg）**，多任务（Phase 1b：seg + depth + normal）**暂缓**为后续阶段，框架的多任务可扩展性保持不动；③ **TopFormer 暂时搁置**——作 split-computing 时 split-point 不好选：其 SSE/SIM 是并联旁路 + 回灌，缺 RTFormer stage3 那种 conv→attention 串联分界与单张紧致 cut，留作 RTFormer bring-up 受阻时的备选；④ 下文凡涉及「推荐顺序 / Phase 1b 多任务 / Backbone ablation」等多任务·多骨干表述，**当前一律以本条为准**，降级为后续工作；⑤ **数据集主用 ADE20K**（150 类单任务 seg；Cityscapes 降为跨数据集对照/参照，配方各自对齐原文）；⑥ **Phase 2 冻结 teacher 暂用单任务 seg teacher（Phase 1a）**——与「先只做单任务」一致，多任务 teacher 待 Phase 1b 重启后再换。）
> 前次更新：2026-06-01（蒸馏升为**主线训练范式**——feature-MSE 对齐冻结 teacher 当 distortion，bitstream 不变；bottleneck 定在 **RTFormer stage3 之后**（conv→attention 边界）；确认**接收端同为 Hailo-8**，两端对称约束，decoder 也必须 DFC 可编译；Demo 2 改为**蒸馏式监督压缩**，弃纯 VIB 用 EntropyBottleneck；并清理历史冗余）
 
---
 
## 一、项目基本信息
 
- **课题名称**：基于视觉的场景语义感知通信系统
- **学生**：Sidney（MSc 毕业设计）
- **方向定位**：信道无关的语义信源编解码（Channel-Agnostic Semantic Source Coding），面向多任务场景理解
- **大背景**：面向协同感知（Collaborative Perception）的语义通信——Intermediate Fusion 中特征级共享时的语义压缩问题
- **核心任务**：边缘端压缩场景语义并传输，接收端用单一压缩表示支持多种下游感知任务——语义分割（主线）、单目深度估计（第二任务）、表面法线估计（第三任务）
- **应用场景**：面向协同感知的语义压缩传输（自动驾驶 V2X、智慧交通、多机器人协同）
- **Sidney 技术背景**：有 3D 目标检测经验（本科 CenterPoint TensorRT 端到端部署），熟悉 Backbone-Neck-Head 范式与 PyTorch，有几何感知（点云处理）经验
- **职业目标**：车企/具身感知算法端侧部署；找日常实习，6 月初投递；5.20 已正式进实验室
### 协同感知中的定位
 
```
协同感知三大范式：
├── Early Fusion（数据级）：传原始图像/点云，带宽不可行
├── Intermediate Fusion（特征级）：传压缩后的中间特征  ← Sidney 服务这一层
└── Late Fusion（结果级）：传最终检测框/标签，信息损失大
 
Sidney 的定位 = 为 Intermediate Fusion 提供底层语义信源编码能力（不做信道，不做融合）：
  本地 encoder → semantic bottleneck → [任意信道] → 接收端 decoder → 多任务场景理解
  核心问题：特征怎么压缩才能在有限带宽下保留足够语义，供接收端完成多种感知任务
  不涉及：信道编码/传输（队友负责）、多智能体特征融合、联合决策
```
 
### 核心叙事（重要）
 
**"单一压缩表示同时支持语义理解（What is there）、深度几何（Where is it）、表面几何（How it's oriented）三类 dense prediction，信道无关、多任务可插拔。"**
 
- 三者都是 dense prediction，可共享 decoder，bottleneck 被迫学到通用场景表示
- 契合 V2X 需求（BEV 转换、自由空间检测、可通行平面估计都既要深度也要法线）
- 三任务（而非两任务）的额外价值：是 MTL 文献（MTAN / MTI-Net / MultiMAE / PAD-Net）的标准 benchmark setting，评审一眼看出方法成熟度；同时验证 bottleneck 对"语义/粗几何/细几何"三类不同性质任务的普适性
---
 
## 二、系统分工（非常重要）
 
```
Sidney 负责的部分（完整语义感知网络）
├── 发送端（Pi 5 + Hailo-8）
│   ├── 语义编码器（Backbone stem+stage1~3，纯卷积）
│   └── 语义瓶颈（Bottleneck）← 核心创新点，同时是算力切分点
└── 接收端（Pi 5 + Hailo-8，同等算力）
    ├── Transformer-增强 Decoder（RTFormer 风格可部署注意力，重建全局上下文）
    ├── 任务头1：语义分割（Cityscapes 19 / NYUDv2 40 / PASCAL-Context 59，mIoU 主指标）← 主线
    ├── 任务头2：单目深度估计（RMSE / δ<1.25 / AbsRel）← 证明粗几何泛化
    └── 任务头3：表面法线估计（mean/median angular error / % within 11.25°/22.5°/30°）← 证明细几何泛化
 
队友负责的部分
└── 信道编码 + 传输（传统信号处理 LDPC/Turbo 等，Sidney 完全不涉及）
 
系统接口
└── Sidney 的 bottleneck 输出 → 队友的信道编码器输入
    训练态：连续 latent + 学到的熵模型估率（feature-MSE 蒸馏驱动，详见第五节）
    部署态：量化 + 算术编码 → bitstream（可接任意信道编码器）
```
 
**关键前提（一次说清，全文不再重复）**：
- **通信-感知完全解耦，这不是 JSCC**：DNN 只做信源编码，训练时不仿真信道、不注入任何噪声、不输出 analog symbols。导师明确要求 "just consider pure semantics"。
- **两端对称**：发送端和接收端都是 Pi 5 + Hailo-8（26 TOPS）。**没有"更强的接收端"可以堆算力**——decoder 也必须 Hailo 可编译、在 26 TOPS 预算内（这条 2026-06-01 确认，直接约束 decoder 设计）。
- 接收端支持多下游任务是核心叙事；Sidney 贡献核心 = "设计信道无关的语义信源编解码器，使压缩表示能支持多种感知任务"。
---
 
## 三、硬件平台
 
### 主平台：Raspberry Pi 5 + Hailo-8 AI HAT（发送端、接收端各一块）
 
| 特性 | 说明 |
|---|---|
| 算力 | 26 TOPS，低功耗 |
| 支持模型 | YOLOv8n/s、ResNet 系列、SegFormer-B0（Model Zoo 有） |
| Transformer 支持 | 轻量可行（ViT、LeViT、EfficientFormer、SegFormer-B0、RTFormer 类线性注意力）；重型/动态注意力不可行 |
| 量化 | INT8 主体，关键层（LayerNorm、Softmax）可 a16_w16 |
| 部署格式 | Hailo DFC 转换 → HEF；`defuse_attention` 拆 MHA，2025-04 版优化 LayerNorm 量化 |
 
### 不可行的架构（Hailo 约束下）
- Deformable DETR / 动态注意力类（`grid_sample`、动态 offset）
- Query-based 分割（Mask2Former、Panoptic-SegFormer：cross-attention + bipartite matching）
- 大型 Transformer（SegFormer-B5、Swin-L）
- 像素级重建解码器（反卷积还原整图）——**接收端也是 Hailo-8，跑不动**
- 不支持算子：`grid_sample`、`torch.unique`、动态 `einsum`、动态 reshape、非最后一维 Softmax、`roll`
### 约束对架构的影响
- 发送端 = stem + stage1~3（纯 Conv+BN+ReLU，Hailo 极友好）；bottleneck 后才是注意力，且注意力放在**接收端 decoder**
- decoder 的 transformer 增强**也受 Hailo 约束**：只能用 RTFormer 那套可部署注意力（GFA / cross-resolution，静态 shape），且要确认过 DFC、与三个 head 一起在 26 TOPS 内
- 实时性是设计约束，不是 future work
- 部署时 LayerNorm/attention Softmax 用 a16_w16，主干 conv 用 a8_w8
- **Hailo Model Zoo 官方支持的真正价值**：因要插 bottleneck 端到端重训，预编译 HEF 用不上；价值仅在于证明该类算子能过 DFC、提供 .alls 配置参考。不要把"官方支持"当 backbone 选型的决定性优势
---
 
## 四、技术架构（信道无关语义信源编解码）
 
### 核心架构决策
 
图像 → Encoder(stem+stage1~3) → **Bottleneck（stage3 后）** → [任意信道] → Transformer-增强 Decoder → 多任务头。叙事核心：单一压缩语义表示，信道无关，可插拔多种感知任务（语义 + 粗几何 + 细几何）。
 
### 主基座选型（按"部署可改造性"排优先级）
 
> **不要把 backbone 硬编码为单一网络。** 这是按优先级排序的候选池，实施顺序随 DFC bring-up 与时间预算动态调整。评估轴 = "在数学等价改造 + 训练/部署异构结构下，哪个骨干能把最多部署简化锁进**零损失精确档**"。
 
**两种部署简化机制的质量分档（核心判据）**：
- **精确等价改造**（RepVGG 折叠 / BN 吸收 / 冻结线性投影合并）：零精度损失、不重训、有数学保证 —— 主线机制。
- **结构替换 + 蒸馏桥接**（训练 softmax → 部署线性/卷积近似）：有损、需额外训练 —— 仅作 fallback / ablation。
  - 注意：这里的"蒸馏"是**部署桥接蒸馏（角色 A）**，是 fallback；与第五节作为训练主线的 **feature 蒸馏（角色 B）** 是两回事，别混。
**双线平行**：CNN 和 Transformer/attention 两条线平行实施做正交对照（证明 bottleneck 普适性），这条不变。
 
#### Transformer / attention 支：优先级候选池
 
**优先级 1（首选蓝本）：RTFormer 风格（GPU-Friendly / external 线性注意力 + 双分辨率）**
- 唯一卷积半边和注意力半边都能走精确档的：双分辨率卷积分支可 RepVGG/BN 折叠；GFA 外部记忆 K_g/V_g 是冻结参数，两端矩阵乘折进前后 BN，只剩 grouped double-normalization 作非线性。
- 结构细节（读 RTFormer NeurIPS 2022 已确认）：stem + stage1/2/3 是基本残差卷积块，**stage4/5 才是 RTFormer attention 块**；高分支全程 stride 8 保边界，低分支 stride 16/32/32；低分支用 GFA（external，线性，数据无关记忆），高分支用 cross-resolution attention（softmax，K/V 来自低分支，数据相关）；FFN = 两层 3×3 conv（无维度扩张）；seg head 用 DAPPM。
- 与 bottleneck 叙事同构：外部记忆 = 一组学到的全局语义基 ≈ "通用场景表示"。最契合 V2X 实时叙事。
- **代价**：不在 Model Zoo，首次 DFC bring-up（double-norm 算子、可能的非 last-dim softmax 需插 transpose）有一次性工程风险；cross-resolution 跨分支项数据相关，只能部分折叠。
**优先级 2（稳健备选）：TopFormer 风格（MobileNet token pyramid + 池化 token 上小 attention）**
- 主体 depthwise separable conv，极 Hailo 友好；attention 只在池化后小而静态 token 集上跑。卷积半边精确折叠，SSE 的 softmax 只能走蒸馏档但体量极小。RTFormer bring-up 受阻时切换。
**优先级 3（参考，不押在部署路径）：EfficientFormerV2 风格**
- 后期 MHSA 带 attention bias + talking-head，注意力半边要蒸馏又要处理附加项；低延迟是为 Apple ANE/ARM NPU 调的，非 Hailo dataflow。当灵感来源，不作蓝本。
**下界基线：MiT-B0（SegFormer backbone）**
- 3.8M 参数，Cityscapes 76.2 mIoU；GLPN 证明可直接做深度。唯一被 Model Zoo 收录、DFC 可编译性已验证的 transformer。但 SR-softmax 数学不可约，"训练≠部署"打法在它身上不产生额外价值 → 定位为**"训什么部署什么"的安全基线与对照下界**，新蓝本的相对压缩-性能曲线都与它对比。
#### CNN 支：DDRNet-23-slim（保持不变）
- 5.7M 参数，Cityscapes 77.4 mIoU @ 109 FPS；双分辨率 + bilateral fusion；纯 Conv+BN+ReLU，Hailo 量化极友好，conv 半边天然支持 RepVGG/BN 折叠。bottleneck 插入候选位置少（detail/context/fusion），ablation 清晰。**PIDNet-S 作备选**。
- 注：RTFormer 双分辨率主干与 DDRNet 同源，选 RTFormer 可让两支共享更多结构。
**双 backbone 工程原则**：
- 共享 80% 代码（bottleneck、decoder、heads、训练循环、数据 pipeline 统一）；`backbones/` 下各骨干返回 4-scale 特征字典，接口对齐。
- 训练超参不强行统一（Transformer 用 AdamW+cosine，CNN 用 SGD+poly）。
- 每个 backbone 实现 **train/deploy 双模式 forward**：训练态走完整图，部署态走折叠后等价图。
- ablation 报**相对压缩-性能曲线**，不报绝对值。
- **推荐顺序（2026-06-30 收敛）**：当前**主用 RTFormer**（论文部署贡献主体，先把它的单任务 seg 跑透）；MiT-B0 仅作打通/下界基线；DDRNet 作 CNN 对照、TopFormer 暂搁置（split-point 不好选），均列**后续**。优先级随 DFC bring-up 实际结果回退。
### Bottleneck 插入点：RTFormer stage3 之后（核心决策，2026-06-01）
 
用 **三方权衡**（编码代价 ↔ 传输数据量 ↔ 任务精度，源自 SC2 / Ladon）定位：
 
| 插入位置 | 发送端代价 | 可压性 | 多任务通用性 | 判定 |
|---|---|---|---|---|
| 早（stage1/2） | 便宜 | 差（低级特征冗余多） | 中 | ✗ |
| **中（stage3 后）** | **便宜（纯卷积）** | **好（语义已聚集、空间已收缩）** | **好（未 task-collapse）** | **✓ 甜点** |
| 晚（stage4/5 后） | 贵（Hailo 跑 attention，风险） | 最好 | 差（向单任务塌缩） | ✗ |
 
- stage3 恰是 RTFormer 的**卷积→注意力分界**：发送端只跑 stem+s1+s2+s3（纯卷积），attention 阶段（stage4/5 等价）挪到接收端 decoder。网络在此**对半切**，bottleneck 同时是压缩点与算力切分点，两半都受 Hailo 约束。
- **关键认知**：可压性取决于**任务相关熵密度，不是 channel 数**。高分中层特征原始张量可能比原图还大（所以不在那儿插）；深层特征虽 channel 多、空间小，任务相关 bit 反而更集中、更好压。判据 = 插在空间已收缩、任务相关信息已聚集处。
- **待定**：stage3 出来是双分支（高分 stride8 + 低分 stride16）。压低分支 + 高分走轻量旁路，还是先融合再压？—— 第十二节开放问题 #1。
### 整体数据流
 
```
======== 发送端 (Pi 5 + Hailo-8) ========
  Image (3, 512, 1024)
        │
        ▼
  stem + stage1 + stage2 + stage3        [纯卷积，Hailo 友好]
   多尺度特征 F1(H/4) F2(H/8) F3(H/16)
        │
        ▼
  Multi-scale Fusion (轻量, FPN-lite / MLP fusion)
        │
        ▼
  ┌─────────────────────────────────────┐  ← 核心创新点 + 算力切分点
  │ Semantic Bottleneck                 │
  │   z (D, h, w),  D≪C, hw≪HW          │
  │   训练：feature-MSE 蒸馏 + 学到的熵模型(rate)│
  │   部署：量化 + 算术编码 → bitstream   │
  └─────────────────────────────────────┘
        │ bitstream
======== 信道 (队友，Sidney 不涉及) ========
        bitstream → [LDPC/Turbo 等] → 传输 → 解码 → bitstream
======== 接收端 (Pi 5 + Hailo-8，同等算力) ========
        z（解码后）
        │
        ▼
  ┌─────────────────────────────────────┐
  │ Transformer-增强 Decoder            │
  │   RTFormer 风格可部署注意力 + 上采样  │
  │   重建 stage4/5 级全局上下文          │
  │   [必须过 DFC，与 heads 同在 26 TOPS 内]│
  │   共享主体，最后 1-2 层任务分叉        │
  └─────────────────────────────────────┘
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ Seg Head  │  │ Depth Head│  │ Normal Head│
  │ (K,H,W)   │  │ (1,H,W)   │  │ (3,H,W)   │
  │ K=19/40/59│  │ depth(m)  │  │ L2-norm   │
  │ mIoU 主线 │  │ RMSE/δ    │  │ angular err│
  └───────────┘  └───────────┘  └───────────┘
   所有数据集     Cityscapes/NYUDv2  NYUDv2/PASCAL-Context
```
 
**任务-数据集开关矩阵**（trainer 必须按 dataset 条件化决定开哪些 head）：
 
| 数据集 | seg | depth | normal |
|---|---|---|---|
| Cityscapes（室外 V2X 主场景） | ✓ (19 类) | ✓ (双目→深度) | ✗ |
| NYUDv2（室内 MTL 经典） | ✓ (40 类) | ✓ (Kinect) | ✓ |
| PASCAL-Context（物体级 MTL 经典） | ✓ (59 类) | ✗ | ✓ |
 
### 多任务头设计原则
 
- **共享 decoder**：bottleneck → 共享特征解码 → 三个分叉头，参数少，bottleneck 被迫学通用表示。共享主体 3-4 层（含 transformer 增强），分叉点选最后 1-2 层（参考 Panoptic-DeepLab 双头哲学、PAD-Net 三头架构）。
**分割头**（极简）：1×1 conv 输出 K 通道 logits → bilinear 上采样。
 
**深度头**（稍重，保深度边界）：1×1 扩通道 → depthwise 3×3 保细节 → 1×1 输出 1 通道 → bilinear 上采样 → sigmoid × max_depth（Cityscapes 80m / NYUDv2 10m）。参考 FastDepth、Lite-Mono、GLPN。
 
**法线头**（结构同深度头，输出 3 通道单位向量）：…→ 1×1 输出 3 通道 → bilinear 上采样 → 沿 channel L2 normalize。参考 PAD-Net、Eigen 2015、MTAN normal branch。
 
**为什么深度/法线头比分割头重**：分割是分类，邻像素同类容忍度高；深度/法线是连续向量回归，边界几何跳变模糊会让轮廓崩溃（远车融进天空、墙面与桌面法线混淆）。故都需 depthwise conv 保边界，深度配梯度匹配损失，法线对噪声尤敏感。
 
### Sidney 几何感知经验复用
深度估计 = 稠密几何回归，近点云 BEV regression；SiLog/L1 类比 CenterPoint L1 regression；RMSE/δ 类比 mAP 距离阈值；disparity→depth 转换需相机内参/baseline，Sidney 熟悉。
 
### 关键设计思想来源（prior art）
 
- **架构祖先（LIC）**：Ballé end-to-end（ICLR 2017）、Ballé Scale Hyperprior（ICLR 2018，bpp 率失真框架）、Minnen Joint Autoregressive（NeurIPS 2018，熵模型）
- **方法论核心 prior art（ICM）**：Le ICM（ICASSP 2021）、Dubois Lossy Compression for Lossless Prediction（NeurIPS 2021，IB 任务保持压缩）、Choi & Han Scalable Coding for Humans and Machines（TIP 2022，一比特流多任务可插拔）
- **监督压缩 / split inference（最对口）**：Matsubara **Entropic Student**（WACV 2022，**feature 蒸馏 + rate**，SC2 最强单任务方法）、**SC2 Benchmark**（TMLR 2023，评估范式）、**Ladon**（WACV 2025，**首个多任务-head 监督压缩**，与 Sidney 同设定；三步训练：feature 蒸馏预训 encoder-decoder → fine-tune decoder → 冻结后挂任务头）
- **多任务架构**：PAD-Net（CVPR 2018，seg+depth+normal 祖师）、MTI-Net（ECCV 2020）、MultiMAE（ECCV 2022）、Panoptic-DeepLab（CVPR 2020）
- **深度估计**：GLPN（MiT backbone 做深度，最强 prior）、AdaBins、Eigen 2014（SiLog）、FastDepth/Lite-Mono（轻量 head）
- **注意**：Shao Task-Oriented Communication 用 IB 但训练做 JSCC（注入信道噪声），与 Sidney 分离式路线相反，仅在 related work 作对比叙述。
---
 
## 五、损失函数与训练范式
 
### 主线 = feature 蒸馏式监督压缩（Entropic Student / Ladon 配方）
 
> **核心认知（2026-06-01 定）**：distortion 用 **feature-level MSE 对齐冻结 teacher**，不是像素 MSE、也不是直接任务 loss。蒸馏只换 distortion 项，**rate 项 + 熵模型 + 算术编码完全不变 → 照常输出 bitstream**。这把"三任务平衡 × 率约束同时训"降级成"匹配 teacher 特征 + 率"，训练稳得多，且去耦合了多任务平衡。
 
**两阶段训练（照搬 Ladon）**：
 
```python
# teacher = 冻结的无压缩多任务 baseline（Phase 1b 网络）
 
# Stage 1：训 encoder + bottleneck + decoder
#   distortion = 对齐 teacher 选定层特征；rate = 学到的熵模型
L_stage1 = Σ_i ‖h_teacher,i − h_student,i‖²  +  β · L_rate
# h_student 由 decoder(bottleneck(encoder(x)) + 均匀噪声) 给出
 
# Stage 2：冻结 encoder + bottleneck（+ decoder 主体），挂 / 微调任务头
L_stage2 = w_seg·L_seg + w_depth·L_depth + w_normal·L_normal   # 按 dataset 条件化
```
 
**L_rate（与主线一致）**：用 CompressAI `EntropyBottleneck` / `GaussianConditional` —— 均匀噪声 z̃ = μ + U(-0.5, 0.5) 模拟量化，学到的 hyperprior 逐位置预测 σ̂，`L_rate = -log p_θ(z̃)`，与真实编码长度差 <5%，部署时直接 round + 算术编码出 bit。**β（0.001~0.1）是核心超参，β 扫描曲线是论文最重要的图之一。**
 
**任务损失（Stage 2 用）**：
- 分割：CE + Dice + Lovász-Softmax
- 深度：`L_depth = SiLog + 0.1·grad_loss`（Eigen 2014 SiLog + 梯度匹配保边界）
- 法线：`L_normal = (1 - cos)[valid].mean()`，cos 在 L2-normalized 向量上算，valid = `gt_normal.norm(dim=1) > 1e-3`
- 多任务权重初始 (1.0, 0.5, 0.5)，**默认 DWA 或 Uncertainty Weighting（Kendall 2018）自动平衡**；单数据集只对存在的任务前向计 loss
**深度 GT 生成（Cityscapes）**：`depth = 0.222 * 2262.0 / (disparity + eps)`，clip 到 [1e-3, 80]m。
 
### 对照臂与 baseline（论文要画的对比）
 
| 角色 | 做法 | 用途 |
|---|---|---|
| **主线** | feature 蒸馏 distortion + 学熵模型 rate（上面 Stage1/2） | 论文核心结果曲线 |
| **对照臂** | 端到端 task-loss ICM：distortion 直接用任务损失，不蒸馏 | "我们也试了纯任务驱动"，对比蒸馏 vs 任务驱动 |
| **baseline** | 纯 LIC：CompressAI 预训练 codec（不重训，像素 MSE）解码图像 → 再跑任务网络（两阶段） | 主线要超过的对照基线，ICM 文献立足点 |
| **最简 ablation** | 纯 VIB：高斯采样 + 固定 N(0,I) + KL 当 rate | 仅作 ablation；KL 是 bpp 松上界，不直接出 bit |
 
**为什么不用 JSCC 作对照基线**：JSCC 注入信道噪声、输出 analog symbols，与分离式信源编码不可比（苹果对橘子）。related work 只用 DeepJSCC 做对比叙述，不画在同一张 bpp-mIoU 曲线上。
 
### 输出格式
- 训练态：连续 latent + 学到的熵模型估率
- 部署态：z → round → 算术编码 → bitstream（接任意信道编码器）。训练/部署分离是 LIC 业界标准。
### 核心实验曲线
- **bpp vs mIoU**（核心）、**bpp vs 深度指标**（RMSE/δ，泛化性）、**bpp vs 法线指标**、**β vs mIoU**
- **带宽对比**：语义压缩 vs H.264 图像流（基准 2~5 Mbps，不是未压缩视频）
- **Ablation**：bottleneck 维度 D / 蒸馏 vs 任务驱动 vs 纯 VIB / backbone（RTFormer vs DDRNet vs MiT-B0）/ 单任务 vs 多任务
---
 
## 六、关键概念速查
 
### 语义通信
不传全部比特，只传对任务有意义的语义。香农追求 BER，语义通信追求任务完成质量。领域很新（2020 后）。
 
### JSCC vs 分离式（Sidney 路线 = Learned Source Coding for Machines）
- **JSCC（DeepJSCC、Shao Task-Oriented IB、Xie DeepSC）**：信源信道联合训练，输出针对特定信道（analog + 噪声注入），换信道需重训。
- **Sidney 分离式**：信源编码器与信道编码完全解耦，输出 bitstream，接任意信道编码器。**Sidney 不做 JSCC**，导师明确前提。
- **领域定位**：Learned Image Compression（架构祖先）+ Image Coding for Machines（任务驱动信源编码）+ Feature Compression for Split Inference（系统同构）三分支交叉。
- **核心 prior art**：Ballé 2017/2018, Le ICM 2021, Dubois 2021, Choi & Han 2022, Matsubara Entropic Student 2022 / SC2 2023 / Ladon 2025。
- 论文措辞：**不要** "Following Xie/Shao, we propose…"；应 "Following the learned source coding for machines paradigm (Le, Dubois, Choi & Han, Matsubara), and different from JSCC-based semantic communication (Bourtsoulatze, Shao), we propose…"
### 四种损失方案的关系（同一类 rate-distortion VAE，区别只在 distortion/采样/熵模型/rate）
 
| 方案 | distortion | 采样 | 熵模型 | rate | 在本项目角色 |
|---|---|---|---|---|---|
| **LIC** | 像素 MSE | 均匀噪声 | 学到的 hyperprior | -log p_θ(z) | baseline（CompressAI 预训练，两阶段） |
| **VIB** | 任务损失（理论框架） | 高斯采样 | 固定 N(0,I) | KL | 最简 ablation |
| **ICM（任务驱动）** | 任务损失（工程） | 均匀噪声 | 学到的 hyperprior | -log p_θ(z) | 对照臂 |
| **蒸馏式监督压缩** | **feature-MSE vs 冻结 teacher** | 均匀噪声 | 学到的 hyperprior | -log p_θ(z) | **★ 主线** |
 
- 它们不是独立网络，是同一拓扑的不同损失配置。LIC 提供工程（拓扑、量化、算术编码、CompressAI）；VIB 提供理论（β 控压缩-任务 trade-off、KL 作 rate 代理）；ICM = LIC 工程 + 任务驱动 distortion；**蒸馏式 = ICM 工程 + feature 蒸馏 distortion**，是本项目主线（去耦合多任务平衡、训练更稳、对齐 SC2 最强方法）。
- 蒸馏（角色 B，训练 distortion）≠ 部署桥接蒸馏（角色 A，softmax→线性近似，第四节 fallback）。
### 信息瓶颈（IB）
找最小充分统计量 Z：对任务 Y 有用、对输入 X 冗余最少。`min -I(Z;Y) + β·I(Z;X)`。VIB 是其深度实现，β 控 trade-off。
 
### RTFormer（详见第四节优先级 1）
双分辨率；低分支 GFA（external 线性注意力，冻结记忆，可折叠）；高分支 cross-resolution attention；stage4/5 才是 attention，stage1~3 是卷积。
 
### SegFormer MiT（下界基线 backbone）
层次化 Transformer，4 stage 4 分辨率；SR 高效自注意力；Mix-FFN 替位置编码。
 
### 单目深度估计
单 RGB 预测每像素深度。损失 SiLog/BerHu/L1。指标 RMSE/AbsRel/SqRel/δ。Cityscapes 双目 disparity 转，KITTI LiDAR 投影。
 
---
 
## 七、开发路径与时间线
 
> **核心思路**：拆成两个独立 demo 分别 de-risk，再合成主实验，避免三个风险（backbone 选型 / 多任务稳定性 / 信源编码可行性）同时压在 Phase 2。
 
```
【Phase 1a / Demo 1 第一阶段（1 周）】单任务分割打通
├── encoder(先 MiT-B0) + decoder + 分割头
├── Cityscapes seg，mIoU ≥ 68（SegFormer-B0 原版 76 的 90%）
└── 验证数据 pipeline、训练 infra、ONNX 导出链路
 
【Phase 1.5 / Demo 2（1 周）】Learned Source Coding 可行性验证（见下方专节）
 
【Phase 1b / Demo 1 第二阶段（1.5 周）· 暂缓(2026-06-30)】多任务无压缩 baseline（= 主线 teacher）—— 现决定先把 RTFormer 单任务 seg 跑透，本阶段推迟至单任务跑通、训练策略统一后再启
├── 共享 encoder + 分叉 decoder + seg + depth head
├── Cityscapes 双目 disparity → depth GT，自写 dataloader
├── 目标：seg mIoU 不跌超 2 点，depth RMSE < 8m
└── 这是 bottleneck 实验的上界基线（条件A），也是 Phase 2 蒸馏的冻结 teacher
 
【Phase 2（3-4 周）】整合：bottleneck 插入多任务网络（论文主体）
├── 在单任务 seg 网络（当前 Phase 1a；多任务待 Phase 1b 重启）stage3 后插 bottleneck，attention 增强挪到 decoder
├── 主线：feature 蒸馏（对齐冻结 teacher——**当前选 A：单任务 seg teacher / Phase 1a**，多任务 teacher 待 Phase 1b 重启后再换）+ 学熵模型 rate + 两阶段挂头
├── 对照臂：端到端 task-loss ICM；baseline：纯 LIC；ablation：纯 VIB
├── 核心：bpp vs mIoU、β 扫描、bpp vs 深度/法线
└── Backbone ablation（RTFormer vs DDRNet vs MiT-B0）
 
【Phase 3（2-3 周）】部署 + 论文写作
├── Hailo-8 PTQ + .alls 调优（目标掉点 < 2 mIoU），发送端 + 接收端两块都要 bring-up
├── 可选：实测延迟/功耗、KITTI 跨域
└── 论文写作
 
【Phase 4（可选）】扩展：全景分割 / joint multi-dataset training，写 future work
```
 
**硬性门槛**：每个 demo 达验证标准即停，不追 SOTA，把时间留给 Phase 2。
 
### Demo 2 设计（蒸馏式监督压缩，CIFAR-10）— 2026-06-01 重设计
 
> 原计划 ResNet-18 + 纯 VIB。现因主线改蒸馏，Demo 2 升级为 **Phase 2 主线的单任务图像级缩小版**：bitstream 机器件不变，只把 distortion 换成 feature-MSE，一次性 de-risk **bitstream 管道 + 蒸馏配方 + 中层切分拓扑**三件 Phase 2 会原样复用的部件。
 
```
Teacher: ResNet-18 在 CIFAR-10 训好，冻结
切分点 = layer2 之后（类比 RTFormer stage3）
 
发送端: stem + layer1 + layer2 → Bottleneck (CompressAI EntropyBottleneck)
                                    │ bitstream
接收端: Decoder(几层 conv 还原 layer2 维度特征) → 冻结 teacher layer3 + layer4 + fc
蒸馏目标: teacher 在 layer2 后的特征 (feature-MSE)
 
Stage 1: 训 encoder+bottleneck+decoder，L = ‖h_t^(layer2) − h_s‖² + β·rate
Stage 2: 冻结 encoder/bottleneck，微调 classifier（下游复用冻结 teacher 时几乎不用做）
```
 
**rate 机器件用 EntropyBottleneck，不用纯 VIB**：KL 只是 bpp 松上界、不直接出 bit；EntropyBottleneck（均匀噪声 + 学熵模型 + range coder）既是 rate 又直接出 bitstream，且与 Phase 2 一致。
 
**验证 checkpoint（达标即停）**：
1. bitstream roundtrip 正确：`compress` → 写文件 → 读 → `decompress` → 分类，结果一致
2. 估计 bpp ≈ 实际文件比特数（数量级一致，理想 <10–20% 偏差）
3. β 扫描出单调 rate-accuracy 曲线
4. 蒸馏配方收敛：feature-MSE 能把 teacher 特征压到低 bpp 还保住精度（某 β 下掉点 <2%）
**加分项（近零成本）**：① 前置半天 sanity——随机特征张量过 EntropyBottleneck，单独验 compress/decompress/文件 roundtrip + bpp≈文件大小，隔离掉算术编码风险再上网络；② 留 task-loss 开关（换 distortion 项），在最简设定下看"蒸馏 vs 任务驱动"曲线差异，为 Phase 2 选型攒数据。
 
### 框架选型（已确定）
- 主代码库纯 PyTorch（不用 MMSegmentation 作主框架）；backbone 从官方仓库摘独立 .py
- 信源编码部件：CompressAI（EntropyBottleneck、GaussianConditional、arithmetic coder）
- 数据：自写 CityscapesMultiTaskDataset / CIFAR10Loader；infra：PyTorch loop + WandB/TensorBoard + OmegaConf
### Hailo 部署链路（已确认）
```
PyTorch fp32 → 静态 ONNX(opset 11-14) → onnxsim → DFC parse → HAR
→ 写 .alls(LayerNorm/Softmax a16_w16) → PTQ + calibration(512 张)
→ Emulator 评估 → compile → HEF → HailoRT 推理
```
**不在 PyTorch 做 QAT**，Hailo 期望干净 fp32 ONNX；需 QAT 用 Hailo QFT。发送端、接收端两块 HEF 分别 bring-up。
 
---
 
## 八、数据集
 
| 角色 | 数据集 | 任务覆盖 | 用途 |
|---|---|---|---|
| 主训练集 1（V2X） | **Cityscapes** | seg(19) + depth(双目) | 核心曲线 bpp vs mIoU/depth；契合 V2X |
| 主训练集 2（室内 MTL） | **NYUDv2** | seg(40) + depth(Kinect) + normal | 三任务全开，MTL 标准 benchmark，验证法线 |
| 主训练集 3（物体级 MTL） | **PASCAL-Context** | seg(59) + normal | 物体级语义 + 法线，证明不依赖室内外特化 |
| 跨域评估（可选） | KITTI | seg + depth(LiDAR) | zero-shot 跨域，扩展章节 |
| Future Work | COCO Panoptic | panoptic | 全景分割，future work |
 
- **Cityscapes**：seg 用 `gtFine/*_labelTrainIds.png`（19 类）；depth 从 disparity 转（上式，clip [1e-3,80]m）；**不启用 normal head**（无原生 GT）。
- **NYUDv2**：官方 .mat（1449 张）+ Silberman split（795/654）；seg 用 Gupta 40 类；depth Kinect（clip [1e-3,10]m）；normal 用 Silberman 工具箱从深度推。**推荐直接用 [MTAN repo](https://github.com/lorenmt/mtan) 预处理数据**（seg+depth+normal 都生成好）。输入 (288,384) 或更大。
- **PASCAL-Context**：VOC2010 + 59 类扩展标注；normal 用 [InvPT/MTI-Net 预处理数据](https://github.com/prismformore/Multi-Task-Transformer)；不启用 depth。输入 (512,512)。
- **KITTI 扩展**：Cityscapes 训好直接 zero-shot KITTI（seg+depth），或少量 fine-tune；比原图 baseline / 压缩方案的跨域掉点。无法线标注。
- **训练策略**：Demo 1 三数据集独立训练独立评估（三 config 三 checkpoint），不做 joint multi-dataset training（列 future work）；Phase 2 核心曲线在 Cityscapes，NYUDv2/PASCAL-Context 作补充实验（ablation/supplementary）。
- ⚠️ 已放弃：NTU RGB+D（行为识别）、nuScenes 单目深度（复杂度过高）。
---
 
## 九、评估指标
 
**分割（主）**：mIoU（K=19/40/59）主线；Pixel Acc 辅助；Per-class IoU 诊断长尾。
 
**深度**：RMSE(m)、AbsRel `mean(|p-g|/g)`、δ<1.25 / 1.25² / 1.25³。只用 valid mask（Cityscapes disparity>0；NYUDv2 depth∈[1e-3,10]）。
 
**法线**：在 angular error space `arccos(clip(<p,g>,-1,1))`（度）；Mean / Median Angular Error；% within 11.25°/22.5°/30°（Eigen 2015, MTAN 标准）；valid = `gt_normal.norm(dim=1)>1e-3`。
 
**语义通信效率（核心贡献指标）**：bpp（训练估计/部署实测）、bpp vs mIoU/depth/normal 曲线、β vs mIoU、带宽对比 vs H.264（2~5 Mbps）。
 
**对比实验设计**：
```
【Cityscapes 主曲线（核心 V2X）】
A: 原图 → 无压缩网络 → mIoU                （上界基线 = teacher）
B: 端到端语义压缩 → decoder → 分割头 → mIoU  （主方案）
C: B + 深度头 → seg mIoU + depth 指标        （粗几何泛化）
【NYUDv2 三任务（MTL 文献位置）】
D: 原图三任务 baseline                       （MTL 上界）
E: 压缩三任务，bottleneck 供三 head          （证明对法线也 work）
【PASCAL-Context】
F: seg + normal 两任务，重复主方案           （领域多样性）
【KITTI 跨域（可选）】
G: Cityscapes 训的压缩 model 直接 eval KITTI
 
结论：B 大幅降带宽下 mIoU 接近 A（核心）；C/E/F 证明同一压缩表示对粗/细几何都有效（普适性）；
E 把方法放到 MTAN/MTI-Net/MultiMAE 同 benchmark，比较有说服力。
```
 
---
 
## 十、paper_reading_list 章节 ↔ 算法设计映射
 
> 详细论文列表见同目录 `paper_reading_list.md`。
 
| 章节 | 对应算法部分 | 必读 ⭐ |
|---|---|---|
| 一、理论基础（VIB/IB/Ballé） | L_rate 设计 + bpp 率失真框架 + bottleneck 理论 | Alemi VIB, Ballé Hyperprior |
| 2.1 综述定位 | 引言 + related work 坐标 | Gündüz JSAC 2023 |
| 2.2.1 LIC | 发送端架构整体 | Ballé 2017/2018 |
| 2.2.2 ICM | **方法论核心 prior art** | Le ICM 2021, Dubois 2021, Choi & Han 2022 |
| 2.2.3 Feature Compression / Split Inference | **主线训练配方** + 系统拓扑 + 评估范式 | **Matsubara Entropic Student / SC2 / Ladon** |
| 2.3 IB 非 JSCC 应用 | VIB 非 JSCC 理论补强 | Achille 2018, Federici 2020 |
| 2.4 JSCC 阵营 | "为什么不做 JSCC"立场段（仅对比） | Bourtsoulatze DeepJSCC |
| 三、语义分割 | backbone 选型 + 分割头 + 分割损失 | SegFormer, DDRNet, **RTFormer** |
| 四、单目深度 | 深度头 + L_depth + 深度指标 | GLPN, AdaBins, Eigen 2014, FastDepth |
| 五、多任务学习 | 共享 encoder + 分叉 decoder + 损失平衡 | PAD-Net, MTI-Net, MultiMAE, Kendall |
| 六、协同感知 V2X | 应用背景（引言引用） | V2X-ViT, Where2comm |
| 七、Hailo-8 部署 | Phase 3 工程 | Hailo DFC 文档 + Model Zoo 配置 |
 
**反向速查（实现某模块读哪节）**：
- 写 encoder（RTFormer/DDRNet/MiT-B0）→ 第三节
- 写共享 decoder + 分叉 heads → 第五节 5.1 + 第四节 4.3
- 实现 bottleneck（熵模型 + rate）→ 第一节 + 2.2.1
- **实现蒸馏配方（feature-MSE + 两阶段挂头）→ 2.2.3（Entropic Student / Ladon）**
- 设计 L_depth → 第四节；平衡多任务 → 第五节 5.3（Kendall/CAGrad/GradNorm）
- bottleneck → 信道接口（量化 + bitstream）→ 2.2.1 + 2.2.3
- 写 related work → 2.1 + 2.2.x + 2.4
- 评估 mIoU/深度/bpp 曲线 → 2.2.3（SC2 范式）+ 第四节
- Hailo 部署 → 第七节
**阅读顺序**：Week1 理论根基（VIB+Ballé+Gündüz）→ Week2 骨架（SegFormer+DDRNet+RTFormer+Eigen+AdaBins+GLPN）→ Week3 多任务（PAD-Net+MTI-Net+MultiMAE+Kendall）→ Week4 信源编码主战场（Le+Dubois+Choi&Han+**Matsubara SC2/Ladon**）→ Week5 对比叙述（DeepJSCC+Shao 借框架不借范式）→ Week6+ 按需。
 
---
 
## 十一、给 Claude 的行为提示
 
- Sidney 是 MSc 学生，有 3D 检测背景（本科 CenterPoint TensorRT 部署），熟 Backbone-Neck-Head，可用作类比；几何感知经验对深度估计适用（点云 BEV regression ↔ 深度 dense regression）。
- Sidney 有写小说习惯，喜欢把技术和应用场景结合，讨论可用场景化语言；保持对话风格，不要过度列表。
- **系统核心叙事**：信道无关的语义信源编解码器，单一压缩表示同时支持分割（语义）+ 深度（粗几何）+ 法线（细几何）。
- **方法论路线**：Sidney 属 **Learned Source Coding for Machines**（Le ICM, Dubois, Choi & Han, Matsubara），**不属于** JSCC（Bourtsoulatze, Shao, Xie）。引 prior art 时不要把 Shao/Xie 当正面方法参考，只在 related work 立场段引用。**这不是 JSCC**，不联合训信道、不注入噪声、训练时 pure semantics。
- **训练主线 = feature 蒸馏式监督压缩**（对齐冻结 teacher 的 feature-MSE 当 distortion + 学熵模型 rate，两阶段挂头）。蒸馏照常出 bitstream。任务驱动 ICM = 对照臂，纯 LIC = baseline，纯 VIB = 最简 ablation。**两种蒸馏别混**：角色 B（训练 distortion，主线）vs 角色 A（softmax→线性部署桥接，fallback）。
- **两端都是 Pi 5 + Hailo-8（26 TOPS），对称**。没有更强接收端可堆算力 → decoder 的 transformer 增强也必须 DFC 可编译、与 heads 同在 26 TOPS 内。
- **bottleneck 插在 RTFormer stage3 之后**（conv→attention 边界）：发送端纯卷积、attention 增强放接收端 decoder；bottleneck = 压缩点 + 算力切分点。可压性看任务相关熵密度，不是 channel 数。
- **双 backbone 平行**：attention 支候选池——P1 RTFormer 风格（线性/external 注意力可精确折叠，首选蓝本）、P2 TopFormer、参考 EfficientFormerV2、MiT-B0 作下界基线；CNN 支 DDRNet-23-slim（PIDNet-S 备选）。正交对照证明 bottleneck 普适性。选型判据 = 谁能把最多部署简化锁进零损失精确档。
- **任务-数据集开关矩阵**：Cityscapes seg+depth；NYUDv2 seg+depth+normal；PASCAL-Context seg+normal。代码按 dataset 条件化前向，别假设所有 head 都开。法线 GT：Cityscapes 不启用；NYUDv2 用 MTAN repo 预处理；PASCAL-Context 用 InvPT/MTI-Net 数据。
- 深度/法线头比分割头稍重（都需 depthwise conv 保边界）。带宽对比基准是 H.264（2~5 Mbps）。应用动机是 V2X 协同感知（不提机场安防）。硬件是 Hailo-8 AI HAT（不提 IMX500、8MB、.rpk）。
- **Bottleneck 是论文核心创新点**，创新叙事不要放在 backbone 选型或 decoder 设计上。
- **Hailo Model Zoo 官方支持**：要插 bottleneck 重训，预编译 HEF 用不上；价值仅在证明算子能编译 + .alls 参考，不是 backbone 选型决定性优势。
- **已放弃/勿再提**：目标检测 + CenterPoint 检测头 + AP 指标（任务历史：分割+检测 → 分割+深度 → 分割+深度+法线）；NTU + 行为识别；信道噪声注入训练；nuScenes 单目深度。
### 待与导师对齐的开放问题（更新 2026-06-01）
 
**已定、仅需告知导师**：
- 第二任务改单目深度（dense+dense 共享 decoder 更合理，语义+几何契合 V2X）
- 三任务三数据集扩展（+NYUDv2 +PASCAL-Context +法线；MTL benchmark 可比位置 + 验证细几何）
- 双 backbone 平行（RTFormer + DDRNet 正交对照，非主/备）
- 方法论定位 Learned Source Coding for Machines（非 JSCC；Shao 降为对比叙述）
- **训练主线改 feature 蒸馏式监督压缩**（对齐 Entropic Student / Ladon）
- **bottleneck 定在 stage3 后，两端对称 Hailo-8**
**仍开放、需决策**：
1. stage3 双分支怎么压：压低分支 + 高分旁路，还是先融合再压？
2. bottleneck 具体结构（单一紧凑 A / 层次化 C / 混合）
3. RTFormer 外部记忆在多任务下的容量是否够（经验问题，可作发现写）
4. 是否需 Ballé 严格 hyperprior 估 bpp（还是当前熵模型足够）
5. 部署阶段是否需实际产 bitstream（还是论文层面 bpp 估计即可）
6. 是否做 Hailo-8 实测延迟/功耗报告
7. KITTI 跨域是否论文必要章节（还是时间允许时扩展）