# 方向决策 · 主用 RTFormer + 先只做单任务（2026-06-30）

> 决策记录。同步改动：`backbone-demo/CLAUDE.md` 顶部横幅、`instruction.md` 顶部 changelog +
> 推荐顺序 + Phase 1b 标注、记忆 [[project-direction-rtformer-single-task]] /
> [[topformer-split-point-poor-fit]]。

## 决策

1. **主用 RTFormer** 作 backbone。论文部署贡献主体集中在它身上；MiT-B0 仅作打通/下界基线，
   DDRNet 作 CNN 对照，均列后续。
2. **先只做单任务分割（seg）**。多任务（depth / normal，原 Phase 1b）暂缓为后续阶段。
   **框架的多任务可扩展性保持不动**——`heads` 仍是 dict、`bottleneck` 空槽不变、4-scale
   接口不变；只是当前实现/调参/交付优先级收敛到「RTFormer × 单任务 seg」。
3. **TopFormer 暂时搁置**，留作 RTFormer bring-up 受阻时的备选。
4. **数据集主用 ADE20K**（150 类单任务 seg）。Cityscapes 降为跨数据集对照/参照，配方各自对齐原文。
5. **Phase 2 蒸馏冻结 teacher 暂用单任务 seg teacher（Phase 1a，选项 A）**——与「先只做单任务」一致；
   多任务 teacher 待 Phase 1b 重启后再换。

## 为什么搁置 TopFormer（split-point 不好选）

项目主线是 split computing + 学习式熵瓶颈（CompressAI EntropyBottleneck），切分点要**同时**满足：
(a) **conv→attention 串联分界**——发送端纯卷积、Hailo 友好，attention 整段挪到接收端 decoder；
(b) **干净图割 + 单张紧致张量**——过 bottleneck 的就一张语义特征，对上 demo2 验过的单张量接口。

- **RTFormer** 在 `stage3 之后`恰好对齐两者（stem+s1~s3 纯卷积，stage4/5 才是 attention），
  是整个叙事的甜点。唯一瑕疵是双分支（高 stride8 + 低 stride16），即开放问题 #1。
- **TopFormer** 两条都打破：其 attention（SSE）不是串在主干末端，而是**并联**在池化分支上吃掉
  全部 4 个尺度、再用 SIM **注入回**高分支。
  - 想保持「发送端纯卷积」→ 必须切在 SSE 前 → 要送 4 路多尺度高分辨率特征，且接收端 SIM 还要
    高分辨率 skip → 多张量过线，违反单张量接口，bpp 被高分支主导。
  - 它唯一紧致的语义张量 `sem`(1/32) 是 **branch 不是 cut**：高分支 skip 绕过它，压它不省带宽。
- 真要在 TopFormer 上做，只有两条退路（都不如 RTFormer 干净）：
  (A) 切在 token pyramid 的 t8 后，s4 走旁路/额外传；
  (B) 把 SSE+SIM 整体当接收端，4 尺度走多尺度 bottleneck（推广单张量接口）。
  两者本质都是提前回答 RTFormer 的开放问题 #1。

> 注：TopFormer 难的是**拓扑/割的干净度**，不是算子友好度——其发送端全 depthwise-separable
> 卷积，折叠/量化反而更省心。卡点纯在「单张紧致 cut」这一条。

## 当前主线与对齐配方

- 主交付 config：`configs/ade20k_rtformer.yaml`（faithful baseline，已 37.65 / 37.90 slide）+
  `configs/ade20k_rtformer_paperhead.yaml`（A/B 隔离 seg head，对标 paper 42.1）。
- ADE20K 配方：AdamW / lr 1e-4 / wd 0.05 / 127 ep(~160k iters @ eff batch 16) / ImageNet 预训练 / 同增强。
- Cityscapes 作**跨数据集对照**：`configs/cityscapes_seg.yaml`，对齐 RTFormer 原文 Cityscapes 配方
  （arXiv:2210.07124：lr 4e-4 / wd 0.0125 / poly0.9 / 484 ep / scale 0.5–2.0 + crop 512×1024 + hflip /
  不用 OHEM）。注：此前 Cityscapes 64.1 mIoU 那次用了遗留半残 config、无增强，已作废为不可比基线
  （见 `notes/rtformer_seg_singletask_notes.md`）。
- **Phase 2 衔接**：冻结 teacher 暂用上面跑出的**单任务 seg teacher**（选项 A）；待 Phase 1b 多任务重启后再换多任务 teacher。
