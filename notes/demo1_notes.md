# Demo 1 笔记 — 多任务感知 baseline 框架（无压缩蓝本）

> 状态：**框架完成 + 冒烟 5/5 + 全量多任务训练跑通**。
> 更新：2026-06-03 ｜ 环境：`sidney-dev` 容器（NGC pytorch 25.02，torch 2.4.0a0，py3.10），GPU 1（A100-80G）

## 0. 全量训练结果（cityscapes_multitask，2026-06-03）

MiT-B0（ImageNet 预训练）+ mlp_fusion neck + 3 层 conv decoder + seg/depth 双头，
uncertainty 加权，200 epochs（74,200 步，~4.7h，单 A100），512×1024，**无数据增强**。
500 张 val 评估：

| 指标 | 结果 | bar | 判定 |
|---|---|---|---|
| seg mIoU | **60.98%** | 单任务 ≥68 | ⚠️ 差 7 点 |
| seg pixel acc | 93.55% | — | — |
| depth RMSE | **6.44 m** | < 8 m | ✅ 达标 |
| depth AbsRel | 0.115 | — | — |
| depth δ<1.25 / δ² / δ³ | 0.867 / 0.964 / 0.988 | — | ✅ |

checkpoint：`/data/sidney/checkpoints/demo1/cs_multitask.pth`（16MB）｜日志：`logs/cs_multitask.log`

**seg 没到 68 的诚实归因（按影响排序）**：
1. **完全无数据增强** —— dataset 只 resize，没有 Cityscapes 标准的 random scale(0.5–2.0)+random crop+hflip。这是单项最大缺口，通常值 +3~6 mIoU。
2. **decoder/neck/head 全从零训** —— 只有 backbone 是 ImageNet 预训练；SegFormer 原版 76 还靠更强 MLP head + 更长 schedule + 大 crop。
3. **缺单任务对照** —— Phase 1b 的 bar 本是"相对单任务跌 <2 点"，但还没跑 `cityscapes_seg` 单任务基线，无法判断是多任务拖累还是 setup 本身上限。
depth 已达标，说明 pipeline/损失/指标都正确，差距在分割侧的训练配方（增强）而非框架 bug。

**补 bar 的下一步（按性价比）**：① 加标准 seg 增强重训（最值）；② 跑单任务 seg 基线做对照；③ 需要再上更大 crop / 更长 schedule。

---

## 1. 这是什么

按 `CLAUDE.md` 蓝本搭的**简洁、模块化、YAML 驱动**的多任务稠密预测训练框架，对应项目 Phase 1a（单任务分割）+ Phase 1b（seg+depth 无压缩 baseline，即后续蒸馏的冻结 teacher）。

核心设计原则（全部落实）：
- **无注册表/装饰器**：每类组件 = 模块级 dict + `build_X(cfg)` 工厂，加模块 = 一行 import + 一行 dict 条目，可 grep、无魔法。
- **YAML 拼装**：OmegaConf，`_base_` 继承 + override 合并，`build_model(cfg)` 全程按 cfg 装配。
- **纯 PyTorch**：手写训练 loop，无 Lightning/MMSeg。
- **bottleneck 空槽**：neck 与 decoder 之间留 `identity` no-op，Phase 2 直接换 EntropyBottleneck，框架不重构。

## 2. 目录结构

```
backbone-demo/
  configs/   base.yaml  cityscapes_seg.yaml  cityscapes_multitask.yaml  synthetic_smoke.yaml
  src/
    models/  backbones/{mit_b0,rtformer,topformer}  necks/{fpn_lite,mlp_fusion}
             bottlenecks/identity  decoders/conv_decoder  heads/{seg,depth,normal}  multitask_net.py
    datasets/  synthetic  cityscapes        losses/  task_losses(seg/depth/normal) + weighting
    engine/    trainer  evaluator           utils/  config(OmegaConf)  metrics  logging
  tools/     train.py  eval.py  export_onnx.py  smoke_test.py
  notes/     demo1_notes.md
```

## 3. 固定接口（保证可互换）

- `backbone.forward(img) -> {"s4","s8","s16","s32"}`（按 stride），任何 backbone 都返回这套键。
  每个 backbone 还暴露 `.out_channels` 字典；`MultiTaskNet` 据此把通道数注入 neck→decoder→heads，**换骨干零改动**。
- `neck(feats) -> 单个 stride-4 融合特征`；`bottleneck(f) -> f`（demo1 no-op）；`decoder(f) -> 共享特征`。
- `heads[task](d) -> 任务输出`（head 内部 bilinear 上采样到输入分辨率，静态 scale_factor 利于 ONNX）。
- `dataset.__getitem__ -> {"image", "targets":{task:tensor}, "tasks":[...]}`；某任务无 GT 就不进 batch，不计 loss。

## 4. 已实现组件

| 类别 | 实现 | 说明 |
|---|---|---|
| backbones | `mit_b0` / `rtformer` / `topformer` | mit_b0 忠实 SegFormer 编码器（下界基线）；rtformer 双分辨率+深层 GFA external attention（首选蓝本，可折叠记忆）；topformer 池化 token 小注意力（备选）。三者均 4-scale 接口。 |
| necks | `mlp_fusion`（默认）/ `fpn_lite` | 轻量多尺度融合，输出单 stride-4 特征。 |
| bottleneck | `identity` | **唯一空槽**，Phase 2 接口同此。 |
| decoder | `conv_decoder` | 共享卷积主体 3-4 层，**预留 transformer 增强 hook**（Phase 2 在此插 RTFormer 块，现为 no-op；`transformer_enhance=true` 会显式 NotImplementedError）。 |
| heads | `seg` / `depth` / `normal` | seg 极简 1×1；depth/normal 带 depthwise 3×3 保边界；depth `sigmoid×max_depth`，normal L2-normalize。 |
| datasets | `cityscapes` / `synthetic` | cityscapes seg 用 labelTrainIds、depth 从 disparity（`0.222*2262/(disp)` clip[1e-3,80]）；synthetic 供冒烟。 |
| losses | seg(CE+Dice+Lovász) / depth(SiLog+0.1·grad) / normal(1-cos·valid) | 多任务加权：`fixed` / `uncertainty`(Kendall) / `dwa`，YAML 配。 |
| engine | trainer（手写 loop，poly/cosine LR，AMP，grad clip）/ evaluator（mIoU/depth/normal 条件化） | uncertainty 的可学习 log_var 已纳入优化器。 |
| tools | train / eval / export_onnx / smoke_test | export 静态 shape opset13，dict→tuple 包装，过 onnx.checker。 |

## 5. 验收标准核对（`CLAUDE.md` §验收）

| # | 标准 | 结果 |
|---|---|---|
| 1 | 改 `backbone.name` 一行，三骨干都能训 | ✅ smoke `[#1]` mit_b0/rtformer/topformer 各训 3 步 OK |
| 2 | 增删 `heads:` 一个任务，trainer 自动只跑激活 head | ✅ smoke `[#2]` 删 depth 后仅 seg；含双头模型只请求 seg 时只计 seg loss |
| 3 | Phase 1a 训分割；Phase 1b 加 depth，条件 loss 正常 | ✅ smoke `[#3]` seg+depth 双 loss 正常下降；conditional loss 验证通过 |
| 4 | `export_onnx.py` 静态 shape 导出无报错 | ✅ smoke `[#4]` + CLI 均过 onnx.checker |
| 5 | 结束后生成 md 笔记 | ✅ 本文件 |

**冒烟测试命令**（无需任何数据集）：
```bash
CUDA_VISIBLE_DEVICES=1 python tools/smoke_test.py        # -> 5/5 passed
# CLI 三件套（合成数据）
CUDA_VISIBLE_DEVICES=1 python tools/train.py  configs/synthetic_smoke.yaml --max-steps 5 --out checkpoints/smoke.pth
CUDA_VISIBLE_DEVICES=1 python tools/eval.py   configs/synthetic_smoke.yaml --ckpt checkpoints/smoke.pth
CUDA_VISIBLE_DEVICES=1 python tools/export_onnx.py configs/synthetic_smoke.yaml --shape 1 3 64 128 --out /tmp/m.onnx
```

## 6. 真训怎么跑（数据到位后）

```bash
# Phase 1a 单任务分割（目标 mIoU >= 68）
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/cityscapes_seg.yaml
# 换骨干：只改一行（或 CLI 覆盖）
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/cityscapes_seg.yaml --set model.backbone.name=rtformer
# Phase 1b 多任务 seg+depth（teacher；目标 seg 跌<2、depth RMSE<8m）
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/cityscapes_multitask.yaml
```

## 7. 待办 / 已知限制（诚实记录）

- **Cityscapes 数据未到位**：交付 config 已就绪，但需要登录账号下载（见下）。在此之前只能合成数据冒烟，**无法验证 mIoU≥68 这类精度 bar**——那需要真数据 + 数小时训练。
- **无 ImageNet 预训练权重**（离线环境）：mit_b0 等目前从零初始化。要冲精度 bar 应载入官方 MiT-B0 预训练。
- **backbone 为忠于结构的精简实现**：rtformer/topformer 抓住了各自的关键结构（双分辨率+external attention / 池化 token attention），但非论文逐行复刻；作为蓝本与正交对照足够，冲 SOTA 前需对齐论文细节。
- **NYUDv2 已下载**：`/data/sidney/datasets/nyudv2/nyu_depth_v2_labeled.mat`（2.97GB，v7.3 HDF5，校验通过）。三任务扩展时需写 `nyudv2.py` adapter（seg40/depth/normal）+ 装 h5py。
- normal head/loss/metric 已实现但 Cityscapes 不启用（无原生 GT），等 NYUDv2/PASCAL-Context。

## 8. 下一步（Phase 2 接口已备好）

- bottleneck 槽：把 `identity` 换成 CompressAI `EntropyBottleneck`，forward 额外返回 rate；trainer 加 `β·L_rate` 项。
- decoder 的 `transformer_hook`：插入可部署 RTFormer/GFA 块（接收端 attention 增强）。
- 蒸馏：本框架 Phase 1b 产出即 Phase 2 的冻结 teacher，feature-MSE 对齐。
