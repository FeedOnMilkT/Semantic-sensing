# semantic-sensing · 正式工程（语义感知通信系统 / ELEC0054）

> 这是从 `demo/backbone-demo` 毕业出来的**正式算法开发与训练工程**。de-risk 阶段的探索保留在
> `demo/`（冻结参照，勿改）；从现在起所有算法开发、训练、消融都在本目录进行。
> 完整项目背景读 `instruction.md`（项目主知识文档，每次新对话先读它恢复认知）。
> 开工前也务必先读容器全局规范：`/workspace/CLAUDE.md`（磁盘/GPU/落盘纪律）。

---

## ⚑ 当前方向（2026-06-30）

- **主用 RTFormer** 作 backbone（论文部署贡献主体）；MiT-B0 仅作打通/下界基线，DDRNet/TopFormer 列后续。
- **数据集主用 ADE20K**（150 类单任务 seg）；Cityscapes 降为跨数据集对照。
- **先只做单任务分割（seg）**；多任务（depth/normal，原 Phase 1b）暂缓——但**框架的多任务可扩展性
  保持不动**（`heads` 是 dict、`bottleneck` 空槽、4-scale 接口都在）。
- **Phase 2 蒸馏的冻结 teacher 暂用单任务 seg teacher**（选项 A）。
- **TopFormer 暂搁置**：作 split-computing 时 split-point 不好选（SSE/SIM 是并联旁路+回灌，缺
  RTFormer stage3 那种 conv→attention 串联分界与单张紧致 cut）。
- 详见 `notes/direction_2026-06-30_rtformer_singletask.md`。

---

## ⚑ 当前工作进度（2026-07-06）

**正在执行的主线 workflow：`notes/phase2_fused_latent_workflow_2026-07-04.md`**（新对话恢复工作先读它）。

- 背景：2026-07-02 三 split 全 val 重评（`notes/phase2_eval_2026-07-02_three_splits.md`）发现
  ① mid 双 latent 独立编码为分支间互信息重复付费（同 β 下 bpp ≈ early 的 3~3.5 倍）；
  ② Stage1 纯 feature-MSE 目标导致高码率端 mIoU 平台期（加 bit 不涨分）。
- 对策（详见 workflow 文档）：**方案 A** `rtformer_fused` 融合单 latent bottleneck（L-MSFC 式
  交错融合，两支捏成一条 y，单 hyperprior）攻 rate 侧；**方案 B** Stage1 掺任务损失
  （`--task-weight`，ICM 式失真项）攻 quality 侧。A/B 正交，先各自单变量验证再叠加。
- 进度（2026-07-06）：workflow §6 全部完成，**步骤 6 决策落定：主线 = 纯 fused，组合臂
  （fused+task-weight）负结果关闭、保留为消融**。fused β sweep {0.003, 0.01, 0.03} 三点
  （0.4256/0.3533、0.2852/0.3534、0.1434/0.3503）全在旧 15 点 pareto 前沿之上，支配整条
  distill mid 曲线；对 JPEG 重叠质量区 BD-rate **−69.2%**（`tools/bd_rate.py`，本轮新增）。
  组合点 tw0.2/tw0.5（β=0.01）被纯 fused β=0.03 双指标支配且负效应随 w 单调——方案 B 适用
  条件修订为"仅对 rate 侧有结构性冗余的 bottleneck 有效"（distill 上 +0.70 点仍成立）。
  两轮评测统一在 `notes/phase2_eval_2026-07-05_fused_taskweight.md`（第二轮章节含支配关系
  判定与 BD 数字）；曲线 `_orch/report_s2_fullval_v3.{csv,png}`（v2 保留）。
- 下一步（候选，均待定未开跑）：fused β=0.001 探高码率端、β=0.003 修复后代码重跑对齐、
  Cityscapes 跨数据集、Hailo `act: convbn` 导出，见 07-05 笔记末尾"后续候选"。

---

## 硬性约束（务必遵守，与 demo 一脉相承）

1. **不要注册表 / 装饰器**。每类组件用**模块级 dict + `build_X(cfg)`** 工厂：加一个新模块 =
   一行 import + 一行 dict 条目。要可 grep、无魔法（不要 mmcv/mmengine 那套）。
2. **YAML 拼装网络**。OmegaConf 读 YAML，`_base_` + override 合并（按文件自身目录解析）。
   一个顶层 `build_model(cfg)` / `MultiTaskNet(cfg)` 完全按 cfg 装配。
3. **纯 PyTorch**。手写训练 loop，不用 Lightning、不把 MMSeg 当主框架。
4. **bottleneck 槽（统一契约 `forward(f) -> (f_hat, info)`）**。`neck` 与 `decoder` 之间：
   `identity`(no-op，info=None，seg 默认) / `compression`(Phase 2 通信 bottleneck——学习式
   g_a/g_s + CompressAI `EntropyBottleneck`，已实现于 `src/models/bottlenecks/compression.py`)。
   CompressAI **惰性导入**（仅建 compression 时才 import），seg 训练零依赖。
5. **力求简洁**，达目标即停，不过度工程、不追 SOTA。

## 无注册表的工厂模式

```python
# src/models/backbones/__init__.py
_BACKBONES = {"mit_b0": MiTB0, "rtformer": RTFormer, "topformer": TopFormer}
def build_backbone(cfg): ...   # pretrained 在此统一处理
```
neck / bottleneck / decoder / head / dataset / loss 全部同此模式，各自 `build_*`。

## 顶层装配

```python
class MultiTaskNet(nn.Module):
    def __init__(self, cfg):
        self.backbone   = build_backbone(cfg.backbone)
        self.neck       = build_neck(cfg.neck)
        self.bottleneck = build_bottleneck(cfg.bottleneck)   # 当前 identity；Phase 2 在此压缩
        self.decoder    = build_decoder(cfg.decoder)
        self.heads      = nn.ModuleDict({t: build_head(h) for t, h in cfg.heads.items()})
    def forward(self, x, tasks=None):
        f = self.bottleneck(self.neck(self.backbone(x)))     # backbone -> dict{s4,s8,s16,s32}
        d = self.decoder(f)
        return {t: self.heads[t](d) for t in (tasks or self.heads)}
```

## 固定接口（保证可互换）

- **backbone.forward(img) → dict**，键固定 `{"s4","s8","s16","s32"}`（按 stride）。任何 backbone 都返回这套键。
- **head** 按任务名索引；trainer 只对 dataset 当前激活的任务前向并计 loss。
- **dataset.__getitem__** 返回 image + `targets` dict + 该样本激活的任务列表。

---

## 目录布局

```
semantic-sensing/
  CLAUDE.md            本文件
  instruction.md       项目主知识文档（背景/系统分工/Phase 计划/相关工作）
  src/
    models/  backbones/(rtformer*主用 / mit_b0 / topformer搁置)  necks/
             bottlenecks/(identity 空槽 / compression=Phase2 通信bottleneck)  decoders/  heads/  multitask_net.py
    datasets/(ade20k*主用 / cityscapes / synthetic)  losses/  engine/(trainer,evaluator)  utils/
  configs/
    base.yaml  ade20k_rtformer{,_bs32_lr4e4,_paperhead}.yaml  cityscapes_seg.yaml   ← 活跃
    _legacy/   多任务/depth/segformer-teacher 等暂缓实验（自洽，含一份 base.yaml）
  tools/   train.py  eval.py  smoke_test.py  pretrain_imagenet.py  export_onnx.py  export_modules*.py  _common.py
  notes/   研究记录（方向决策、各 ablation 笔记）
  deploy/  hailo/parse_modules.py（Hailo DFC parse 脚本；ONNX 产物不入库，用 export_modules 现导）
```

## 模型 / 落盘（重要：遵守容器磁盘纪律）

- **模型权重一律放大盘**，不进 /workspace（小系统盘）：`/data/sidney/checkpoints/semantic-sensing/`
  - `ade20k/`：当前 3 个 ADE20K seg run（`ade20k_rtformer{,_bs32_lr4e4,_paperhead}.pth` + `.last` + 训练日志）。
  - `pretrained/`：`rtformer_in1k.pth`（RTFormer ImageNet 预训练，val top1 71.8%，训练必需）、`mit_b0_imagenet_hf.bin`。
- ONNX / HAR / logs 都不入 git（见 `.gitignore`）。
- HF/torch/pip 缓存路径已由容器环境变量指向大盘，勿改。

## 当前模型现状（截至迁移时）

| ckpt | mIoU | 备注 |
|---|---|---|
| `ade20k_rtformer` | 37.65 / 37.90(slide) | faithful baseline，generic head |
| `ade20k_rtformer_bs32_lr4e4` | **38.23** / 36.94(slide) | 推 LR/batch，whole-img 最好 |
| `ade20k_rtformer_paperhead` | 36.55 | paper 头 A/B（**反证：头不是差距来源**） |

离 paper 42.1 约 4 点，瓶颈大概率在 ImageNet 预训练（71.8% 偏低），非 seg 端。详见
`notes/direction_2026-06-30_rtformer_singletask.md`。**作 Phase 2 teacher 不必复现 42.1**；要精进
teacher 的唯一高 ROI 动作是补/换预训练，且应在跑最终蒸馏数字之前定死 teacher（蒸馏会冻结它）。

## 快速校验

```bash
cd /workspace/semantic-sensing
python tools/smoke_test.py                      # 框架 import + 前向/训练 sanity（seg，5/5）
python tools/smoke_student.py                   # Phase2 通信 bottleneck student 冒烟（4/4）
# 主线训练（避开故障 GPU2；batch_size 为 per-GPU）：
CUDA_VISIBLE_DEVICES=1,3,4,5 torchrun --nproc_per_node=4 tools/train.py \
  configs/ade20k_rtformer.yaml --out /data/sidney/checkpoints/semantic-sensing/ade20k/ade20k_rtformer.pth
```
