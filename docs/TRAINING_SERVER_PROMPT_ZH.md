# ImageNet 骨干预训练：服务器 Codex 提示词

将下面提示词交给服务器上的 Codex。它只执行 RTFormer ImageNet-1K 分类预训练和导出可被 ADE20K 骨干加载器直接读取的 backbone checkpoint；不训练 ADE20K teacher，不启动 codec Stage 1/2。

---

## 可直接使用的提示词

你在单卡 NVIDIA 服务器上工作。用户提供的 GPU 名称是 RTX PRO 6000，但必须通过命令读取本机实际型号、驱动、compute capability、显存及 PyTorch CUDA 能力；不要根据名称硬编码 GPU 架构。用户估计 dummy-data 实验约 17–20 小时，这只是用户估算，真实数据加载与训练总耗时需实测，不能承诺。

### 目标与边界

只执行本仓库的 RTFormer ImageNet-1K 分类骨干预训练，使用 `configs/imagenet_rtformer_original.yaml` 配方入口，最终产出可由 `src/models/backbones/rtformer.py::load_pretrained()` 严格校验并加载的 trunk 权重。ImageNet 主 recipe 按 RTFormer 原论文补充材料 Table 6：300 epochs、AdamW lr=5e-4、WD=.04、betas=(.9,.999)、cosine min lr=5e-6、5-epoch linear warmup from 5e-7、RandAug(9,.5)、color jitter=.4、mixup alpha=.2、cutmix alpha=1.0、random erasing=.25、drop path=0.0。单卡使用 physical batch 128 × accumulation 8，目标有效 batch=1024。用户指定 BF16 autocast 是工程覆盖，不是原文精度。原文没有列明 label smoothing、crop/interpolation、Mixup/CutMix 切换概率或分类 head；当前实现分别采用无 label smoothing、timm bicubic ImageNet crop、mix/cut 各 0.5、GAP 拼接 stage5 两支后的 1000 类线性 head，这些是显式实现推断。分类只训练 stage1–5 trunk和临时 head；导出不含分类 head，也不含分类阶段未使用的 DAPPM。由于本轮替换为官方 RTFormer-Base topology，旧简化 trunk 的 ImageNet 权重会被严格拒绝，必须重新预训练。论文 §5.4 的最终 ADE20K teacher 是另一个阶段：11-way（10目标+背景）160,000 steps；本次不启动 teacher、codec、Stage 2、部署导出，也不上传数据或权重。不要混用 ADE teacher 的官方150类参考配置或 codec Stage 1 的100 epochs。

数据文件本身不需要是 BF16。HF ImageNet 样本解码为 RGB 图像；当前 loader 经变换得到归一化 FP32 输入张量，分类标签为整数类别 ID。BF16 指 CUDA autocast 下模型运算使用的混合精度，不是需要另行下载的“BF16 数据集”，也不改变 label dtype。

### 1. 保护工作区并检查硬件

从训练仓库根目录执行。先运行 `git status --short --branch`、`git diff --stat`、`git diff --check` 并检查 `git diff`；必须保留既有用户改动。不要执行 `git reset`、`git clean`、覆盖配置或无条件同步。若工作区有改动，先分辨用户改动与当前任务所需变更，只在确认目标文件内容后继续。

记录下列输出到本次 run 目录，但不要记录 token：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
python3 --version
df -h /data
```

`compute_cap` 查询字段若驱动不支持，则通过下面安装后的 `torch.cuda.get_device_capability()` 读取。实际数据盘可能不叫 `/data`；先查看挂载点并选择有足够空间的路径。不要把机器上的训练数据、缓存、日志、权重放入 git 仓库。

### 2. 用 uv 创建隔离环境并选择匹配的 CUDA wheel

使用 Python 3.11 的仓库外/忽略目录环境，并把 Hugging Face 缓存放到有容量的数据盘：

```bash
export RUN_DIR=/data/semantic-sensing-runs/imagenet-rtformer-$(date +%Y%m%d-%H%M%S)
export HF_HOME=/data/huggingface/semantic-sensing
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$RUN_DIR" "$HF_HOME"
uv python install 3.11
uv venv --python 3.11 .venv
```

根据 `nvidia-smi` 驱动和 PyTorch 设备兼容表选择官方 wheel，不要安装旧的通用 `torch>=2.2` 或未经验证的 CUDA 12.1 wheel。以 2026-10-12 官方信息为参考：PyTorch 2.12.0 的官方 CUDA 13.0 wheel 与 torchvision 0.27.0 可通过下面命令安装；PyTorch 2.12 文档对 Blackwell 等新 GPU 建议 CUDA 13.0+，Linux 驱动至少 580.65.06。若实际 GPU/driver 不满足，服务器 Codex 必须根据官方兼容矩阵为**实际 compute capability 和当前 driver**选择受支持的官方 wheel 并复测，不得修改系统 driver、不得选一个不支持该 GPU 的旧版 PyTorch；没有安全可用组合则停止并报告。

```bash
uv pip install --python .venv/bin/python torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

安装后必须探测实际组合，再决定能否开训：

```bash
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "本次任务要求 CUDA；未检测到 CUDA"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, "capability", torch.cuda.get_device_capability(i),
          "memory_GiB", round(p.total_memory / 2**30, 1))
print("torch", torch.__version__, "CUDA runtime", torch.version.cuda,
      "BF16", torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), "当前 GPU/wheel 不支持 BF16；停止并报告"
PY
```

仅在硬件和 driver 允许该组合且上述探测通过后，记录 torch/torchvision/CUDA runtime 版本，再把该 GPU wheel 固定为解析 constraint，避免 timm 的依赖解析悄悄替换 CUDA wheel：

```bash
.venv/bin/python - <<'PY' > "$RUN_DIR/torch-constraints.txt"
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY
uv pip install --python .venv/bin/python -r requirements-imagenet.txt \
  --constraint "$RUN_DIR/torch-constraints.txt"
.venv/bin/python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
uv pip freeze --python .venv/bin/python > "$RUN_DIR/requirements.lock.txt"
.venv/bin/python -m pip --version
```

记录 Python、torch/torchvision、CUDA runtime、driver、GPU 型号/capability、BF16 检测及 `requirements.lock.txt`。这份 freeze 是本次已解析环境快照；重建时应先按记录的官方 CUDA wheel 安装 torch/torchvision，再按快照复原其余包。不得将未经服务器检查的本机/Mac wheel 当作 CUDA 兼容证据。

### 3. 检查 ImageNet 授权、数据、空间

`ILSVRC/imagenet-1k` 是 Hugging Face gated dataset。先在浏览器申请并获得数据集访问许可、接受数据条款，再在服务器通过官方 CLI 登录。禁止把 token 写进脚本、命令历史、仓库、日志或本提示词，也不要打印 token：

```bash
.venv/bin/hf auth login
.venv/bin/hf auth whoami
```

如果账号未获许可、`whoami` 未登录，或脚本 `load_dataset()` 收到 401/403，立即停止并报告需要用户完成授权；不要绕过门禁、尝试非授权镜像或索取/回显 token。保留已有 HF cache 并复用它，不清空缓存。授权成功后检查数据盘剩余空间，显式下载并验证两个 split：

```bash
HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  .venv/bin/python tools/pretrain_imagenet.py --config configs/imagenet_rtformer_original.yaml --device cuda --precision bf16 \
  --prepare-data-only --workers 0 --out "$RUN_DIR/rtformer_in1k.pth"
```

该命令会按 HF dataset builder 下载/准备 train/validation 数据到 HF cache，检查标准 ImageNet-1K 的 train=1,281,167、validation=50,000、两 split 标签元数据各有 1000 类且 class names 完全相同；仅读取 Arrow 元数据，不遍历解码每一张图。若 HF 公开 builder 版本有差异，不能弱化为“非空即可”：先查清 split/config，明确报告实测长度和元数据差异，经用户确认数据仍为完整 ImageNet-1K 后再继续。此命令不建模、不训练、不写 checkpoint。若是 cache 残缺或下载中断，修复 cache/download 后复查；不要开始 dry-run 或 300-epoch 训练。

### 4. 真实数据小批 dry-run

先执行正式入口的一次真实数据小批测试：

```bash
HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  .venv/bin/python tools/pretrain_imagenet.py --config configs/imagenet_rtformer_original.yaml --device cuda --precision bf16 \
  --batch-size 2 --workers 0 --dry-run --out "$RUN_DIR/rtformer_in1k.pth"
```

该模式要从真实 ImageNet 训练集和 validation 集各取一个 batch，执行训练 batch 的 forward、loss、backward、optimizer step，以及 validation batch forward；检查 loss/top-1 为有限值后退出，不写 checkpoint。若它失败，先解决授权、缓存、图像解码、label、CUDA/BF16 或显存问题；不得直接改成 FP32 或悄悄缩小正式 batch 后宣称 BF16 batch-128 已验证。

batch 2 通过后再用正式 batch 做一次真实数据显存/吞吐小测，仍只执行一 train + 一 validation batch，不写checkpoint：

```bash
HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  .venv/bin/python tools/pretrain_imagenet.py --config configs/imagenet_rtformer_original.yaml --device cuda --precision bf16 \
  --batch-size 128 --workers 0 --dry-run --out "$RUN_DIR/rtformer_in1k.pth"
```

两种 dry-run 都成功，才开始正式 300 epochs。正式训练按每 8 个 microbatch 累积；epoch末尾不足8个的尾窗仍会做一次更新，并以该窗实际样本总数归一梯度。batch-128 dry-run 若 OOM，记录显存峰值并诊断；正式运行若 physical batch 128 无法容纳，先定位显存、模型或worker瓶颈，不得静默改写有效 batch。

### 5. 运行正式预训练

服务器单卡正式命令：

```bash
set -o pipefail
HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  .venv/bin/python tools/pretrain_imagenet.py --config configs/imagenet_rtformer_original.yaml --device cuda --precision bf16 \
  --batch-size 128 --accumulation-steps 8 --epochs 300 --workers 10 --eval-every 5 \
  --out "$RUN_DIR/rtformer_in1k.pth" 2>&1 | tee "$RUN_DIR/train.log"
```

脚本使用原文 AdamW/cosine 和增强超参数，外加物理 batch 128 × accumulation 8。尾窗按其样本实际总数归一；scheduler 和 global step 只在 optimizer update 后推进。MixUp/CutMix 生成 soft labels，以 soft-target cross entropy 训练；验证仍用原始整数标签计算 top-1。Mix/Cut 0.5切换、无 label smoothing、timm crop/interpolation 是未由原文指定的实现默认。BF16 是用户指定的运行精度覆盖；输入图像先解码为 RGB，再转 normalized FP32 tensors，不需要BF16格式数据集。若 OOM，先保存现场并诊断显存、其他进程、workers、数据内存和共享内存；调整 microbatch 时必须同步修改累积步数并保持有效 batch 1024，否则记录为偏离原文的实验。

用户预计的 dummy-data 时间约 17–20 小时只用于规划参考；ImageNet 下载、解码、增强、验证与checkpoint开销会改变实际时长。以脚本 ETA 和实测吞吐为准。

### 6. 断点恢复与产物

入口每个 epoch 原子更新完整训练 checkpoint：`<输出名>.last.pth`，包含模型、优化器、精度 scaler、epoch、最佳精度、随机状态和 schedule。中断后只在相同 world size、updates/epoch、batch、累积步数、LR、WD、warmup、precision 配置下执行 `--resume`；checkpoint保存global optimizer step，scheduler从global step继续。示例：

```bash
  .venv/bin/python tools/pretrain_imagenet.py --config configs/imagenet_rtformer_original.yaml \
  --device cuda --precision bf16 --batch-size 128 --accumulation-steps 8 --epochs 300 \
  --workers 10 --eval-every 5 --resume --out "$RUN_DIR/rtformer_in1k.pth"
```

`rtformer_in1k.pth` 仅在验证 top-1 提升时更新，保存 ImageNet 分类阶段完整训练过的 RTFormer stage1–5 trunk tensors（包括两个 stage5 分支；不含临时 GAP+1000类分类 head、不含未参与分类 forward 的 DAPPM）。`.last.pth` 是可恢复的训练态完整 checkpoint，不应作为 ADE backbone 初始化文件。日志、环境快照与 checkpoint 放在 `$RUN_DIR`，不得提交到 git。

### 7. 完成后核对并汇报

确认 checkpoint 存在且可用 `torch.load(path, map_location="cpu", weights_only=True)` 读取 `backbone` tensor dict；用 `RTFormer().load_pretrained(path)` 做严格 trunk coverage/shape 检查，DAPPM/segmentation head 未在分类阶段训练。记录并汇报：实际 GPU/driver/capability/VRAM，torch/vision/CUDA 版本与 BF16 结果，ImageNet splits 行数及许可/cache状态，dry-run结果，300 epochs完成数、耗时、吞吐、最佳 validation top-1、last/best checkpoint 路径、加载 smoke test结果、是否中断/OOM。若只完成一部分，明确说明；不要启动 ADE20K teacher、codec Stage 1/2，不上传或 push 文件。

### 官方参考

- [PyTorch previous versions / 官方 wheel 命令](https://pytorch.org/get-started/previous-versions/)
- [PyTorch 2.12 release：新 GPU 与驱动要求](https://pytorch.org/blog/pytorch-2-12-release-blog/)
- [uv Python 管理](https://docs.astral.sh/uv/guides/install-python/)
- [uv CLI reference](https://docs.astral.sh/uv/reference/cli/)
- [Hugging Face gated datasets](https://huggingface.co/docs/hub/datasets-gated)
- [RTFormer original supplement, Table 6 ImageNet recipe](https://papers.nips.cc/paper_files/paper/2022/file/30e10e671c5e43edb67eb257abb6c3ea-Supplemental-Conference.pdf)
- [timm RandAugment and Mixup/CutMix implementation](https://github.com/huggingface/pytorch-image-models/tree/main/timm/data)
