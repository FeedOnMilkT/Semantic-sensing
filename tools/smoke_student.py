#!/usr/bin/env python
"""Student smoke test — exercises the Phase-2 communication bottleneck end to end.

The "student" = a frozen seg teacher with the learned communication bottleneck
spliced into its split point (the neck->decoder slot, the RTFormer-stage3 analogue):

    x -> [frozen backbone+neck] -> f -> g_a -> EntropyBottleneck -> g_s -> f_hat
      -> [frozen decoder+heads] -> task preds

Checks (synthetic data, no dataset, no pretrained weights):
  #1 train: rate-distortion loss (feature_mse + beta*bpp) backprops; the codec and aux
     optimizers both step; distortion decreases over N steps; bpp is finite.
  #2 full split-inference forward: the frozen teacher tail consumes f_hat -> seg preds.
  #3 framework integration: build_model(bottleneck=compression) forwards and exposes
     model.bottleneck_info (likelihoods).
  #4 deployment: real bitstream roundtrip (compress -> bytes -> file -> decompress) is
     exact under rounding, and estimated bpp ~= actual file bpp (< 20%).

Exits non-zero on any failure.
"""
import os
import sys
import tempfile
import traceback

from _common import pick_device
from omegaconf import OmegaConf
import torch

from src import _compat  # noqa: F401  applies the CompressAI import shim first
from src.utils import load_config
from src.models import build_model
from src.models.bottlenecks.compression import CommBottleneck
from src.losses.compression import rate_distortion_loss

SIZE = (64, 128)               # tiny image for speed
LATENT, HIDDEN, NUM_DOWN = 48, 96, 1
DEVICE = pick_device()
RESULTS = []
STATE = {}


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        RESULTS.append(True)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        RESULTS.append(False)


def _seg_cfg(bottleneck="identity"):
    cfg = load_config("configs/cityscapes_seg.yaml")
    cfg.model.backbone.args.pretrained = None        # hermetic: no disk dependency
    if bottleneck == "compression":
        cfg.model.bottleneck = OmegaConf.create(
            {"name": "compression",
             "args": {"latent_channels": LATENT, "hidden": HIDDEN, "num_down": NUM_DOWN}})
    return cfg


def setup():
    torch.manual_seed(0)
    teacher = build_model(_seg_cfg("identity").model).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    cin = teacher.neck.out_channels
    bottleneck = CommBottleneck(in_channels=cin, latent_channels=LATENT,
                                hidden=HIDDEN, num_down=NUM_DOWN).to(DEVICE)
    STATE.update(teacher=teacher, bottleneck=bottleneck, cin=cin)


def c1_train_rate_distortion():
    teacher, bottleneck = STATE["teacher"], STATE["bottleneck"]
    B = 2
    num_pixels = B * SIZE[0] * SIZE[1]
    opt = torch.optim.Adam(bottleneck.codec_parameters(), lr=1e-3)
    aux_opt = torch.optim.Adam(bottleneck.aux_parameters(), lr=1e-3)
    bottleneck.train()
    dists = []
    for step in range(80):
        x = torch.randn(B, 3, *SIZE, device=DEVICE)
        with torch.no_grad():
            f = teacher.neck(teacher.backbone(x))           # frozen split feature
        f_hat, info = bottleneck(f)
        total, parts = rate_distortion_loss(f_hat, f, info, beta=0.01, num_pixels=num_pixels)
        opt.zero_grad(); total.backward(); opt.step()
        aux = bottleneck.aux_loss()
        aux_opt.zero_grad(); aux.backward(); aux_opt.step()
        dists.append(parts["distortion"].item())
        if step % 20 == 0:
            print(f"      step {step:02d}  dist={parts['distortion']:.4f} "
                  f"bpp={parts['bpp']:.4f} aux={aux.item():.1f}")
    assert torch.isfinite(parts["bpp"]), "bpp not finite"
    assert dists[-1] < dists[0] * 0.9, f"distortion did not drop: {dists[0]:.4f} -> {dists[-1]:.4f}"


def c2_full_split_forward():
    teacher, bottleneck = STATE["teacher"], STATE["bottleneck"]
    B = 2
    x = torch.randn(B, 3, *SIZE, device=DEVICE)
    bottleneck.eval()
    with torch.no_grad():
        f = teacher.neck(teacher.backbone(x))
        f_hat, _ = bottleneck(f)
        d = teacher.decoder(f_hat)
        preds = {t: teacher.heads[t](d) for t in teacher.heads}
    seg = preds["seg"]
    assert f_hat.shape == f.shape, f"f_hat {tuple(f_hat.shape)} != f {tuple(f.shape)}"
    assert seg.shape[0] == B and seg.dim() == 4, f"bad seg preds {tuple(seg.shape)}"


def c3_framework_integration():
    model = build_model(_seg_cfg("compression").model).to(DEVICE)
    x = torch.randn(2, 3, *SIZE, device=DEVICE)
    preds = model(x)
    assert "seg" in preds, "seg head missing from preds"
    assert model.bottleneck_info is not None and "likelihoods" in model.bottleneck_info, \
        "compression bottleneck did not expose likelihoods via model.bottleneck_info"


def c4_bitstream_roundtrip():
    # the rANS range coder runs on CPU -> move the codec there for compress/decompress
    bottleneck = STATE["bottleneck"].cpu().eval()
    bottleneck.update(force=True)                       # build CDF tables
    cin = STATE["cin"]
    B = 4
    num_pixels = B * SIZE[0] * SIZE[1]
    torch.manual_seed(1)
    f = torch.randn(B, cin, SIZE[0] // 4, SIZE[1] // 4)   # feature at stride-4 of the image
    with torch.no_grad():
        f_hat_fwd, info = bottleneck(f)
        est_bpp = CommBottleneck.estimated_bpp(info, num_pixels).item()
        payload = bottleneck.compress(f)
        with tempfile.TemporaryDirectory() as d:
            total_bytes = 0
            for i, s in enumerate(payload["strings"]):
                p = os.path.join(d, f"{i}.bin")
                with open(p, "wb") as fh:
                    fh.write(s)
                total_bytes += os.path.getsize(p)
            read = []
            for i in range(len(payload["strings"])):
                with open(os.path.join(d, f"{i}.bin"), "rb") as fh:
                    read.append(fh.read())
        payload["strings"] = read
        f_hat_dec = bottleneck.decompress(payload)
    actual_bpp = total_bytes * 8 / num_pixels
    max_diff = (f_hat_fwd - f_hat_dec).abs().max().item()
    rel = abs(est_bpp - actual_bpp) / actual_bpp
    print(f"      est_bpp={est_bpp:.5f}  actual_bpp={actual_bpp:.5f} ({total_bytes}B)  "
          f"rel_err={rel * 100:.1f}%  roundtrip_max|diff|={max_diff:.2e}")
    assert max_diff < 1e-4, f"roundtrip not exact: {max_diff:.2e}"
    assert rel < 0.20, f"estimated bpp far from actual: {rel * 100:.1f}%"


if __name__ == "__main__":
    print(f"[smoke_student] device={DEVICE} size={SIZE} latent={LATENT} num_down={NUM_DOWN}")
    setup()
    check("#1 train: rate-distortion drops, codec+aux step", c1_train_rate_distortion)
    check("#2 full split-inference forward (frozen teacher tail)", c2_full_split_forward)
    check("#3 framework integration: build_model(compression) + bottleneck_info", c3_framework_integration)
    check("#4 bitstream roundtrip exact + estimated bpp ~= actual", c4_bitstream_roundtrip)
    n_ok = sum(RESULTS)
    print(f"\n[smoke_student] {n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
