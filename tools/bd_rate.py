#!/usr/bin/env python
"""Bjøntegaard-delta metrics between two series of an eval_rd_curve.py report CSV.

Treats mIoU as the quality metric. Reports both directions:
  - BD-rate : avg % rate difference at equal quality, integrated over the two
              curves' OVERLAPPING quality interval (Bjøntegaard in the log-rate
              domain; monotone PCHIP interpolation, the modern reference variant —
              plain cubic polyfit overshoots badly on our near-flat mIoU curves).
              Undefined (n/a) when the quality ranges do not overlap (e.g. the test
              curve is strictly better everywhere) — stated instead of extrapolating.
  - BD-mIoU : avg quality difference at equal rate, integrated over the overlapping
              log-rate interval (same construction with the axes swapped).

BD-rate inverts the curve to rate(quality), which requires monotonicity; dominated
points (higher rate, no quality gain) are pareto-filtered for that fit only.
BD-mIoU uses all points (quality(rate) needs no inversion).

  python tools/bd_rate.py --csv .../_orch/report_s2_fullval_v3.csv \
      --anchor mid --test mid-fused
"""
import argparse

import numpy as np


def load_series(csv_path, label):
    """Rows of `label` as (rate, quality) arrays sorted by rate. Works for both
    student series (early/mid/mid-fused/...) and the jpeg rows."""
    pts = []
    for ln in open(csv_path).read().splitlines()[1:]:
        p = ln.split(",")
        if p[0] == label and p[3] != "-":
            pts.append((float(p[3]), float(p[4])))    # bpp, mIoU
    pts.sort()
    return np.array([q[0] for q in pts]), np.array([q[1] for q in pts])


def _pareto(r, q):
    """Keep only points where quality strictly increases with rate: a point that adds
    rate but no quality over the previous kept point is dominated and dropped."""
    keep_r, keep_q = [], []
    for ri, qi in zip(r, q):                          # r is sorted ascending
        if not keep_q or qi > keep_q[-1]:
            keep_r.append(ri); keep_q.append(qi)
    return np.array(keep_r), np.array(keep_q)


def _bd(x1, y1, x2, y2):
    """Avg (curve2 - curve1) over the overlapping x interval; None if no overlap.
    Monotone PCHIP interpolation (needs strictly increasing x)."""
    from scipy.interpolate import PchipInterpolator
    lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    if hi <= lo:
        return None
    a1 = PchipInterpolator(x1, y1).antiderivative()
    a2 = PchipInterpolator(x2, y2).antiderivative()
    return ((a2(hi) - a2(lo)) - (a1(hi) - a1(lo))) / (hi - lo)


def bd_rate(r_anchor, q_anchor, r_test, q_test):
    ra, qa = _pareto(r_anchor, q_anchor)
    rt, qt = _pareto(r_test, q_test)
    if len(ra) < 2 or len(rt) < 2:
        return None                                   # curve degenerates after pareto filter
    d = _bd(qa, np.log(ra), qt, np.log(rt))           # x=quality, y=log rate
    return None if d is None else (np.exp(d) - 1.0) * 100.0


def bd_quality(r_anchor, q_anchor, r_test, q_test):
    return _bd(np.log(r_anchor), q_anchor, np.log(r_test), q_test)   # x=log rate, y=quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="report CSV from tools/eval_rd_curve.py")
    ap.add_argument("--anchor", required=True, help="reference series label (e.g. mid, jpeg)")
    ap.add_argument("--test", required=True, help="series measured against the anchor")
    args = ap.parse_args()

    ra, qa = load_series(args.csv, args.anchor)
    rt, qt = load_series(args.csv, args.test)
    for name, r, q in [(args.anchor, ra, qa), (args.test, rt, qt)]:
        if len(r) < 2:
            raise SystemExit(f"series '{name}': need >=2 points, got {len(r)}")
        print(f"{name:12s} n={len(r)}  bpp [{r.min():.4f}, {r.max():.4f}]  "
              f"mIoU [{q.min():.4f}, {q.max():.4f}]")

    br = bd_rate(ra, qa, rt, qt)
    bq = bd_quality(ra, qa, rt, qt)
    print(f"BD-rate ({args.test} vs {args.anchor}): "
          + (f"{br:+.1f}%" if br is not None else "n/a (quality ranges do not overlap)"))
    print(f"BD-mIoU ({args.test} vs {args.anchor}): "
          + (f"{bq:+.4f}" if bq is not None else "n/a (rate ranges do not overlap)"))


if __name__ == "__main__":
    main()
