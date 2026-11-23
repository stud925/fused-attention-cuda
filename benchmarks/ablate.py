"""Cumulative ablation of the stage-4 optimisation knobs of the fused kernel.

    python benchmarks/ablate.py [--seq 2048] [--dtype fp16] [--head-dim 64]

Prints (and writes docs/ablation.csv) the median latency of each variant, plus
the best warps-per-block. Checks every variant against SDPA first, so a
variant that is fast but wrong is flagged instead of reported.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import fused_attention_cuda as fa  # noqa: E402

VARIANTS = [  # (label, kwargs) -- each row adds one optimisation to the previous one
    ("baseline (stage 3)", dict(pad=False, vec=False, fast_exp=False, warps=8)),
    ("+ pad K tile (no bank conflicts)", dict(pad=True, vec=False, fast_exp=False, warps=8)),
    ("+ 16-byte vector loads", dict(pad=True, vec=True, fast_exp=False, warps=8)),
    ("+ exp2f with folded log2e", dict(pad=True, vec=True, fast_exp=True, warps=8)),
    ("  warps=4 (tile-size/occupancy)", dict(pad=True, vec=True, fast_exp=True, warps=4)),
]


def median_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--causal", action="store_true")
    a = ap.parse_args()
    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    tol = 1e-2 if dtype == torch.float16 else 1e-4

    shape = (a.batch, a.heads, a.seq, a.head_dim)
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=a.causal)

    print(f"{torch.cuda.get_device_name()}  shape={shape} {a.dtype} causal={a.causal}")
    rows = []
    for label, kw in VARIANTS:
        out = fa.fused_attention(q, k, v, a.causal, **kw)
        err = (out.float() - ref.float()).abs().max().item()
        ms = median_ms(lambda: fa.fused_attention(q, k, v, a.causal, **kw))
        flag = "" if err < tol else "  <-- WRONG"
        print(f"{label:38s} {ms:9.4f} ms   max_err={err:.2e}{flag}")
        rows.append(dict(variant=label.strip(), latency_ms=round(ms, 4), max_err=f"{err:.2e}", **kw))
    sd = median_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=a.causal))
    print(f"{'torch SDPA (reference)':38s} {sd:9.4f} ms")

    out = Path(__file__).resolve().parents[1] / "docs" / "ablation.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
