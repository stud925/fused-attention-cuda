"""Tiny driver for Nsight Compute / Nsight Systems: runs ONE implementation a few times.

    python benchmarks/profile_target.py --impl fused --seq 2048 [--no-pad --no-vec --no-fast-exp --warps 4]
    python benchmarks/profile_target.py --impl naive --seq 1024

See docs/profile.sh for the ncu / nsys command lines.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import fused_attention_cuda as fa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["naive", "tiled", "fused"], required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--no-vec", action="store_true")
    ap.add_argument("--no-fast-exp", action="store_true")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()

    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    q, k, v = (torch.randn(a.batch, a.heads, a.seq, a.head_dim, device="cuda", dtype=dtype)
               for _ in range(3))
    if a.impl == "fused":
        fn = lambda: fa.fused_attention(q, k, v, False, pad=not a.no_pad, vec=not a.no_vec,
                                        fast_exp=not a.no_fast_exp, warps=a.warps)
    else:
        fn = lambda: getattr(fa, f"{a.impl}_attention")(q, k, v)
    fn()  # warm-up (also triggers any lazy init) -- skip it in ncu with --launch-skip
    torch.cuda.synchronize()
    for _ in range(a.iters):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
