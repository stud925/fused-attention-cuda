"""Latency + peak-memory sweep over sequence length: naive vs tiled vs fused vs torch SDPA.

    python benchmarks/bench.py                       # defaults, writes docs/bench.csv + PNGs
    python benchmarks/bench.py --dtype fp32 --heads 8 --batch 4

Latency: CUDA events, warmup runs, median of --iters runs.
Memory : peak torch.cuda.max_memory_allocated during the call, minus what was
         already allocated (the q/k/v inputs), i.e. the *extra* memory the op needs.
Rows that run out of memory are recorded with empty values (not invented).
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

IMPLS = {
    "naive": fa.naive_attention,
    "tiled": fa.tiled_attention,
    "fused": fa.fused_attention,
    "sdpa": lambda q, k, v, causal=False: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
}


def measure(fn, q, k, v, causal, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v, causal=causal)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(q, k, v, causal=causal)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def peak_extra_mem(fn, q, k, v, causal):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn(q, k, v, causal=causal)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--seq-min", type=int, default=128)
    ap.add_argument("--seq-max", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--impls", nargs="+", default=list(IMPLS))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "bench.csv"))
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    seqs = []
    n = a.seq_min
    while n <= a.seq_max:
        seqs.append(n)
        n *= 2

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}  cuda {torch.version.cuda}")
    rows = []
    for N in seqs:
        for name in a.impls:
            fn = IMPLS[name]
            row = dict(impl=name, seq_len=N, batch=a.batch, heads=a.heads, head_dim=a.head_dim,
                       dtype=a.dtype, causal=a.causal, latency_ms="", extra_mem_mb="")
            try:
                torch.manual_seed(0)
                q, k, v = (torch.randn(a.batch, a.heads, N, a.head_dim, device="cuda", dtype=dtype)
                           for _ in range(3))
                row["extra_mem_mb"] = round(peak_extra_mem(fn, q, k, v, a.causal) / 2**20, 2)
                row["latency_ms"] = round(measure(fn, q, k, v, a.causal, a.warmup, a.iters), 4)
            except torch.cuda.OutOfMemoryError:
                print(f"  {name:6s} N={N}: OOM")
            finally:
                q = k = v = None
                torch.cuda.empty_cache()
            print(f"  {name:6s} N={N:5d}  {row['latency_ms']!s:>10} ms  {row['extra_mem_mb']!s:>10} MB")
            rows.append(row)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)
    if not a.no_plot:
        plot(rows, Path(a.out).parent, a)


def plot(rows, outdir, a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"B={a.batch} H={a.heads} D={a.head_dim} {a.dtype}" + (" causal" if a.causal else "")
    for key, ylabel, fname in [("latency_ms", "median latency (ms)", "latency_vs_seq.png"),
                               ("extra_mem_mb", "extra peak memory (MiB)", "memory_vs_seq.png")]:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for name in dict.fromkeys(r["impl"] for r in rows):
            pts = [(r["seq_len"], r[key]) for r in rows if r["impl"] == name and r[key] != ""]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)
    print("wrote plots to", outdir)


if __name__ == "__main__":
    main()
