# fused-attention-cuda

A FlashAttention-style fused attention **forward** kernel written from scratch in
CUDA C++, exposed as a PyTorch op, and benchmarked honestly against a naive
three-kernel baseline and PyTorch's `scaled_dot_product_attention` (SDPA).
This is a learning/portfolio project: clarity, correctness and honest profiling
matter more than beating SDPA.

> **Status:** code complete through stage 5 (see below) but **not yet built or
> run** — it was written without GPU access. The results, versions and charts
> sections are placeholders until the commands below have been run on real
> hardware. Nothing here is a measured number yet.

## What it computes

`O = softmax(Q Kᵀ / √d) V` for `Q, K, V` of shape `[batch, heads, seq_len, head_dim]`,
`head_dim ∈ {64, 128}`, fp32 or fp16 (fp32 accumulation), optional causal mask.

## Why online softmax avoids the N×N matrix

The naive implementation computes the full score matrix `S = QKᵀ` (`N×N` per
head), writes it to global memory, reads it back for softmax, writes `P`, and
reads it again for `PV`. Memory is O(N²) and the kernel is bound by traffic to
that matrix.

Softmax only needs, per row, the max `m` and the sum `l = Σ exp(s − m)`. Both can
be maintained *incrementally* while streaming over tiles of K/V: when a new tile
raises the max from `m` to `m'`, the running sum and the running output are
rescaled by `exp(m − m')`. After the last tile `o / l` equals `softmax(S) V`
exactly. The scores of one 32-key tile live in registers and are discarded, so
extra memory is O(N·d) (just Q, K, V, O) instead of O(N²). Details and the exact
thread mapping are in the header comment of [`csrc/fused.cu`](csrc/fused.cu).

## Layout

```
csrc/naive.cu     stage 1  three separate kernels (QKᵀ, softmax, PV)
csrc/tiled.cu     stage 2  shared-memory tiled GEMMs, coalesced loads
csrc/fused.cu     stage 3-4 fused online-softmax kernel + optimisation knobs
csrc/bindings.cpp PyTorch bindings
python/fused_attention_cuda/   wrapper: naive_attention, tiled_attention, fused_attention
tests/            pytest correctness vs SDPA (fp32 1e-4, fp16 1e-2)
benchmarks/       bench.py (sweep -> CSV + plots), ablate.py, profile_target.py
docs/             profiling.md, profile.sh, bench.csv + charts (after you run them)
```

## Build and run

Requirements: NVIDIA GPU, CUDA toolkit with `nvcc` (must support your GPU's
architecture — e.g. DGX Spark / GB10 is `sm_121`, needs CUDA ≥ 12.9), a CUDA
build of PyTorch for your platform, `pytest`, `matplotlib`.

```bash
export TORCH_CUDA_ARCH_LIST="12.1"            # your GPU's compute capability
pip install -e . --no-build-isolation          # or skip: it JIT-builds on first import
pytest -q tests                                # correctness first
python benchmarks/ablate.py                    # optimisation ablation
python benchmarks/bench.py                     # sweep N=128..8192 -> docs/bench.csv + PNGs
bash docs/profile.sh                           # Nsight Compute / Systems data
```

```python
from fused_attention_cuda import fused_attention
out = fused_attention(q, k, v, causal=False)   # q,k,v: [B, H, N, D] CUDA fp16/fp32
```

## Tested environment

| | |
|---|---|
| GPU | — |
| Driver / CUDA toolkit | — |
| PyTorch | — |
| Python | — |

## Results

Not run yet. After `python benchmarks/bench.py`:

![latency](docs/latency_vs_seq.png) ![memory](docs/memory_vs_seq.png)

| seq_len | naive (ms) | tiled (ms) | fused (ms) | SDPA (ms) | naive mem (MiB) | fused mem (MiB) | SDPA mem (MiB) |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

If SDPA is faster, that is expected (it uses Tensor Cores and async copies; this
kernel uses FP32 FMAs). `docs/profiling.md` is where the profiler evidence goes.

## What I learned

_To be written after running and profiling._

## What I'd do next

- Tensor Core (WMMA/MMA) path for QKᵀ and PV — the biggest known gap to SDPA.
- Double-buffered `cp.async` tile loads to overlap memory and compute.
- More than one query row per warp / larger tiles to raise K/V reuse.
- Backward pass (recompute-based, as in FlashAttention).
