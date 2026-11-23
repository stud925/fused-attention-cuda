"""Python wrapper around the CUDA extension.

If the compiled extension (`pip install -e .`) is not present, it is built on
the fly with torch.utils.cpp_extension.load (slow the first time, cached after).
"""
from pathlib import Path

try:
    from . import _C
except ImportError:  # not installed -> JIT build from ../../csrc
    from torch.utils.cpp_extension import load

    _csrc = Path(__file__).resolve().parents[2] / "csrc"
    _C = load(
        name="fused_attention_cuda_jit",
        sources=[str(p) for p in sorted(_csrc.glob("*.cpp")) + sorted(_csrc.glob("*.cu"))],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
        verbose=True,
    )


def naive_attention(q, k, v, causal=False):
    """Three-kernel baseline; materialises the full N x N score matrix."""
    return _C.naive_attention(q, k, v, causal)


def fused_attention(q, k, v, causal=False, *, pad=True, vec=True, fast_exp=True, warps=8):
    """Fused FlashAttention-style forward (online softmax, no N x N matrix).

    q, k, v: [batch, heads, seq_len, head_dim], fp32 or fp16, head_dim 64 or 128.
    pad / vec / fast_exp / warps are the stage-4 optimisation knobs (see
    csrc/fused.cu); they change speed, not results.
    """
    return _C.fused_attention(q, k, v, causal, pad, vec, fast_exp, warps)


def tiled_attention(q, k, v, causal=False):
    """Same three-step structure as naive, with shared-memory tiled GEMMs."""
    return _C.tiled_attention(q, k, v, causal)
