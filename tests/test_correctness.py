"""Correctness vs torch SDPA. Run:  pytest -q tests"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

import fused_attention_cuda as fa  # noqa: E402

IMPLS = ["naive", "tiled", "fused"]

# (batch, heads, seq_len, head_dim): includes N not a multiple of any tile size
SHAPES = [
    (1, 1, 64, 64),
    (2, 4, 128, 64),
    (1, 2, 100, 64),
    (2, 3, 257, 64),
    (1, 2, 512, 64),
    (1, 1, 64, 128),
    (2, 2, 200, 128),
    (1, 2, 512, 128),
]
TOL = {torch.float32: 1e-4, torch.float16: 1e-2}


def get_impl(name):
    return {
        "naive": fa.naive_attention,
        "tiled": getattr(fa, "tiled_attention", None),
        "fused": getattr(fa, "fused_attention", None),
    }[name]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(impl, shape, dtype, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    out = get_impl(impl)(q, k, v, causal=causal)
    # SDPA runs in the same dtype (what the spec asks for).
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert out.dtype == dtype and out.shape == q.shape
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL[dtype], f"{impl} max abs err {err:.3e} >= {TOL[dtype]}"


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("vec", [False, True])
@pytest.mark.parametrize("fast_exp", [False, True])
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_knob_combinations(pad, vec, fast_exp, warps, dtype, causal):
    """Every optimisation knob combination must stay correct (re-run after each change)."""
    torch.manual_seed(1)
    shape = (2, 2, 197, 128)  # N deliberately not a multiple of the tile/warp sizes
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    out = fa.fused_attention(q, k, v, causal, pad=pad, vec=vec, fast_exp=fast_exp, warps=warps)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL[dtype], f"max abs err {err:.3e}"


def test_fused_does_not_materialise_scores():
    """Peak extra memory of the fused kernel should be ~ the output tensor, not N*N."""
    B, H, N, D = 1, 4, 4096, 64
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fa.fused_attention(q, k, v)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    assert extra < 4 * q.numel() * q.element_size()  # far below B*H*N*N*4 bytes
