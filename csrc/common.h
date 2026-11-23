// Shared declarations for the host-side launchers defined in the .cu files.
//
// Layout convention everywhere: Q, K, V, O are contiguous [B, H, N, D]; we
// flatten (B, H) into one "BH" batch index since every (b, h) pair is an
// independent attention problem.
#pragma once
#include <cuda_runtime.h>

// ---- Stage 1: naive (three separate kernels, materialises S = N x N) -------
// S is a caller-provided workspace of BH*N*N floats (allocated through PyTorch
// so that its size shows up in torch.cuda.max_memory_allocated).
void naive_forward(const float* q, const float* k, const float* v, float* S,
                   float* out, int BH, int N, int D, bool causal,
                   cudaStream_t stream);

// Row-wise softmax in place over S[BH, N, N]; shared by naive and tiled.
void launch_softmax(float* S, int BH, int N, bool causal, cudaStream_t stream);

// ---- Stage 2: same three-step structure but with shared-memory tiled GEMMs -
void tiled_forward(const float* q, const float* k, const float* v, float* S,
                   float* out, int BH, int N, int D, bool causal,
                   cudaStream_t stream);

// ---- Stage 3/4: fused FlashAttention-style kernel --------------------------
// dtype: 0 = fp32, 1 = fp16. Optimisation knobs (see fused.cu):
//   pad   : pad the K smem tile row by 1 float (removes bank conflicts)
//   vec   : 16-byte vectorised global loads of the K/V tiles
//   fast_exp : fold log2(e) into the Q scale and use exp2f
//   warps : warps (= query rows) per thread block, 4 or 8
void fused_forward(const void* q, const void* k, const void* v, void* out,
                   int dtype, int BH, int N, int D, bool causal, bool pad,
                   bool vec, bool fast_exp, int warps, cudaStream_t stream);
