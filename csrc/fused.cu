// Stages 3-4: fused FlashAttention-style forward kernel with online softmax.
//
// ---------------------------------------------------------------------------
// Why no N x N matrix
// ---------------------------------------------------------------------------
// For one query row q_i we need  o_i = sum_j softmax_j(s_ij) v_j,  s_ij = q_i.k_j/sqrt(D).
// Softmax needs the row max and row sum, which naively requires all of s_i*
// first. Online softmax instead streams over K/V in tiles keeping, per row,
//   m : running max of scores seen so far
//   l : running sum of exp(s - m)
//   o : running un-normalised output  sum exp(s - m) v
// When a new tile raises the max from m to m', everything accumulated so far is
// rescaled by alpha = exp(m - m'):   l <- l*alpha + sum exp(s-m'),  o <- o*alpha + sum exp(s-m') v.
// After the last tile o / l is exactly softmax(s) V. Scores only ever exist in
// registers for one 32-key tile, so global memory traffic is O(N*D), not O(N^2).
//
// ---------------------------------------------------------------------------
// Thread mapping  (block = W warps, grid = (ceil(N/W), BH))
// ---------------------------------------------------------------------------
// * One WARP owns one query row (so one block owns W consecutive rows and all
//   W warps share the same K/V tiles from shared memory).
// * K/V are processed in tiles of BC = 32 keys.
// * Score phase: LANE l computes the score for key (kv0 + l): a D-long dot
//   product of the row's q (shared memory, broadcast: all lanes read the same
//   address) with K-tile row l. The 32 scores of the tile are therefore spread
//   one per lane, so the row max / sum are plain warp shuffle reductions.
// * PV phase: lane l owns output features d = l, l+32, ... (E = D/32 of them,
//   kept in registers). For each key j the probability p_j is broadcast from
//   lane j with __shfl_sync and every lane FMAs it into its features using
//   V-tile row j (consecutive lanes -> consecutive addresses: conflict free).
//
// ---------------------------------------------------------------------------
// Optimisation knobs (stage 4) -- template parameters so each can be toggled
// and profiled in isolation:
//   PAD  : K-tile row stride D+1 instead of D. In the score phase lane l reads
//          Ks[l*stride + d]; with stride D (multiple of 32) all 32 lanes hit
//          the same bank (32-way conflict). stride D+1 spreads them over all
//          banks.
//   VEC  : load K/V tiles with 16-byte (uint4) global loads instead of scalar
//          loads: 4x (fp32) / 8x (fp16) fewer load instructions.
//   EXP2 : fold log2(e) into the Q scaling so softmax uses exp2f (a single
//          hardware EX2 instruction) instead of expf (range reduction + EX2).
//   W    : warps per block (rows per block / K-V tile reuse vs occupancy).
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <type_traits>

#include "common.h"

namespace {

constexpr int BC = 32;  // keys per tile == warp size (one score per lane)
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

template <typename T, int D, int W, bool PAD, bool VEC, bool EXP2>
__global__ void __launch_bounds__(W * 32)
    fused_fwd(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
              T* __restrict__ O, int N, bool causal, float scale) {
  constexpr int KS = D + (PAD ? 1 : 0);  // K tile row stride (floats)
  constexpr int E = D / 32;              // output features per lane
  constexpr int NT = W * 32;

  // All tiles live in fp32 shared memory regardless of T (accumulate in fp32).
  __shared__ float Ks[BC * KS];
  __shared__ float Vs[BC * D];
  __shared__ float Qs[W * D];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const size_t base = (size_t)blockIdx.y * N * D;  // start of this (b,h)
  const int row = blockIdx.x * W + warp;           // query row owned by this warp
  const bool row_ok = row < N;                     // rows >= N still join barriers/shuffles

  // Fold 1/sqrt(D) (and log2 e for the exp2 path) into q once.
  const float c = EXP2 ? scale * LOG2E : scale;
  for (int d = lane; d < D; d += 32)
    Qs[warp * D + d] = row_ok ? to_f(Q[base + (size_t)row * D + d]) * c : 0.f;

  float o[E];
#pragma unroll
  for (int e = 0; e < E; ++e) o[e] = 0.f;
  float m = -INFINITY, l = 0.f;

  // Causal: this block never needs keys beyond its last row.
  const int row_hi = min(N, (int)(blockIdx.x + 1) * W);
  const int kv_end = causal ? row_hi : N;  // uniform across the block

  for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
    __syncthreads();  // previous tile fully consumed before overwrite

    // ---- cooperative K/V tile load (global -> fp32 shared) ---------------
    if constexpr (VEC) {
      constexpr int VN = 16 / sizeof(T);  // elements per 16-byte load
      constexpr int RV = D / VN;          // vectors per row
      for (int idx = tid; idx < BC * RV; idx += NT) {
        int r = idx / RV, cc = (idx % RV) * VN, j = kv0 + r;
        if (j < N) {
          uint4 kk = *reinterpret_cast<const uint4*>(K + base + (size_t)j * D + cc);
          uint4 vv = *reinterpret_cast<const uint4*>(V + base + (size_t)j * D + cc);
          const T* kp = reinterpret_cast<const T*>(&kk);
          const T* vp = reinterpret_cast<const T*>(&vv);
#pragma unroll
          for (int i = 0; i < VN; ++i) {
            Ks[r * KS + cc + i] = to_f(kp[i]);
            Vs[r * D + cc + i] = to_f(vp[i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < VN; ++i) {
            Ks[r * KS + cc + i] = 0.f;
            Vs[r * D + cc + i] = 0.f;
          }
        }
      }
    } else {
      for (int idx = tid; idx < BC * D; idx += NT) {
        int r = idx / D, cc = idx % D, j = kv0 + r;
        Ks[r * KS + cc] = j < N ? to_f(K[base + (size_t)j * D + cc]) : 0.f;
        Vs[r * D + cc] = j < N ? to_f(V[base + (size_t)j * D + cc]) : 0.f;
      }
    }
    __syncthreads();

    // ---- scores: lane l <-> key kv0 + l ------------------------------------
    const int j = kv0 + lane;
    const bool valid = row_ok && j < N && (!causal || j <= row);
    float acc = 0.f;
    {
      const float* kr = &Ks[lane * KS];
      const float* qr = &Qs[warp * D];
#pragma unroll
      for (int d = 0; d < D; ++d) acc = fmaf(qr[d], kr[d], acc);
    }
    const float s = valid ? acc : -INFINITY;

    // ---- online softmax update (all lanes hold identical m, l) -------------
    const float m_new = fmaxf(m, warp_max(s));
    // If nothing valid has been seen yet m_new = -inf; use 0 as the shift to
    // avoid (-inf) - (-inf) = NaN. exp(-inf) = 0 then gives alpha = p = 0.
    const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
    float alpha, p;
    if constexpr (EXP2) {
      alpha = exp2f(m - m_use);
      p = exp2f(s - m_use);
    } else {
      alpha = expf(m - m_use);
      p = expf(s - m_use);
    }
    l = l * alpha + warp_sum(p);
    m = m_new;

    // ---- o = o*alpha + P_tile @ V_tile -------------------------------------
#pragma unroll
    for (int e = 0; e < E; ++e) o[e] *= alpha;
#pragma unroll
    for (int jj = 0; jj < BC; ++jj) {
      const float pj = __shfl_sync(0xffffffff, p, jj);  // broadcast p of key jj
      const float* vr = &Vs[jj * D + lane];
#pragma unroll
      for (int e = 0; e < E; ++e) o[e] = fmaf(pj, vr[e * 32], o[e]);
    }
  }

  if (row_ok) {
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e)
      O[base + (size_t)row * D + e * 32 + lane] = from_f<T>(o[e] * inv);
  }
}

template <typename T, int D, int W, bool PAD, bool VEC, bool EXP2>
void launch(const void* q, const void* k, const void* v, void* out, int BH, int N, bool causal,
            cudaStream_t stream) {
  dim3 grid((N + W - 1) / W, BH);
  fused_fwd<T, D, W, PAD, VEC, EXP2><<<grid, W * 32, 0, stream>>>(
      static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v),
      static_cast<T*>(out), N, causal, 1.f / sqrtf((float)D));
}

template <typename F>
void bsw(bool b, F&& f) {
  if (b) f(std::true_type{});
  else f(std::false_type{});
}

// Runtime flags -> compile-time template arguments.
template <typename T, int D>
void dispatch(const void* q, const void* k, const void* v, void* out, int BH, int N, bool causal,
              bool pad, bool vec, bool fast_exp, int warps, cudaStream_t stream) {
  bsw(pad, [&](auto P) {
    bsw(vec, [&](auto Vc) {
      bsw(fast_exp, [&](auto X) {
        constexpr bool p_ = decltype(P)::value, v_ = decltype(Vc)::value, x_ = decltype(X)::value;
        if (warps == 4) launch<T, D, 4, p_, v_, x_>(q, k, v, out, BH, N, causal, stream);
        else launch<T, D, 8, p_, v_, x_>(q, k, v, out, BH, N, causal, stream);
      });
    });
  });
}

}  // namespace

void fused_forward(const void* q, const void* k, const void* v, void* out, int dtype, int BH,
                   int N, int D, bool causal, bool pad, bool vec, bool fast_exp, int warps,
                   cudaStream_t stream) {
  auto go = [&](auto tag) {
    using T = decltype(tag);
    if (D == 64) dispatch<T, 64>(q, k, v, out, BH, N, causal, pad, vec, fast_exp, warps, stream);
    else dispatch<T, 128>(q, k, v, out, BH, N, causal, pad, vec, fast_exp, warps, stream);
  };
  if (dtype == 0) go(float{});
  else go(__half{});
}
