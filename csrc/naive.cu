// Stage 1: naive attention baseline.
//
//   S = (Q K^T) * 1/sqrt(D)      kernel 1  (one thread per S element)
//   P = softmax_row(S)           kernel 2  (one block per row)
//   O = P V                      kernel 3  (one thread per O element)
//
// Deliberately simple and un-optimised: every thread reads its operands
// straight from global memory, and S (BH x N x N floats) lives in global
// memory between kernels. This is the correctness reference for the fancier
// kernels and the baseline for speedup / memory claims.
#include <cfloat>
#include <cmath>

#include "common.h"

namespace {

constexpr int TILE = 16;

// Thread mapping: x -> key index j (fast axis), y -> query index i.
// Writes S[bh, i, j] are coalesced across x. Reads of Q[i, :] are the same
// address for all threads with equal y (broadcast); reads of K[j, :] are
// strided by D across x (uncoalesced) -- this is one thing the tiled stage
// fixes.
__global__ void qk_naive(const float* Q, const float* K, float* S, int N, int D,
                         float scale) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t bh = blockIdx.z;
  if (i >= N || j >= N) return;
  const float* q = Q + (bh * N + i) * D;
  const float* k = K + (bh * N + j) * D;
  float acc = 0.f;
  for (int d = 0; d < D; ++d) acc += q[d] * k[d];
  S[(bh * N + i) * N + j] = acc * scale;
}

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// Block-wide reductions through a 32-float shared scratch (one slot per warp).
template <bool IS_MAX>
__device__ float block_reduce(float v, float* red) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = IS_MAX ? warp_max(v) : warp_sum(v);
  __syncthreads();  // red[] may still be in use from a previous reduction
  if (lane == 0) red[warp] = v;
  __syncthreads();
  int nw = (blockDim.x + 31) >> 5;
  v = lane < nw ? red[lane] : (IS_MAX ? -FLT_MAX : 0.f);
  return IS_MAX ? warp_max(v) : warp_sum(v);
}

// One block per (bh, i) row, in place. Three passes over the row: max, exp+sum,
// normalise. For causal, entries j > i are never read and are written as 0.
__global__ void softmax_rows(float* S, int N, bool causal) {
  size_t row = blockIdx.x;
  int i = (int)(row % N);
  float* r = S + row * N;
  int lim = causal ? i + 1 : N;
  __shared__ float red[32];

  float m = -FLT_MAX;
  for (int j = threadIdx.x; j < lim; j += blockDim.x) m = fmaxf(m, r[j]);
  m = block_reduce<true>(m, red);

  float s = 0.f;
  for (int j = threadIdx.x; j < lim; j += blockDim.x) {
    float e = expf(r[j] - m);
    r[j] = e;
    s += e;
  }
  s = block_reduce<false>(s, red);

  float inv = 1.f / s;
  for (int j = threadIdx.x; j < N; j += blockDim.x) r[j] = j < lim ? r[j] * inv : 0.f;
}

// Thread mapping: x -> output feature d (fast axis), y -> query index i.
// V[j, d] reads are coalesced across x; P[i, j] is a broadcast across x.
__global__ void pv_naive(const float* P, const float* V, float* O, int N, int D) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t bh = blockIdx.z;
  if (i >= N || d >= D) return;
  const float* p = P + (bh * N + i) * N;
  const float* v = V + bh * N * D;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) acc += p[j] * v[(size_t)j * D + d];
  O[(bh * N + i) * D + d] = acc;
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

void launch_softmax(float* S, int BH, int N, bool causal, cudaStream_t stream) {
  softmax_rows<<<(unsigned)((size_t)BH * N), 256, 0, stream>>>(S, N, causal);
}

void naive_forward(const float* q, const float* k, const float* v, float* S,
                   float* out, int BH, int N, int D, bool causal,
                   cudaStream_t stream) {
  float scale = 1.f / sqrtf((float)D);
  dim3 blk(TILE, TILE);
  dim3 g1(cdiv(N, TILE), cdiv(N, TILE), BH);
  qk_naive<<<g1, blk, 0, stream>>>(q, k, S, N, D, scale);
  launch_softmax(S, BH, N, causal, stream);
  dim3 g3(cdiv(D, TILE), cdiv(N, TILE), BH);
  pv_naive<<<g3, blk, 0, stream>>>(S, v, out, N, D);
}
