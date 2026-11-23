// Stage 2: same three-step attention, but QK^T and PV use a shared-memory
// tiled batched GEMM. The softmax kernel is shared with naive.cu.
//
// Memory-hierarchy lesson: in the naive kernels each thread re-reads its whole
// row/column from global memory (D or N loads per output). Here a 32x32 thread
// block cooperatively stages a 32x32 tile of A and of B into shared memory
// once, and every element is then reused 32 times from shared memory.
//
//   C[M,Nn] = alpha * A[M,K] * op(B)       (batched over blockIdx.z)
//   TB=true : B is stored [Nn,K] and op(B) = B^T   (used for Q K^T)
//   TB=false: B is stored [K,Nn]                   (used for P V)
#include <cmath>

#include "common.h"

namespace {

constexpr int T = 32;

template <bool TB>
__global__ void bgemm(const float* A, const float* B, float* C, int M, int Nn, int K,
                      float alpha, size_t sA, size_t sB, size_t sC) {
  // +1 column of padding: the transposed store into Bs and the column reads
  // of As/Bs would otherwise hit the same banks (32-way conflict).
  __shared__ float As[T][T + 1];
  __shared__ float Bs[T][T + 1];

  A += blockIdx.z * sA;
  B += blockIdx.z * sB;
  C += blockIdx.z * sC;

  // tx -> output column (fast axis), ty -> output row.
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;

  float acc = 0.f;
  for (int t = 0; t < K; t += T) {
    // A tile: consecutive tx read consecutive k -> coalesced.
    As[ty][tx] = (row < M && t + tx < K) ? A[(size_t)row * K + t + tx] : 0.f;
    if (TB) {
      // B stored [Nn,K]: to stay coalesced, tx walks k (contiguous in memory)
      // and ty walks n; we store transposed so Bs[k][n].
      int n = blockIdx.x * T + ty;
      Bs[tx][ty] = (n < Nn && t + tx < K) ? B[(size_t)n * K + t + tx] : 0.f;
    } else {
      // B stored [K,Nn]: tx walks n (contiguous) -> coalesced, Bs[k][n].
      Bs[ty][tx] = (t + ty < K && col < Nn) ? B[(size_t)(t + ty) * Nn + col] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);  // As: broadcast, Bs: stride-1
    __syncthreads();
  }
  if (row < M && col < Nn) C[(size_t)row * Nn + col] = alpha * acc;
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

void tiled_forward(const float* q, const float* k, const float* v, float* S,
                   float* out, int BH, int N, int D, bool causal,
                   cudaStream_t stream) {
  float scale = 1.f / sqrtf((float)D);
  dim3 blk(T, T);
  // S[N,N] = scale * Q[N,D] * K[N,D]^T
  dim3 g1(cdiv(N, T), cdiv(N, T), BH);
  bgemm<true><<<g1, blk, 0, stream>>>(q, k, S, N, N, D, scale, (size_t)N * D, (size_t)N * D,
                                      (size_t)N * N);
  launch_softmax(S, BH, N, causal, stream);
  // O[N,D] = P[N,N] * V[N,D]
  dim3 g2(cdiv(D, T), cdiv(N, T), BH);
  bgemm<false><<<g2, blk, 0, stream>>>(S, v, out, N, D, N, 1.f, (size_t)N * N, (size_t)N * D,
                                       (size_t)N * D);
}
