// PyTorch bindings. Kernel code lives in the .cu files; this file only checks
// arguments, allocates outputs/workspaces through PyTorch's caching allocator
// (so torch.cuda.max_memory_allocated sees them) and launches on the current
// stream.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "common.h"

namespace {

void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4, "expected [batch, heads, seq_len, head_dim]");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v shapes must match");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, v dtypes must match");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16,
              "only float32 and float16 are supported");
}

using BaselineFn = void (*)(const float*, const float*, const float*, float*, float*, int,
                            int, int, bool, cudaStream_t);

// naive / tiled work in fp32 only: fp16 inputs are upcast, result is cast back.
torch::Tensor run_baseline(const torch::Tensor& q_, const torch::Tensor& k_,
                           const torch::Tensor& v_, bool causal, BaselineFn fn) {
  check_qkv(q_, k_, v_);
  auto q = q_.to(torch::kFloat32).contiguous();
  auto k = k_.to(torch::kFloat32).contiguous();
  auto v = v_.to(torch::kFloat32).contiguous();
  int64_t B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
  auto S = torch::empty({B, H, N, N}, q.options());  // the N x N matrix
  auto out = torch::empty_like(q);
  fn(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), S.data_ptr<float>(),
     out.data_ptr<float>(), (int)(B * H), (int)N, (int)D, causal,
     at::cuda::getCurrentCUDAStream());
  return out.to(q_.scalar_type());
}

}  // namespace

torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  return run_baseline(q, k, v, causal, naive_forward);
}

torch::Tensor tiled_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  return run_baseline(q, k, v, causal, tiled_forward);
}

// Fused kernel: no workspace, output written directly in the input dtype.
torch::Tensor fused_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                              bool pad, bool vec, bool fast_exp, int64_t warps) {
  check_qkv(q, k, v);
  int64_t D = q.size(3);
  TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128, got ", D);
  TORCH_CHECK(warps == 4 || warps == 8, "warps must be 4 or 8");
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  auto out = torch::empty_like(q);
  int64_t BH = q.size(0) * q.size(1), N = q.size(2);
  fused_forward(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                q.scalar_type() == torch::kFloat16 ? 1 : 0, (int)BH, (int)N, (int)D, causal, pad,
                vec, fast_exp, (int)warps, at::cuda::getCurrentCUDAStream());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention", &fused_attention, "Fused online-softmax attention", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("causal") = false, py::arg("pad") = false,
        py::arg("vec") = false, py::arg("fast_exp") = false, py::arg("warps") = 8);
  m.def("naive_attention", &naive_attention, "Naive 3-kernel attention (fp32 internally)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
  m.def("tiled_attention", &tiled_attention, "3-kernel attention with tiled GEMMs (fp32 internally)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
}
