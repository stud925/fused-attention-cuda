"""Build with:  pip install -e . --no-build-isolation

Set TORCH_CUDA_ARCH_LIST (e.g. "12.1" for DGX Spark / GB10) to pick the target
architecture; if unset PyTorch builds for the GPUs it can see.
"""
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    "csrc/bindings.cpp",
    "csrc/naive.cu",
    "csrc/tiled.cu",
    "csrc/fused.cu",
]

setup(
    name="fused-attention-cuda",
    version="0.1.0",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CUDAExtension(
            "fused_attention_cuda._C",
            sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-lineinfo"],  # lineinfo: source view in ncu
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
