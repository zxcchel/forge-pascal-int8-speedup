"""
setup.py — builds zimage_ext CUDA extension for sm61 (Pascal)

Usage:
    python setup.py build_ext --inplace

Or via pip (recommended):
    pip install -e .

After build, a .pyd file appears in this directory.
Import with: import zimage_ext
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# sm61 = Pascal (GTX 1060, Titan X Pascal)
# Only compile for sm61 — speeds up build significantly
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")

setup(
    name="zimage_ext",
    version="0.1.0",
    description="Fused CUDA kernels for Z-Image / Lumina NextDiT on sm61 (Pascal)",
    ext_modules=[
        CUDAExtension(
            name="zimage_ext",
            sources=[
                "zimage_bindings.cpp",
                "zimage_block.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "/O2",          # MSVC optimization
                    "/std:c++17",   # C++17
                ],
                "nvcc": [
                    "-O3",
                    "-arch=sm_61",          # Pascal only
                    "--use_fast_math",       # fast sin/cos/exp/rsqrt
                    "-lineinfo",             # debug info without -G overhead
                    "--expt-relaxed-constexpr",
                    # fp8 support
                    # suppress some nvcc warnings
                    "-Xcudafe", "--diag_suppress=186",
                    "-allow-unsupported-compiler",
                    "-Wno-deprecated-gpu-targets",
                    "--ptxas-options=-v",
                ],
            },
            include_dirs=[
                # CUDA include path — adjust if needed
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)