import subprocess
import os
import sys
from packaging.version import parse, Version
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# package name managed by pip, which can be remove by `pip uninstall tiny_pkg`
PACKAGE_NAME = "ada_cute_fp8_flash_attention"

ext_modules = []
generator_flag = []
cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_89,code=sm_89")

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    cxx_flags = ["/O2", "/std:c++17", "/Zc:__cplusplus", "/permissive-"] + generator_flag
    nvcc_host_flags = [
        "-Xcompiler", "/Zc:__cplusplus",
        "-Xcompiler", "/permissive-",
    ]
else:
    cxx_flags = ["-O3", "-std=c++17"] + generator_flag
    nvcc_host_flags = []

# cuda module
ext_modules.append(
    CUDAExtension(
        # package name for import
        name="ada_cute_fp8_flash_attention",
        sources=[
            "csrc/attention_api.cpp",
            "csrc/flash_attention.cu",
            "csrc/flash_api.cpp",
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            # add nvcc compile flags
            "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "--use_fast_math",
                    "-lineinfo",
                    "--ptxas-options=-v",
                    "--ptxas-options=-O2",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",

                ]
                + nvcc_host_flags
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "include",
            Path(this_dir) / "deps/cutlass/include",
            Path(this_dir) / "deps/cutlass/tools/utils/include" ,
            Path(this_dir) / "deps/cutlass/examples/common" ,
            # Path(this_dir) / "some" / "thing" / "more",
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="Attention mechanism implement by CUDA",
    ext_modules=ext_modules,
    cmdclass={ "build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)




