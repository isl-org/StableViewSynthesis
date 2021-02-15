from setuptools import setup
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
)
import os
from pathlib import Path
import sys

import boilerplate as bp

bp_proj = bp.Project(
    name="ext",
    functions=[
        bp.Function(
            name="map_to_list_nn",
            fwd_input=[
                bp.Tensor("features", ["bs", "nv", "c", "height", "width"]),
                bp.Tensor(
                    "nbidx",
                    ["nelems"],
                    dtype="torch::kInt",
                    dispatch_dtype="int",
                ),
                bp.Tensor("srcpos", ["nelems", 2]),
                bp.Primitive("bs", "int", pass_to_kernel=False),
                bp.Primitive("nv", "int", pass_to_kernel=False),
                bp.Primitive("height", "int", pass_to_kernel=False),
                bp.Primitive("width", "int", pass_to_kernel=False),
            ],
            fwd_output=[bp.Tensor("out", ["nelems", "c"])],
            fwd_kernels=[bp.Kernel(length=["nelems"])],
            bwd_input=[
                bp.Tensor(
                    "nbidx",
                    ["nelems"],
                    dtype="torch::kInt",
                    dispatch_dtype="int",
                ),
                bp.Tensor("srcpos", ["nelems", 2]),
                bp.Primitive("bs", "int", pass_to_kernel=False),
                bp.Primitive("nv", "int", pass_to_kernel=False),
                bp.Primitive("height", "int", pass_to_kernel=False),
                bp.Primitive("width", "int", pass_to_kernel=False),
                bp.Tensor("grad_out", ["nelems", "c"]),
            ],
            bwd_output=[
                bp.Tensor(
                    "grad_features",
                    ["bs", "nv", "c", "height", "width"],
                    dtype="grad_out",
                )
            ],
            bwd_kernels=[bp.Kernel(length=["nelems"])],
        ),
        bp.Function(
            name="map_to_list_bl",
            fwd_input=[
                bp.Tensor("features", ["bs", "nv", "c", "height", "width"]),
                bp.Tensor(
                    "nbidx",
                    ["nelems"],
                    dtype="torch::kInt",
                    dispatch_dtype="int",
                ),
                bp.Tensor("srcpos", ["nelems", 2]),
                bp.Primitive("bs", "int", pass_to_kernel=False),
                bp.Primitive("nv", "int", pass_to_kernel=False),
                bp.Primitive("height", "int", pass_to_kernel=False),
                bp.Primitive("width", "int", pass_to_kernel=False),
            ],
            fwd_output=[bp.Tensor("out", ["nelems", "c"])],
            fwd_kernels=[bp.Kernel(length=["nelems"])],
            bwd_input=[
                bp.Tensor(
                    "nbidx",
                    ["nelems"],
                    dtype="torch::kInt",
                    dispatch_dtype="int",
                ),
                bp.Tensor("srcpos", ["nelems", 2]),
                bp.Primitive("bs", "int", pass_to_kernel=False),
                bp.Primitive("nv", "int", pass_to_kernel=False),
                bp.Primitive("height", "int", pass_to_kernel=False),
                bp.Primitive("width", "int", pass_to_kernel=False),
                bp.Tensor("grad_out", ["nelems", "c"]),
            ],
            bwd_output=[
                bp.Tensor(
                    "grad_features",
                    ["bs", "nv", "c", "height", "width"],
                    dtype="grad_out",
                )
            ],
            bwd_kernels=[bp.Kernel(length=["nelems"])],
        ),
        bp.Function(
            name="map_to_list_bl_seq",
            fwd_input=[
                bp.Tensor("features", ["c", "height", "width"]),
                bp.Tensor(
                    "tgtidx",
                    ["nelems_feat"],
                    dtype="torch::kInt",
                    dispatch_dtype="int",
                ),
                bp.Tensor("srcpos", ["nelems_feat", 2]),
                bp.Tensor("out", ["nelems", "c"]),
            ],
            fwd_output=[],
            fwd_kernels=[
                bp.Kernel(
                    length=["nelems_feat"], dispatch_scalar_type_name="out"
                )
            ],
        ),
        bp.Function(
            name="list_to_map",
            fwd_input=[
                bp.Tensor("features", ["nelems", "channels"]),
                bp.Tensor(
                    "tgtidx",
                    ["nelems"],
                    dtype="torch::kInt",
                    dispatch_dtype="int",
                ),
                bp.Primitive("bs", "int", pass_to_kernel=False),
                bp.Primitive("height", "int", pass_to_kernel=False),
                bp.Primitive("width", "int", pass_to_kernel=False),
            ],
            fwd_output=[
                bp.Tensor("out_sum", ["bs", "channels", "height", "width"]),
                bp.Tensor("out_mask", ["bs", 1, "height", "width"]),
            ],
            fwd_kernels=[bp.Kernel(length=["nelems"])],
            bwd_input=[
                bp.Tensor(
                    "grad_out_sum", ["bs", "channels", "height", "width"]
                ),
                bp.Tensor(
                    "tgtidx",
                    ["nelems"],
                    dtype="torch::kInt",
                    dispatch_dtype="int",
                ),
                bp.Primitive("bs", "int", pass_to_kernel=False),
                bp.Primitive("height", "int", pass_to_kernel=False),
                bp.Primitive("width", "int", pass_to_kernel=False),
            ],
            bwd_output=[
                bp.Tensor(
                    "grad_features",
                    ["nelems", "channels"],
                    dtype="grad_out_sum",
                )
            ],
            bwd_kernels=[bp.Kernel(length=["nelems"])],
        ),
    ],
)


include_dirs = [
    str(Path(".").absolute()),
    str(Path("./include").absolute()),
    str(bp_proj.out_dir),
]

cpp_args = []

# only needed for mac
if sys.platform == "darwin":
    os.environ["CC"] = "clang++"
    os.environ["CXX"] = "clang++"
    cpp_args.append("-std=c++14")
    cpp_args.append("-stdlib=libc++")

nvcc_args = [
    "-arch=sm_30",
    "-gencode=arch=compute_30,code=sm_30",
    "-gencode=arch=compute_35,code=sm_35",
    # '-gencode=arch=compute_50,code=sm_50',
    # '-gencode=arch=compute_52,code=sm_52',
    # '-gencode=arch=compute_60,code=sm_60',
    # '-gencode=arch=compute_61,code=sm_61',
    # '-gencode=arch=compute_70,code=sm_70',
    # '-gencode=arch=compute_70,code=compute_70',
]

cpp_sources = [
    # str(bp_proj.get_cpu_path()),
    "ext_cpu.cpp"
]

cuda_sources = [
    # str(bp_proj.get_cuda_path()),
    # str(bp_proj.get_kernel_path()),
    "ext_cuda.cpp",
    "ext_kernel.cu",
]

cpp_ext = CppExtension(
    f"{bp_proj.name}_cpu",
    cpp_sources,
    extra_compile_args={"cxx": cpp_args},
    extra_link_args=cpp_args,
)
ext_modules = [cpp_ext]
if torch.cuda.is_available():
    cuda_ext = CUDAExtension(
        f"{bp_proj.name}_cuda",
        cuda_sources,
        extra_compile_args={"cxx": cpp_args, "nvcc": nvcc_args},
    )
    ext_modules.append(cuda_ext)


def dummy(self, extension):
    self._add_compile_flag(extension, "-D_GLIBCXX_USE_CXX11_ABI=1")


BuildExtension._add_gnu_abi_flag_if_binary = dummy

setup(
    name=bp_proj.name,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    include_dirs=include_dirs,
)
