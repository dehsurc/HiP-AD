import os
import site

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=[],
    extra_args=[],
    extra_include_path=[],
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}
    include_dirs = list(extra_include_path)

    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        include_dirs.extend(
            [
                os.path.join(cuda_home, "include"),
                os.path.join(cuda_home, "targets", "x86_64-linux", "include"),
            ]
        )

    for site_dir in site.getsitepackages():
        include_dirs.extend(
            [
                os.path.join(site_dir, "nvidia", "cuda_runtime", "include"),
                os.path.join(site_dir, "nvidia", "cublas", "include"),
                os.path.join(site_dir, "nvidia", "cusparse", "include"),
                os.path.join(site_dir, "nvidia", "cusolver", "include"),
                os.path.join(site_dir, "nvidia", "curand", "include"),
            ]
        )

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=[p for p in include_dirs if os.path.isdir(p)],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="deformable_aggregation_ext",
        ext_modules=[
            make_cuda_ext(
                "deformable_aggregation_ext",
                module=".",
                sources=[
                    f"src/deformable_aggregation.cpp",
                    f"src/deformable_aggregation_cuda.cu",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
