#!/usr/bin/env python
"""Build the vendored detectron2 extension for the local machine.

The upstream repository ships a prebuilt detectron2._C extension for
x86_64/Python 3.9. This setup file rebuilds the extension in place for the
current interpreter, CPU architecture, and CUDA toolkit.
"""

import glob
import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


THIS_DIR = Path(__file__).resolve().parent


def get_extensions():
    extensions_dir = THIS_DIR / "detectron2" / "layers" / "csrc"
    main_source = extensions_dir / "vision.cpp"
    sources = [str(main_source)]
    sources += glob.glob(str(extensions_dir / "**" / "*.cpp"), recursive=True)
    sources = list(dict.fromkeys(sources))

    cuda_sources = glob.glob(str(extensions_dir / "**" / "*.cu"), recursive=True)
    extension = CppExtension
    define_macros = []
    extra_compile_args = {"cxx": ["-std=c++17"]}

    if (
        torch.cuda.is_available()
        and CUDA_HOME is not None
        and Path(CUDA_HOME).is_dir()
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += cuda_sources
        define_macros.append(("WITH_CUDA", None))
        extra_compile_args["nvcc"] = [
            "-std=c++17",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        cc = os.environ.get("CC")
        if cc:
            extra_compile_args["nvcc"].append(f"-ccbin={cc}")

    return [
        extension(
            "detectron2._C",
            sources,
            include_dirs=[str(extensions_dir)],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="idm-vton-gradio-demo",
    version="0.1.0",
    packages=find_packages(where=str(THIS_DIR)),
    package_dir={"": str(THIS_DIR)},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
