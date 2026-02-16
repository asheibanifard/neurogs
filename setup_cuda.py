"""
Setup script for compiling CUDA extensions for NeuroGS
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture for current GPU
# For RTX 4090: sm_89, For A100: sm_80, For V100: sm_70
cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '7.0;7.5;8.0;8.6;8.9;9.0')

setup(
    name='neurogs_cuda',
    ext_modules=[
        CUDAExtension(
            name='neurogs_cuda',
            sources=[
                'cuda_kernels/cuda_extension.cpp',
                'cuda_kernels/gaussian_splatting_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xptxas', '-v',
                    '--expt-relaxed-constexpr',
                    f'-gencode=arch=compute_75,code=sm_75',  # RTX 2080, Quadro RTX 8000
                    f'-gencode=arch=compute_80,code=sm_80',  # A100
                    f'-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    f'-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
