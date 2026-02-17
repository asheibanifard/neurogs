from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture from environment or use common defaults
cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '7.0 7.5 8.0 8.6')

setup(
    name='gaussian_eval_cuda',
    ext_modules=[
        CUDAExtension(
            name='gaussian_eval_cuda',
            sources=['gaussian_eval_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
