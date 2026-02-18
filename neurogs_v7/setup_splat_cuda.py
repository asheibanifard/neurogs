from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="splat_cuda",
    ext_modules=[
        CUDAExtension(
            "splat_cuda",
            ["splat_cuda.cu"],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
