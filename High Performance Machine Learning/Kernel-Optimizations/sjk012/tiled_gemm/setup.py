from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('tiled_gemm', ['tiled_gemm.pyx'],
            include_dirs = [numpy.get_include()],
            extra_compile_args=['-fopenmp', '-O3'],
            extra_link_args=['-fopenmp'])
]

setup(
    ext_modules = cythonize(extensions, annotate=True),
)
