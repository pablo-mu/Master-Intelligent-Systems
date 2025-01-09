from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('max_pool_bwd', ['max_pool_bwd.pyx'],
            include_dirs = [numpy.get_include()],
            extra_compile_args=['-fopenmp', '-O3'],
            extra_link_args=['-fopenmp'])
]

setup(
    ext_modules = cythonize(extensions, compiler_directives={"language_level" : "3"}, annotate=True),
)

