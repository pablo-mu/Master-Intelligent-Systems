from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('col2im_cython', ['col2im_cython.pyx'],
            include_dirs = [numpy.get_include()],
            extra_compile_args=['-fopenmp', '-O3'],
            extra_link_args=['-fopenmp'])
]

setup(
    ext_modules = cythonize(extensions, compiler_directives={"language_level" : "3"}, annotate=True),
)

