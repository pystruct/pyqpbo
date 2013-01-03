from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("pyqpbo", ["pyqpbo.pyx"], language="c++",
                             include_dirs=["qpbo_src", np.get_include()],
                             libraries=["qpbo"], library_dirs=["."])])
