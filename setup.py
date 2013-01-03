import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

qpbo_directory = "qpbo_src"

files = ["QPBO.cpp", "QPBO_extra.cpp", "QPBO_maxflow.cpp",
         "QPBO_postprocessing.cpp"]

files = [os.path.join(qpbo_directory, f) for f in files]
files.insert(0, "pyqpbo.pyx")

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("pyqpbo", files, language="c++",
                             include_dirs=[qpbo_directory, np.get_include()],
                             library_dirs=[qpbo_directory],
                             extra_compile_args=["-fpermissive"])])
