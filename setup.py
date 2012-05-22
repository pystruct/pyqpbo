from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("pyqpbo", ["pyqpbo.pyx", "QPBO.cpp"],
        language="c++", include_dirs=["../../tools/QPBO-v1.3.src/"])])
