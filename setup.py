from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

qpbo_dir = "../../tools/QPBO-v1.3.src/"
files = [qpbo_dir + f for f in ["QPBO.cpp", "QPBO.cpp", "QPBO_maxflow.cpp",
    "QPBO_postprocessing.cpp", "QPBO_extra.cpp"]]

files.insert(0, "pyqpbo.pyx")

setup(cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("pyqpbo", files,
        language="c++", include_dirs=["../../tools/QPBO-v1.3.src/"])])
