import os
import tarfile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import urllib

# fetch and unpack the archive. Not the nicest way...

urllib.urlretrieve("http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz",
                   "QPBO-v1.3.src.tar.gz")
tfile = tarfile.open("QPBO-v1.3.src.tar.gz", 'r:gz')
tfile.extractall('.')

qpbo_directory = "QPBO-v1.3.src"

files = ["QPBO.cpp", "QPBO_extra.cpp", "QPBO_maxflow.cpp",
         "QPBO_postprocessing.cpp"]

files = [os.path.join(qpbo_directory, f) for f in files]
files.insert(0, "pyqpbo.pyx")

setup(cmdclass={'build_ext': build_ext},
      requires=['Cython'],
      ext_modules=[Extension("pyqpbo", files, language="c++",
                             include_dirs=[qpbo_directory, np.get_include()],
                             library_dirs=[qpbo_directory],
                             extra_compile_args=["-fpermissive"])])
