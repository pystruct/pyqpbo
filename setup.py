import os
import tarfile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import urllib

# fetch and unpack the archive. Not the nicest way...

urllib.urlretrieve("http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz",
                   "QPBO-v1.3.src.tar.gz")
tfile = tarfile.open("QPBO-v1.3.src.tar.gz", 'r:gz')
tfile.extractall('.')

qpbo_directory = "QPBO-v1.3.src"

extensions = Extension('pyqpbo', 
                       ['pyqpbo.pyx'], 
                       language='c++', 
                       include_dirs=[qpbo_directory, np.get_include()],
                       library_dirs=[qpbo_directory],
                       extra_compile_args=["-fpermissive"])

setup(ext_modules = cythonize([extensions]))
