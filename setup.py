import os
import tarfile
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import numpy as np
import urllib


class QPBOInstall(build_ext):
    def run(self):
        # fetch and unpack the archive. Not the nicest way...
        urllib.urlretrieve("http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz",
                           "QPBO-v1.3.src.tar.gz")
        tfile = tarfile.open("QPBO-v1.3.src.tar.gz", 'r:gz')
        tfile.extractall('.')
        build_ext.run(self)


qpbo_directory = "QPBO-v1.3.src"

files = ["QPBO.cpp", "QPBO_extra.cpp", "QPBO_maxflow.cpp",
         "QPBO_postprocessing.cpp"]

files = [os.path.join(qpbo_directory, f) for f in files]
files.insert(0, "src/pyqpbo.cpp")

setup(name='pyqpbo',
      packages=['pyqpbo'],
      version="0.1",
      author="Andreas Mueller",
      author_email="t3kcit@gmail.com",
      url="http://pystruct.github.io",
      cmdclass={"build_ext": QPBOInstall},
      ext_modules=[
          Extension('pyqpbo.pyqpbo', sources=files, language='c++',
                    include_dirs=[qpbo_directory, np.get_include()],
                    library_dirs=[qpbo_directory],
                    extra_compile_args=["-fpermissive"])])
