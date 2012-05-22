from distutils.core import setup
from numpy.distutils.core import setup as npsetup
from distutils.extension import Extension
from Cython.Distutils import build_ext

qpbo_dir = "../../tools/QPBO-v1.3.src/"


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('qpbo', parent_package, top_path)

    sources = [qpbo_dir + f for f in ["QPBO.cpp", "QPBO.cpp",
        "QPBO_maxflow.cpp", "QPBO_postprocessing.cpp", "QPBO_extra.cpp"]]
    config.add_extension('libqpbo', sources=sources, language='c++')
    return config

if __name__ == '__main__':
    # first we build the gco library
    npsetup(**configuration(top_path='').todict())

    # then we build our cython extension
    setup(cmdclass={'build_ext': build_ext},
        ext_modules=[Extension("pyqpbo", ['pyqpbo.pyx'], language="c++",
        include_dirs=[qpbo_dir], libraries=['qpbo'], library_dirs=['qpbo'])])
